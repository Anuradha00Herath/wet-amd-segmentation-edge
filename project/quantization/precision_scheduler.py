"""
precision_scheduler.py
----------------------
Assigns precision (INT8 / FP16 / FP32) to ONNX node groups.

Translates a precision map (group → precision) into the
`nodes_to_exclude` list needed by ORT's quantize_static().

Note on FP16:
  ORT's CPU provider does not support native FP16 compute.
  "FP16" groups are kept at FP32 in the ONNX graph and
  ORT will run them efficiently using float32 kernels.
  True FP16 requires GPU (CUDAExecutionProvider) or NPU.
  We document this clearly so the precision map is still
  research-meaningful for hardware-aware deployment planning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.logger import get_logger
from quantization.layer_profiler import ONNX_LAYER_GROUPS

_log = get_logger(__name__)

# Supported precision levels
PRECISION_INT8 = "int8"
PRECISION_FP16 = "fp16"   # kept FP32 on CPU; true FP16 on GPU/NPU
PRECISION_FP32 = "fp32"


@dataclass
class MixedPrecisionConfig:
    """
    Maps each layer group to a target precision.

    Groups: encoder_stem, encoder_blocks, encoder_head,
            decoder, segmentation_head, other
    """
    precision_map: Dict[str, str] = field(default_factory=lambda: {
        "encoder_stem":       PRECISION_INT8,
        "encoder_blocks":     PRECISION_INT8,
        "encoder_head":       PRECISION_INT8,
        "decoder":            PRECISION_INT8,
        "segmentation_head":  PRECISION_FP32,
        "other":              PRECISION_INT8,
    })
    name: str = "mixed_precision"


# Pre-defined experiment configurations
MIXED_PRECISION_EXPERIMENTS = {
    "all_int8": MixedPrecisionConfig(
        name="all_int8",
        precision_map={g: PRECISION_INT8 for g in ONNX_LAYER_GROUPS},
    ),
    "encoder_int8_decoder_fp32": MixedPrecisionConfig(
        name="encoder_int8_decoder_fp32",
        precision_map={
            "encoder_stem":      PRECISION_INT8,
            "encoder_blocks":    PRECISION_INT8,
            "encoder_head":      PRECISION_INT8,
            "decoder":           PRECISION_FP32,
            "segmentation_head": PRECISION_FP32,
            "other":             PRECISION_INT8,
        },
    ),
    "encoder_fp32_decoder_int8": MixedPrecisionConfig(
        name="encoder_fp32_decoder_int8",
        precision_map={
            "encoder_stem":      PRECISION_FP32,
            "encoder_blocks":    PRECISION_FP32,
            "encoder_head":      PRECISION_FP32,
            "decoder":           PRECISION_INT8,
            "segmentation_head": PRECISION_INT8,
            "other":             PRECISION_INT8,
        },
    ),
    "sensitive_fp32_rest_int8": MixedPrecisionConfig(
        name="sensitive_fp32_rest_int8",
        precision_map={
            "encoder_stem":      PRECISION_FP32,   # usually most sensitive
            "encoder_blocks":    PRECISION_INT8,
            "encoder_head":      PRECISION_INT8,
            "decoder":           PRECISION_INT8,
            "segmentation_head": PRECISION_FP32,
            "other":             PRECISION_INT8,
        },
    ),
    "seghead_fp32_rest_int8": MixedPrecisionConfig(
        name="seghead_fp32_rest_int8",
        precision_map={
            "encoder_stem":      PRECISION_INT8,
            "encoder_blocks":    PRECISION_INT8,
            "encoder_head":      PRECISION_INT8,
            "decoder":           PRECISION_INT8,
            "segmentation_head": PRECISION_FP32,
            "other":             PRECISION_INT8,
        },
    ),
}


class PrecisionScheduler:
    """
    Converts a MixedPrecisionConfig into a list of ONNX nodes to exclude
    from INT8 quantization (i.e. keep at FP32).

    Usage:
        scheduler = PrecisionScheduler(mp_cfg, node_groups)
        excluded  = scheduler.get_excluded_nodes()
        # pass excluded to quantize_excluding_nodes()
    """

    def __init__(
        self,
        mp_cfg: MixedPrecisionConfig,
        node_groups: Dict[str, List[str]],
    ) -> None:
        self.mp_cfg      = mp_cfg
        self.node_groups = node_groups

    def get_excluded_nodes(self) -> List[str]:
        """
        Return node names that should be EXCLUDED from INT8 quantization
        (i.e. kept at FP32 or FP16).

        Returns:
            List of node name strings.
        """
        excluded = []
        for group, precision in self.mp_cfg.precision_map.items():
            if precision in (PRECISION_FP32, PRECISION_FP16):
                nodes = self.node_groups.get(group, [])
                excluded.extend(nodes)
                if nodes:
                    _log.info(
                        f"  {group}: {precision.upper()} "
                        f"({len(nodes)} nodes excluded from INT8)"
                    )
        return excluded

    def summary(self) -> str:
        """Return a human-readable precision assignment summary."""
        lines = [f"Mixed Precision Config: {self.mp_cfg.name}"]
        for group, precision in self.mp_cfg.precision_map.items():
            n = len(self.node_groups.get(group, []))
            lines.append(f"  {group:<25} → {precision.upper():<6} ({n} nodes)")
        return "\n".join(lines)


def config_from_sensitivity(
    sensitivity_results: list,
    top_k_fp32: int = 2,
    all_groups: Optional[List[str]] = None,
) -> MixedPrecisionConfig:
    """
    Auto-generate a MixedPrecisionConfig from sensitivity results:
    top-k most sensitive groups → FP32, rest → INT8.

    Args:
        sensitivity_results: Output of LayerSensitivityAnalyser.run().
        top_k_fp32:          Number of most sensitive groups to keep FP32.
        all_groups:          All group names (for groups not in results).

    Returns:
        MixedPrecisionConfig.
    """
    sensitive_groups = [r["group"] for r in sensitivity_results[:top_k_fp32]]
    all_g = all_groups or list(ONNX_LAYER_GROUPS.keys())

    precision_map = {
        g: PRECISION_FP32 if g in sensitive_groups else PRECISION_INT8
        for g in all_g
    }

    name = f"top{top_k_fp32}_sensitive_fp32"
    _log.info(f"Auto config '{name}': FP32 groups = {sensitive_groups}")

    return MixedPrecisionConfig(name=name, precision_map=precision_map)
