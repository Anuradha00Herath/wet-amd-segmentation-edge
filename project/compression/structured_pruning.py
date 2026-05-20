"""
structured_pruning.py
---------------------
Structured channel pruning for UNet++ EfficientNet-B4 segmentation models.

Approach:
  - Use torch.nn.utils.prune for structured L1 channel pruning
  - Support per-layer, group-wise, and global pruning modes
  - Preserve skip connections by pruning encoder/decoder independently
  - Make pruning permanent (remove masks) before export

Reuses:
  - compression.pruning_utils  (score_conv_channels, get_prune_mask)
  - utils.logger               (get_logger)
  - utils.config               (Config)
"""

import copy
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from utils.config import Config
from utils.logger import get_logger
from compression.pruning_utils import (
    get_conv_layers, model_size_mb, count_parameters,
    score_conv_channels, export_pruned_onnx,
)

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Layer group definitions                                                     #
# --------------------------------------------------------------------------- #

PRUNING_GROUPS = {
    "encoder":          "_encoder",
    "decoder":          "_decoder",
    "segmentation_head": "segmentation_head",
}


def _get_group(name: str) -> str:
    for group, prefix in PRUNING_GROUPS.items():
        if prefix in name:
            return group
    return "other"


# --------------------------------------------------------------------------- #
#  Structured pruning functions                                               #
# --------------------------------------------------------------------------- #

def prune_conv_layer(
    module: nn.Conv2d,
    prune_ratio: float,
    method: str = "l1_structured",
    dim: int = 0,
) -> nn.Conv2d:
    """
    Apply structured pruning to a single Conv2d layer.

    Args:
        module:      Conv2d layer to prune.
        prune_ratio: Fraction of output channels to prune.
        method:      Pruning method ('l1_structured' or 'random_structured').
        dim:         Dimension to prune (0 = output channels).

    Returns:
        Pruned module (with mask attached).
    """
    n_prune = max(1, int(module.out_channels * prune_ratio))
    n_prune = min(n_prune, module.out_channels - 1)  # keep at least 1

    if method == "l1_structured":
        prune.ln_structured(module, name="weight", amount=n_prune, n=1, dim=dim)
    else:
        prune.random_structured(module, name="weight", amount=n_prune, dim=dim)

    return module


def make_pruning_permanent(model: nn.Module) -> nn.Module:
    """
    Remove pruning masks and make weights permanent.
    Required before ONNX export or fine-tuning evaluation.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass  # not pruned
    return model


# --------------------------------------------------------------------------- #
#  Per-layer pruning                                                           #
# --------------------------------------------------------------------------- #

def prune_model_layerwise(
    model: nn.Module,
    prune_ratio: float,
    skip_layers: Optional[List[str]] = None,
    method: str = "l1_structured",
) -> Tuple[nn.Module, Dict]:
    """
    Apply uniform structured pruning to all Conv2d layers.

    Args:
        model:        Model to prune (modified in-place on a deep copy).
        prune_ratio:  Fraction of channels to prune per layer.
        skip_layers:  Layer name substrings to skip (e.g. ['segmentation_head']).
        method:       Pruning method.

    Returns:
        (pruned_model, stats_dict)
    """
    skip_layers = skip_layers or ["segmentation_head"]
    pruned      = copy.deepcopy(model)
    stats       = {"pruned_layers": [], "skipped_layers": [], "total_params_before": count_parameters(model)}

    for name, module in pruned.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if any(skip in name for skip in skip_layers):
            stats["skipped_layers"].append(name)
            continue
        if module.out_channels <= 4:   # skip very small layers
            continue

        prune_conv_layer(module, prune_ratio, method)
        stats["pruned_layers"].append({
            "name":         name,
            "out_channels": module.out_channels,
            "prune_ratio":  prune_ratio,
        })

    make_pruning_permanent(pruned)
    stats["total_params_after"] = count_parameters(pruned)
    stats["compression_ratio"]  = stats["total_params_before"] / max(stats["total_params_after"], 1)

    _log.info(
        f"Layer-wise pruning {prune_ratio:.0%}: "
        f"{stats['total_params_before']:,} → {stats['total_params_after']:,} params "
        f"({stats['compression_ratio']:.2f}× compression)"
    )
    return pruned, stats


# --------------------------------------------------------------------------- #
#  Global pruning                                                              #
# --------------------------------------------------------------------------- #

def prune_model_global(
    model: nn.Module,
    prune_ratio: float,
    skip_layers: Optional[List[str]] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Apply global structured pruning — prune least important channels
    across ALL layers simultaneously.

    Args:
        model:       Model to prune.
        prune_ratio: Global fraction of channels to prune.
        skip_layers: Layers to skip.

    Returns:
        (pruned_model, stats_dict)
    """
    skip_layers = skip_layers or ["segmentation_head"]
    pruned      = copy.deepcopy(model)

    # Collect all conv layers eligible for pruning
    parameters_to_prune = []
    for name, module in pruned.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 4:
            if not any(skip in name for skip in skip_layers):
                parameters_to_prune.append((module, "weight"))

    if not parameters_to_prune:
        return pruned, {}

    # Global L1 structured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_ratio,
    )

    make_pruning_permanent(pruned)

    stats = {
        "n_layers_pruned": len(parameters_to_prune),
        "total_params_before": count_parameters(model),
        "total_params_after":  count_parameters(pruned),
    }
    stats["compression_ratio"] = stats["total_params_before"] / max(stats["total_params_after"], 1)

    _log.info(
        f"Global pruning {prune_ratio:.0%}: "
        f"{stats['total_params_before']:,} → {stats['total_params_after']:,} params"
    )
    return pruned, stats


# --------------------------------------------------------------------------- #
#  Group-wise pruning (encoder/decoder separately)                            #
# --------------------------------------------------------------------------- #

def prune_model_groupwise(
    model: nn.Module,
    group_ratios: Dict[str, float],
) -> Tuple[nn.Module, Dict]:
    """
    Apply different pruning ratios to different architectural groups.

    Args:
        model:        Model to prune.
        group_ratios: Dict mapping group name → prune ratio.
                      e.g. {"encoder": 0.3, "decoder": 0.2, "segmentation_head": 0.0}

    Returns:
        (pruned_model, stats_dict)
    """
    pruned = copy.deepcopy(model)
    stats  = {"groups": {}, "total_params_before": count_parameters(model)}

    for name, module in pruned.named_modules():
        if not isinstance(module, nn.Conv2d) or module.out_channels <= 4:
            continue
        group = _get_group(name)
        ratio = group_ratios.get(group, 0.0)
        if ratio > 0:
            prune_conv_layer(module, ratio)
            stats["groups"][group] = stats["groups"].get(group, 0) + 1

    make_pruning_permanent(pruned)
    stats["total_params_after"]  = count_parameters(pruned)
    stats["compression_ratio"]   = stats["total_params_before"] / max(stats["total_params_after"], 1)

    _log.info(f"Group-wise pruning: {stats['compression_ratio']:.2f}× compression")
    return pruned, stats


# --------------------------------------------------------------------------- #
#  Main pruning pipeline                                                       #
# --------------------------------------------------------------------------- #

class StructuredPruningPipeline:
    """
    End-to-end structured pruning pipeline.

    Usage:
        pipeline = StructuredPruningPipeline(model, cfg)
        results  = pipeline.run_pruning_sweep(ratios=[0.1, 0.2, 0.3, 0.5])
    """

    def __init__(self, model: nn.Module, cfg: Config) -> None:
        self.model = model
        self.cfg   = cfg

    def prune(
        self,
        prune_ratio: float,
        mode: str = "layerwise",
        group_ratios: Optional[Dict[str, float]] = None,
        skip_layers: Optional[List[str]] = None,
    ) -> Tuple[nn.Module, Dict]:
        """
        Prune the model.

        Args:
            prune_ratio:  Fraction of channels to prune.
            mode:         'layerwise', 'global', or 'groupwise'.
            group_ratios: Required for 'groupwise' mode.
            skip_layers:  Layers to skip.

        Returns:
            (pruned_model, stats)
        """
        if mode == "global":
            return prune_model_global(self.model, prune_ratio, skip_layers)
        elif mode == "groupwise" and group_ratios:
            return prune_model_groupwise(self.model, group_ratios)
        else:
            return prune_model_layerwise(self.model, prune_ratio, skip_layers)

    def export_pruned(
        self,
        pruned_model: nn.Module,
        prune_ratio: float,
        mode: str = "layerwise",
    ) -> str:
        """Export pruned model to ONNX."""
        name = f"pruned_{mode}_{int(prune_ratio*100)}pct.onnx"
        path = os.path.join(self.cfg.checkpoints_dir, name)
        return export_pruned_onnx(pruned_model.cpu(), self.cfg, path)

    def run_pruning_sweep(
        self,
        ratios: List[float] = [0.1, 0.2, 0.3, 0.5, 0.7],
        mode: str = "layerwise",
        skip_layers: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Run pruning at multiple ratios and return stats for each.

        Returns:
            List of stats dicts with compression metrics.
        """
        results = []
        for ratio in ratios:
            _log.info(f"\n── Pruning ratio: {ratio:.0%} ──────────────")
            pruned_model, stats = self.prune(ratio, mode, skip_layers=skip_layers)
            stats["prune_ratio"] = ratio
            stats["size_mb"]     = model_size_mb(pruned_model)
            stats["mode"]        = mode
            results.append(stats)
            _log.info(f"  Size: {stats['size_mb']:.2f} MB | Compression: {stats['compression_ratio']:.2f}×")

        return results