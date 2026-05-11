"""
sensitivity_analysis.py
-----------------------
Layer-wise quantization sensitivity analysis via ONNX Runtime.

Approach (ONNX-compatible, works with EfficientNet-B4):
  For each layer group (encoder, decoder, etc.):
    1. Quantize EVERYTHING except that group → measure Dice
    2. The group causing the LARGEST Dice drop when excluded
       is the MOST sensitive (needs to stay in higher precision)

This is the standard "leave-one-out" sensitivity analysis used in
academic quantization research (e.g. HAWQv2, BRECQ).

Reuses:
  - quantization.quant_utils     (create_ort_session, onnx_size_mb)
  - quantization.static_quant    (benchmark_ort_session)
  - quantization.calibration     (OCTCalibrationReader)
  - quantization.quant_config    (QuantConfig)
  - evaluation.metrics           (compute_all_metrics, MetricAccumulator)
  - utils.logger                 (get_logger, CSVLogger)
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from utils.config import Config
from utils.logger import get_logger, CSVLogger
from utils.dataset import build_dataloaders
from quantization.quant_utils import create_ort_session, onnx_size_mb
from quantization.quant_config import QuantConfig
from quantization.calibration import OCTCalibrationReader
from quantization.layer_profiler import ONNX_LAYER_GROUPS, get_onnx_nodes_by_group
from evaluation.metrics import compute_all_metrics, MetricAccumulator

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Selective quantization (exclude a node list)                                #
# --------------------------------------------------------------------------- #

def quantize_excluding_nodes(
    fp32_onnx_path: str,
    output_path: str,
    cfg: Config,
    qcfg: QuantConfig,
    nodes_to_exclude: List[str],
) -> str:
    """
    Quantize all ops EXCEPT the specified nodes (which stay FP32).

    Args:
        fp32_onnx_path:   FP32 ONNX model.
        output_path:      Output INT8 ONNX path.
        cfg:              Config.
        qcfg:             QuantConfig.
        nodes_to_exclude: List of node names to keep at FP32.

    Returns:
        Path to quantized model.
    """
    try:
        from onnxruntime.quantization import (
            quantize_static, QuantFormat, QuantType, CalibrationMethod
        )
        import onnx
    except ImportError:
        raise ImportError("Run: pip install onnx onnxruntime")

    model_proto = onnx.load(fp32_onnx_path)
    input_name  = model_proto.graph.input[0].name

    calib_reader = OCTCalibrationReader(
        cfg, n_batches=qcfg.calib_batches,
        input_name=input_name, batch_size=1,
    )

    quantize_static(
        model_input             = fp32_onnx_path,
        model_output            = output_path,
        calibration_data_reader = calib_reader,
        quant_format            = QuantFormat.QOperator,
        per_channel             = qcfg.per_channel,
        weight_type             = QuantType.QInt8,
        activation_type         = QuantType.QUInt8,
        calibrate_method        = CalibrationMethod.MinMax,
        reduce_range            = qcfg.reduce_range,
        nodes_to_exclude        = nodes_to_exclude,   # ← key parameter
    )
    return output_path


# --------------------------------------------------------------------------- #
#  Evaluate ORT session on test set                                            #
# --------------------------------------------------------------------------- #

def _evaluate_session(session, cfg: Config) -> Dict[str, float]:
    """Run test set evaluation on an ORT session. Returns metrics dict."""
    _, _, test_loader = build_dataloaders(cfg)
    input_name = session.get_inputs()[0].name
    acc = MetricAccumulator()

    for images, masks in test_loader:
        for i in range(images.shape[0]):
            img_np    = images[i:i+1].numpy().astype(np.float32)
            logits_np = session.run(None, {input_name: img_np})[0]
            pred      = torch.from_numpy(logits_np).argmax(dim=1)
            acc.update(compute_all_metrics(pred, masks[i:i+1]))

    return acc.mean()


# --------------------------------------------------------------------------- #
#  Sensitivity analyser                                                        #
# --------------------------------------------------------------------------- #

class LayerSensitivityAnalyser:
    """
    Leave-one-out layer sensitivity analysis.

    For each layer group:
      - Quantize all OTHER groups to INT8
      - Keep this group at FP32
      - Measure Dice score
      - Sensitivity = (group_dice - fully_int8_dice)
        Higher = more sensitive = needs higher precision

    Usage:
        analyser = LayerSensitivityAnalyser(fp32_onnx_path, cfg, qcfg)
        results  = analyser.run()
        analyser.save_csv(results)
    """

    def __init__(
        self,
        fp32_onnx_path: str,
        cfg: Config,
        qcfg: QuantConfig,
        tmp_dir: Optional[str] = None,
    ) -> None:
        self.fp32_onnx_path = fp32_onnx_path
        self.cfg    = cfg
        self.qcfg   = qcfg
        self.tmp_dir = tmp_dir or cfg.checkpoints_dir

        # Get node groups from the ONNX graph
        self.node_groups = get_onnx_nodes_by_group(fp32_onnx_path)
        _log.info(f"Node groups found: {list(self.node_groups.keys())}")

    def _get_fully_int8_dice(self) -> float:
        """Baseline: fully INT8 quantized model Dice."""
        out_path = os.path.join(self.tmp_dir, "_sensitivity_full_int8.onnx")
        quantize_excluding_nodes(
            self.fp32_onnx_path, out_path,
            self.cfg, self.qcfg,
            nodes_to_exclude=[],   # exclude nothing = quantize everything
        )
        session = create_ort_session(out_path)
        metrics = _evaluate_session(session, self.cfg)
        os.remove(out_path)
        return metrics["mean_dice"]

    def run(self) -> List[Dict[str, Any]]:
        """
        Run leave-one-out sensitivity analysis for all layer groups.

        Returns:
            List of result dicts sorted by sensitivity (most sensitive first).
        """
        _log.info("Computing fully INT8 baseline Dice ...")
        full_int8_dice = self._get_fully_int8_dice()
        _log.info(f"Fully INT8 Dice: {full_int8_dice:.4f}")

        results = []

        for group_name, node_names in self.node_groups.items():
            if not node_names:
                continue

            _log.info(f"Testing group: {group_name} ({len(node_names)} nodes) ...")
            out_path = os.path.join(
                self.tmp_dir, f"_sensitivity_{group_name}.onnx"
            )

            try:
                quantize_excluding_nodes(
                    self.fp32_onnx_path, out_path,
                    self.cfg, self.qcfg,
                    nodes_to_exclude=node_names,  # keep this group FP32
                )
                session = create_ort_session(out_path)
                metrics = _evaluate_session(session, self.cfg)

                dice_with_fp32 = metrics["mean_dice"]
                sensitivity    = dice_with_fp32 - full_int8_dice

                results.append({
                    "group":           group_name,
                    "n_nodes":         len(node_names),
                    "dice_with_fp32":  round(dice_with_fp32, 6),
                    "full_int8_dice":  round(full_int8_dice, 6),
                    "sensitivity":     round(sensitivity, 6),
                    "iou_with_fp32":   round(metrics.get("mean_iou", 0), 6),
                    "pixel_acc":       round(metrics.get("pixel_acc", 0), 6),
                    "size_mb":         round(onnx_size_mb(out_path), 2),
                })

                _log.info(
                    f"  {group_name}: Dice={dice_with_fp32:.4f} "
                    f"(+{sensitivity:.4f} vs full INT8)"
                )

            except Exception as e:
                _log.warning(f"  {group_name} failed: {e}")
            finally:
                if os.path.exists(out_path):
                    os.remove(out_path)

        # Sort by sensitivity descending (most sensitive first)
        results.sort(key=lambda r: r["sensitivity"], reverse=True)
        return results

    def save_csv(self, results: List[Dict[str, Any]]) -> str:
        """Save sensitivity results to CSV."""
        if not results:
            return ""
        path    = os.path.join(self.cfg.csv_dir, "sensitivity_analysis.csv")
        csv_log = CSVLogger(path, fieldnames=list(results[0].keys()))
        for row in results:
            csv_log.log(row)
        _log.info(f"Sensitivity CSV → {path}")
        return path

    def get_sensitivity_ranking(
        self, results: List[Dict[str, Any]]
    ) -> List[str]:
        """Return group names ranked from most to least sensitive."""
        return [r["group"] for r in results]
