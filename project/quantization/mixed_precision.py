"""
mixed_precision.py
------------------
Mixed precision quantization pipeline.

Applies different precision levels to different ONNX node groups
using ORT's nodes_to_exclude parameter.

Reuses:
  - quantization.sensitivity_analysis  (quantize_excluding_nodes, _evaluate_session)
  - quantization.precision_scheduler   (PrecisionScheduler, MIXED_PRECISION_EXPERIMENTS)
  - quantization.layer_profiler        (get_onnx_nodes_by_group)
  - quantization.quant_utils           (create_ort_session, onnx_size_mb, compression_ratio)
  - quantization.static_quant          (benchmark_ort_session)
  - utils.logger                       (get_logger, CSVLogger)
"""

import os
from typing import Any, Dict, List, Optional

from utils.config import Config
from utils.logger import get_logger, CSVLogger
from quantization.quant_config import QuantConfig
from quantization.layer_profiler import get_onnx_nodes_by_group
from quantization.precision_scheduler import (
    PrecisionScheduler, MixedPrecisionConfig, MIXED_PRECISION_EXPERIMENTS
)
from quantization.sensitivity_analysis import (
    quantize_excluding_nodes, _evaluate_session
)
from quantization.quant_utils import create_ort_session, onnx_size_mb, compression_ratio
from quantization.static_quant import benchmark_ort_session

_log = get_logger(__name__)


class MixedPrecisionPipeline:
    """
    Run multiple mixed-precision experiments and compare results.

    Usage:
        pipeline = MixedPrecisionPipeline(fp32_onnx_path, cfg, qcfg)
        results  = pipeline.run_experiments(experiments)
        pipeline.save_csv(results)
    """

    def __init__(
        self,
        fp32_onnx_path: str,
        cfg: Config,
        qcfg: QuantConfig,
        fp32_dice: float = 0.0,
    ) -> None:
        self.fp32_onnx_path = fp32_onnx_path
        self.cfg            = cfg
        self.qcfg           = qcfg
        self.fp32_dice      = fp32_dice
        self.node_groups    = get_onnx_nodes_by_group(fp32_onnx_path)
        self.fp32_size_mb   = onnx_size_mb(fp32_onnx_path)

    def run_single(
        self,
        mp_cfg: MixedPrecisionConfig,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run one mixed-precision experiment.

        Args:
            mp_cfg:      Precision configuration.
            output_path: Where to save the quantized ONNX.

        Returns:
            Result dict with accuracy and performance metrics.
        """
        if output_path is None:
            output_path = os.path.join(
                self.cfg.checkpoints_dir, f"mp_{mp_cfg.name}.onnx"
            )

        scheduler = PrecisionScheduler(mp_cfg, self.node_groups)
        excluded  = scheduler.get_excluded_nodes()

        _log.info(f"\nExperiment: {mp_cfg.name}")
        _log.info(scheduler.summary())
        _log.info(f"Excluding {len(excluded)} nodes from INT8 quantization")

        # Quantize
        quantize_excluding_nodes(
            self.fp32_onnx_path, output_path,
            self.cfg, self.qcfg,
            nodes_to_exclude=excluded,
        )

        # Evaluate
        session    = create_ort_session(output_path)
        metrics    = _evaluate_session(session, self.cfg)
        latency    = benchmark_ort_session(
            session, self.cfg,
            n_runs=self.cfg.benchmark_runs,
            warmup=self.cfg.benchmark_warmup,
            input_name=session.get_inputs()[0].name,
        )
        size_mb    = onnx_size_mb(output_path)
        comp_ratio = compression_ratio(self.fp32_size_mb, size_mb)
        dice_drop  = self.fp32_dice - metrics.get("mean_dice", 0)

        result = {
            "experiment":    mp_cfg.name,
            "fp32_groups":   [
                g for g, p in mp_cfg.precision_map.items()
                if p != "int8"
            ],
            "mean_dice":     round(metrics.get("mean_dice", 0), 6),
            "mean_iou":      round(metrics.get("mean_iou",  0), 6),
            "pixel_acc":     round(metrics.get("pixel_acc", 0), 6),
            "dice_drop":     round(dice_drop, 6),
            "size_mb":       round(size_mb,   2),
            "compression":   round(comp_ratio, 2),
            "mean_ms":       round(latency.get("mean_ms", 0), 2),
            "fps":           round(latency.get("fps",      0), 1),
            "n_excluded":    len(excluded),
            "onnx_path":     output_path,
        }

        _log.info(
            f"  → Dice={result['mean_dice']:.4f} "
            f"(drop={dice_drop:+.4f}) | "
            f"Size={size_mb:.1f}MB | FPS={result['fps']:.1f}"
        )
        return result

    def run_experiments(
        self,
        experiments: Optional[Dict[str, MixedPrecisionConfig]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run all mixed-precision experiments.

        Args:
            experiments: Dict of name → MixedPrecisionConfig.
                         Defaults to MIXED_PRECISION_EXPERIMENTS.

        Returns:
            List of result dicts sorted by Dice score descending.
        """
        if experiments is None:
            experiments = MIXED_PRECISION_EXPERIMENTS

        results = []
        for name, mp_cfg in experiments.items():
            try:
                result = self.run_single(mp_cfg)
                results.append(result)
            except Exception as e:
                _log.warning(f"Experiment '{name}' failed: {e}")

        # Sort by Dice descending
        results.sort(key=lambda r: r["mean_dice"], reverse=True)
        return results

    def save_csv(self, results: List[Dict[str, Any]]) -> str:
        """Save all experiment results to CSV."""
        if not results:
            return ""

        # Flatten fp32_groups list for CSV
        flat_results = []
        for r in results:
            row = dict(r)
            row["fp32_groups"] = "|".join(r.get("fp32_groups", []))
            row.pop("onnx_path", None)
            flat_results.append(row)

        path    = os.path.join(self.cfg.csv_dir, "mixed_precision_results.csv")
        csv_log = CSVLogger(path, fieldnames=list(flat_results[0].keys()))
        for row in flat_results:
            csv_log.log(row)
        _log.info(f"Mixed precision CSV → {path}")
        return path

    def get_pareto_optimal(
        self,
        results: List[Dict[str, Any]],
        objective_x: str = "mean_ms",
        objective_y: str = "mean_dice",
    ) -> List[Dict[str, Any]]:
        """
        Return Pareto-optimal experiments (best accuracy-latency tradeoff).

        A result is Pareto-optimal if no other result is better in BOTH
        accuracy and latency simultaneously.

        Args:
            results:     All experiment results.
            objective_x: Metric to minimise (e.g. mean_ms, size_mb).
            objective_y: Metric to maximise (e.g. mean_dice).

        Returns:
            List of Pareto-optimal results.
        """
        pareto = []
        for r in results:
            dominated = False
            for other in results:
                if other is r:
                    continue
                # Other dominates r if: better or equal on both objectives
                if (other[objective_x] <= r[objective_x] and
                        other[objective_y] >= r[objective_y] and
                        (other[objective_x] < r[objective_x] or
                         other[objective_y] > r[objective_y])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r)
        return sorted(pareto, key=lambda r: r[objective_y], reverse=True)
