"""
static_quant.py
---------------
Post-Training Static INT8 Quantization via ONNX Runtime.

Pipeline:
  FP32 PyTorch model
      ↓  export_to_onnx()
  FP32 ONNX model
      ↓  quantize_static() [ORT]  ← calibration data
  INT8 ONNX model
      ↓  create_ort_session()
  INT8 ORT InferenceSession

Why ORT instead of torch.quantization:
  - EfficientNet-B4 SiLU activations are unsupported by PyTorch
    static quantization observers.
  - ORT handles arbitrary op graphs including SiLU natively.
  - This approach is validated to work with smp UNet++ models.

Reuses:
  - quantization.quant_utils   (export, session creation)
  - quantization.calibration   (OCTCalibrationReader)
  - evaluation.*               (benchmarking, metrics, reports)
  - utils.*                    (config, logger, dataset)
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from utils.config import Config
from utils.logger import get_logger, CSVLogger
from quantization.quant_utils import (
    export_to_onnx, create_ort_session,
    run_ort_inference, onnx_size_mb, compression_ratio,
)
from quantization.calibration import OCTCalibrationReader
from quantization.quant_config import QuantConfig

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Static INT8 quantization                                                    #
# --------------------------------------------------------------------------- #

def quantize_static_onnx(
    fp32_onnx_path: str,
    output_path: str,
    cfg: Config,
    qcfg: QuantConfig,
) -> str:
    """
    Apply ONNX Runtime static INT8 quantization with calibration.

    Args:
        fp32_onnx_path: Path to the FP32 ONNX model.
        output_path:    Path to save the INT8 ONNX model.
        cfg:            Project Config.
        qcfg:           QuantConfig.

    Returns:
        Path to the quantized INT8 ONNX model.
    """
    try:
        from onnxruntime.quantization import (  # type: ignore
            quantize_static, CalibrationDataReader,
            QuantFormat, QuantType, CalibrationMethod,
        )
    except ImportError:
        raise ImportError(
            "onnxruntime not installed.\n"
            "Run: pip install onnxruntime"
        )

    # Get input name from the ONNX model
    import onnx  # type: ignore
    model_proto = onnx.load(fp32_onnx_path)
    input_name  = model_proto.graph.input[0].name

    # Build calibration reader
    calib_reader = OCTCalibrationReader(
        cfg,
        n_batches=qcfg.calib_batches,
        input_name=input_name,
    )

    _log.info(f"Running static INT8 quantization → {output_path}")

    quantize_static(
        model_input          = fp32_onnx_path,
        model_output         = output_path,
        calibration_data_reader = calib_reader,
        quant_format         = QuantFormat.QOperator,
        per_channel          = qcfg.per_channel,
        weight_type          = QuantType.QInt8,
        activation_type      = QuantType.QUInt8,
        calibrate_method     = CalibrationMethod.MinMax,
        reduce_range         = qcfg.reduce_range,
        optimize_model       = True,
    )

    size_mb = onnx_size_mb(output_path)
    _log.info(f"INT8 model saved → {output_path}  ({size_mb:.2f} MB)")
    return output_path


def quantize_dynamic_onnx(
    fp32_onnx_path: str,
    output_path: str,
) -> str:
    """
    Apply ONNX Runtime dynamic INT8 quantization (no calibration needed).

    Faster to apply than static but typically less accurate.

    Args:
        fp32_onnx_path: Path to the FP32 ONNX model.
        output_path:    Path to save the INT8 ONNX model.

    Returns:
        Path to the quantized model.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore
    except ImportError:
        raise ImportError("Run: pip install onnxruntime")

    _log.info(f"Running dynamic INT8 quantization → {output_path}")

    quantize_dynamic(
        model_input  = fp32_onnx_path,
        model_output = output_path,
        weight_type  = QuantType.QInt8,
    )

    size_mb = onnx_size_mb(output_path)
    _log.info(f"Dynamic INT8 model saved → {output_path}  ({size_mb:.2f} MB)")
    return output_path


# --------------------------------------------------------------------------- #
#  ORT latency benchmarking                                                    #
# --------------------------------------------------------------------------- #

def benchmark_ort_session(
    session,
    cfg: Config,
    n_runs: int = 100,
    warmup: int = 10,
    input_name: str = "input",
) -> Dict[str, float]:
    """
    Measure ORT session inference latency.

    Args:
        session:    ORT InferenceSession.
        cfg:        Config (for input shape).
        n_runs:     Timed runs.
        warmup:     Warm-up runs.
        input_name: Input node name.

    Returns:
        Dict with mean_ms, std_ms, min_ms, max_ms, fps.
    """
    import statistics, time

    dummy = np.random.randn(
        1, cfg.in_channels, *cfg.image_size
    ).astype(np.float32)

    latencies = []

    # Warm-up
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    # Timed runs
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1e3)

    mean_ms = statistics.mean(latencies)
    return {
        "mean_ms":   mean_ms,
        "std_ms":    statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min_ms":    min(latencies),
        "max_ms":    max(latencies),
        "median_ms": statistics.median(latencies),
        "fps":       1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
#  Accuracy evaluation via ORT                                                 #
# --------------------------------------------------------------------------- #

def evaluate_ort_model(
    session,
    cfg: Config,
    input_name: str = "input",
) -> Dict[str, float]:
    """
    Evaluate an ORT model on the test set and return segmentation metrics.

    Reuses evaluation.metrics — no duplicate metric code.

    Args:
        session:    ORT InferenceSession.
        cfg:        Config.
        input_name: Input node name.

    Returns:
        Dict of segmentation metrics.
    """
    from utils.dataset import build_dataloaders
    from evaluation.metrics import compute_all_metrics, MetricAccumulator

    _, _, test_loader = build_dataloaders(cfg)
    acc = MetricAccumulator()

    for images, masks in test_loader:
        images_np  = images.numpy().astype(np.float32)
        logits_np  = session.run(None, {input_name: images_np})[0]
        pred_mask  = torch.from_numpy(logits_np).argmax(dim=1)
        acc.update(compute_all_metrics(pred_mask, masks))

    return acc.mean()


# --------------------------------------------------------------------------- #
#  Full PTQ pipeline                                                           #
# --------------------------------------------------------------------------- #

class PTQPipeline:
    """
    End-to-end PTQ pipeline:
      1. Export FP32 → ONNX
      2. Static INT8 quantization with calibration
      3. (Optional) Dynamic INT8 quantization
      4. Benchmark both models
      5. Evaluate accuracy
      6. Generate comparison report

    Usage:
        pipeline = PTQPipeline(cfg, qcfg, device)
        results  = pipeline.run(fp32_model)
    """

    def __init__(
        self,
        cfg: Config,
        qcfg: QuantConfig,
        device: torch.device,
    ) -> None:
        self.cfg    = cfg
        self.qcfg   = qcfg
        self.device = device

        self.fp32_onnx_path   = os.path.join(cfg.checkpoints_dir, qcfg.onnx_filename)
        self.int8_static_path = os.path.join(
            cfg.checkpoints_dir, "baseline_int8_static.onnx"
        )
        self.int8_dynamic_path = os.path.join(
            cfg.checkpoints_dir, "baseline_int8_dynamic.onnx"
        )

    def run(self, fp32_model: torch.nn.Module) -> Dict[str, Any]:
        """
        Run the complete PTQ pipeline.

        Args:
            fp32_model: Loaded FP32 PyTorch model.

        Returns:
            Dict containing all benchmark and accuracy results.
        """
        results: Dict[str, Any] = {}

        # ── Step 1: Export FP32 → ONNX ──────────────────────────────────────
        _log.info("Step 1: Exporting FP32 model to ONNX ...")
        self.fp32_onnx_path = export_to_onnx(
            fp32_model, self.cfg,
            filename=self.qcfg.onnx_filename,
            opset_version=self.qcfg.opset_version,
        )
        results["fp32_onnx_path"]   = self.fp32_onnx_path
        results["fp32_onnx_size_mb"] = onnx_size_mb(self.fp32_onnx_path)

        # ── Step 2: FP32 ONNX benchmark ─────────────────────────────────────
        _log.info("Step 2: Benchmarking FP32 ONNX ...")
        fp32_session = create_ort_session(
            self.fp32_onnx_path, self.qcfg.execution_providers
        )
        input_name = fp32_session.get_inputs()[0].name

        results["fp32_latency"] = benchmark_ort_session(
            fp32_session, self.cfg,
            n_runs=self.cfg.benchmark_runs,
            warmup=self.cfg.benchmark_warmup,
            input_name=input_name,
        )
        results["fp32_accuracy"] = evaluate_ort_model(
            fp32_session, self.cfg, input_name=input_name
        )

        # ── Step 3: Static INT8 quantization ────────────────────────────────
        _log.info("Step 3: Applying static INT8 quantization ...")
        self.int8_static_path = quantize_static_onnx(
            self.fp32_onnx_path, self.int8_static_path,
            self.cfg, self.qcfg,
        )
        results["int8_static_path"]    = self.int8_static_path
        results["int8_static_size_mb"] = onnx_size_mb(self.int8_static_path)
        results["static_compression"]  = compression_ratio(
            results["fp32_onnx_size_mb"], results["int8_static_size_mb"]
        )

        # ── Step 4: Static INT8 benchmark ───────────────────────────────────
        _log.info("Step 4: Benchmarking static INT8 model ...")
        int8_static_session = create_ort_session(
            self.int8_static_path, self.qcfg.execution_providers
        )
        int8_input_name = int8_static_session.get_inputs()[0].name

        results["int8_static_latency"] = benchmark_ort_session(
            int8_static_session, self.cfg,
            n_runs=self.cfg.benchmark_runs,
            warmup=self.cfg.benchmark_warmup,
            input_name=int8_input_name,
        )
        results["int8_static_accuracy"] = evaluate_ort_model(
            int8_static_session, self.cfg, input_name=int8_input_name
        )

        # ── Step 5: Dynamic INT8 (optional) ─────────────────────────────────
        if "dynamic" in self.qcfg.compare_backends:
            _log.info("Step 5: Applying dynamic INT8 quantization ...")
            self.int8_dynamic_path = quantize_dynamic_onnx(
                self.fp32_onnx_path, self.int8_dynamic_path
            )
            results["int8_dynamic_size_mb"] = onnx_size_mb(self.int8_dynamic_path)
            results["dynamic_compression"]  = compression_ratio(
                results["fp32_onnx_size_mb"], results["int8_dynamic_size_mb"]
            )

            int8_dyn_session   = create_ort_session(
                self.int8_dynamic_path, self.qcfg.execution_providers
            )
            dyn_input_name = int8_dyn_session.get_inputs()[0].name

            results["int8_dynamic_latency"]  = benchmark_ort_session(
                int8_dyn_session, self.cfg,
                input_name=dyn_input_name,
            )
            results["int8_dynamic_accuracy"] = evaluate_ort_model(
                int8_dyn_session, self.cfg, input_name=dyn_input_name
            )

        # ── Step 6: Save CSV summary ─────────────────────────────────────────
        self._save_comparison_csv(results)

        return results

    def _save_comparison_csv(self, results: Dict[str, Any]) -> None:
        """Save FP32 vs INT8 comparison to CSV."""
        rows = [
            {
                "model":        "fp32_onnx",
                "size_mb":      results.get("fp32_onnx_size_mb", 0),
                "compression":  1.0,
                "mean_dice":    results["fp32_accuracy"].get("mean_dice", 0),
                "mean_iou":     results["fp32_accuracy"].get("mean_iou",  0),
                "pixel_acc":    results["fp32_accuracy"].get("pixel_acc", 0),
                "mean_ms":      results["fp32_latency"].get("mean_ms",    0),
                "fps":          results["fp32_latency"].get("fps",         0),
            },
            {
                "model":        "int8_static_onnx",
                "size_mb":      results.get("int8_static_size_mb", 0),
                "compression":  results.get("static_compression",  0),
                "mean_dice":    results["int8_static_accuracy"].get("mean_dice", 0),
                "mean_iou":     results["int8_static_accuracy"].get("mean_iou",  0),
                "pixel_acc":    results["int8_static_accuracy"].get("pixel_acc", 0),
                "mean_ms":      results["int8_static_latency"].get("mean_ms",    0),
                "fps":          results["int8_static_latency"].get("fps",         0),
            },
        ]

        if "int8_dynamic_accuracy" in results:
            rows.append({
                "model":       "int8_dynamic_onnx",
                "size_mb":     results.get("int8_dynamic_size_mb", 0),
                "compression": results.get("dynamic_compression",  0),
                "mean_dice":   results["int8_dynamic_accuracy"].get("mean_dice", 0),
                "mean_iou":    results["int8_dynamic_accuracy"].get("mean_iou",  0),
                "pixel_acc":   results["int8_dynamic_accuracy"].get("pixel_acc", 0),
                "mean_ms":     results["int8_dynamic_latency"].get("mean_ms",    0),
                "fps":         results["int8_dynamic_latency"].get("fps",         0),
            })

        csv_path = os.path.join(self.cfg.csv_dir, "ptq_comparison.csv")
        csv_log  = CSVLogger(csv_path, fieldnames=list(rows[0].keys()))
        for row in rows:
            csv_log.log(row)

        _log.info(f"PTQ comparison CSV → {csv_path}")