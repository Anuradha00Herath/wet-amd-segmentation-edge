"""
pruning_profiler.py
-------------------
Benchmarking pruned models vs baseline and quantized models.

Reuses:
  - quantization.quant_utils      (create_ort_session, onnx_size_mb)
  - quantization.static_quant     (quantize_static_onnx)
  - quantization.quant_config     (QuantConfig)
  - evaluation.metrics            (compute_all_metrics, MetricAccumulator)
  - compression.pruning_utils     (export_pruned_onnx, model_size_mb)
  - utils.logger                  (get_logger, CSVLogger)
"""

import os
import time
import statistics
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from utils.config import Config
from utils.logger import get_logger, CSVLogger
from utils.dataset import build_dataloaders
from quantization.quant_utils import create_ort_session, onnx_size_mb
from quantization.quant_config import QuantConfig
from evaluation.metrics import compute_all_metrics, MetricAccumulator

_log = get_logger(__name__)


def evaluate_onnx_accuracy(onnx_path: str, cfg: Config) -> Dict[str, float]:
    """Evaluate ONNX model on test set."""
    session    = create_ort_session(onnx_path)
    input_name = session.get_inputs()[0].name
    _, _, test_loader = build_dataloaders(cfg)
    acc = MetricAccumulator()

    for images, masks in test_loader:
        for i in range(images.shape[0]):
            img_np    = images[i:i+1].numpy().astype(np.float32)
            logits_np = session.run(None, {input_name: img_np})[0]
            pred      = torch.from_numpy(logits_np).argmax(dim=1)
            acc.update(compute_all_metrics(pred, masks[i:i+1]))

    return acc.mean()


def benchmark_onnx_latency(
    onnx_path: str,
    cfg: Config,
    n_runs: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """Measure ONNX inference latency."""
    session    = create_ort_session(onnx_path)
    input_name = session.get_inputs()[0].name
    dummy      = np.random.randn(1, cfg.in_channels, *cfg.image_size).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1e3)

    mean_ms = statistics.mean(latencies)
    return {
        "mean_ms": round(mean_ms, 2),
        "std_ms":  round(statistics.stdev(latencies), 2),
        "fps":     round(1000 / mean_ms, 1),
    }


def profile_pruned_model(
    onnx_path: str,
    cfg: Config,
    label: str,
    baseline_dice: float = 0.0,
    n_runs: int = 100,
) -> Dict[str, Any]:
    """
    Full profiling of a pruned ONNX model.

    Args:
        onnx_path:      Path to pruned ONNX model.
        cfg:            Config.
        label:          Descriptive label for reporting.
        baseline_dice:  FP32 baseline Dice for drop calculation.
        n_runs:         Latency benchmark runs.

    Returns:
        Dict with accuracy + performance metrics.
    """
    _log.info(f"Profiling: {label}")

    accuracy = evaluate_onnx_accuracy(onnx_path, cfg)
    latency  = benchmark_onnx_latency(onnx_path, cfg, n_runs)
    size_mb  = onnx_size_mb(onnx_path)

    result = {
        "label":       label,
        "onnx_path":   onnx_path,
        "size_mb":     round(size_mb, 2),
        "mean_dice":   round(accuracy.get("mean_dice", 0), 6),
        "mean_iou":    round(accuracy.get("mean_iou",  0), 6),
        "pixel_acc":   round(accuracy.get("pixel_acc", 0), 6),
        "dice_drop":   round(baseline_dice - accuracy.get("mean_dice", 0), 6),
        "mean_ms":     latency["mean_ms"],
        "std_ms":      latency["std_ms"],
        "fps":         latency["fps"],
    }

    _log.info(
        f"  Dice={result['mean_dice']:.4f} (drop={result['dice_drop']:+.4f}) | "
        f"Size={size_mb:.1f}MB | FPS={result['fps']:.1f}"
    )
    return result


def profile_pruning_sweep(
    pruned_onnx_paths: Dict[str, str],
    cfg: Config,
    baseline_dice: float,
    n_runs: int = 100,
) -> List[Dict[str, Any]]:
    """
    Profile multiple pruned models (one per pruning ratio).

    Args:
        pruned_onnx_paths: Dict mapping label → onnx_path.
        cfg:               Config.
        baseline_dice:     FP32 baseline Dice score.
        n_runs:            Latency runs per model.

    Returns:
        List of result dicts sorted by Dice descending.
    """
    results = []
    for label, path in pruned_onnx_paths.items():
        if not os.path.exists(path):
            _log.warning(f"Skipping {label}: {path} not found.")
            continue
        result = profile_pruned_model(path, cfg, label, baseline_dice, n_runs)
        results.append(result)

    results.sort(key=lambda r: r["mean_dice"], reverse=True)
    return results


def save_profiling_csv(results: List[Dict[str, Any]], cfg: Config) -> str:
    """Save profiling results to CSV."""
    if not results:
        return ""
    path    = os.path.join(cfg.csv_dir, "pruning_profiling_results.csv")
    csv_log = CSVLogger(path, fieldnames=[
        k for k in results[0].keys() if k != "onnx_path"
    ])
    for r in results:
        row = {k: v for k, v in r.items() if k != "onnx_path"}
        csv_log.log(row)
    _log.info(f"Profiling CSV → {path}")
    return path


def print_profiling_table(results: List[Dict[str, Any]]) -> None:
    """Print a comparison table."""
    print("\n" + "=" * 85)
    print(f"  {'Label':<30} {'Dice':>8} {'Drop':>8} {'Size MB':>9} {'FPS':>7} {'ms':>8}")
    print("-" * 85)
    for r in results:
        print(
            f"  {r['label']:<30} {r['mean_dice']:>8.4f} "
            f"{r['dice_drop']:>+8.4f} {r['size_mb']:>9.2f} "
            f"{r['fps']:>7.1f} {r['mean_ms']:>8.2f}"
        )
    print("=" * 85 + "\n")