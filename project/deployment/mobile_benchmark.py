"""
mobile_benchmark.py
-------------------
Colab-side mobile deployment benchmarking.
Reuses: quantization.quant_utils, evaluation.profiler, utils.logger
"""

import os
import time
import statistics
from typing import Any, Dict, List, Optional

import numpy as np

from utils.config import Config
from utils.logger import get_logger, CSVLogger
from quantization.quant_utils import onnx_size_mb
from deployment.export_mobile_onnx import DEPLOY_MODELS

_log = get_logger(__name__)


def benchmark_single_model(
    onnx_path: str,
    cfg: Config,
    n_runs: int = 100,
    warmup: int = 10,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Benchmark a single ONNX model simulating mobile single-image inference."""
    import onnxruntime as ort
    import psutil

    label = label or os.path.basename(onnx_path)

    # Model loading time
    t0 = time.perf_counter()
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
    load_time_ms = (time.perf_counter() - t0) * 1e3

    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, cfg.in_channels, *cfg.image_size).astype(np.float32)

    # Cold start
    t0 = time.perf_counter()
    _ = session.run(None, {input_name: dummy})
    cold_start_ms = (time.perf_counter() - t0) * 1e3

    # Warm-up
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss / 1e6

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1e3)

    rss_after = process.memory_info().rss / 1e6
    mean_ms   = statistics.mean(latencies)

    return {
        "label":           label,
        "size_mb":         round(onnx_size_mb(onnx_path), 2),
        "load_time_ms":    round(load_time_ms, 1),
        "cold_start_ms":   round(cold_start_ms, 2),
        "mean_ms":         round(mean_ms, 2),
        "std_ms":          round(statistics.stdev(latencies), 2),
        "min_ms":          round(min(latencies), 2),
        "max_ms":          round(max(latencies), 2),
        "fps":             round(1000.0 / mean_ms, 1),
        "ram_mb":          round(rss_after, 1),
        "ram_increase_mb": round(max(0, rss_after - rss_before), 1),
    }


def benchmark_all_models(
    cfg: Config,
    deploy_dir: Optional[str] = None,
    n_runs: int = 100,
) -> List[Dict[str, Any]]:
    """Benchmark all deployment models."""
    results = []
    for model_name, filename in DEPLOY_MODELS.items():
        path = os.path.join(cfg.checkpoints_dir, filename)
        if not os.path.exists(path):
            _log.warning(f"Skipping {model_name}: not found.")
            continue
        _log.info(f"Benchmarking {model_name} ...")
        result = benchmark_single_model(path, cfg, n_runs=n_runs, label=model_name)
        results.append(result)
        if deploy_dir:
            opt_path = os.path.join(deploy_dir, filename.replace(".onnx", "_mobile_opt.onnx"))
            if os.path.exists(opt_path):
                opt_r = benchmark_single_model(opt_path, cfg, n_runs=n_runs, label=f"{model_name}_opt")
                results.append(opt_r)
    results.sort(key=lambda r: r["mean_ms"])
    return results


def save_benchmark_csv(results: List[Dict[str, Any]], cfg: Config) -> str:
    if not results:
        return ""
    path = os.path.join(cfg.csv_dir, "mobile_benchmark_results.csv")
    csv  = CSVLogger(path, fieldnames=list(results[0].keys()))
    for r in results:
        csv.log(r)
    _log.info(f"Mobile benchmark CSV -> {path}")
    return path


def print_mobile_benchmark_table(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 90)
    print(f"  {'Model':<25} {'Size MB':>9} {'Load ms':>9} {'Cold ms':>9} {'Mean ms':>9} {'FPS':>7} {'RAM MB':>8}")
    print("-" * 90)
    fp32_ms = next((r["mean_ms"] for r in results if r["label"] == "fp32"), None)
    for r in results:
        speedup = f"  ({fp32_ms/r['mean_ms']:.2f}x)" if fp32_ms and r["label"] != "fp32" else ""
        print(
            f"  {r['label']:<25} {r['size_mb']:>9.2f} {r['load_time_ms']:>9.1f} "
            f"{r['cold_start_ms']:>9.2f} {r['mean_ms']:>9.2f} {r['fps']:>7.1f} "
            f"{r['ram_mb']:>8.1f}{speedup}"
        )
    print("=" * 90 + "\n")
