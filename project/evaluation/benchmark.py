"""
benchmark.py
------------
Inference benchmarking utilities.

Measures:
  - Mean latency (ms)
  - Standard deviation of latency
  - Frames per second (FPS)
  - Peak memory usage (CUDA or CPU RSS)

Design: The Benchmarker class separates timing logic from metric
computation, allowing it to be reused for baseline, PTQ, QAT, and
ONNX models in later phases.
"""

import gc
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from utils.config import Config
from utils.logger import get_logger

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Core latency timer                                                           #
# --------------------------------------------------------------------------- #

def _sync(device: torch.device) -> None:
    """Synchronise CUDA if needed before reading wall-clock time."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_runs: int = 100,
    warmup: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Time a model over repeated runs on a fixed input.

    Args:
        model:        nn.Module in eval mode.
        input_tensor: Pre-built input tensor (B, C, H, W).
        n_runs:       Number of timed runs.
        warmup:       Warm-up runs (excluded from timing).
        device:       Target device. Inferred from input_tensor if None.

    Returns:
        Dict with keys: mean_ms, std_ms, min_ms, max_ms, fps.
    """
    if device is None:
        device = input_tensor.device

    model.eval()
    input_tensor = input_tensor.to(device)

    latencies: List[float] = []

    with torch.no_grad():
        # Warm-up
        for _ in range(warmup):
            _ = model(input_tensor)
        _sync(device)

        # Timed runs
        for _ in range(n_runs):
            _sync(device)
            t0 = time.perf_counter()
            _ = model(input_tensor)
            _sync(device)
            latencies.append((time.perf_counter() - t0) * 1e3)  # ms

    mean_ms = statistics.mean(latencies)
    std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
    }


# --------------------------------------------------------------------------- #
#  Memory profiling                                                             #
# --------------------------------------------------------------------------- #

def get_memory_usage_mb(device: torch.device) -> float:
    """
    Return current memory usage in MB.

    For CUDA: allocated GPU memory.
    For CPU:  process RSS via psutil (if installed), else 0.
    """
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1e6

    try:
        import psutil, os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1e6
    except ImportError:
        return 0.0


# --------------------------------------------------------------------------- #
#  Benchmarker                                                                  #
# --------------------------------------------------------------------------- #

class Benchmarker:
    """
    Full-suite benchmarker: latency + memory + parameter stats.

    Usage:
        bench = Benchmarker(cfg, device)
        results = bench.run(model, label="baseline_fp32")
        bench.print_results(results)
    """

    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

    def _build_dummy_input(self) -> torch.Tensor:
        b = 1
        c = self.cfg.in_channels
        h, w = self.cfg.image_size
        return torch.randn(b, c, h, w)

    def run(
        self,
        model: nn.Module,
        label: str = "model",
        input_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark a single model.

        Args:
            model:        Model to benchmark (moved to self.device).
            label:        Descriptive label for logging.
            input_tensor: Custom input; default is a random dummy of cfg size.

        Returns:
            Dict containing all benchmark metrics.
        """
        from utils.helpers import count_parameters, model_size_mb, estimate_flops

        model = model.to(self.device).eval()
        if input_tensor is None:
            input_tensor = self._build_dummy_input()

        _log.info(f"Benchmarking '{label}' on {self.device} ...")

        # Clear cache
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

        # Latency
        latency = time_inference(
            model, input_tensor,
            n_runs=self.cfg.benchmark_runs,
            warmup=self.cfg.benchmark_warmup,
            device=self.device,
        )

        # Peak memory (CUDA only reliable)
        if self.device.type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / 1e6
        else:
            peak_mem_mb = get_memory_usage_mb(self.device)

        # FLOPs
        flops = estimate_flops(model, input_tensor.to(self.device))

        results = {
            "label": label,
            "device": str(self.device),
            "n_params": count_parameters(model, trainable_only=False),
            "model_size_mb": model_size_mb(model),
            "flops": flops,
            "peak_mem_mb": peak_mem_mb,
            **latency,
        }
        return results

    @staticmethod
    def print_results(results: Dict[str, Any]) -> None:
        print("\n" + "=" * 55)
        print(f"  Benchmark: {results['label']}")
        print("=" * 55)
        print(f"  Device            : {results['device']}")
        print(f"  Parameters        : {results['n_params']:,}")
        print(f"  Model size        : {results['model_size_mb']:.2f} MB")
        if results.get("flops"):
            print(f"  FLOPs             : {results['flops']:,}")
        print(f"  Mean latency      : {results['mean_ms']:.2f} ms ± {results['std_ms']:.2f}")
        print(f"  Min / Max latency : {results['min_ms']:.2f} / {results['max_ms']:.2f} ms")
        print(f"  FPS               : {results['fps']:.1f}")
        print(f"  Peak memory       : {results['peak_mem_mb']:.2f} MB")
        print("=" * 55 + "\n")

    def compare(
        self,
        models_and_labels: List[Tuple[nn.Module, str]],
    ) -> List[Dict[str, Any]]:
        """
        Benchmark multiple models and print a comparison table.

        Args:
            models_and_labels: List of (model, label) tuples.

        Returns:
            List of result dicts in the same order.
        """
        all_results = []
        for model, label in models_and_labels:
            results = self.run(model, label=label)
            all_results.append(results)

        # Comparison table
        print("\n" + "=" * 80)
        print(f"  {'Label':<25} {'FPS':>8} {'Mean ms':>10} {'Params':>12} {'MB':>8}")
        print("-" * 80)
        for r in all_results:
            print(
                f"  {r['label']:<25} "
                f"{r['fps']:>8.1f} "
                f"{r['mean_ms']:>10.2f} "
                f"{r['n_params']:>12,} "
                f"{r['model_size_mb']:>8.2f}"
            )
        print("=" * 80 + "\n")

        return all_results
