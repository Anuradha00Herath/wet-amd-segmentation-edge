"""
profiler.py
-----------
Model profiling utilities for Phase 2 benchmarking.

Measures:
  - FLOPs (via thop)
  - Parameter count
  - Model size (MB)
  - Runtime RAM (via tracemalloc)
  - CPU utilisation % (via psutil)
  - Inference latency (mean/std/min/max)
  - FPS
  - torch.profiler operator breakdown

All functions are model-agnostic and reusable across FP32, PTQ, QAT,
mixed-precision, and ONNX phases.
"""

import os
import time
import tracemalloc
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from utils.config import Config
from utils.logger import get_logger

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  FLOPs & parameter counting                                                  #
# --------------------------------------------------------------------------- #

def count_flops(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> Optional[int]:
    """
    Estimate FLOPs via thop. Returns None if thop is not installed.

    Args:
        model:        Model in eval mode.
        input_tensor: Example input (B, C, H, W).

    Returns:
        Integer FLOPs count or None.
    """
    try:
        from thop import profile  # type: ignore
        flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
        return int(flops)
    except Exception:
        return None


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count total or trainable parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Approximate model size in MB (float32 = 4 bytes per param)."""
    return count_parameters(model) * 4 / 1e6


# --------------------------------------------------------------------------- #
#  RAM profiling via tracemalloc                                               #
# --------------------------------------------------------------------------- #

def profile_ram(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> Dict[str, float]:
    """
    Measure peak RAM increase during a single forward pass using tracemalloc.

    Args:
        model:        Model in eval mode.
        input_tensor: Example input on CPU.

    Returns:
        Dict with keys: peak_ram_mb, current_ram_mb.
    """
    model  = model.cpu().eval()
    tensor = input_tensor.cpu()

    tracemalloc.start()
    with torch.no_grad():
        _ = model(tensor)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "peak_ram_mb":    peak    / 1e6,
        "current_ram_mb": current / 1e6,
    }


# --------------------------------------------------------------------------- #
#  CPU utilisation                                                              #
# --------------------------------------------------------------------------- #

def profile_cpu_utilisation(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_runs: int = 20,
) -> Dict[str, float]:
    """
    Measure CPU utilisation during repeated inference using psutil.

    Args:
        model:        Model in eval mode.
        input_tensor: Example input.
        n_runs:       Number of inference runs to average over.

    Returns:
        Dict with keys: mean_cpu_pct, max_cpu_pct.
    """
    try:
        import psutil
    except ImportError:
        _log.warning("psutil not installed. CPU utilisation unavailable.")
        return {"mean_cpu_pct": 0.0, "max_cpu_pct": 0.0}

    process = psutil.Process(os.getpid())
    cpu_readings: List[float] = []

    model.eval()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(input_tensor)
            cpu_readings.append(process.cpu_percent(interval=None))

    return {
        "mean_cpu_pct": sum(cpu_readings) / len(cpu_readings),
        "max_cpu_pct":  max(cpu_readings),
    }


# --------------------------------------------------------------------------- #
#  Latency profiling                                                            #
# --------------------------------------------------------------------------- #

def profile_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_runs: int = 100,
    warmup: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Measure per-inference latency with warm-up.

    Args:
        model:        Model in eval mode.
        input_tensor: Input tensor (B, C, H, W).
        n_runs:       Timed runs.
        warmup:       Warm-up runs (not timed).
        device:       Target device.

    Returns:
        Dict with keys: mean_ms, std_ms, min_ms, max_ms, fps, median_ms.
    """
    import statistics

    if device is None:
        device = input_tensor.device

    model        = model.to(device).eval()
    input_tensor = input_tensor.to(device)

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    latencies: List[float] = []

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        _sync()

        for _ in range(n_runs):
            _sync()
            t0 = time.perf_counter()
            _ = model(input_tensor)
            _sync()
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
#  torch.profiler operator breakdown                                            #
# --------------------------------------------------------------------------- #

def profile_operators(
    model: nn.Module,
    input_tensor: torch.Tensor,
    cfg: Config,
    n_warmup: int = 3,
    n_active: int = 5,
    trace_name: str = "profile_trace",
    export_chrome_trace: bool = True,
) -> None:
    """
    Run torch.profiler and print top-20 operators by CPU time.
    Optionally exports a Chrome trace JSON for visual inspection.

    Args:
        model:               Model in eval mode.
        input_tensor:        Example input (B, C, H, W).
        cfg:                 Config (used for results path).
        n_warmup:            Warm-up steps.
        n_active:            Active profiling steps.
        trace_name:          Base filename for JSON trace.
        export_chrome_trace: Save .json to cfg.results_dir.
    """
    device     = input_tensor.device
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    schedule = torch.profiler.schedule(
        wait=0, warmup=n_warmup, active=n_active, repeat=1
    )

    model.eval()
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(n_warmup + n_active):
                _ = model(input_tensor)
                prof.step()

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    if export_chrome_trace:
        os.makedirs(cfg.results_dir, exist_ok=True)
        path = os.path.join(cfg.results_dir, f"{trace_name}.json")
        prof.export_chrome_trace(path)
        _log.info(f"Chrome trace saved → {path}")
        _log.info("View at: https://ui.perfetto.dev/")


# --------------------------------------------------------------------------- #
#  Unified ModelProfiler                                                       #
# --------------------------------------------------------------------------- #

class ModelProfiler:
    """
    All-in-one profiler: FLOPs, params, size, RAM, CPU%, latency.

    Usage:
        profiler = ModelProfiler(cfg, device)
        report   = profiler.run(model, label="baseline_fp32")
        profiler.print_report(report)
    """

    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg    = cfg
        self.device = device

    def _dummy_input(self) -> torch.Tensor:
        c    = self.cfg.in_channels
        h, w = self.cfg.image_size
        return torch.randn(1, c, h, w)

    def run(
        self,
        model: nn.Module,
        label: str = "model",
        input_tensor: Optional[torch.Tensor] = None,
        n_runs: int = 100,
        warmup: int = 10,
    ) -> Dict[str, Any]:
        """
        Run full profiling suite and return results dict.

        Args:
            model:        Model to profile.
            label:        Descriptive label.
            input_tensor: Custom input; defaults to random dummy.
            n_runs:       Latency runs.
            warmup:       Warm-up runs.

        Returns:
            Dict with all profiling metrics.
        """
        if input_tensor is None:
            input_tensor = self._dummy_input()

        _log.info(f"Profiling '{label}' ...")

        model.eval()

        # Static metrics
        n_params  = count_parameters(model)
        size_mb   = model_size_mb(model)
        flops     = count_flops(model.cpu(), input_tensor.cpu())

        # RAM (CPU)
        ram = profile_ram(model, input_tensor)

        # Latency
        latency = profile_latency(
            model, input_tensor,
            n_runs=n_runs, warmup=warmup, device=self.device,
        )

        # CPU utilisation
        cpu_util = profile_cpu_utilisation(
            model.cpu(), input_tensor.cpu(), n_runs=20
        )

        # Peak CUDA memory
        peak_gpu_mb = 0.0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model.to(self.device)(input_tensor.to(self.device))
            peak_gpu_mb = torch.cuda.max_memory_allocated() / 1e6

        return {
            "label":          label,
            "device":         str(self.device),
            "n_params":       n_params,
            "model_size_mb":  size_mb,
            "flops":          flops,
            "peak_ram_mb":    ram["peak_ram_mb"],
            "peak_gpu_mb":    peak_gpu_mb,
            "mean_cpu_pct":   cpu_util["mean_cpu_pct"],
            "max_cpu_pct":    cpu_util["max_cpu_pct"],
            **latency,
        }

    @staticmethod
    def print_report(report: Dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print(f"  Profile: {report['label']}")
        print("=" * 60)
        print(f"  Device            : {report['device']}")
        print(f"  Parameters        : {report['n_params']:,}")
        print(f"  Model size        : {report['model_size_mb']:.2f} MB")
        if report.get("flops"):
            print(f"  FLOPs             : {report['flops']:,}")
        print(f"  Peak RAM          : {report['peak_ram_mb']:.2f} MB")
        if report["peak_gpu_mb"] > 0:
            print(f"  Peak GPU mem      : {report['peak_gpu_mb']:.2f} MB")
        print(f"  Mean latency      : {report['mean_ms']:.2f} ± {report['std_ms']:.2f} ms")
        print(f"  Median latency    : {report['median_ms']:.2f} ms")
        print(f"  Min / Max         : {report['min_ms']:.2f} / {report['max_ms']:.2f} ms")
        print(f"  FPS               : {report['fps']:.1f}")
        print(f"  CPU utilisation   : {report['mean_cpu_pct']:.1f}% (max {report['max_cpu_pct']:.1f}%)")
        print("=" * 60 + "\n")

    def estimate_memory_footprint(
        self, model: nn.Module, input_tensor: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Convenience wrapper returning size + RAM dict."""
        if input_tensor is None:
            input_tensor = self._dummy_input()
        ram = profile_ram(model, input_tensor)
        return {"model_size_mb": model_size_mb(model), **ram}


# Keep old function name for backward compatibility with Phase 1 imports
def profile_model(
    model: nn.Module,
    input_tensor: torch.Tensor,
    cfg: Config,
    **kwargs,
) -> None:
    """Backward-compatible wrapper around profile_operators."""
    profile_operators(model, input_tensor, cfg, **kwargs)


def estimate_memory_footprint(
    model: nn.Module, input_tensor: torch.Tensor
) -> Dict[str, float]:
    """Backward-compatible wrapper."""
    return {
        "model_size_mb": model_size_mb(model),
        **profile_ram(model, input_tensor),
    }
