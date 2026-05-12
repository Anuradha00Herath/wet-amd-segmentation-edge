"""
utils/profiler.py
-----------------
Lightweight, reusable profiling utilities.

All utilities are self-contained and import-safe — they degrade gracefully
when optional libraries (pynvml, psutil) are unavailable.

Classes / functions
-------------------
- Timer              : context-manager wall-clock timer.
- CPUMemorySnapshot  : peak RSS before / after a block.
- GPUMemorySnapshot  : peak CUDA allocated memory.
- SystemSnapshot     : CPU % + RAM summary at a point in time.
- profile_forward    : one-shot forward-pass profiler (time + memory).
"""

from __future__ import annotations

import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Wall-clock timer ──────────────────────────────────────────────────────────

class Timer:
    """
    Context-manager and manual start/stop timer.

    Usage::

        with Timer() as t:
            do_work()
        print(t.elapsed_ms)  # milliseconds
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._end:   Optional[float] = None

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> "Timer":
        self._end = time.perf_counter()
        return self

    @property
    def elapsed_s(self) -> float:
        if self._start is None or self._end is None:
            raise RuntimeError("Timer has not been started and stopped.")
        return self._end - self._start

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000.0

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


# ── CPU / system memory ───────────────────────────────────────────────────────

def _get_rss_mb() -> float:
    """Current process RSS in MB using psutil (returns NaN if unavailable)."""
    try:
        import psutil, os
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 ** 2)
    except Exception:
        return float("nan")


@dataclass
class CPUMemorySnapshot:
    """
    Measure peak RSS increase during a code block.

    Usage::

        snap = CPUMemorySnapshot()
        snap.before()
        do_work()
        snap.after()
        print(snap.delta_mb)
    """
    rss_before_mb: float = 0.0
    rss_after_mb:  float = 0.0

    def before(self) -> None:
        gc.collect()
        self.rss_before_mb = _get_rss_mb()

    def after(self) -> None:
        self.rss_after_mb = _get_rss_mb()

    @property
    def delta_mb(self) -> float:
        return self.rss_after_mb - self.rss_before_mb


# ── GPU memory ────────────────────────────────────────────────────────────────

@dataclass
class GPUMemorySnapshot:
    """
    Record peak CUDA allocated memory during a code block.

    Usage::

        snap = GPUMemorySnapshot("cuda:0")
        snap.before()
        model(x)
        snap.after()
        print(snap.peak_mb)
    """
    device: str = "cuda:0"
    peak_mb: float = 0.0

    def before(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def after(self) -> None:
        if torch.cuda.is_available():
            self.peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        else:
            self.peak_mb = float("nan")


# ── System snapshot ───────────────────────────────────────────────────────────

def system_snapshot() -> Dict[str, float]:
    """
    Return a point-in-time snapshot of system resource usage.

    Returns
    -------
    dict with keys:
        cpu_percent      — instantaneous CPU utilisation (%).
        ram_used_mb      — used RAM in MB.
        ram_total_mb     — total RAM in MB.
        gpu_util_percent — GPU utilisation % (NaN if unavailable).
        gpu_mem_used_mb  — GPU memory used in MB (NaN if unavailable).
    """
    result: Dict[str, float] = {
        "cpu_percent":      float("nan"),
        "ram_used_mb":      float("nan"),
        "ram_total_mb":     float("nan"),
        "gpu_util_percent": float("nan"),
        "gpu_mem_used_mb":  float("nan"),
    }

    try:
        import psutil
        result["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        vm = psutil.virtual_memory()
        result["ram_used_mb"]  = vm.used  / (1024 ** 2)
        result["ram_total_mb"] = vm.total / (1024 ** 2)
    except Exception as e:
        logger.debug("psutil unavailable: %s", e)

    if torch.cuda.is_available():
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
            result["gpu_util_percent"] = util.gpu
            result["gpu_mem_used_mb"]  = mem.used / (1024 ** 2)
        except Exception as e:
            logger.debug("pynvml unavailable: %s", e)

    return result


# ── One-shot forward-pass profiler ────────────────────────────────────────────

def profile_forward(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 256, 256),
    device: str = "cpu",
    num_runs: int = 50,
) -> Dict[str, float]:
    """
    Profile a single forward pass: timing + memory.

    Parameters
    ----------
    model       : model in eval mode.
    input_shape : (B, C, H, W).
    device      : torch device string.
    num_runs    : number of timed runs (mean is reported).

    Returns
    -------
    dict with keys:
        mean_ms, std_ms — latency statistics.
        peak_memory_mb  — peak memory during forward (CPU or GPU).
        device
    """
    import statistics

    model  = model.to(device).eval()
    dummy  = torch.randn(*input_shape).to(device)
    times  = []

    def _sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _sync(); _ = model(dummy); _sync()

    # Timed runs
    with torch.no_grad():
        for _ in range(num_runs):
            _sync()
            t0 = time.perf_counter()
            _  = model(dummy)
            _sync()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    # Memory
    if device.startswith("cuda"):
        snap = GPUMemorySnapshot(device)
    else:
        snap_cpu = CPUMemorySnapshot()
        snap_cpu.before()

    with torch.no_grad():
        if device.startswith("cuda"):
            snap.before()
            _ = model(dummy)
            snap.after()
            peak_mb = snap.peak_mb
        else:
            snap_cpu.before()
            _ = model(dummy)
            snap_cpu.after()
            peak_mb = snap_cpu.delta_mb

    result = {
        "mean_ms":       statistics.mean(times),
        "std_ms":        statistics.stdev(times) if len(times) > 1 else 0.0,
        "peak_memory_mb": peak_mb,
        "device":        device,
    }
    logger.info(
        "profile_forward → mean=%.3f ms  peak_mem=%.2f MB  device=%s",
        result["mean_ms"], result["peak_memory_mb"], device,
    )
    return result
