"""
evaluation/metrics_throughput.py
---------------------------------
Throughput benchmarking: FPS and per-frame inference latency.

Design decisions
----------------
- A configurable number of warm-up batches are discarded before timing
  starts, avoiding JIT / cache cold-start artefacts.
- CUDA synchronisation (``torch.cuda.synchronize``) is called before
  every timing boundary so GPU wall-clock time is captured correctly.
- Timing is accumulated over the full test loader so results reflect
  real-world throughput (not a single-batch micro-benchmark).
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _sync(device: str) -> None:
    """Block until all CUDA kernels finish (no-op on CPU)."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def benchmark_throughput(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
    num_warmup_batches: int = 3,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Measure inference throughput over a DataLoader.

    Parameters
    ----------
    model               : model in eval mode, already on ``device``.
    data_loader         : provides (image, mask) batches.
    device              : torch device string.
    num_warmup_batches  : batches to discard before timing begins.
    max_batches         : cap total timed batches (None = all).

    Returns
    -------
    dict with keys:
        fps             — frames per second.
        ms_per_frame    — milliseconds per frame.
        total_images    — number of images timed.
        total_time_s    — total wall-clock seconds for timed batches.
    """
    model.eval()
    total_images = 0
    total_time_s = 0.0
    batch_num    = 0

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            bs     = images.size(0)

            if batch_num < num_warmup_batches:
                # Warm-up: run but do not time
                _sync(device)
                _ = model(images)
                _sync(device)
                batch_num += 1
                continue

            # Timed inference
            _sync(device)
            t0 = time.perf_counter()
            _  = model(images)
            _sync(device)
            t1 = time.perf_counter()

            total_time_s += (t1 - t0)
            total_images += bs
            batch_num    += 1

            if max_batches is not None and (batch_num - num_warmup_batches) >= max_batches:
                break

    if total_images == 0:
        logger.warning("No images were timed (dataset smaller than warmup?).")
        return {"fps": float("nan"), "ms_per_frame": float("nan"),
                "total_images": 0, "total_time_s": 0.0}

    fps          = total_images / total_time_s
    ms_per_frame = (total_time_s / total_images) * 1000.0

    logger.info(
        "Throughput → FPS: %.2f | ms/frame: %.3f | images: %d",
        fps, ms_per_frame, total_images,
    )
    return {
        "fps":           fps,
        "ms_per_frame":  ms_per_frame,
        "total_images":  total_images,
        "total_time_s":  total_time_s,
    }


def single_image_latency(
    model: nn.Module,
    input_shape: tuple = (1, 1, 256, 256),
    device: str = "cpu",
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, float]:
    """
    Measure single-image (batch=1) latency statistics via repeated runs.

    Useful for edge-device deployment profiling.

    Returns
    -------
    dict with keys: mean_ms, std_ms, min_ms, max_ms, median_ms.
    """
    import statistics

    model.eval()
    dummy  = torch.randn(*input_shape).to(device)
    times  = []

    with torch.no_grad():
        for _ in range(num_warmup):
            _sync(device)
            _ = model(dummy)
            _sync(device)

        for _ in range(num_runs):
            _sync(device)
            t0 = time.perf_counter()
            _  = model(dummy)
            _sync(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

    result = {
        "mean_ms":   statistics.mean(times),
        "std_ms":    statistics.stdev(times),
        "min_ms":    min(times),
        "max_ms":    max(times),
        "median_ms": statistics.median(times),
    }
    logger.info(
        "Single-image latency → mean=%.3f ms  std=%.3f ms", 
        result["mean_ms"], result["std_ms"],
    )
    return result
