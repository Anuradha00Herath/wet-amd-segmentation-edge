"""
profiler.py
-----------
PyTorch-native model profiling utilities.

Wraps torch.profiler to generate:
  - Operator-level CPU/CUDA time breakdown
  - Memory allocation traces
  - Chrome-trace JSON files for visualisation in chrome://tracing
"""

import os
from typing import Callable, Optional

import torch
import torch.nn as nn

from utils.config import Config
from utils.logger import get_logger

_log = get_logger(__name__)


def profile_model(
    model: nn.Module,
    input_tensor: torch.Tensor,
    cfg: Config,
    n_warmup: int = 3,
    n_active: int = 5,
    trace_name: str = "profile_trace",
    export_chrome_trace: bool = True,
) -> None:
    """
    Profile a model using torch.profiler and optionally export a
    Chrome trace for visual inspection.

    Args:
        model:               Model in eval mode.
        input_tensor:        Example input (B, C, H, W).
        cfg:                 Project Config (used for results path).
        n_warmup:            Warm-up steps before profiling.
        n_active:            Number of steps to actively record.
        trace_name:          Base filename for the JSON trace.
        export_chrome_trace: Save .json trace to cfg.results_dir.
    """
    device = input_tensor.device
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    trace_path = os.path.join(cfg.results_dir, f"{trace_name}.json")

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
            for step in range(n_warmup + n_active):
                _ = model(input_tensor)
                prof.step()

    # Print top-20 operators by self CPU time
    _log.info("Top-20 ops by self CPU time:")
    print(
        prof.key_averages(group_by_input_shape=False).table(
            sort_by="self_cpu_time_total", row_limit=20
        )
    )

    if export_chrome_trace:
        os.makedirs(cfg.results_dir, exist_ok=True)
        prof.export_chrome_trace(trace_path)
        _log.info(f"Chrome trace saved to {trace_path}")
        _log.info("Open with chrome://tracing or https://ui.perfetto.dev/")


def estimate_memory_footprint(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> dict:
    """
    Estimate the model's memory footprint during a single forward pass.

    Returns:
        Dict with 'param_mb', 'activation_mb' (CUDA only), 'total_mb'.
    """
    from utils.helpers import model_size_mb

    param_mb = model_size_mb(model)

    if input_tensor.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated() / 1e6
        with torch.no_grad():
            _ = model(input_tensor)
        after = torch.cuda.max_memory_allocated() / 1e6
        activation_mb = max(0.0, after - before - param_mb)
    else:
        activation_mb = 0.0  # CPU RSS measurement is imprecise; skip

    return {
        "param_mb": param_mb,
        "activation_mb": activation_mb,
        "total_mb": param_mb + activation_mb,
    }
