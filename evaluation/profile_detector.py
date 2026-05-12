"""
evaluation/profile_detector.py
--------------------------------
Standalone profiling utilities for the YOLOv8n ROI detector.

Wraps Phase 1's generic profiler (utils/profiler.py) and
metrics_compute.py with YOLOv8-specific adaptors:

- YOLOv8's underlying nn.Module is extracted for FLOPs / param counting.
- Memory profiling uses a 3-channel input (YOLOv8 expects RGB).
- A human-readable profiling report is printed and saved to CSV.
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def extract_yolo_module(checkpoint_path: str) -> torch.nn.Module:
    """
    Load a YOLOv8 checkpoint and return the underlying ``nn.Module``
    so standard PyTorch profiling tools can be applied.

    Parameters
    ----------
    checkpoint_path : path to best.pt / last.pt.

    Returns
    -------
    torch.nn.Module  in eval mode.
    """
    from ultralytics import YOLO
    yolo  = YOLO(checkpoint_path)
    model = yolo.model
    model.eval()
    return model


def profile_yolo_detector(
    checkpoint_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    device: str = "cpu",
    num_latency_runs: int = 100,
) -> dict:
    """
    Full profiling pass for a YOLOv8n checkpoint.

    Measures
    --------
    - GFLOPs
    - Total parameters (M)
    - Peak memory during forward pass (MB)
    - Mean / std latency (ms) — single image
    - FPS

    Parameters
    ----------
    checkpoint_path   : path to .pt file.
    input_shape       : (B, C, H, W) — B should be 1 for latency.
    device            : "cpu" or "cuda".
    num_latency_runs  : repeated runs for stable latency estimate.

    Returns
    -------
    dict  All profiling results (floats and strings).
    """
    from evaluation.metrics_compute import count_flops, count_parameters, measure_memory_mb
    from utils.profiler              import profile_forward

    model = extract_yolo_module(checkpoint_path).to(device)

    # ── FLOPs ─────────────────────────────────────────────────────────────────
    gflops = count_flops(model, input_shape=input_shape, device=device)

    # ── Parameters ────────────────────────────────────────────────────────────
    params = count_parameters(model)

    # ── Memory + latency ──────────────────────────────────────────────────────
    fwd = profile_forward(
        model, input_shape=input_shape, device=device, num_runs=num_latency_runs
    )

    result = {
        "checkpoint":     checkpoint_path,
        "device":         device,
        "gflops":         gflops,
        "total_params_M": params["total_params_M"],
        "memory_mb":      fwd["peak_memory_mb"],
        "mean_ms":        fwd["mean_ms"],
        "std_ms":         fwd["std_ms"],
        "fps":            1000.0 / (fwd["mean_ms"] + 1e-9),
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    _print_profile_report(result)
    return result


def _print_profile_report(r: dict) -> None:
    sep = "─" * 50
    print(f"\n{sep}")
    print("  YOLO DETECTOR PROFILING REPORT")
    print(sep)
    print(f"  Checkpoint  : {r['checkpoint']}")
    print(f"  Device      : {r['device']}")
    print(f"  GFLOPs      : {r['gflops']:.4f}")
    print(f"  Params (M)  : {r['total_params_M']:.2f}")
    print(f"  Memory (MB) : {r['memory_mb']:.2f}")
    print(f"  Latency     : {r['mean_ms']:.3f} ± {r['std_ms']:.3f} ms")
    print(f"  FPS         : {r['fps']:.2f}")
    print(sep + "\n")


def save_profile_csv(result: dict, csv_path: str, append: bool = True) -> None:
    """
    Append profiling result to a CSV file.

    Parameters
    ----------
    result   : dict returned by ``profile_yolo_detector``.
    csv_path : output file path.
    append   : if True, append to existing file.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    headers = list(result.keys())
    mode    = "a" if (append and os.path.isfile(csv_path)) else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if mode == "w":
            writer.writeheader()
        writer.writerow(result)
    logger.info("Profiling results saved → %s", csv_path)
