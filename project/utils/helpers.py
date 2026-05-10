"""
helpers.py
----------
General-purpose utility functions shared across the project.

Includes:
  - Device selection
  - Automatic directory creation
  - Dataset statistics computation
  - Parameter / FLOPs counting
  - Pretty-printing utilities
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# --------------------------------------------------------------------------- #
#  Device management                                                            #
# --------------------------------------------------------------------------- #

def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Return the best available device.

    Args:
        prefer_gpu: Use GPU when available (default True).

    Returns:
        torch.device – 'cuda', 'mps', or 'cpu'.
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # Apple Silicon
            return torch.device("mps")
    return torch.device("cpu")


def device_info(device: torch.device) -> str:
    """Return a human-readable description of the current device."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        mem = torch.cuda.get_device_properties(device).total_memory / 1e9
        return f"CUDA – {name} ({mem:.1f} GB)"
    return device.type.upper()


# --------------------------------------------------------------------------- #
#  Directory helpers                                                            #
# --------------------------------------------------------------------------- #

def ensure_dirs(paths: List[str]) -> None:
    """Create each directory path if it does not exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Model introspection                                                          #
# --------------------------------------------------------------------------- #

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of (trainable) parameters in a model.

    Args:
        model:          PyTorch model.
        trainable_only: Only count parameters with requires_grad=True.

    Returns:
        Integer parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Return approximate model size in megabytes (float32 assumed)."""
    n_params = count_parameters(model, trainable_only=False)
    return n_params * 4 / 1e6   # 4 bytes per float32


def estimate_flops(model: nn.Module, input_tensor: torch.Tensor) -> Optional[int]:
    """
    Estimate FLOPs via thop (if installed) or return None.

    Args:
        model:        PyTorch model in eval mode.
        input_tensor: Example input with batch dimension.

    Returns:
        FLOPs count or None if thop is not available.
    """
    try:
        from thop import profile  # type: ignore
        flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
        return int(flops)
    except ImportError:
        return None


# --------------------------------------------------------------------------- #
#  Dataset statistics                                                           #
# --------------------------------------------------------------------------- #

def compute_dataset_stats(loader: DataLoader) -> Tuple[List[float], List[float]]:
    """
    Compute per-channel mean and std over an image DataLoader.

    Returns:
        (mean, std) each as a list of floats, one value per channel.
    """
    mean = torch.zeros(0)
    std = torch.zeros(0)
    n_batches = 0

    for images, _ in loader:
        # images: (B, C, H, W)
        b, c, h, w = images.shape
        if n_batches == 0:
            mean = torch.zeros(c)
            m2 = torch.zeros(c)

        images = images.view(b, c, -1)  # (B, C, H*W)
        mean += images.mean(dim=[0, 2])
        m2 += images.std(dim=[0, 2])
        n_batches += 1

    mean /= n_batches
    std = m2 / n_batches
    return mean.tolist(), std.tolist()


# --------------------------------------------------------------------------- #
#  Timing                                                                       #
# --------------------------------------------------------------------------- #

class Timer:
    """Simple wall-clock timer with lap support."""

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self.laps: List[float] = []

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def lap(self) -> float:
        assert self._start is not None, "Call Timer.start() first."
        elapsed = time.perf_counter() - self._start
        self.laps.append(elapsed)
        self._start = time.perf_counter()
        return elapsed

    def stop(self) -> float:
        return self.lap()

    @property
    def mean_lap(self) -> float:
        return sum(self.laps) / len(self.laps) if self.laps else 0.0


# --------------------------------------------------------------------------- #
#  Pretty-printing                                                              #
# --------------------------------------------------------------------------- #

def print_model_summary(model: nn.Module, input_size: Tuple, device: torch.device) -> None:
    """Print parameter count, model size, and (optionally) FLOPs."""
    n_params = count_parameters(model)
    size_mb = model_size_mb(model)

    dummy = torch.zeros(1, *input_size).to(device)
    flops = estimate_flops(model.to(device), dummy)

    print("=" * 50)
    print(f"  Trainable parameters : {n_params:,}")
    print(f"  Model size (float32) : {size_mb:.2f} MB")
    if flops is not None:
        print(f"  Estimated FLOPs      : {flops:,}")
    print("=" * 50)


def print_dict(d: Dict, title: str = "") -> None:
    """Pretty-print a dictionary."""
    if title:
        print(f"\n{title}")
        print("-" * len(title))
    for k, v in d.items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:.6f}")
        else:
            print(f"  {k:<30} {v}")
