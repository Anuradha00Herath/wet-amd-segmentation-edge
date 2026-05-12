"""
evaluation/metrics_compute.py
------------------------------
Computational cost metrics: FLOPs, parameter count, memory usage.

Uses ``ptflops`` (preferred) with ``thop`` as fallback so the code works
even if one library has an API-breaking update.

Design decisions
----------------
- All functions accept a model + dummy input and return plain dicts.
- Memory is measured as the *peak* resident memory during a forward pass,
  not just model weights — this captures activation overhead too.
- GPU memory is read from ``torch.cuda`` when a CUDA device is active.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── FLOPs ──────────────────────────────────────────────────────────────────────

def count_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 256, 256),
    device: str = "cpu",
) -> float:
    """
    Count multiply-accumulate operations (GFLOPs) for a single forward pass.

    Tries ``ptflops`` first; falls back to ``thop``.

    Parameters
    ----------
    model        : model in eval mode.
    input_shape  : (B, C, H, W) — batch size must be 1.
    device       : torch device string.

    Returns
    -------
    float  GFLOPs (giga multiply-accumulate operations).
    """
    model = model.to(device).eval()

    # ── ptflops ───────────────────────────────────────────────────────────────
    try:
        from ptflops import get_model_complexity_info
        # ptflops expects (C, H, W) without batch dim
        input_res = tuple(input_shape[1:])
        macs, _   = get_model_complexity_info(
            model, input_res,
            as_strings=False, print_per_layer_stat=False, verbose=False,
        )
        gflops = macs / 1e9
        logger.info("FLOPs (ptflops): %.4f GFLOPs", gflops)
        return gflops
    except Exception as e:
        logger.warning("ptflops failed (%s); trying thop.", e)

    # ── thop fallback ─────────────────────────────────────────────────────────
    try:
        from thop import profile
        dummy = torch.randn(*input_shape).to(device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        gflops  = macs / 1e9
        logger.info("FLOPs (thop): %.4f GFLOPs", gflops)
        return gflops
    except Exception as e:
        logger.error("thop also failed (%s). Returning NaN.", e)
        return float("nan")


# ── Parameters ────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters.

    Returns
    -------
    dict with keys: total_params (int), trainable_params (int),
                    total_params_M (float), trainable_params_M (float).
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params":       total,
        "trainable_params":   trainable,
        "total_params_M":     total     / 1e6,
        "trainable_params_M": trainable / 1e6,
    }


# ── Memory ────────────────────────────────────────────────────────────────────

def measure_memory_mb(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 256, 256),
    device: str = "cpu",
) -> dict:
    """
    Measure peak memory usage during a forward pass in MB.

    For CPU: uses Python's ``tracemalloc``.
    For GPU: uses ``torch.cuda.max_memory_allocated``.

    Returns
    -------
    dict with keys:
        memory_mb (float) — peak allocated during forward pass.
        device (str)
    """
    model  = model.to(device).eval()
    dummy  = torch.randn(*input_shape).to(device)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(dummy)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_mb    = peak_bytes / (1024 ** 2)
        logger.info("Peak GPU memory: %.2f MB", peak_mb)
        return {"memory_mb": peak_mb, "device": device}

    # CPU path — tracemalloc
    import tracemalloc
    gc.collect()
    tracemalloc.start()
    with torch.no_grad():
        _ = model(dummy)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 ** 2)
    logger.info("Peak CPU memory (tracemalloc): %.2f MB", peak_mb)
    return {"memory_mb": peak_mb, "device": device}


# ── Convenience aggregator ─────────────────────────────────────────────────────

def compute_all_compute_metrics(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 256, 256),
    device: str = "cpu",
) -> dict:
    """
    Compute FLOPs, parameter count, and memory in a single call.

    Returns
    -------
    dict with keys: gflops, total_params_M, trainable_params_M, memory_mb.
    """
    flops  = count_flops(model, input_shape, device)
    params = count_parameters(model)
    mem    = measure_memory_mb(model, input_shape, device)
    return {
        "gflops":             flops,
        "total_params_M":     params["total_params_M"],
        "trainable_params_M": params["trainable_params_M"],
        "memory_mb":          mem["memory_mb"],
    }
