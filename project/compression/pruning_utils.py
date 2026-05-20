"""
pruning_utils.py
----------------
Utility functions for structured channel pruning.

Provides:
  - Channel importance scoring (L1, L2, gradient, Taylor)
  - Mask application helpers
  - Pruned model size estimation
  - Channel count inspection
  - ONNX export of pruned models

Reuses:
  - utils.logger    (get_logger)
  - utils.config    (Config)
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from utils.logger import get_logger

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Channel importance scoring                                                  #
# --------------------------------------------------------------------------- #

def l1_channel_importance(weight: torch.Tensor) -> torch.Tensor:
    """
    L1-norm importance: sum of absolute weights per output channel.
    Shape: (out_channels, in_channels, kH, kW) → (out_channels,)
    """
    return weight.abs().sum(dim=(1, 2, 3))


def l2_channel_importance(weight: torch.Tensor) -> torch.Tensor:
    """L2-norm importance per output channel."""
    return weight.pow(2).sum(dim=(1, 2, 3)).sqrt()


def taylor_channel_importance(
    weight: torch.Tensor,
    grad: torch.Tensor,
) -> torch.Tensor:
    """
    Taylor first-order importance: |weight × gradient| per channel.
    Requires one backward pass to populate gradients.
    """
    return (weight * grad).abs().sum(dim=(1, 2, 3))


IMPORTANCE_FNS = {
    "l1":     l1_channel_importance,
    "l2":     l2_channel_importance,
}


def score_conv_channels(
    module: nn.Conv2d,
    method: str = "l1",
    grad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute channel importance scores for a Conv2d module.

    Args:
        module: Conv2d layer.
        method: Scoring method ('l1', 'l2', 'taylor').
        grad:   Weight gradient tensor (required for 'taylor').

    Returns:
        Tensor of shape (out_channels,) with importance scores.
    """
    w = module.weight.data
    if method == "taylor":
        if grad is None:
            _log.warning("Taylor pruning requires gradients — falling back to L1.")
            return l1_channel_importance(w)
        return taylor_channel_importance(w, grad)
    return IMPORTANCE_FNS.get(method, l1_channel_importance)(w)


# --------------------------------------------------------------------------- #
#  Channel mask helpers                                                        #
# --------------------------------------------------------------------------- #

def get_prune_mask(
    scores: torch.Tensor,
    prune_ratio: float,
) -> torch.Tensor:
    """
    Return a boolean mask (True = keep) for channels below the prune threshold.

    Args:
        scores:      Importance scores (out_channels,).
        prune_ratio: Fraction of channels to prune (0.0–0.9).

    Returns:
        Boolean tensor of shape (out_channels,).
    """
    n_prune = max(1, int(len(scores) * prune_ratio))
    n_keep  = max(1, len(scores) - n_prune)   # always keep at least 1 channel
    threshold = scores.topk(n_keep, largest=True).values.min()
    return scores >= threshold


def count_remaining_channels(mask: torch.Tensor) -> int:
    return int(mask.sum().item())


# --------------------------------------------------------------------------- #
#  Model parameter / size utilities                                            #
# --------------------------------------------------------------------------- #

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Estimated FP32 model size in MB."""
    return count_parameters(model) * 4 / 1e6


def get_conv_layers(model: nn.Module) -> Dict[str, nn.Conv2d]:
    """Return all Conv2d layers as {name: module}."""
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    }


def conv_channel_summary(model: nn.Module) -> List[Dict]:
    """Return per-Conv2d channel counts."""
    rows = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            rows.append({
                "name":        name,
                "in_channels": m.in_channels,
                "out_channels": m.out_channels,
                "kernel":      list(m.kernel_size),
                "params":      m.weight.numel(),
            })
    return rows


# --------------------------------------------------------------------------- #
#  ONNX export for pruned model                                               #
# --------------------------------------------------------------------------- #

def export_pruned_onnx(
    model: nn.Module,
    cfg,
    output_path: str,
    opset: int = 18,
) -> str:
    """
    Export a pruned PyTorch model to ONNX.

    Args:
        model:       Pruned model (on CPU).
        cfg:         Config (for image_size, in_channels).
        output_path: Output ONNX path.
        opset:       ONNX opset version.

    Returns:
        Path to exported ONNX file.
    """
    model.eval()
    dummy = torch.zeros(1, cfg.in_channels, *cfg.image_size)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model, dummy, output_path,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(output_path) / 1e6
    _log.info(f"Pruned ONNX → {output_path} ({size_mb:.2f} MB)")
    return output_path