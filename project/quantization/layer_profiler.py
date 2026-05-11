"""
layer_profiler.py
-----------------
Per-layer profiling utilities for sensitivity analysis.

Profiles:
  - Parameter count and size per named module
  - Activation statistics (mean, std, min, max) via forward hooks
  - Per-layer latency contribution via torch.profiler
  - ONNX node group identification for sensitivity experiments

Reuses:
  - evaluation.profiler  (model_size_mb, count_parameters)
  - utils.logger         (get_logger)
"""

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from utils.config import Config
from utils.logger import get_logger

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Module-level parameter stats                                                #
# --------------------------------------------------------------------------- #

def profile_module_parameters(model: nn.Module) -> List[Dict[str, Any]]:
    """
    Return per-module parameter counts and sizes.

    Args:
        model: PyTorch model.

    Returns:
        List of dicts with keys: name, type, n_params, size_kb.
    """
    rows = []
    for name, module in model.named_modules():
        n = sum(p.numel() for p in module.parameters(recurse=False))
        if n == 0:
            continue
        rows.append({
            "name":    name,
            "type":    type(module).__name__,
            "n_params": n,
            "size_kb": n * 4 / 1e3,   # float32
        })
    return sorted(rows, key=lambda r: r["n_params"], reverse=True)


# --------------------------------------------------------------------------- #
#  Activation statistics via forward hooks                                     #
# --------------------------------------------------------------------------- #

class ActivationProfiler:
    """
    Collect per-layer activation statistics during a forward pass.

    Usage:
        profiler = ActivationProfiler(model)
        profiler.register()
        with torch.no_grad():
            _ = model(dummy_input)
        stats = profiler.stats()
        profiler.remove()
    """

    def __init__(self, model: nn.Module, target_types=(nn.Conv2d,)) -> None:
        self.model        = model
        self.target_types = target_types
        self._handles: list = []
        self._stats: Dict[str, Dict] = {}

    def _make_hook(self, name: str):
        def hook(module, input, output):
            with torch.no_grad():
                t = output.detach().float()
                self._stats[name] = {
                    "mean":  t.mean().item(),
                    "std":   t.std().item(),
                    "min":   t.min().item(),
                    "max":   t.max().item(),
                    "shape": list(t.shape),
                }
        return hook

    def register(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, self.target_types):
                h = module.register_forward_hook(self._make_hook(name))
                self._handles.append(h)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def stats(self) -> Dict[str, Dict]:
        return self._stats


# --------------------------------------------------------------------------- #
#  ONNX node group identification                                              #
# --------------------------------------------------------------------------- #

ONNX_LAYER_GROUPS = {
    "encoder_stem":    ["/encoder/_conv_stem"],
    "encoder_blocks":  ["/encoder/_blocks"],
    "encoder_head":    ["/encoder/_conv_head"],
    "decoder":         ["/decoder"],
    "segmentation_head": ["/segmentation_head"],
}


def get_onnx_nodes_by_group(onnx_path: str) -> Dict[str, List[str]]:
    """
    Return ONNX node names grouped by architectural component.

    Args:
        onnx_path: Path to FP32 ONNX model.

    Returns:
        Dict mapping group_name → list of node names.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError("Run: pip install onnx")

    model  = onnx.load(onnx_path)
    groups = defaultdict(list)

    for node in model.graph.node:
        name = node.name or ""
        assigned = False
        for group, prefixes in ONNX_LAYER_GROUPS.items():
            if any(p in name for p in prefixes):
                groups[group].append(name)
                assigned = True
                break
        if not assigned:
            groups["other"].append(name)

    return dict(groups)


def list_quantizable_nodes(onnx_path: str) -> List[str]:
    """
    Return names of all Conv nodes in the ONNX graph (quantizable ops).

    Args:
        onnx_path: Path to FP32 ONNX model.

    Returns:
        List of Conv node name strings.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError("Run: pip install onnx")

    model = onnx.load(onnx_path)
    return [n.name for n in model.graph.node if n.op_type == "Conv"]
