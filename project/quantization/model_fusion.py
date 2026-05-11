"""
model_fusion.py
---------------
Layer fusion utilities for ONNX export optimisation.

For smp UNet++ with EfficientNet-B4, layer fusion is handled
automatically by ONNX Runtime's optimize_model=True flag during
quantization. This module provides:

  1. Pre-export model preparation (eval mode, CPU move)
  2. ONNX graph optimisation (via onnxruntime.transformers or onnx optimizer)
  3. Fusion diagnostics (list fuseable patterns found in an ONNX graph)

Note: torch.quantization.fuse_modules() is NOT used here because
EfficientNet-B4's SiLU activations and complex skip connections are
incompatible with PyTorch's static fusion rules.
"""

import os
from typing import List, Optional

import torch
import torch.nn as nn

from utils.logger import get_logger

_log = get_logger(__name__)


def prepare_model_for_export(
    model: nn.Module,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Set model to eval mode and move to export device (CPU).

    Args:
        model:  PyTorch model.
        device: Target device. Defaults to CPU for ONNX portability.

    Returns:
        Model ready for torch.onnx.export().
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device).eval()

    # Disable gradient computation — not needed for export
    for param in model.parameters():
        param.requires_grad = False

    _log.info("Model prepared for ONNX export (eval, CPU, no grad).")
    return model


def optimise_onnx_graph(
    onnx_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Apply ONNX graph optimisations (constant folding, operator fusion).

    Uses onnxruntime's built-in session options for graph optimisation.
    This includes Conv-BN fusion, GELU approximation, etc.

    Args:
        onnx_path:   Path to input ONNX model.
        output_path: Path to save optimised model. Overwrites input if None.

    Returns:
        Path to optimised ONNX model.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        raise ImportError("Run: pip install onnxruntime")

    if output_path is None:
        base, ext = os.path.splitext(onnx_path)
        output_path = f"{base}_optimised{ext}"

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    sess_options.optimized_model_filepath = output_path

    # Creating a session with these options triggers optimisation + saves
    _ = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    size_mb = os.path.getsize(output_path) / 1e6
    _log.info(f"Optimised ONNX saved → {output_path}  ({size_mb:.2f} MB)")
    return output_path


def list_onnx_ops(onnx_path: str, top_n: int = 20) -> List[str]:
    """
    List the unique op types in an ONNX graph.

    Useful for diagnosing which ops are present before quantization.

    Args:
        onnx_path: Path to ONNX model.
        top_n:     Max number of ops to print.

    Returns:
        List of op type strings.
    """
    try:
        import onnx  # type: ignore
    except ImportError:
        raise ImportError("Run: pip install onnx")

    model  = onnx.load(onnx_path)
    ops    = list({node.op_type for node in model.graph.node})
    ops.sort()

    print(f"\nONNX ops in {os.path.basename(onnx_path)} ({len(ops)} unique):")
    for op in ops[:top_n]:
        print(f"  {op}")
    if len(ops) > top_n:
        print(f"  ... and {len(ops) - top_n} more")

    return ops