"""
quant_utils.py
--------------
Shared utilities for ONNX export and ONNX Runtime quantization.

Reuses:
  - utils.helpers  (model_size_mb, count_parameters)
  - utils.logger   (get_logger, save_checkpoint)
  - evaluation.benchmark (time_inference for latency)

No Phase 1/2 code is duplicated here.
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from utils.config import Config
from utils.logger import get_logger

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  ONNX export                                                                 #
# --------------------------------------------------------------------------- #

def export_to_onnx(
    model: nn.Module,
    cfg: Config,
    filename: Optional[str] = None,
    opset_version: int = 17,
    device: Optional[torch.device] = None,
) -> str:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model:         Model in eval mode.
        cfg:           Project Config.
        filename:      Output filename. Defaults to 'baseline_fp32.onnx'.
        opset_version: ONNX opset version.
        device:        Export device (CPU recommended for portability).

    Returns:
        Path to the saved ONNX file.
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device).eval()

    dummy_input = torch.randn(
        1, cfg.in_channels, *cfg.image_size, device=device
    )

    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    fname = filename or "baseline_fp32.onnx"
    path  = os.path.join(cfg.checkpoints_dir, fname)

    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        verbose=False,
    )

    size_mb = os.path.getsize(path) / 1e6
    _log.info(f"ONNX model exported → {path}  ({size_mb:.2f} MB)")
    return path


# --------------------------------------------------------------------------- #
#  ONNX Runtime inference session                                              #
# --------------------------------------------------------------------------- #

def create_ort_session(
    onnx_path: str,
    providers: list = None,
):
    """
    Create an ONNX Runtime InferenceSession.

    Args:
        onnx_path:  Path to .onnx file.
        providers:  Execution providers list.

    Returns:
        onnxruntime.InferenceSession
    """
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        raise ImportError(
            "onnxruntime not installed. Run: pip install onnxruntime"
        )

    if providers is None:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_path, providers=providers)
    _log.info(
        f"ORT session created: {onnx_path} | "
        f"providers={session.get_providers()}"
    )
    return session


# --------------------------------------------------------------------------- #
#  ONNX Runtime inference                                                      #
# --------------------------------------------------------------------------- #

def run_ort_inference(
    session,
    image: np.ndarray,
    input_name: str = "input",
) -> np.ndarray:
    """
    Run a single inference with an ORT session.

    Args:
        session:    ORT InferenceSession.
        image:      Input array (B, C, H, W) float32.
        input_name: Input node name.

    Returns:
        Output logits array (B, num_classes, H, W).
    """
    outputs = session.run(None, {input_name: image})
    return outputs[0]


# --------------------------------------------------------------------------- #
#  ONNX file size utility                                                      #
# --------------------------------------------------------------------------- #

def onnx_size_mb(onnx_path: str) -> float:
    """Return ONNX file size in MB."""
    return os.path.getsize(onnx_path) / 1e6


# --------------------------------------------------------------------------- #
#  Compression ratio                                                           #
# --------------------------------------------------------------------------- #

def compression_ratio(fp32_size_mb: float, quant_size_mb: float) -> float:
    """
    Compute compression ratio: fp32_size / quant_size.
    A ratio of 4.0 means the quantized model is 4× smaller.
    """
    if quant_size_mb <= 0:
        return 0.0
    return fp32_size_mb / quant_size_mb