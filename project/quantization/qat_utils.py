"""
qat_utils.py
------------
Shared utilities for the QAT pipeline.

Reuses Phase 1/2/3 utilities — no duplicate code.
New additions specific to QAT:
  - Weight fake-quantization hooks
  - QAT checkpoint save/load
  - QAT → ONNX export wrapper
"""

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from utils.config import Config
from utils.logger import get_logger, save_checkpoint, load_checkpoint
from quantization.fake_quant_config import FakeQuantConfig, fake_quantize_tensor
from quantization.quant_utils import export_to_onnx  # reuse Phase 3 export

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Weight fake-quantization hook                                               #
# --------------------------------------------------------------------------- #

class FakeQuantWeightHook:
    """
    Forward pre-hook that fake-quantizes Conv2d weights before each
    forward pass during QAT training.

    Registered via model.register_forward_pre_hook() — removed automatically
    when training is complete.

    Usage:
        hook = FakeQuantWeightHook(fq_cfg)
        handle = model.register_forward_pre_hook(hook)
        # ... train ...
        handle.remove()  # remove before ONNX export
    """

    def __init__(self, fq_cfg: FakeQuantConfig) -> None:
        self.fq_cfg = fq_cfg

    def __call__(self, module: nn.Module, inputs):
        if not self.fq_cfg.weight_quant:
            return
        if isinstance(module, nn.Conv2d) and module.weight is not None:
            with torch.no_grad():
                module.weight.data = fake_quantize_tensor(
                    module.weight.data,
                    num_bits=self.fq_cfg.num_bits,
                    symmetric=self.fq_cfg.weight_symmetric,
                    per_channel=self.fq_cfg.weight_per_channel,
                )


def register_fake_quant_hooks(
    model: nn.Module,
    fq_cfg: FakeQuantConfig,
) -> list:
    """
    Register weight fake-quantization hooks on all Conv2d layers.

    Args:
        model:   nn.Module to hook.
        fq_cfg:  FakeQuantConfig.

    Returns:
        List of hook handles (call handle.remove() to unregister).
    """
    handles = []
    hook    = FakeQuantWeightHook(fq_cfg)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_pre_hook(hook))
    _log.info(f"Registered fake-quant hooks on {len(handles)} Conv2d layers.")
    return handles


def remove_hooks(handles: list) -> None:
    """Remove all registered hooks."""
    for h in handles:
        h.remove()
    _log.info(f"Removed {len(handles)} fake-quant hooks.")


# --------------------------------------------------------------------------- #
#  QAT checkpoint utilities                                                    #
# --------------------------------------------------------------------------- #

def save_qat_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: Dict[str, float],
    cfg: Config,
    is_best: bool = False,
    label: str = "qat",
) -> str:
    """
    Save a QAT training checkpoint.

    Args:
        model:     QAT model.
        optimizer: Optimizer state.
        epoch:     Current epoch.
        metrics:   Validation metrics dict.
        cfg:       Config.
        is_best:   Save as best checkpoint if True.
        label:     Checkpoint label.

    Returns:
        Path to saved checkpoint.
    """
    state = {
        "epoch":      epoch,
        "model_state": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "dice":       metrics.get("mean_dice", 0.0),
        "metrics":    metrics,
        "config":     cfg.__dict__,
        "qat":        True,
    }

    last_path = os.path.join(cfg.checkpoints_dir, f"{label}_last.pth")
    best_path = os.path.join(cfg.checkpoints_dir, f"{label}_best.pth")

    save_checkpoint(state, last_path,
                    is_best=is_best, best_filepath=best_path)
    return best_path if is_best else last_path


def load_qat_checkpoint(
    model: nn.Module,
    cfg: Config,
    label: str = "qat",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Load the best QAT checkpoint into a model.

    Args:
        model:  nn.Module (same architecture as when saved).
        cfg:    Config.
        label:  Checkpoint label.
        device: Target device.

    Returns:
        Model with loaded weights.
    """
    if device is None:
        device = torch.device("cpu")

    path  = os.path.join(cfg.checkpoints_dir, f"{label}_best.pth")
    state = load_checkpoint(path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    _log.info(
        f"Loaded QAT checkpoint from {path} "
        f"(epoch {state['epoch']}, dice {state['dice']:.4f})"
    )
    return model


# --------------------------------------------------------------------------- #
#  QAT → ONNX export                                                          #
# --------------------------------------------------------------------------- #

def export_qat_to_onnx(
    model: nn.Module,
    cfg: Config,
    handles: Optional[list] = None,
    filename: str = "qat_fp32.onnx",
) -> str:
    """
    Export QAT-trained model to ONNX (FP32 weights, quant-aware fine-tuned).

    Removes fake-quant hooks before export so the ONNX graph is clean.
    The exported ONNX is then quantized via ORT (same as Phase 3 PTQ).

    Args:
        model:    QAT-trained model.
        cfg:      Config.
        handles:  Hook handles to remove before export.
        filename: Output filename.

    Returns:
        Path to exported ONNX file.
    """
    if handles:
        remove_hooks(handles)

    dummy = torch.randn(1, cfg.in_channels, *cfg.image_size)
    model = model.cpu().eval()

    onnx_path = os.path.join(cfg.checkpoints_dir, filename)

    with torch.no_grad():
        torch.onnx.utils.export(
            model, dummy, onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            training=torch.onnx.TrainingMode.EVAL,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        )

    size_mb = os.path.getsize(onnx_path) / 1e6
    _log.info(f"QAT ONNX exported → {onnx_path}  ({size_mb:.2f} MB)")
    return onnx_path