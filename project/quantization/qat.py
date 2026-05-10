"""
qat.py
------
Quantization-Aware Training (QAT) – Phase 2 placeholder.

QAT simulates quantization noise during training so the model
learns representations that are robust to INT8 precision.

Workflow:
  1. Load float32 baseline (or start from scratch).
  2. Fuse layers.
  3. Set qconfig and call torch.quantization.prepare_qat().
  4. Train for cfg.qat_epochs with simulated quantization.
  5. Convert to INT8 for deployment.

References:
  https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.config import Config
from utils.logger import get_logger

_log = get_logger(__name__)


def prepare_qat(
    model: nn.Module,
    backend: str = "qnnpack",
) -> nn.Module:
    """
    Insert fake-quantization modules for QAT.

    Args:
        model:   Float32 model with fused layers.
        backend: Quantization backend.

    Returns:
        QAT-prepared model.

    Note:
        Placeholder – fully implement in Phase 2.
    """
    raise NotImplementedError("QAT preparation is a Phase 2 feature.")
    # torch.backends.quantized.engine = backend
    # model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    # torch.quantization.prepare_qat(model, inplace=True)
    # return model


def qat_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> nn.Module:
    """
    Fine-tuning loop with fake quantization enabled.

    Note:
        Placeholder – fully implement in Phase 2.
    """
    raise NotImplementedError("QAT training loop is a Phase 2 feature.")


def convert_qat_model(model: nn.Module) -> nn.Module:
    """
    Convert a QAT model to a true INT8 quantized model.

    Note:
        Placeholder – implement in Phase 2.
    """
    raise NotImplementedError("QAT conversion is a Phase 2 feature.")
    # model.eval()
    # torch.quantization.convert(model, inplace=True)
    # return model
