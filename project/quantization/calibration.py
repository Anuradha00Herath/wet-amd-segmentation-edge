"""
calibration.py
--------------
Calibration utilities for Post-Training Quantization (PTQ).

In PTQ, the model observes a small representative subset of the
training data ("calibration set") so that activation ranges can be
measured and used to set quantization parameters.

Phase 1: scaffolding and interfaces only.
Phase 2: integrate with static_quant.py.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.config import Config
from utils.logger import get_logger

_log = get_logger(__name__)


def run_calibration(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    n_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> None:
    """
    Feed calibration batches through a prepared (quantization-aware)
    model to collect activation statistics.

    The model must already have observers inserted (via
    torch.quantization.prepare) before calling this function.

    Args:
        model:     Prepared (observer-inserted) model in eval mode.
        loader:    DataLoader over the calibration dataset.
        cfg:       Project Config.
        n_batches: Number of batches to use. Defaults to cfg.ptq_calib_batches.
        device:    Target device.
    """
    from typing import Optional  # local import for type hints

    if device is None:
        device = torch.device("cpu")

    n_batches = n_batches or cfg.ptq_calib_batches
    model.eval()
    model.to(device)

    _log.info(f"Running PTQ calibration for {n_batches} batches ...")
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= n_batches:
                break
            images = images.to(device)
            _ = model(images)

    _log.info("Calibration complete.")


def get_calibration_loader(cfg: Config) -> DataLoader:
    """
    Return a small DataLoader suitable for PTQ calibration.

    Uses the full inference loader (no augmentation) limited to
    cfg.ptq_calib_batches batches.
    """
    from utils.dataset import build_inference_loader
    return build_inference_loader(cfg, batch_size=cfg.batch_size)
