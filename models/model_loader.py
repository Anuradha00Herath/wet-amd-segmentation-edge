"""
models/model_loader.py
----------------------
Factory for loading the segmentation model and restoring checkpoints.

Keeping model construction here (rather than inline in scripts) means
every script — inference, evaluation, future ROI-guided pipeline — all
use a single, consistent loading path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


def build_model(cfg: SimpleNamespace) -> torch.nn.Module:
    """
    Instantiate the segmentation model from config.

    Currently supports any architecture available in
    ``segmentation_models_pytorch`` (UnetPlusPlus, Unet, DeepLabV3+, …).
    To swap architectures, change ``model.architecture`` in config.yaml —
    no code changes needed.

    Parameters
    ----------
    cfg : loaded config namespace.

    Returns
    -------
    torch.nn.Module  (not yet moved to device)
    """
    arch  = cfg.model.architecture
    klass = getattr(smp, arch, None)
    if klass is None:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Available: {[a for a in dir(smp) if not a.startswith('_')]}"
        )

    model = klass(
        encoder_name          = cfg.model.encoder,
        encoder_weights       = cfg.model.encoder_weights,
        in_channels           = cfg.data.in_channels,
        classes               = cfg.classes.num_classes,
        activation            = cfg.model.activation,
        decoder_channels      = cfg.model.decoder_channels,
        decoder_use_batchnorm = True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Built %s/%s | params=%.2fM",
        arch, cfg.model.encoder, total_params / 1e6,
    )
    return model


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
) -> dict:
    """
    Load a saved checkpoint into *model* (in-place).

    The checkpoint format produced by the baseline training notebook is::

        {
            "model_state": state_dict,
            "epoch":       int,
            "dice":        float,
        }

    Parameters
    ----------
    model           : model instance (already built with ``build_model``).
    checkpoint_path : path to the ``.pth`` file.
    device          : torch device string.

    Returns
    -------
    dict  The full checkpoint dict (caller can inspect epoch / dice etc.).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Handle both raw state-dicts and wrapped checkpoint dicts
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        epoch = ckpt.get("epoch", "?")
        dice  = ckpt.get("dice",  float("nan"))
        logger.info("Loaded checkpoint: epoch=%s  val_dice=%.4f", epoch, dice)
    else:
        # Assume it's a bare state_dict
        model.load_state_dict(ckpt)
        ckpt  = {"model_state": ckpt}
        logger.info("Loaded raw state_dict from %s", path)

    return ckpt


def get_model(cfg: SimpleNamespace, device: Optional[str] = None) -> torch.nn.Module:
    """
    Convenience wrapper: build + load checkpoint + move to device.

    Parameters
    ----------
    cfg    : loaded config namespace.
    device : override device (defaults to ``cfg.inference.device``).

    Returns
    -------
    torch.nn.Module  Ready for inference (eval mode).
    """
    device = device or cfg.inference.device
    model  = build_model(cfg)
    load_checkpoint(model, cfg.model.checkpoint, device=device)
    model  = model.to(device)
    model.eval()
    logger.info("Model moved to device=%s and set to eval mode", device)
    return model
