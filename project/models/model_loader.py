"""
model_loader.py
---------------
Unified model loading, checkpoint management, and inference wrappers.

All model construction and weight loading is centralised here so
experiment scripts only need to call a single function.
"""

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.baseline_model import BaselineModel, build_baseline, NUM_CLASSES
from utils.config import Config
from utils.logger import get_logger, load_checkpoint

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Model registry                                                               #
# --------------------------------------------------------------------------- #

_MODEL_REGISTRY: Dict[str, Any] = {
    "baseline": build_baseline,
    # "lightweight_unet": build_lightweight_unet,  # add future variants here
}


def get_model(model_name: str, in_channels: int = 1, out_channels: int = NUM_CLASSES) -> nn.Module:
    """
    Instantiate a model by name from the registry.

    Args:
        model_name:   Key in the model registry (e.g. 'baseline').
        in_channels:  Input channels (1 for grayscale OCT).
        out_channels: Output classes (6 for wetAMD segmentation).

    Returns:
        nn.Module with random weights.

    Raises:
        KeyError: If model_name is not found in registry.
    """
    if model_name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[model_name](in_channels=in_channels, out_channels=out_channels)


# --------------------------------------------------------------------------- #
#  Checkpoint loading                                                           #
# --------------------------------------------------------------------------- #

def load_model(
    cfg: Config,
    checkpoint_path: Optional[str] = None,
    model_name: Optional[str] = None,
    strict: bool = True,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Instantiate a model and load weights from a checkpoint.

    Args:
        cfg:             Project Config.
        checkpoint_path: Explicit .pth path. Falls back to cfg.baseline_checkpoint.
        model_name:      Model key (falls back to cfg.model_name).
        strict:          Passed to model.load_state_dict().
        device:          Target device. Defaults to CPU for portability.

    Returns:
        Model with loaded weights, set to eval mode.
    """
    if device is None:
        device = torch.device("cpu")

    name = model_name or cfg.model_name
    model = get_model(name, cfg.in_channels, cfg.out_channels)

    ckpt_path = checkpoint_path or os.path.join(
        cfg.checkpoints_dir, cfg.baseline_checkpoint
    )

    if os.path.isfile(ckpt_path):
        state = load_checkpoint(ckpt_path, map_location=device)
        # Support both checkpoint formats
        if "model_state" in state:
            model.load_state_dict(state["model_state"], strict=strict)
            _log.info(f"Loaded '{name}' from {ckpt_path} (epoch {state.get('epoch', '?')}, dice {state.get('dice', '?')})")
        elif "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=strict)
            _log.info(f"Loaded '{name}' from {ckpt_path} (epoch {state.get('epoch', '?')})")
        else:
            model.load_state_dict(state, strict=strict)
            _log.info(f"Loaded '{name}' state dict from {ckpt_path}")
    else:
        _log.warning(
            f"Checkpoint not found at {ckpt_path}. "
            "Model initialised with random weights."
        )

    model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
#  Inference wrapper                                                            #
# --------------------------------------------------------------------------- #

class SegmentationInference:
    """
    High-level inference wrapper for 6-class OCT segmentation.

    Args:
        model:  Loaded nn.Module in eval mode.
        device: Target device.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model  = model
        self.device = device

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on a (possibly batched) image tensor.

        Args:
            image: Float tensor (B, 1, H, W) or (1, H, W).

        Returns:
            probs:      Softmax probabilities  (B, 6, H, W)  – CPU tensor.
            pred_mask:  Predicted class index  (B, H, W)     – CPU tensor (int64).
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image  = image.to(self.device)
        logits = self.model(image)                      # (B, 6, H, W)
        probs  = torch.softmax(logits, dim=1)           # (B, 6, H, W)
        pred_mask = torch.argmax(probs, dim=1)          # (B, H, W)

        return probs.cpu(), pred_mask.cpu()

    def predict_single(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single image (1, H, W) → probs (6, H, W), pred_mask (H, W)."""
        probs, pred_mask = self.predict(image.unsqueeze(0))
        return probs.squeeze(0), pred_mask.squeeze(0)


# --------------------------------------------------------------------------- #
#  Placeholder: quantized model loader (Phase 2+)                              #
# --------------------------------------------------------------------------- #

def load_quantized_model(
    cfg: Config,
    checkpoint_path: str,
    quantization_type: str = "ptq",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Placeholder for loading quantized (PTQ / QAT) model checkpoints.
    Full implementation in Phase 2.
    """
    raise NotImplementedError(
        "load_quantized_model is a Phase 2 feature. "
        "Implement in quantization/static_quant.py or quantization/qat.py."
    )