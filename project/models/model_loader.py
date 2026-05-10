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

from models.baseline_model import BaselineUNet, build_baseline
from utils.config import Config
from utils.logger import get_logger, load_checkpoint

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Model registry                                                               #
# --------------------------------------------------------------------------- #
# Register new model variants here.  Each entry maps a string key to a
# callable that accepts (in_channels, out_channels) and returns an nn.Module.

_MODEL_REGISTRY: Dict[str, Any] = {
    "baseline": build_baseline,
    # "mobile_unet": build_mobile_unet,  # add future variants here
}


def get_model(model_name: str, in_channels: int = 1, out_channels: int = 1) -> nn.Module:
    """
    Instantiate a model by name.

    Args:
        model_name:   Key in the model registry (e.g. 'baseline').
        in_channels:  Input channels.
        out_channels: Output channels.

    Returns:
        Uninitialised (random weights) nn.Module.

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
        # Support both raw state_dict and wrapped checkpoints
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=strict)
            _log.info(
                f"Loaded '{name}' weights from {ckpt_path} "
                f"(epoch {state.get('epoch', '?')})"
            )
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
    High-level inference wrapper with pre/post-processing.

    Args:
        model:     Loaded nn.Module in eval mode.
        device:    Target device.
        threshold: Sigmoid threshold for binary mask (default 0.5).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on a (possibly batched) image tensor.

        Args:
            image: Float tensor (B, C, H, W) or (C, H, W).

        Returns:
            (probability_map, binary_mask) both as CPU tensors.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        logits = self.model(image)              # (B, 1, H, W)
        probs = torch.sigmoid(logits)           # (B, 1, H, W)
        binary = (probs >= self.threshold).float()

        return probs.cpu(), binary.cpu()

    def predict_single(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method for single image (C, H, W) → (1, H, W) outputs."""
        probs, binary = self.predict(image.unsqueeze(0))
        return probs.squeeze(0), binary.squeeze(0)


# --------------------------------------------------------------------------- #
#  Placeholder: quantized model loader (Phase 2+)                             #
# --------------------------------------------------------------------------- #

def load_quantized_model(
    cfg: Config,
    checkpoint_path: str,
    quantization_type: str = "ptq",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Placeholder for loading quantized (PTQ / QAT) model checkpoints.

    Args:
        cfg:               Project Config.
        checkpoint_path:   Path to quantized model checkpoint.
        quantization_type: One of 'ptq', 'qat', 'mixed'.
        device:            Target device.

    Returns:
        Quantized model in eval mode.

    Note:
        Full implementation added in Phase 2 (quantization/).
    """
    raise NotImplementedError(
        "load_quantized_model is a Phase 2 feature. "
        "Implement in quantization/static_quant.py or quantization/qat.py."
    )
