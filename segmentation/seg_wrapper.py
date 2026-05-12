"""
segmentation/seg_wrapper.py
----------------------------
Generic, architecture-agnostic segmentation model wrapper for Phase 3.

Design decisions
----------------
- Does NOT hardcode UNet++ or EfficientNet-B4. The architecture is read
  from config, so swapping to DeepLabV3+ or any SMP model requires only
  a YAML change.
- Accepts raw numpy arrays (greyscale uint8) and returns class maps (H×W
  int numpy arrays) so the pipeline layer never touches PyTorch tensors.
- Preprocessing (normalise, to-tensor) is done internally so callers
  need no albumentations dependency.
- Mixed precision (autocast) is applied automatically on CUDA.
- Reuses Phase 1 ``models/model_loader.py`` for weight loading — zero
  duplication.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


class SegmentationWrapper:
    """
    Inference wrapper for any segmentation model loadable via
    ``segmentation_models_pytorch``.

    Parameters
    ----------
    cfg      : loaded pipeline config namespace (``roi_pipeline_config.yaml``).
    device   : override device (defaults to ``cfg.segmentation.device``).
    """

    def __init__(self, cfg, device: Optional[str] = None) -> None:
        from models.model_loader import build_model, load_checkpoint

        self.device    = device or cfg.segmentation.device
        self.num_classes = cfg.classes.num_classes

        # ── Build model from pipeline config ──────────────────────────────────
        # model_loader.build_model expects cfg.model.* fields.
        # We adapt by building a temporary namespace-compatible object.
        model_cfg = _make_model_cfg(cfg)
        self.model = build_model(model_cfg)
        load_checkpoint(self.model, cfg.segmentation.checkpoint, device=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        # ── Preprocessing pipeline ────────────────────────────────────────────
        self._transform = A.Compose([
            A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
            ToTensorV2(),
        ])
        self._use_amp = (self.device != "cpu")
        logger.info(
            "SegmentationWrapper ready | device=%s | amp=%s | classes=%d",
            self.device, self._use_amp, self.num_classes,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(
        self,
        image: np.ndarray,
        target_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Run segmentation on a single greyscale image.

        Parameters
        ----------
        image       : H×W uint8 greyscale numpy array.
        target_size : resize image to this square size before inference.
                      If None, use image as-is.

        Returns
        -------
        np.ndarray  H×W uint8 class map (original spatial size if resized).
        """
        orig_h, orig_w = image.shape[:2]

        if target_size is not None:
            img_in = cv2.resize(image, (target_size, target_size))
        else:
            img_in = image

        tensor = self._preprocess(img_in)                    # (1,1,H,W)
        logits = self._forward(tensor)                       # (1,C,H,W)
        pred   = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

        # Resize prediction back to input image size
        if target_size is not None and (pred.shape[0] != orig_h or pred.shape[1] != orig_w):
            pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return pred

    def predict_batch(
        self,
        images: List[np.ndarray],
        target_size: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Run segmentation on a list of greyscale images.

        Parameters
        ----------
        images      : list of H×W uint8 greyscale arrays.
        target_size : optional resize before inference.

        Returns
        -------
        List[np.ndarray]  class maps, one per input image.
        """
        return [self.predict(img, target_size) for img in images]

    def predict_tensor(self, images: np.ndarray) -> torch.Tensor:
        """
        Run segmentation on a pre-stacked (N,1,H,W) float tensor.
        Returns raw logits (N,C,H,W) on CPU — for metric computation.

        Parameters
        ----------
        images : (N,1,H,W) float32 normalised tensor.

        Returns
        -------
        torch.Tensor  (N,C,H,W) logits on CPU.
        """
        t = torch.as_tensor(images).to(self.device)
        return self._forward(t).cpu()

    # ── Internals ──────────────────────────────────────────────────────────────

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Normalise and convert to (1,1,H,W) tensor."""
        aug    = self._transform(image=image)
        tensor = aug["image"].unsqueeze(0).to(self.device)   # (1,1,H,W)
        return tensor

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional AMP."""
        with torch.no_grad():
            if self._use_amp:
                with torch.cuda.amp.autocast():
                    return self.model(tensor)
            return self.model(tensor)


# ── Config adaptor ─────────────────────────────────────────────────────────────

def _make_model_cfg(cfg):
    """
    Build a SimpleNamespace compatible with ``models/model_loader.build_model``
    from the pipeline config's ``segmentation`` section.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        model=SimpleNamespace(
            architecture    = cfg.segmentation.architecture,
            encoder         = cfg.segmentation.encoder,
            encoder_weights = cfg.segmentation.encoder_weights,
            decoder_channels= cfg.segmentation.decoder_channels,
            activation      = cfg.segmentation.activation,
            checkpoint      = cfg.segmentation.checkpoint,
        ),
        data=SimpleNamespace(in_channels=cfg.segmentation.in_channels),
        classes=SimpleNamespace(num_classes=cfg.classes.num_classes),
    )
