"""
transforms.py
-------------
Image and mask transformation pipelines for OCT segmentation.

Design decisions:
  - torchvision.transforms.v2 is used where possible for joint
    image+mask transforms (requires torchvision >= 0.15).
  - Albumentations is listed as an optional augmentation backend;
    swap in the AlbumentationsWrapper if Albumentations is installed.
  - All pipelines return (image_tensor, mask_tensor) tuples.
  - Normalisation statistics are read from Config so there is one
    place to update them.
"""

from typing import Callable, Optional, Tuple

import torch
from torchvision import transforms as T

from utils.config import Config


# --------------------------------------------------------------------------- #
#  Helper: convert grayscale PIL image and binary PIL mask to tensors          #
# --------------------------------------------------------------------------- #

class MaskToTensor:
    """Convert a PIL grayscale mask image to a binary float32 tensor."""

    def __call__(self, mask) -> torch.Tensor:
        # PIL L-mode image → (H, W) → (1, H, W) float32 in {0, 1}
        t = T.functional.pil_to_tensor(mask).float()  # (1, H, W) uint8
        return (t > 127).float()                       # binarise


# --------------------------------------------------------------------------- #
#  Standard pipelines                                                           #
# --------------------------------------------------------------------------- #

def get_train_transform(cfg: Config) -> Tuple[Callable, Callable]:
    """
    Return (image_transform, mask_transform) for training.

    Augmentations included (extend as needed):
      - Random horizontal flip
      - Random vertical flip
      - Random rotation ±10°
      - Resize to cfg.image_size
      - Normalise images

    Args:
        cfg: Project Config instance.

    Returns:
        Tuple of (image_transform, mask_transform) callables.
    """
    image_transform = T.Compose([
        T.Resize(cfg.image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ToTensor(),                              # → (C, H, W) float32 [0, 1]
        T.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
    ])

    mask_transform = T.Compose([
        T.Resize(cfg.image_size, interpolation=T.InterpolationMode.NEAREST),
        MaskToTensor(),
    ])

    return image_transform, mask_transform


def get_val_transform(cfg: Config) -> Tuple[Callable, Callable]:
    """
    Return (image_transform, mask_transform) for validation / inference.
    No random augmentations applied.

    Args:
        cfg: Project Config instance.

    Returns:
        Tuple of (image_transform, mask_transform) callables.
    """
    image_transform = T.Compose([
        T.Resize(cfg.image_size),
        T.ToTensor(),
        T.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
    ])

    mask_transform = T.Compose([
        T.Resize(cfg.image_size, interpolation=T.InterpolationMode.NEAREST),
        MaskToTensor(),
    ])

    return image_transform, mask_transform


def denormalize(tensor: torch.Tensor, cfg: Config) -> torch.Tensor:
    """
    Reverse normalisation for visualisation purposes.

    Args:
        tensor: Normalised image tensor (C, H, W) or (B, C, H, W).
        cfg:    Config containing norm_mean and norm_std.

    Returns:
        Denormalised tensor clamped to [0, 1].
    """
    mean = torch.tensor(cfg.norm_mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(cfg.norm_std, dtype=tensor.dtype, device=tensor.device)

    if tensor.dim() == 4:                # (B, C, H, W)
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
    else:                                # (C, H, W)
        mean = mean[:, None, None]
        std = std[:, None, None]

    return (tensor * std + mean).clamp(0, 1)


# --------------------------------------------------------------------------- #
#  Optional: Albumentations-backed joint transforms                            #
# --------------------------------------------------------------------------- #

class AlbumentationsWrapper:
    """
    Thin wrapper for Albumentations joint image+mask transforms.

    Usage (requires `pip install albumentations`):

        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        aug = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2(),
        ])
        wrapper = AlbumentationsWrapper(aug, cfg)
        image_t, mask_t = wrapper(pil_image, pil_mask)
    """

    def __init__(self, albumentation_transform, cfg: Config) -> None:
        self.transform = albumentation_transform
        self.cfg = cfg

    def __call__(self, image, mask) -> Tuple[torch.Tensor, torch.Tensor]:
        import numpy as np

        image_np = np.array(image)    # (H, W) or (H, W, C)
        mask_np = np.array(mask)      # (H, W)

        augmented = self.transform(image=image_np, mask=mask_np)
        image_t: torch.Tensor = augmented["image"]
        mask_t: torch.Tensor = torch.from_numpy(augmented["mask"]).unsqueeze(0).float()
        mask_t = (mask_t > 127).float()

        return image_t, mask_t
