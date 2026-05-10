"""
transforms.py
-------------
Albumentations-based image and mask transformation pipelines for
6-class OCT wetAMD segmentation.

Design decisions:
  - Albumentations is used for ALL transforms (train and val) because
    it handles joint image + mask augmentation correctly.
  - Images are grayscale OCT (H, W) numpy arrays loaded by cv2.
  - Masks are class index arrays (H, W) int64 from rgb_mask_to_class().
  - ToTensorV2 converts image → (1, H, W) float32, mask stays (H, W) int64.
  - Normalisation matches the original notebook: mean=0.5, std=0.5.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.config import Config


def get_train_transform(cfg: Config) -> A.Compose:
    """
    Augmentation pipeline for training.

    Augmentations match the original notebook:
      - Horizontal flip
      - Vertical flip
      - Random rotate 90°
      - Shift / scale / rotate
      - Random brightness & contrast
      - Gaussian noise
      - Gaussian blur
      - Grid distortion
      - CLAHE
      - Normalize + ToTensorV2

    Args:
        cfg: Project Config (image_size, norm_mean, norm_std).

    Returns:
        Albumentations Compose pipeline.
    """
    h, w = cfg.image_size

    return A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1,
            rotate_limit=15, p=0.4
        ),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.GridDistortion(p=0.2),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.Normalize(
            mean=cfg.norm_mean,
            std=cfg.norm_std,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),   # image → (1, H, W) float32; mask stays (H, W)
    ])


def get_val_transform(cfg: Config) -> A.Compose:
    """
    Minimal pipeline for validation, test, and inference.
    No augmentations — only resize and normalise.

    Args:
        cfg: Project Config.

    Returns:
        Albumentations Compose pipeline.
    """
    h, w = cfg.image_size

    return A.Compose([
        A.Resize(h, w),
        A.Normalize(
            mean=cfg.norm_mean,
            std=cfg.norm_std,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])