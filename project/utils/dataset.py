"""
dataset.py
----------
Custom PyTorch Dataset for the OCT wetAMD binary segmentation task.

Layout expected on disk:
    data/images/  – OCT image files (PNG / JPG / TIF)
    data/mask/    – Corresponding binary mask files (same filenames)

Design decisions:
  - Images and masks are matched by filename stem (no assumptions about
    subfolder structure beyond the two top-level directories above).
  - Train / validation split is performed with a seeded random split
    so it is reproducible across runs.
  - Transforms are injected at construction time, keeping Dataset
    decoupled from the transform logic in transforms.py.
"""

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from utils.config import Config
from utils.seed import get_generator
from utils.transforms import get_train_transform, get_val_transform


# Supported image extensions
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _find_image_paths(directory: str) -> List[Path]:
    """Return sorted list of image paths in directory."""
    paths = [
        p for p in sorted(Path(directory).iterdir())
        if p.suffix.lower() in _IMAGE_EXTS
    ]
    if not paths:
        raise FileNotFoundError(
            f"No images found in {directory}. "
            f"Expected extensions: {_IMAGE_EXTS}"
        )
    return paths


class OCTSegmentationDataset(Dataset):
    """
    Dataset for OCT binary segmentation.

    Args:
        images_dir:       Path to the directory containing OCT images.
        masks_dir:        Path to the directory containing binary masks.
        image_transform:  Transform / augmentation applied to images.
        mask_transform:   Transform applied to masks (e.g. resize + binarise).
        indices:          Optional list of integer indices to use as a subset.
                          When None, all matched image/mask pairs are used.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        indices: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.image_paths = _find_image_paths(images_dir)
        self.masks_dir = Path(masks_dir)

        # Validate that a mask exists for every image
        self._validate_pairs()

        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]

        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def _validate_pairs(self) -> None:
        missing = []
        for img_path in self.image_paths:
            mask_path = self._mask_path_for(img_path)
            if not mask_path.exists():
                missing.append(str(mask_path))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} mask(s) not found. First missing:\n  {missing[0]}"
            )

    def _mask_path_for(self, image_path: Path) -> Path:
        """Resolve mask path from image path by matching stem."""
        # Try same extension first, then common mask extensions
        for ext in [image_path.suffix, ".png", ".jpg", ".tif"]:
            candidate = self.masks_dir / (image_path.stem + ext)
            if candidate.exists():
                return candidate
        # Return expected path even if not found (will raise in _validate_pairs)
        return self.masks_dir / (image_path.stem + image_path.suffix)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self._mask_path_for(img_path)

        image = Image.open(img_path).convert("L")   # greyscale OCT
        mask = Image.open(mask_path).convert("L")   # greyscale mask

        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask

    def __repr__(self) -> str:
        return (
            f"OCTSegmentationDataset("
            f"n_samples={len(self)}, "
            f"image_transform={self.image_transform is not None}, "
            f"mask_transform={self.mask_transform is not None})"
        )


# --------------------------------------------------------------------------- #
#  DataLoader factory                                                           #
# --------------------------------------------------------------------------- #

def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders with seeded random split.

    Args:
        cfg: Project Config instance.

    Returns:
        (train_loader, val_loader) tuple.
    """
    # Build full dataset (no transforms yet – need indices first)
    img_paths = _find_image_paths(cfg.images_dir)
    n_total = len(img_paths)
    n_val = max(1, int(n_total * cfg.val_split))
    n_train = n_total - n_val

    # Reproducible split
    train_indices, val_indices = _split_indices(n_total, n_val, seed=cfg.seed)

    train_img_t, train_mask_t = get_train_transform(cfg)
    val_img_t, val_mask_t = get_val_transform(cfg)

    train_ds = OCTSegmentationDataset(
        cfg.images_dir, cfg.masks_dir,
        image_transform=train_img_t,
        mask_transform=train_mask_t,
        indices=train_indices,
    )
    val_ds = OCTSegmentationDataset(
        cfg.images_dir, cfg.masks_dir,
        image_transform=val_img_t,
        mask_transform=val_mask_t,
        indices=val_indices,
    )

    g = get_generator(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        generator=g,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_loader, val_loader


def build_inference_loader(cfg: Config, batch_size: int = 1) -> DataLoader:
    """
    DataLoader covering the entire dataset without augmentation.
    Used for calibration and full-dataset evaluation.

    Args:
        cfg:        Project Config instance.
        batch_size: Batch size (default 1 for sequential inference).

    Returns:
        DataLoader over the full dataset.
    """
    val_img_t, val_mask_t = get_val_transform(cfg)
    ds = OCTSegmentationDataset(
        cfg.images_dir, cfg.masks_dir,
        image_transform=val_img_t,
        mask_transform=val_mask_t,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )


# --------------------------------------------------------------------------- #
#  Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _split_indices(n: int, n_val: int, seed: int) -> Tuple[List[int], List[int]]:
    """Return deterministic train / val index lists."""
    import random
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    return indices[n_val:], indices[:n_val]
