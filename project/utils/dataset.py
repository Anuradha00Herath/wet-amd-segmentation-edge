"""
dataset.py
----------
Custom PyTorch Dataset for the OCT wetAMD 6-class segmentation task.

Layout expected on disk:
    data/images/  – OCT grayscale image files (PNG / JPG / TIF)
    data/mask/    – Corresponding RGB colour-coded mask files (same filenames)

Mask class mapping (RGB → class index):
    [0,   0,   0  ] → 0  Background
    [255, 0,   255] → 1  Retinal Layer
    [255, 255, 0  ] → 2  PED
    [255, 0,   0  ] → 3  SRF
    [0,   0,   255] → 4  IRF
    [0,   255, 0  ] → 5  RPE

Split: 70% train / 15% val / 15% test (seeded, reproducible)
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.config import Config
from utils.seed import get_generator
from utils.transforms import get_train_transform, get_val_transform


# --------------------------------------------------------------------------- #
#  Class colour map                                                             #
# --------------------------------------------------------------------------- #

NUM_CLASSES = 6

CLASS_INFO = {
    0: {"name": "Background",    "rgb": [0,   0,   0  ]},
    1: {"name": "Retinal Layer", "rgb": [255, 0,   255]},
    2: {"name": "PED",           "rgb": [255, 255, 0  ]},
    3: {"name": "SRF",           "rgb": [255, 0,   0  ]},
    4: {"name": "IRF",           "rgb": [0,   0,   255]},
    5: {"name": "RPE",           "rgb": [0,   255, 0  ]},
}

CLASS_NAMES = [CLASS_INFO[i]["name"] for i in range(NUM_CLASSES)]


# --------------------------------------------------------------------------- #
#  RGB mask → class index                                                      #
# --------------------------------------------------------------------------- #

def rgb_mask_to_class(mask_bgr: np.ndarray, tolerance: int = 40) -> np.ndarray:
    """
    Convert a BGR colour-coded mask (as loaded by cv2) to a class index map.

    Args:
        mask_bgr:  BGR uint8 array (H, W, 3).
        tolerance: Maximum colour distance to accept a pixel as a class match.

    Returns:
        Class index array (H, W) dtype int64.
    """
    mask_rgb  = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    h, w, _   = mask_rgb.shape
    pixels    = mask_rgb.reshape(-1, 3).astype(np.float32)

    best_dist = np.full(len(pixels), np.inf)
    best_cls  = np.zeros(len(pixels), dtype=np.int64)

    for cls_id, info in CLASS_INFO.items():
        ref    = np.array(info["rgb"], dtype=np.float32)
        dist   = np.linalg.norm(pixels - ref, axis=1)
        better = dist < best_dist
        best_dist[better] = dist[better]
        best_cls[better]  = cls_id

    return best_cls.reshape(h, w)


# --------------------------------------------------------------------------- #
#  Supported extensions                                                        #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class OCTSegmentationDataset(Dataset):
    """
    Dataset for OCT 6-class segmentation with RGB colour-coded masks.

    Args:
        images_dir: Path to OCT grayscale images.
        masks_dir:  Path to RGB colour-coded mask images.
        transform:  Albumentations Compose pipeline (joint image + mask).
        indices:    Optional subset indices for train/val/test splitting.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform=None,
        indices: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.image_paths = _find_image_paths(images_dir)
        self.masks_dir   = Path(masks_dir)
        self.transform   = transform

        self._validate_pairs()

        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]

    def _validate_pairs(self) -> None:
        missing = []
        for img_path in self.image_paths:
            if not self._mask_path_for(img_path).exists():
                missing.append(str(self._mask_path_for(img_path)))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} mask(s) not found. First missing:\n  {missing[0]}"
            )

    def _mask_path_for(self, image_path: Path) -> Path:
        for ext in [image_path.suffix, ".png", ".jpg", ".tif"]:
            candidate = self.masks_dir / (image_path.stem + ext)
            if candidate.exists():
                return candidate
        return self.masks_dir / (image_path.stem + image_path.suffix)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path  = self.image_paths[idx]
        mask_path = self._mask_path_for(img_path)

        # Load image as grayscale, mask as BGR colour
        image    = cv2.imread(str(img_path),  cv2.IMREAD_GRAYSCALE)  # (H, W)
        mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)      # (H, W, 3)

        # Convert RGB mask → class index map (H, W) int64
        mask = rgb_mask_to_class(mask_bgr)

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]        # (1, H, W) float32 tensor
            mask  = aug["mask"].long()  # (H, W) int64 tensor
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
            mask  = torch.tensor(mask,  dtype=torch.long)

        return image, mask

    def __repr__(self) -> str:
        return (
            f"OCTSegmentationDataset("
            f"n_samples={len(self)}, "
            f"transform={self.transform is not None})"
        )


# --------------------------------------------------------------------------- #
#  Train / Val / Test split                                                    #
# --------------------------------------------------------------------------- #

def _three_way_split(
    n: int,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Reproducible 3-way index split.

    Args:
        n:          Total number of samples.
        train_frac: Fraction for training (e.g. 0.70).
        val_frac:   Fraction for validation (e.g. 0.15).
        seed:       Random seed.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    rng     = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


# --------------------------------------------------------------------------- #
#  DataLoader factories                                                        #
# --------------------------------------------------------------------------- #

def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders with seeded 70/15/15 split.

    Args:
        cfg: Project Config instance.

    Returns:
        (train_loader, val_loader, test_loader) tuple.
    """
    img_paths = _find_image_paths(cfg.images_dir)
    n_total   = len(img_paths)

    train_idx, val_idx, test_idx = _three_way_split(
        n_total, cfg.train_split, cfg.val_split, cfg.seed
    )

    train_transform = get_train_transform(cfg)
    val_transform   = get_val_transform(cfg)

    train_ds = OCTSegmentationDataset(
        cfg.images_dir, cfg.masks_dir,
        transform=train_transform, indices=train_idx,
    )
    val_ds = OCTSegmentationDataset(
        cfg.images_dir, cfg.masks_dir,
        transform=val_transform, indices=val_idx,
    )
    test_ds = OCTSegmentationDataset(
        cfg.images_dir, cfg.masks_dir,
        transform=val_transform, indices=test_idx,
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
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_loader, val_loader, test_loader


def build_inference_loader(cfg: Config, batch_size: int = 1) -> DataLoader:
    """
    DataLoader over the full dataset without augmentation.
    Used for PTQ calibration and full-dataset evaluation.
    """
    val_transform = get_val_transform(cfg)
    ds = OCTSegmentationDataset(
        cfg.images_dir, cfg.masks_dir,
        transform=val_transform,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )