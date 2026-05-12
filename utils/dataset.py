"""
utils/dataset.py
----------------
OCT dataset and data-loading helpers.

Key design decisions
--------------------
- OCTDataset is split-agnostic; the caller passes pre-split index lists
  (or uses random_split) and sets `transform` accordingly.
- RGB → class-id conversion uses the same L2-nearest-colour logic as the
  baseline notebook, parameterised via CLASS_INFO from the config.
- Normalisation uses albumentations so the same pipeline handles both
  spatial augmentation and tensor conversion.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


# ── Colour conversion ──────────────────────────────────────────────────────────

def rgb_mask_to_class(
    mask_bgr: np.ndarray,
    class_info: Dict[int, dict],
    tolerance: int = 40,
) -> np.ndarray:
    """
    Convert an RGB-encoded mask (read with cv2 → BGR) to a per-pixel
    integer class map using nearest-colour matching.

    Parameters
    ----------
    mask_bgr   : H×W×3 uint8 array in BGR order (as returned by cv2.imread).
    class_info : ``{class_id: {"name": str, "rgb": (R,G,B)}}``
    tolerance  : kept for API consistency; not used as a hard threshold
                 (we always pick the nearest colour).

    Returns
    -------
    np.ndarray : H×W uint8 class-id map.
    """
    mask_rgb  = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    h, w, _   = mask_rgb.shape
    pixels    = mask_rgb.reshape(-1, 3).astype(np.float32)
    best_dist = np.full(len(pixels), np.inf)
    best_cls  = np.zeros(len(pixels), dtype=np.uint8)

    for cls_id, info in class_info.items():
        ref  = np.array(info["rgb"], dtype=np.float32)
        dist = np.linalg.norm(pixels - ref, axis=1)
        better = dist < best_dist
        best_dist[better] = dist[better]
        best_cls[better]  = cls_id

    return best_cls.reshape(h, w)


def mask_to_rgb(
    mask_np: np.ndarray,
    class_colors: List[List[int]],
) -> np.ndarray:
    """
    Convert an integer class map to a colour RGB image for visualisation.

    Parameters
    ----------
    mask_np      : H×W int array.
    class_colors : list of [R,G,B] lists, indexed by class id.

    Returns
    -------
    np.ndarray : H×W×3 uint8 RGB image.
    """
    h, w = mask_np.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for c, colour in enumerate(class_colors):
        rgb[mask_np == c] = colour
    return rgb


# ── Dataset ────────────────────────────────────────────────────────────────────

class OCTDataset(Dataset):
    """
    PyTorch Dataset for OCT greyscale images with RGB-encoded segmentation masks.

    Parameters
    ----------
    image_dir   : directory containing .png / .jpg / .tif / .bmp images.
    mask_dir    : directory containing matching mask images (same filename).
    class_info  : colour-to-class mapping dict.
    image_size  : spatial size to resize both image and mask to (square).
    transform   : albumentations Compose pipeline applied to (image, mask).
                  If None, a minimal no-aug pipeline is used.
    """

    _EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        class_info: Dict[int, dict],
        image_size: int = 256,
        transform: Optional[Callable] = None,
    ) -> None:
        self.image_dir  = image_dir
        self.mask_dir   = mask_dir
        self.class_info = class_info
        self.image_size = image_size
        self.transform  = transform or _val_transforms()
        self.ids: List[str] = sorted(
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in self._EXTENSIONS
        )
        if len(self.ids) == 0:
            raise RuntimeError(f"No images found in {image_dir}")
        logger.info("OCTDataset: %d images found in %s", len(self.ids), image_dir)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname    = self.ids[idx]
        img_path = os.path.join(self.image_dir, fname)
        msk_path = os.path.join(self.mask_dir,  fname)

        img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask_bgr = cv2.imread(msk_path, cv2.IMREAD_COLOR)

        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        if mask_bgr is None:
            raise FileNotFoundError(f"Cannot read mask: {msk_path}")

        img      = cv2.resize(img,      (self.image_size, self.image_size))
        mask_bgr = cv2.resize(mask_bgr, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_NEAREST)
        mask     = rgb_mask_to_class(mask_bgr, self.class_info)

        aug      = self.transform(image=img, mask=mask)
        image_t  = aug["image"]                  # float32 tensor (1,H,W)
        mask_t   = aug["mask"].long()            # int64  tensor (H,W)

        return image_t, mask_t

    def get_filename(self, idx: int) -> str:
        """Return original filename for the given index."""
        return self.ids[idx]


# ── Transforms ─────────────────────────────────────────────────────────────────

def _common_norm() -> list:
    """Shared normalisation + tensor conversion steps."""
    return [
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2(),
    ]


def train_transforms() -> A.Compose:
    """Augmentation pipeline for training splits."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, p=0.4),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.GridDistortion(p=0.2),
        A.CLAHE(clip_limit=4.0, p=0.3),
        *_common_norm(),
    ])


def _val_transforms() -> A.Compose:
    """Minimal pipeline for validation / test / inference splits."""
    return A.Compose(_common_norm())


val_transforms  = _val_transforms   # alias for external import
test_transforms = _val_transforms   # alias for external import


# ── DataLoader factory ─────────────────────────────────────────────────────────

def build_dataloaders(
    cfg,                            # SimpleNamespace from config_loader
    class_info: Dict[int, dict],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from config.

    Parameters
    ----------
    cfg        : loaded config namespace.
    class_info : colour-to-class mapping dict.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    full_ds = OCTDataset(
        image_dir  = cfg.data.image_dir,
        mask_dir   = cfg.data.mask_dir,
        class_info = class_info,
        image_size = cfg.data.image_size,
        transform  = _val_transforms(),   # default; overridden below
    )
    n       = len(full_ds)
    n_train = int(cfg.data.train_split * n)
    n_val   = int(cfg.data.val_split   * n)
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(cfg.project.seed)
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=generator
    )

    # Set per-split transforms by patching the shared dataset
    train_ds.dataset.transform = train_transforms()
    val_ds.dataset.transform   = _val_transforms()
    test_ds.dataset.transform  = _val_transforms()

    pin = torch.cuda.is_available()
    kw  = dict(num_workers=2, pin_memory=pin)

    train_loader = DataLoader(train_ds, batch_size=cfg.inference.batch_size,
                              shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.inference.batch_size,
                              shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.inference.batch_size,
                              shuffle=False, **kw)

    logger.info("Splits → Train:%d  Val:%d  Test:%d", n_train, n_val, n_test)
    return train_loader, val_loader, test_loader
