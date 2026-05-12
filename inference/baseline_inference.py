"""
inference/baseline_inference.py
--------------------------------
Batch inference pipeline for the baseline segmentation model.

Responsibilities
----------------
1. Load model from checkpoint.
2. Run batch inference over a DataLoader (or directory of images).
3. Optionally save predicted mask PNG files.
4. Return (logits, predictions, ground_truth) tensors for metric computation.

Design decisions
----------------
- Inference is separated from evaluation so it can be re-used in future
  ROI-guided phases without modification.
- Predictions are saved as colour-coded RGB PNGs (same palette as training).
- Timing is collected here as a by-product; dedicated benchmarking lives in
  metrics_throughput.py which gives more statistically rigorous numbers.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

_DEFAULT_COLORS: List[List[int]] = [
    [0,   0,   0  ],
    [255, 0,   255],
    [255, 255, 0  ],
    [255, 0,   0  ],
    [0,   0,   255],
    [0,   255, 0  ],
]


def _mask_to_rgb(mask_np: np.ndarray, class_colors: List[List[int]]) -> np.ndarray:
    """Convert H×W class map → H×W×3 uint8 RGB."""
    h, w = mask_np.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for c, col in enumerate(class_colors):
        rgb[mask_np == c] = col
    return rgb


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


# ── Core batch runner ──────────────────────────────────────────────────────────

def run_batch_inference(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
    save_predictions: bool = False,
    output_dir: str = "results/predictions",
    class_colors: Optional[List[List[int]]] = None,
    num_warmup_batches: int = 3,
) -> Dict:
    """
    Run inference over all batches in *data_loader*.

    Parameters
    ----------
    model                : model in eval mode, on *device*.
    data_loader          : DataLoader yielding (image, mask) pairs.
    device               : torch device string.
    save_predictions     : if True, write colour-coded PNGs to *output_dir*.
    output_dir           : directory for prediction PNGs.
    class_colors         : RGB colour list (defaults to _DEFAULT_COLORS).
    num_warmup_batches   : batches to run before timing starts.

    Returns
    -------
    dict with keys:
        all_logits   : (N, C, H, W) float tensor (on CPU).
        all_preds    : (N, H, W)    int   tensor (on CPU).
        all_targets  : (N, H, W)    int   tensor (on CPU).
        inference_time_s : total wall time for timed batches (s).
        num_images   : N.
    """
    colors = class_colors or _DEFAULT_COLORS
    if save_predictions:
        os.makedirs(output_dir, exist_ok=True)

    model.eval()
    all_logits:  List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    total_time_s = 0.0
    batch_idx    = 0
    saved_count  = 0

    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc="Inference", unit="batch"):
            images = images.to(device)
            masks  = masks.to(device)

            if batch_idx < num_warmup_batches:
                # Warm-up: not timed
                _sync(device)
                logits = model(images)
                _sync(device)
            else:
                _sync(device)
                t0     = time.perf_counter()
                logits = model(images)
                _sync(device)
                t1     = time.perf_counter()
                total_time_s += (t1 - t0)

            all_logits.append(logits.cpu())
            all_targets.append(masks.cpu())

            # ── Save predictions ──────────────────────────────────────────────
            if save_predictions:
                preds_np = logits.argmax(dim=1).cpu().numpy()
                for pred in preds_np:
                    rgb = _mask_to_rgb(pred, colors)
                    out = os.path.join(output_dir, f"pred_{saved_count:04d}.png")
                    cv2.imwrite(out, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    saved_count += 1

            batch_idx += 1

    all_logits_t  = torch.cat(all_logits,  dim=0)
    all_targets_t = torch.cat(all_targets, dim=0)
    all_preds_t   = all_logits_t.argmax(dim=1)

    num_timed = batch_idx - num_warmup_batches
    logger.info(
        "Inference complete: %d images | %.3f s (timed %d batches) | saved=%d",
        len(all_logits_t), total_time_s, max(num_timed, 0), saved_count,
    )

    return {
        "all_logits":       all_logits_t,
        "all_preds":        all_preds_t,
        "all_targets":      all_targets_t,
        "inference_time_s": total_time_s,
        "num_images":       len(all_logits_t),
    }


# ── Single image ───────────────────────────────────────────────────────────────

def predict_single(
    model: nn.Module,
    image_path: str,
    image_size: int,
    device: str = "cpu",
    class_colors: Optional[List[List[int]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a single OCT image file.

    Parameters
    ----------
    model       : model in eval mode.
    image_path  : path to greyscale OCT image.
    image_size  : resize target (square).
    device      : torch device string.
    class_colors: optional colour list.

    Returns
    -------
    (pred_class_map, pred_rgb)  — both H×W (×3) numpy arrays.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    colors = class_colors or _DEFAULT_COLORS
    img    = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img = cv2.resize(img, (image_size, image_size))

    transform = A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    tensor = transform(image=img)["image"].unsqueeze(0).to(device)  # (1,1,H,W)

    model.eval()
    with torch.no_grad():
        logits    = model(tensor)
    pred_cls  = logits.argmax(dim=1).squeeze().cpu().numpy()
    pred_rgb  = _mask_to_rgb(pred_cls, colors)

    return pred_cls, pred_rgb
