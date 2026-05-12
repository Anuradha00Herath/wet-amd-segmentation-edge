"""
pipelines/coord_mapper.py
--------------------------
ROI coordinate mapping utilities for Phase 3.

Problem
-------
After detecting an ROI (x1,y1,x2,y2), cropping it, and running
segmentation on the crop, the predicted class map is in "crop space"
(e.g. 256×256). It must be:

  1. Scaled back to the cropped ROI's actual pixel size.
  2. Placed at the correct (x1,y1) offset in the full original image.
  3. The rest of the full-image canvas is filled with background (class 0),
     or optionally with a fallback prediction.

This module handles all of those transformations, with explicit checks
for edge cases (degenerate boxes, out-of-bounds crops, aspect-ratio padding).

Functions
---------
- map_roi_pred_to_full      : core mapping function.
- merge_roi_predictions     : merge multiple ROI masks into one.
- compute_effective_padding : compute padding from "percent" mode.
- validate_coords           : sanity-check bbox before use.
- DebugMapper               : logs every mapping step (for debugging).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_coords(
    x1: int, y1: int, x2: int, y2: int,
    img_w: int, img_h: int,
    label: str = "",
) -> bool:
    """
    Sanity-check bbox coordinates.

    Returns True if valid, False + warning log if degenerate.
    """
    if x1 >= x2 or y1 >= y2:
        logger.warning("%s Degenerate bbox: (%d,%d,%d,%d)", label, x1, y1, x2, y2)
        return False
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        logger.warning(
            "%s Bbox partially out of image (%dx%d): (%d,%d,%d,%d)",
            label, img_w, img_h, x1, y1, x2, y2,
        )
        # Still valid — out-of-bounds will be clamped
    return True


# ── Padding computation ───────────────────────────────────────────────────────

def compute_effective_padding(
    x1: int, y1: int, x2: int, y2: int,
    padding_mode: str = "fixed",
    padding_px: int = 40,
    padding_percent: float = 0.20,
    img_w: int = 256, img_h: int = 256,
) -> int:
    """
    Compute effective padding in pixels.

    Parameters
    ----------
    padding_mode    : "fixed" → use padding_px directly.
                      "percent" → padding_px = padding_percent × max(w,h).
    padding_px      : fixed pixel padding.
    padding_percent : fractional padding (0.20 = 20% of bbox size).
    img_w, img_h    : image dimensions (used for clamping, not computation).

    Returns
    -------
    int  effective padding in pixels.
    """
    if padding_mode == "percent":
        box_max = max(x2 - x1, y2 - y1)
        return int(box_max * padding_percent)
    return padding_px


# ── Core mapping ──────────────────────────────────────────────────────────────

def map_roi_pred_to_full(
    roi_pred:     np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    orig_h: int,  orig_w: int,
    background_class: int = 0,
) -> np.ndarray:
    """
    Map a segmentation prediction from crop space back to full image space.

    Steps
    -----
    1. Resize ``roi_pred`` (which is in seg-model output space, e.g. 256×256)
       to the actual pixel size of the ROI crop (x2-x1) × (y2-y1).
    2. Clamp ROI coordinates to image boundaries.
    3. Place the resized prediction into a full-size canvas filled with
       ``background_class``.

    Parameters
    ----------
    roi_pred         : H'×W' uint8 class map from the segmentation model.
    x1,y1,x2,y2     : ROI bounding box in original image coordinates
                       (post-padding, pre-clamp).
    orig_h, orig_w   : original image dimensions.
    background_class : class id to fill outside the ROI region.

    Returns
    -------
    np.ndarray  H×W uint8 full-resolution class map.
    """
    # ── Clamp to image boundaries ─────────────────────────────────────────────
    cx1 = max(0, x1);      cy1 = max(0, y1)
    cx2 = min(orig_w, x2); cy2 = min(orig_h, y2)

    if cx1 >= cx2 or cy1 >= cy2:
        logger.warning(
            "map_roi_pred_to_full: ROI (%d,%d,%d,%d) entirely outside image "
            "(%d×%d). Returning background.", x1, y1, x2, y2, orig_w, orig_h,
        )
        return np.full((orig_h, orig_w), background_class, dtype=np.uint8)

    roi_w = cx2 - cx1
    roi_h = cy2 - cy1

    # ── Resize prediction to ROI pixel size ───────────────────────────────────
    if roi_pred.shape != (roi_h, roi_w):
        resized = cv2.resize(
            roi_pred.astype(np.uint8),
            (roi_w, roi_h),
            interpolation=cv2.INTER_NEAREST,   # preserve class ids — no blending
        )
    else:
        resized = roi_pred.astype(np.uint8)

    # ── Place into full canvas ────────────────────────────────────────────────
    full = np.full((orig_h, orig_w), background_class, dtype=np.uint8)
    full[cy1:cy2, cx1:cx2] = resized

    logger.debug(
        "Mapped ROI pred %s → full %s | roi_bbox=(%d,%d,%d,%d)",
        roi_pred.shape, full.shape, cx1, cy1, cx2, cy2,
    )
    return full


# ── Multi-ROI merge ───────────────────────────────────────────────────────────

def merge_roi_predictions(
    full_preds: List[np.ndarray],
    strategy: str = "last_wins",
) -> np.ndarray:
    """
    Merge multiple full-resolution ROI predictions into one mask.

    Parameters
    ----------
    full_preds : list of H×W uint8 class maps (same size).
    strategy   : "last_wins" — later predictions overwrite earlier ones
                               (use when ROIs are non-overlapping).
                 "max_class" — take the highest class id per pixel
                               (useful for ordered class hierarchy).

    Returns
    -------
    np.ndarray  H×W uint8 merged class map.
    """
    if not full_preds:
        raise ValueError("merge_roi_predictions: empty prediction list.")

    merged = full_preds[0].copy()
    for pred in full_preds[1:]:
        if strategy == "last_wins":
            mask = pred > 0          # overwrite only where non-background
            merged[mask] = pred[mask]
        elif strategy == "max_class":
            merged = np.maximum(merged, pred)
        else:
            raise ValueError(f"Unknown merge strategy: {strategy!r}")
    return merged


# ── Debug mapper ──────────────────────────────────────────────────────────────

@dataclass
class MappingRecord:
    """Stores one mapping operation's inputs/outputs for debugging."""
    roi_pred_shape:  tuple
    bbox:            Tuple[int,int,int,int]
    full_shape:      Tuple[int,int]
    unique_classes:  List[int]
    non_bg_pixels:   int


class DebugMapper:
    """
    Wrapper around ``map_roi_pred_to_full`` that records every operation
    for visual debugging and coordinate mismatch analysis.

    Usage::

        mapper = DebugMapper()
        full = mapper.map(roi_pred, x1, y1, x2, y2, H, W)
        mapper.print_summary()
    """

    def __init__(self) -> None:
        self.records: List[MappingRecord] = []

    def map(
        self,
        roi_pred: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        orig_h: int, orig_w: int,
        background_class: int = 0,
    ) -> np.ndarray:
        full = map_roi_pred_to_full(
            roi_pred, x1, y1, x2, y2, orig_h, orig_w, background_class
        )
        self.records.append(MappingRecord(
            roi_pred_shape = roi_pred.shape,
            bbox           = (x1, y1, x2, y2),
            full_shape     = (orig_h, orig_w),
            unique_classes = list(np.unique(full).astype(int)),
            non_bg_pixels  = int((full != background_class).sum()),
        ))
        return full

    def print_summary(self) -> None:
        print(f"\n{'─'*55}")
        print(f"  DebugMapper: {len(self.records)} mapping(s)")
        print(f"{'─'*55}")
        for i, r in enumerate(self.records):
            print(f"  [{i}] ROI pred: {r.roi_pred_shape} → full: {r.full_shape}")
            print(f"       bbox: {r.bbox}")
            print(f"       classes: {r.unique_classes}  non-bg px: {r.non_bg_pixels}")
        print()
