"""
inference/roi_cropper.py
-------------------------
Reusable ROI cropping utilities — designed for Phase 3 integration.

This module is Phase-agnostic: it takes pixel coordinates and a numpy
image and returns cropped patches. It does NOT import ultralytics or
any Phase 2–specific code, so Phase 3 can import it without changes.

Functions
---------
- crop_roi            : crop + optional resize preserving aspect ratio.
- safe_pad_coords     : clamp coordinates to image boundaries.
- map_coords_to_original : scale Phase 3 segmentation output back to
                           original image space.
- batch_crop_rois     : crop multiple images from a list of detections.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Coordinate helpers ─────────────────────────────────────────────────────────

def safe_pad_coords(
    x1: int, y1: int, x2: int, y2: int,
    padding_px: int,
    img_w: int, img_h: int,
) -> Tuple[int, int, int, int]:
    """
    Apply padding to bbox coordinates, clamped to image boundaries.

    Parameters
    ----------
    x1, y1, x2, y2 : pixel coordinates (top-left, bottom-right).
    padding_px      : pixels to expand on each side.
    img_w, img_h    : image dimensions.

    Returns
    -------
    (x1, y1, x2, y2)  padded and clamped.
    """
    return (
        max(0,     x1 - padding_px),
        max(0,     y1 - padding_px),
        min(img_w, x2 + padding_px),
        min(img_h, y2 + padding_px),
    )


# ── Crop ──────────────────────────────────────────────────────────────────────

def crop_roi(
    image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    target_size: Optional[int] = None,
    keep_aspect_ratio: bool = True,
    padding_px: int = 0,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop a Region of Interest from an image.

    Parameters
    ----------
    image             : H×W or H×W×3 numpy array.
    x1, y1, x2, y2   : crop coordinates in original image space.
    target_size       : if set, resize the crop to this square size.
    keep_aspect_ratio : if True and target_size is set, pad to square
                        rather than stretching.
    padding_px        : extra padding applied before cropping.

    Returns
    -------
    crop        : the cropped (and optionally resized) numpy array.
    final_coords: (x1, y1, x2, y2) actually used for the crop (post-padding).
    """
    h, w = image.shape[:2]
    cx1, cy1, cx2, cy2 = safe_pad_coords(
        x1, y1, x2, y2, padding_px, img_w=w, img_h=h
    )

    # Guard against degenerate boxes
    if cx1 >= cx2 or cy1 >= cy2:
        logger.warning("Degenerate crop box (%d,%d,%d,%d) — returning full image.",
                       cx1, cy1, cx2, cy2)
        cx1, cy1, cx2, cy2 = 0, 0, w, h

    crop = image[cy1:cy2, cx1:cx2]

    if target_size is not None:
        if keep_aspect_ratio:
            crop = _resize_with_padding(crop, target_size)
        else:
            crop = cv2.resize(crop, (target_size, target_size),
                              interpolation=cv2.INTER_LINEAR)

    return crop, (cx1, cy1, cx2, cy2)


def _resize_with_padding(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize image to fit within target_size × target_size, padding with zeros
    to maintain aspect ratio.
    """
    h, w  = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if image.ndim == 2:
        canvas = np.zeros((target_size, target_size), dtype=image.dtype)
    else:
        canvas = np.zeros((target_size, target_size, image.shape[2]), dtype=image.dtype)

    pad_y = (target_size - new_h) // 2
    pad_x = (target_size - new_w) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    return canvas


# ── Coordinate mapping ────────────────────────────────────────────────────────

def map_coords_to_original(
    x: float, y: float,
    roi_x1: int, roi_y1: int,
    roi_w: int, roi_h: int,
    resized_w: int, resized_h: int,
) -> Tuple[float, float]:
    """
    Map a point (x, y) from resized ROI space back to original image space.

    Used in Phase 3 to project segmentation mask predictions back onto
    the full OCT image.

    Parameters
    ----------
    x, y            : coordinates in resized ROI image space.
    roi_x1, roi_y1  : top-left corner of the ROI in original image.
    roi_w, roi_h    : ROI dimensions in original image.
    resized_w, resized_h : dimensions the ROI was resized to.

    Returns
    -------
    (orig_x, orig_y)  coordinates in original image space.
    """
    scale_x = roi_w / resized_w
    scale_y = roi_h / resized_h
    orig_x  = roi_x1 + x * scale_x
    orig_y  = roi_y1 + y * scale_y
    return orig_x, orig_y


def map_mask_to_original(
    roi_mask: np.ndarray,
    roi_x1: int, roi_y1: int,
    orig_h: int, orig_w: int,
) -> np.ndarray:
    """
    Resize and place an ROI-space segmentation mask back into original image space.

    Parameters
    ----------
    roi_mask      : H'×W' segmentation mask (integer class map) in ROI space.
    roi_x1,roi_y1 : top-left corner of the ROI in original image.
    orig_h, orig_w: original image dimensions.

    Returns
    -------
    np.ndarray  full-resolution H×W mask with the ROI region filled in
                and background set to 0 (background class).
    """
    roi_h = roi_mask.shape[0]
    roi_w = roi_mask.shape[1]
    # Determine actual ROI dimensions in original space
    roi_x2 = min(orig_w, roi_x1 + roi_w)
    roi_y2 = min(orig_h, roi_y1 + roi_h)
    actual_w = roi_x2 - roi_x1
    actual_h = roi_y2 - roi_y1

    resized = cv2.resize(
        roi_mask.astype(np.uint8),
        (actual_w, actual_h),
        interpolation=cv2.INTER_NEAREST,  # no blending of class ids
    )

    full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    full_mask[roi_y1:roi_y2, roi_x1:roi_x2] = resized
    return full_mask


# ── Batch crop ────────────────────────────────────────────────────────────────

def batch_crop_rois(
    images: List[np.ndarray],
    detections_list,        # List[List[Detection]] from roi_inference.py
    target_size: Optional[int] = 256,
    keep_aspect_ratio: bool = True,
    padding_px: int = 0,
    fallback_to_full: bool = True,
) -> List[Tuple[np.ndarray, Optional[Tuple[int,int,int,int]]]]:
    """
    Crop the primary (highest-confidence) detected ROI from each image.

    Parameters
    ----------
    images            : list of H×W greyscale arrays.
    detections_list   : list of Detection lists (one per image).
    target_size       : resize each crop to this square size.
    keep_aspect_ratio : pad rather than stretch when resizing.
    padding_px        : extra padding added before cropping.
    fallback_to_full  : if no detection, crop the full image.

    Returns
    -------
    List of (crop, coords)  where coords = (x1, y1, x2, y2) or None.
    """
    output = []
    for img, dets in zip(images, detections_list):
        h, w = img.shape[:2]

        if not dets:
            if fallback_to_full:
                logger.debug("No detection → using full image as ROI.")
                crop, coords = crop_roi(img, 0, 0, w, h, target_size,
                                        keep_aspect_ratio, padding_px)
            else:
                crop   = np.zeros((target_size or h, target_size or w), dtype=img.dtype)
                coords = None
        else:
            # Use the highest-confidence detection
            best = dets[0]
            crop, coords = crop_roi(img, best.x1, best.y1, best.x2, best.y2,
                                    target_size, keep_aspect_ratio, padding_px)

        output.append((crop, coords))
    return output
