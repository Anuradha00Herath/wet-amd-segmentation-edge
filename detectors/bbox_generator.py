"""
detectors/bbox_generator.py
-----------------------------
Generate YOLO-format bounding boxes from RGB-encoded segmentation masks.

Pipeline per image
------------------
1. Read RGB mask → convert to integer class map (reuses Phase 1 logic).
2. Create a binary lesion map (union of configured lesion class ids).
3. Find connected components with cv2.connectedComponentsWithStats.
4. Filter out tiny components (noise) by minimum area threshold.
5. Optionally merge all components into one union bbox.
6. Apply configurable pixel padding (clamped to image boundaries).
7. Convert pixel bbox → YOLO normalised format  [cx, cy, w, h] ∈ [0,1].

Design decisions
----------------
- Lesion class ids are configurable so the researcher can include/exclude
  structural classes (e.g. Retinal Layer) without touching code.
- ``merge_all_lesions=True`` (default) produces a single ROI per image —
  appropriate for Phase 3 where we crop one region for the segmentation model.
- ``merge_all_lesions=False`` produces per-component boxes — useful for
  analysing individual lesion types.
- All functions are stateless and accept numpy arrays so they can be tested
  and reused in notebooks without I/O.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """
    Axis-aligned bounding box in pixel coordinates.

    Attributes
    ----------
    x1, y1 : top-left corner.
    x2, y2 : bottom-right corner (exclusive).
    class_id : YOLO class id (0 = lesion by default).
    component_area : pixel area of the source connected component.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int = 0
    component_area: int = 0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_yolo(self, img_w: int, img_h: int) -> Tuple[int, float, float, float, float]:
        """
        Convert to YOLO normalised format.

        Returns
        -------
        (class_id, cx, cy, w, h)  — all floats normalised to [0, 1].
        """
        cx = (self.x1 + self.x2) / 2.0 / img_w
        cy = (self.y1 + self.y2) / 2.0 / img_h
        w  = self.width  / img_w
        h  = self.height / img_h
        return (self.class_id, cx, cy, w, h)

    def pad(self, px: int, img_w: int, img_h: int) -> "BoundingBox":
        """Return a new BoundingBox with padding applied, clamped to image."""
        return BoundingBox(
            x1=max(0,     self.x1 - px),
            y1=max(0,     self.y1 - px),
            x2=min(img_w, self.x2 + px),
            y2=min(img_h, self.y2 + px),
            class_id=self.class_id,
            component_area=self.component_area,
        )


# ── Core extraction logic ──────────────────────────────────────────────────────

def extract_lesion_binary(
    class_map: np.ndarray,
    lesion_class_ids: List[int],
) -> np.ndarray:
    """
    Build a binary lesion mask from an integer class map.

    Parameters
    ----------
    class_map        : H×W uint8 integer class map.
    lesion_class_ids : class ids to treat as lesion (e.g. [2,3,4,5]).

    Returns
    -------
    np.ndarray  H×W uint8 binary mask (255 = lesion, 0 = background).
    """
    binary = np.zeros_like(class_map, dtype=np.uint8)
    for cls_id in lesion_class_ids:
        binary[class_map == cls_id] = 255
    return binary


def find_component_boxes(
    binary_mask: np.ndarray,
    min_area_px: int = 50,
) -> List[BoundingBox]:
    """
    Find bounding boxes for all connected components in a binary mask.

    Parameters
    ----------
    binary_mask : H×W uint8 mask (255 = foreground).
    min_area_px : components smaller than this are ignored as noise.

    Returns
    -------
    List[BoundingBox]  One box per qualifying connected component.
    """
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    boxes: List[BoundingBox] = []
    # Label 0 is the background — start from 1
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        x1 = stats[label, cv2.CC_STAT_LEFT]
        y1 = stats[label, cv2.CC_STAT_TOP]
        w  = stats[label, cv2.CC_STAT_WIDTH]
        h  = stats[label, cv2.CC_STAT_HEIGHT]
        boxes.append(BoundingBox(x1=x1, y1=y1, x2=x1+w, y2=y1+h,
                                 component_area=int(area)))
    return boxes


def merge_boxes(boxes: List[BoundingBox], class_id: int = 0) -> Optional[BoundingBox]:
    """
    Merge a list of bounding boxes into a single union bounding box.

    Returns None if boxes is empty.
    """
    if not boxes:
        return None
    x1 = min(b.x1 for b in boxes)
    y1 = min(b.y1 for b in boxes)
    x2 = max(b.x2 for b in boxes)
    y2 = max(b.y2 for b in boxes)
    total_area = sum(b.component_area for b in boxes)
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2,
                       class_id=class_id, component_area=total_area)


# ── RGB mask → BoundingBox list ────────────────────────────────────────────────

def boxes_from_mask(
    mask_bgr: np.ndarray,
    class_info: Dict[int, dict],
    lesion_class_ids: List[int],
    min_area_px: int = 50,
    padding_px: int = 10,
    merge_all: bool = True,
    yolo_class_id: int = 0,
) -> List[BoundingBox]:
    """
    Full pipeline: RGB mask → list of padded BoundingBox objects.

    Parameters
    ----------
    mask_bgr         : H×W×3 BGR uint8 mask (cv2.imread output).
    class_info       : ``{class_id: {"name":str, "rgb":(R,G,B)}}``
    lesion_class_ids : seg class ids to include in lesion binary.
    min_area_px      : minimum component area to keep.
    padding_px       : pixel padding applied to each side.
    merge_all        : if True, merge all components into one union box.
    yolo_class_id    : YOLO class label for output boxes.

    Returns
    -------
    List[BoundingBox]  Empty list if no lesion found.
    """
    from utils.dataset import rgb_mask_to_class  # reuse Phase 1 function

    h, w = mask_bgr.shape[:2]

    # Step 1: RGB → class map
    class_map = rgb_mask_to_class(mask_bgr, class_info)

    # Step 2: binary lesion mask
    binary = extract_lesion_binary(class_map, lesion_class_ids)

    if binary.max() == 0:
        logger.debug("No lesion pixels found in mask.")
        return []

    # Step 3: connected components
    component_boxes = find_component_boxes(binary, min_area_px)
    if not component_boxes:
        logger.debug("All components filtered by min_area_px=%d.", min_area_px)
        return []

    # Step 4: optionally merge
    if merge_all:
        merged = merge_boxes(component_boxes, class_id=yolo_class_id)
        raw_boxes = [merged] if merged else []
    else:
        for b in component_boxes:
            b.class_id = yolo_class_id
        raw_boxes = component_boxes

    # Step 5: pad
    padded = [b.pad(padding_px, img_w=w, img_h=h) for b in raw_boxes]
    return padded


# ── YOLO label file writer ─────────────────────────────────────────────────────

def write_yolo_label(
    boxes: List[BoundingBox],
    label_path: str,
    img_w: int,
    img_h: int,
) -> None:
    """
    Write YOLO-format label file (.txt) for a list of bounding boxes.

    Each line: ``class_id cx cy w h``  (all floats, space-separated).
    An empty file is written if boxes is empty (YOLO expects it for negatives).

    Parameters
    ----------
    boxes      : list of BoundingBox objects.
    label_path : output .txt file path.
    img_w, img_h : image dimensions for normalisation.
    """
    os.makedirs(os.path.dirname(label_path) or ".", exist_ok=True)
    with open(label_path, "w") as f:
        for box in boxes:
            cls, cx, cy, w, h = box.to_yolo(img_w, img_h)
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ── Batch processing ───────────────────────────────────────────────────────────

@dataclass
class BboxGenerationStats:
    """Accumulates statistics across a batch of mask → bbox conversions."""
    total:           int = 0
    with_lesion:     int = 0
    no_lesion:       int = 0
    multi_component: int = 0
    errors:          int = 0
    skipped_filenames: List[str] = field(default_factory=list)

    def report(self) -> str:
        return (
            f"Total: {self.total} | With lesion: {self.with_lesion} | "
            f"No lesion: {self.no_lesion} | Multi-component: {self.multi_component} | "
            f"Errors: {self.errors}"
        )


def generate_all_boxes(
    image_dir: str,
    mask_dir:  str,
    class_info: Dict[int, dict],
    lesion_class_ids: List[int],
    min_area_px: int = 50,
    padding_px:  int = 10,
    merge_all:   bool = True,
) -> Tuple[Dict[str, List[BoundingBox]], BboxGenerationStats]:
    """
    Batch-generate bounding boxes for all images in a directory.

    Parameters
    ----------
    image_dir        : directory containing OCT images.
    mask_dir         : directory containing RGB mask files (same filenames).
    class_info       : seg class colour mapping.
    lesion_class_ids : which seg classes count as lesion.
    min_area_px      : minimum component area.
    padding_px       : bbox padding in pixels.
    merge_all        : merge all components per image.

    Returns
    -------
    results : dict ``{filename: List[BoundingBox]}``
    stats   : BboxGenerationStats summary.
    """
    _EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    filenames = sorted(
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in _EXTENSIONS
    )

    results: Dict[str, List[BoundingBox]] = {}
    stats   = BboxGenerationStats()

    for fname in filenames:
        stats.total += 1
        mask_path  = os.path.join(mask_dir, fname)
        image_path = os.path.join(image_dir, fname)

        if not os.path.exists(mask_path):
            logger.warning("Mask not found for %s — skipping.", fname)
            stats.errors += 1
            stats.skipped_filenames.append(fname)
            continue

        try:
            mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            img      = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if mask_bgr is None or img is None:
                raise IOError("cv2.imread returned None")

            boxes = boxes_from_mask(
                mask_bgr         = mask_bgr,
                class_info       = class_info,
                lesion_class_ids = lesion_class_ids,
                min_area_px      = min_area_px,
                padding_px       = padding_px,
                merge_all        = merge_all,
            )

            results[fname] = boxes

            if not boxes:
                stats.no_lesion += 1
            else:
                stats.with_lesion += 1
                if len(boxes) > 1:
                    stats.multi_component += 1

        except Exception as e:
            logger.error("Error processing %s: %s", fname, e)
            stats.errors += 1
            stats.skipped_filenames.append(fname)
            results[fname] = []

    logger.info("Bbox generation complete: %s", stats.report())
    return results, stats
