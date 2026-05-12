"""
detectors/roi_visualizer.py
-----------------------------
Visualisation utilities specific to Phase 2 ROI detection.

Functions
---------
- visualize_bbox_on_image     : draw boxes on a single OCT image.
- visualize_mask_vs_bbox      : side-by-side mask + bbox overlay.
- save_dataset_samples        : random sample grid from the YOLO dataset.
- visualize_detections        : draw YOLOv8 inference results on an image.
- save_detection_grid         : grid of detection results for a batch.
- plot_detection_metrics      : bar / line charts for detector evaluation.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from detectors.bbox_generator import BoundingBox

logger = logging.getLogger(__name__)

# Colour palette for drawing boxes (BGR for cv2, RGB for matplotlib)
_BOX_COLOR_BGR = (0, 255, 0)      # green
_BOX_COLOR_RGB = (0, 1.0, 0)
_GT_COLOR_BGR  = (255, 0, 0)      # blue  (ground truth)
_PRED_COLOR_BGR= (0, 165, 255)    # orange (prediction)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ── Single image ───────────────────────────────────────────────────────────────

def visualize_bbox_on_image(
    image: np.ndarray,
    boxes: List[BoundingBox],
    color_bgr: Tuple[int, int, int] = _BOX_COLOR_BGR,
    thickness: int = 2,
    label: str = "lesion",
) -> np.ndarray:
    """
    Draw bounding boxes onto a greyscale or RGB image copy.

    Parameters
    ----------
    image     : H×W or H×W×3 numpy array.
    boxes     : list of BoundingBox objects.
    color_bgr : BGR colour for box outline.
    thickness : line thickness.
    label     : text label drawn above each box.

    Returns
    -------
    np.ndarray  BGR colour image with boxes drawn.
    """
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    for box in boxes:
        cv2.rectangle(vis, (box.x1, box.y1), (box.x2, box.y2), color_bgr, thickness)
        cv2.putText(
            vis, label, (box.x1, max(0, box.y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA,
        )
    return vis


def visualize_mask_vs_bbox(
    image: np.ndarray,
    mask_bgr: np.ndarray,
    boxes: List[BoundingBox],
    save_path: str,
    title: str = "",
) -> None:
    """
    Save a 3-panel figure: [OCT | RGB mask | OCT + bbox].

    Parameters
    ----------
    image    : H×W greyscale numpy array.
    mask_bgr : H×W×3 BGR mask array.
    boxes    : generated bounding boxes.
    save_path: output PNG path.
    title    : optional suptitle.
    """
    overlay = visualize_bbox_on_image(image, boxes)
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=10)
    axes[0].imshow(image,   cmap="gray"); axes[0].set_title("OCT Image")
    axes[1].imshow(mask_rgb);             axes[1].set_title("Segmentation Mask")
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Generated BBox ({len(boxes)} boxes)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved mask vs bbox: %s", save_path)


# ── Dataset sample grid ────────────────────────────────────────────────────────

def save_dataset_samples(
    image_dir: str,
    bbox_results: Dict[str, List[BoundingBox]],
    save_path: str,
    n_samples: int = 12,
    seed: int = 42,
    cols: int = 4,
) -> None:
    """
    Save a grid of random samples showing OCT images with their generated bboxes.

    Parameters
    ----------
    image_dir    : directory of source OCT images.
    bbox_results : ``{filename: List[BoundingBox]}``.
    save_path    : output PNG path.
    n_samples    : number of images to show.
    seed         : for reproducible sampling.
    cols         : grid columns.
    """
    rng       = random.Random(seed)
    all_files = [f for f, boxes in bbox_results.items() if boxes]
    samples   = rng.sample(all_files, min(n_samples, len(all_files)))
    rows      = (len(samples) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).reshape(-1)

    for i, fname in enumerate(samples):
        img_path = os.path.join(image_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        overlay = visualize_bbox_on_image(img, bbox_results[fname])
        axes[i].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[i].set_title(Path(fname).stem[:20], fontsize=7)
        axes[i].axis("off")

    for j in range(len(samples), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Generated ROI Bounding Boxes — Sample Preview", fontsize=11)
    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved dataset sample grid: %s", save_path)


# ── YOLOv8 detection results ───────────────────────────────────────────────────

def draw_yolo_detections(
    image: np.ndarray,
    detections: List[dict],
    gt_boxes: Optional[List[BoundingBox]] = None,
    conf_threshold: float = 0.0,
) -> np.ndarray:
    """
    Draw YOLOv8 detection results on an image.

    Parameters
    ----------
    image          : H×W greyscale or H×W×3 BGR array.
    detections     : list of dicts with keys: x1,y1,x2,y2,conf,cls.
    gt_boxes       : optional ground-truth boxes (drawn in blue).
    conf_threshold : skip detections below this confidence.

    Returns
    -------
    np.ndarray  BGR visualisation image.
    """
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    # Draw GT (blue)
    if gt_boxes:
        for box in gt_boxes:
            cv2.rectangle(vis, (box.x1, box.y1), (box.x2, box.y2), _GT_COLOR_BGR, 2)
            cv2.putText(vis, "GT", (box.x1, max(0, box.y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _GT_COLOR_BGR, 1)

    # Draw predictions (orange)
    for det in detections:
        if det.get("conf", 1.0) < conf_threshold:
            continue
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        conf = det.get("conf", 0.0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), _PRED_COLOR_BGR, 2)
        cv2.putText(vis, f"{conf:.2f}", (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _PRED_COLOR_BGR, 1)
    return vis


def save_detection_grid(
    images: List[np.ndarray],
    detections_list: List[List[dict]],
    gt_boxes_list: Optional[List[List[BoundingBox]]],
    save_path: str,
    filenames: Optional[List[str]] = None,
    cols: int = 4,
) -> None:
    """
    Save a grid of detection result visualisations.

    Parameters
    ----------
    images           : list of H×W greyscale arrays.
    detections_list  : list of detection dicts per image.
    gt_boxes_list    : optional list of GT box lists per image.
    save_path        : output PNG path.
    filenames        : optional filenames for titles.
    cols             : grid columns.
    """
    n    = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        gt = gt_boxes_list[i] if gt_boxes_list else None
        vis = draw_yolo_detections(images[i], detections_list[i], gt_boxes=gt)
        axes[i].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        title = (filenames[i][:20] if filenames else f"Sample {i}")
        axes[i].set_title(title, fontsize=7)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    # Legend
    handles = [
        mpatches.Patch(color=np.array([0,0,1.0]), label="Ground Truth"),
        mpatches.Patch(color=np.array([1.0,0.65,0]), label="Prediction"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("ROI Detector Predictions", fontsize=11)
    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved detection grid: %s", save_path)


# ── Metric plots ───────────────────────────────────────────────────────────────

def plot_detection_metrics(
    metrics: dict,
    save_path: str,
    title: str = "ROI Detector Metrics",
) -> None:
    """
    Bar chart of scalar detection metrics (mAP50, Precision, Recall, F1 …).

    Parameters
    ----------
    metrics   : ``{"mAP50": 0.92, "Precision": 0.88, ...}``.
    save_path : output PNG path.
    title     : figure title.
    """
    keys   = [k for k, v in metrics.items() if isinstance(v, (int, float))]
    values = [metrics[k] for k in keys]

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.2), 4))
    bars = ax.bar(keys, values, color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved detection metrics plot: %s", save_path)
