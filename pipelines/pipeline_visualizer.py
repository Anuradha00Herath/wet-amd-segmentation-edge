"""
pipelines/pipeline_visualizer.py
----------------------------------
Visualisation utilities specific to the Phase 3 ROI-guided pipeline.

Functions
---------
- visualize_pipeline_result   : 4-panel: OCT | ROI crop | seg in crop | full pred
- visualize_baseline_vs_roi   : side-by-side baseline vs ROI prediction
- save_pipeline_grid          : grid of N pipeline results
- visualize_failure_case      : highlight failed detection / bad mapping
- save_failure_cases          : batch save failure case visuals
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pipelines.roi_seg_pipeline import PipelineResult

logger = logging.getLogger(__name__)

_DEFAULT_COLORS = [
    [0,   0,   0  ], [255, 0, 255], [255, 255, 0],
    [255, 0,   0  ], [0,   0, 255], [0,   255, 0],
]


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _mask_to_rgb(mask: np.ndarray, colors: list) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in enumerate(colors):
        rgb[mask == c] = col
    return rgb


def _draw_box(img_bgr: np.ndarray, coords, color=(0, 165, 255), thickness=2):
    """Draw a bounding box on a BGR image."""
    if coords is None:
        return img_bgr
    x1, y1, x2, y2 = coords
    return cv2.rectangle(img_bgr.copy(), (x1, y1), (x2, y2), color, thickness)


# ── Single result ──────────────────────────────────────────────────────────────

def visualize_pipeline_result(
    image:      np.ndarray,
    gt_mask:    np.ndarray,
    result:     PipelineResult,
    save_path:  str,
    class_names: List[str],
    class_colors: Optional[list] = None,
    title: str = "",
) -> None:
    """
    5-panel visualisation: OCT | GT | ROI crop | seg in crop | full pred.

    Parameters
    ----------
    image       : H×W greyscale array.
    gt_mask     : H×W integer class map.
    result      : PipelineResult from ROIPipeline.run().
    save_path   : output PNG path.
    class_names : list of class name strings.
    class_colors: optional colour list.
    title       : suptitle string.
    """
    colors = class_colors or _DEFAULT_COLORS

    # Panel images
    gt_rgb   = _mask_to_rgb(gt_mask, colors)
    full_rgb = _mask_to_rgb(result.full_pred, colors)

    # OCT with ROI box overlay
    img_bgr  = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if result.roi_coords:
        img_bgr = _draw_box(img_bgr, result.roi_coords)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ROI crop
    if result.roi_coords and result.roi_pred is not None:
        x1, y1, x2, y2 = result.roi_coords
        crop   = image[y1:y2, x1:x2]
        roi_rgb= _mask_to_rgb(result.roi_pred, colors)
        n_panels = 5
    else:
        crop    = image
        roi_rgb = np.zeros((*image.shape, 3), dtype=np.uint8)
        n_panels= 5

    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 3.5, 3.5))
    fig.suptitle(title or "ROI Pipeline Result", fontsize=10)

    axes[0].imshow(image,    cmap="gray"); axes[0].set_title("OCT Input")
    axes[1].imshow(img_rgb);               axes[1].set_title("Detected ROI")
    axes[2].imshow(gt_rgb);                axes[2].set_title("Ground Truth")
    axes[3].imshow(roi_rgb);               axes[3].set_title("Seg in Crop")
    axes[4].imshow(full_rgb);              axes[4].set_title("Full Prediction")
    for ax in axes:
        ax.axis("off")

    patches = [mpatches.Patch(color=np.array(colors[i])/255.0, label=f"{i}:{n}")
               for i, n in enumerate(class_names)]
    fig.legend(handles=patches, loc="lower center", ncol=min(6, len(class_names)),
               fontsize=7, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved pipeline result: %s", save_path)


# ── Baseline vs ROI comparison ────────────────────────────────────────────────

def visualize_baseline_vs_roi(
    image:          np.ndarray,
    gt_mask:        np.ndarray,
    baseline_pred:  np.ndarray,
    roi_pred:       np.ndarray,
    save_path:      str,
    class_names:    List[str],
    class_colors:   Optional[list] = None,
    title:          str = "",
) -> None:
    """
    4-panel comparison: OCT | GT | Baseline seg | ROI-guided seg.

    Parameters
    ----------
    baseline_pred : H×W class map from full-image segmentation.
    roi_pred      : H×W class map from ROI-guided pipeline.
    """
    colors   = class_colors or _DEFAULT_COLORS
    gt_rgb   = _mask_to_rgb(gt_mask,       colors)
    base_rgb = _mask_to_rgb(baseline_pred, colors)
    roi_rgb  = _mask_to_rgb(roi_pred,      colors)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title or "Baseline vs ROI-Guided", fontsize=10)

    axes[0].imshow(image,    cmap="gray"); axes[0].set_title("OCT Input")
    axes[1].imshow(gt_rgb);                axes[1].set_title("Ground Truth")
    axes[2].imshow(base_rgb);              axes[2].set_title("Baseline (Full Image)")
    axes[3].imshow(roi_rgb);               axes[3].set_title("ROI-Guided")
    for ax in axes:
        ax.axis("off")

    patches = [mpatches.Patch(color=np.array(colors[i])/255.0, label=f"{i}:{n}")
               for i, n in enumerate(class_names)]
    fig.legend(handles=patches, loc="lower center", ncol=min(6, len(class_names)),
               fontsize=7, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Grid of pipeline results ──────────────────────────────────────────────────

def save_pipeline_grid(
    images:       List[np.ndarray],
    gt_masks:     List[np.ndarray],
    results:      List[PipelineResult],
    save_path:    str,
    class_colors: Optional[list] = None,
    max_samples:  int = 8,
    cols:         int = 4,
) -> None:
    """Save a compact grid: [OCT+box | GT | Full pred] per row."""
    colors  = class_colors or _DEFAULT_COLORS
    n       = min(len(images), max_samples)
    rows    = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows * 3, cols, figsize=(cols * 3, rows * 9))
    axes = np.array(axes).reshape(rows * 3, cols)

    for i in range(n):
        row_base = (i // cols) * 3
        col      = i  % cols

        img_bgr = cv2.cvtColor(images[i], cv2.COLOR_GRAY2BGR)
        if results[i].roi_coords:
            img_bgr = _draw_box(img_bgr, results[i].roi_coords)

        axes[row_base,     col].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axes[row_base,     col].set_title("OCT+ROI", fontsize=7)
        axes[row_base + 1, col].imshow(_mask_to_rgb(gt_masks[i], colors))
        axes[row_base + 1, col].set_title("GT", fontsize=7)
        axes[row_base + 2, col].imshow(_mask_to_rgb(results[i].full_pred, colors))
        axes[row_base + 2, col].set_title("Pipeline Pred", fontsize=7)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Pipeline grid saved: %s", save_path)


# ── Failure cases ─────────────────────────────────────────────────────────────

def visualize_failure_case(
    image:      np.ndarray,
    gt_mask:    np.ndarray,
    result:     PipelineResult,
    save_path:  str,
    reason:     str = "",
    class_colors: Optional[list] = None,
) -> None:
    """Save a debugging panel for a failure case."""
    colors   = class_colors or _DEFAULT_COLORS
    img_bgr  = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if result.roi_coords:
        img_bgr = _draw_box(img_bgr, result.roi_coords, color=(0, 0, 255))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    title = f"FAILURE: {reason}" if reason else "Failure Case"
    fig.suptitle(title, fontsize=10, color="red")

    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("OCT + Predicted ROI (red=bad)")
    axes[1].imshow(_mask_to_rgb(gt_mask, colors))
    axes[1].set_title("Ground Truth")
    axes[2].imshow(_mask_to_rgb(result.full_pred, colors))
    axes[2].set_title("Pipeline Output")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Failure case saved: %s", save_path)


def save_failure_cases(
    images:       List[np.ndarray],
    gt_masks:     List[np.ndarray],
    results:      List[PipelineResult],
    gt_boxes:     list,              # List[List[BoundingBox]] from Phase 2
    output_dir:   str,
    iou_threshold: float = 0.3,
    class_colors:  Optional[list] = None,
    max_cases:     int = 20,
) -> List[int]:
    """
    Identify and save failure case visualisations.

    Failure = no detection OR detected bbox IoU with GT bbox < iou_threshold.

    Returns
    -------
    List[int]  indices of failure cases.
    """
    from evaluation.evaluate_detector import box_iou

    failures = []
    saved    = 0

    for i, (img, gt_mask, result, gt_box_list) in enumerate(
        zip(images, gt_masks, results, gt_boxes)
    ):
        reason = ""
        if not result.detected:
            reason = "no_detection"
        elif gt_box_list and result.roi_coords:
            cx1, cy1, cx2, cy2 = result.roi_coords
            best_iou = max(
                box_iou((cx1, cy1, cx2, cy2), (b.x1, b.y1, b.x2, b.y2))
                for b in gt_box_list
            )
            if best_iou < iou_threshold:
                reason = f"poor_iou_{best_iou:.2f}"

        if reason:
            failures.append(i)
            if saved < max_cases:
                save_path = os.path.join(output_dir, f"failure_{saved:04d}_{reason}.png")
                visualize_failure_case(
                    img, gt_mask, result, save_path, reason=reason,
                    class_colors=class_colors,
                )
                saved += 1

    logger.info("Failure cases: %d / %d", len(failures), len(images))
    return failures
