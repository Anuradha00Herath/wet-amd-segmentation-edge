"""
utils/visualizer.py
--------------------
Visualisation helpers for segmentation results.

All functions accept numpy arrays and write PNG files; they do not call
``plt.show()`` so they are safe to run in headless Colab cells.

Functions
---------
- save_prediction_overlay   : OCT + GT + Pred side-by-side with legend.
- save_batch_grid           : grid of N samples from a batch.
- plot_metric_curves        : training history line plots.
- plot_metric_bar           : per-class bar chart for a dict of values.
- plot_confusion_matrix     : confusion matrix heatmap.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Default class colours matching the baseline notebook
_DEFAULT_COLORS: List[List[int]] = [
    [0,   0,   0  ],   # 0 Background
    [255, 0,   255],   # 1 Retinal Layer
    [255, 255, 0  ],   # 2 PED
    [255, 0,   0  ],   # 3 SRF
    [0,   0,   255],   # 4 IRF
    [0,   255, 0  ],   # 5 RPE
]


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _mask_to_rgb(
    mask_np: np.ndarray,
    class_colors: List[List[int]],
) -> np.ndarray:
    """Convert integer class map → H×W×3 uint8 RGB."""
    h, w = mask_np.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for c, colour in enumerate(class_colors):
        rgb[mask_np == c] = colour
    return rgb


def _legend_patches(
    class_names: List[str],
    class_colors: List[List[int]],
) -> List[mpatches.Patch]:
    return [
        mpatches.Patch(color=np.array(class_colors[i]) / 255.0, label=f"{i}: {n}")
        for i, n in enumerate(class_names)
    ]


# ── Per-sample overlay ─────────────────────────────────────────────────────────

def save_prediction_overlay(
    image_np: np.ndarray,
    gt_mask_np: np.ndarray,
    pred_mask_np: np.ndarray,
    save_path: str,
    class_names: List[str],
    class_colors: Optional[List[List[int]]] = None,
    title: str = "",
) -> None:
    """
    Save a 3-panel figure: [OCT image | Ground Truth | Prediction].

    Parameters
    ----------
    image_np     : H×W float/uint8 greyscale image.
    gt_mask_np   : H×W integer class map (ground truth).
    pred_mask_np : H×W integer class map (model prediction).
    save_path    : output file path (.png).
    class_names  : list of class name strings.
    class_colors : RGB colours per class (defaults to _DEFAULT_COLORS).
    title        : optional suptitle string.
    """
    colors = class_colors or _DEFAULT_COLORS
    gt_rgb   = _mask_to_rgb(gt_mask_np,   colors)
    pred_rgb = _mask_to_rgb(pred_mask_np, colors)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title, fontsize=10)

    axes[0].imshow(image_np,  cmap="gray"); axes[0].set_title("OCT Image")
    axes[1].imshow(gt_rgb);                 axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_rgb);               axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")

    patches = _legend_patches(class_names, colors)
    fig.legend(handles=patches, loc="lower center", ncol=len(class_names),
               fontsize=7, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved overlay: %s", save_path)


# ── Batch grid ─────────────────────────────────────────────────────────────────

def save_batch_grid(
    images: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_logits: torch.Tensor,
    save_path: str,
    class_names: List[str],
    class_colors: Optional[List[List[int]]] = None,
    max_samples: int = 8,
) -> None:
    """
    Save a grid of [image | GT | Pred] rows for up to *max_samples* images.

    Parameters
    ----------
    images       : (B,1,H,W) float tensor (normalised).
    gt_masks     : (B,H,W) integer tensor.
    pred_logits  : (B,C,H,W) float tensor (raw logits).
    save_path    : output file path.
    class_names  : list of class name strings.
    class_colors : optional custom colour list.
    max_samples  : cap rows shown (to keep figure manageable).
    """
    colors  = class_colors or _DEFAULT_COLORS
    preds   = pred_logits.argmax(dim=1)
    n       = min(images.size(0), max_samples)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img_np  = images[i].squeeze().cpu().numpy()
        # Denormalise from [-1, 1] → [0, 1]
        img_np  = (img_np * 0.5 + 0.5).clip(0, 1)
        gt_np   = gt_masks[i].cpu().numpy()
        pred_np = preds[i].cpu().numpy()

        gt_classes   = [class_names[c] for c in np.unique(gt_np)   if c < len(class_names)]
        pred_classes = [class_names[c] for c in np.unique(pred_np) if c < len(class_names)]

        axes[i, 0].imshow(img_np,              cmap="gray")
        axes[i, 0].set_title("OCT",            fontsize=8)
        axes[i, 1].imshow(_mask_to_rgb(gt_np,   colors))
        axes[i, 1].set_title(f"GT: {', '.join(gt_classes)}",   fontsize=7)
        axes[i, 2].imshow(_mask_to_rgb(pred_np, colors))
        axes[i, 2].set_title(f"Pred: {', '.join(pred_classes)}", fontsize=7)
        for ax in axes[i]:
            ax.axis("off")

    patches = _legend_patches(class_names, colors)
    fig.legend(handles=patches, loc="lower center", ncol=min(6, len(class_names)),
               fontsize=8, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved batch grid: %s", save_path)


# ── Training curves ────────────────────────────────────────────────────────────

def plot_metric_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Training History",
    freeze_epoch: Optional[int] = None,
) -> None:
    """
    Plot loss and metric curves from a training history dict.

    Parameters
    ----------
    history      : ``{"train_loss": [...], "val_loss": [...], ...}``
    save_path    : output file path.
    title        : figure title.
    freeze_epoch : if set, draws a vertical dashed line (encoder unfreeze).
    """
    keys   = list(history.keys())
    n_axes = len(keys)
    fig, axes = plt.subplots(1, n_axes, figsize=(5 * n_axes, 4))
    if n_axes == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12, fontweight="bold")

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, key, colour in zip(axes, keys, colours):
        vals   = history[key]
        epochs = list(range(1, len(vals) + 1))
        ax.plot(epochs, vals, color=colour, linewidth=2, label=key)
        if freeze_epoch:
            ax.axvline(x=freeze_epoch, color="grey", linestyle="--",
                       alpha=0.7, label="Unfreeze encoder")
        ax.set_title(key.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved metric curves: %s", save_path)


# ── Per-class bar chart ────────────────────────────────────────────────────────

def plot_metric_bar(
    values: List[float],
    class_names: List[str],
    metric_name: str,
    save_path: str,
    class_colors: Optional[List[List[int]]] = None,
) -> None:
    """
    Horizontal bar chart for a per-class metric (e.g. per-class Dice).

    Parameters
    ----------
    values       : metric value per class.
    class_names  : class name strings.
    metric_name  : y-axis / title label.
    save_path    : output file path.
    class_colors : optional bar colours.
    """
    colors = class_colors or _DEFAULT_COLORS
    norm_colors = [np.array(c) / 255.0 for c in colors[:len(values)]]

    fig, ax = plt.subplots(figsize=(8, max(3, len(values) * 0.6)))
    bars = ax.barh(class_names, values, color=norm_colors, edgecolor="black",
                   linewidth=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(metric_name)
    ax.set_title(f"Per-Class {metric_name}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved bar chart: %s", save_path)


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    normalise: bool = True,
) -> None:
    """
    Save a confusion matrix heatmap.

    Parameters
    ----------
    cm          : (N,N) confusion matrix (counts).
    class_names : list of N class name strings.
    save_path   : output file path.
    normalise   : if True, normalise rows to [0,1] (recall per class).
    """
    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        cm_plot  = np.divide(cm.astype(float), row_sums,
                             where=row_sums != 0)
        fmt = ".2f"
        label = "Recall"
    else:
        cm_plot = cm.astype(int)
        fmt = "d"
        label = "Count"

    n   = len(class_names)
    fig, ax = plt.subplots(figsize=(n + 2, n + 1))
    im  = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues",
                    vmin=0, vmax=1 if normalise else None)
    plt.colorbar(im, ax=ax, label=label)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    thresh = cm_plot.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_plot[i, j]
            txt = f"{val:{fmt}}" if not (normalise and np.isnan(val)) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if val > thresh else "black")

    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix: %s", save_path)
