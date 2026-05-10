"""
visualization.py
----------------
Plotting utilities for segmentation results and training curves.

All functions save figures to cfg.plots_dir and optionally display
inline (useful in Google Colab notebooks).
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from utils.config import Config
from utils.transforms import denormalize

# Lazy import matplotlib to avoid errors in headless environments
def _plt():
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------- #
#  Qualitative segmentation overlay                                             #
# --------------------------------------------------------------------------- #

def plot_predictions(
    images: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    cfg: Config,
    n: int = 4,
    filename: str = "predictions.png",
    show: bool = False,
) -> None:
    """
    Save a grid of: OCT image | ground truth | prediction | overlay.

    Args:
        images:     Batch tensor (B, C, H, W) – normalised.
        gt_masks:   Ground truth (B, 1, H, W).
        pred_masks: Predicted binary mask (B, 1, H, W).
        cfg:        Config (for output path and norm stats).
        n:          Number of samples to display (≤ batch size).
        filename:   Output filename.
        show:       Display inline (set True in Colab).
    """
    plt = _plt()

    n = min(n, images.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["OCT Image", "Ground Truth", "Prediction", "Overlay"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight="bold")

    images_disp = denormalize(images[:n], cfg).cpu()

    for i in range(n):
        img = images_disp[i].squeeze().numpy()
        gt = gt_masks[i].squeeze().cpu().numpy()
        pred = pred_masks[i].squeeze().cpu().numpy()

        axes[i][0].imshow(img, cmap="gray")
        axes[i][1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[i][2].imshow(pred, cmap="gray", vmin=0, vmax=1)

        # Overlay: image + red contour for GT, green for prediction
        axes[i][3].imshow(img, cmap="gray")
        axes[i][3].contour(gt, levels=[0.5], colors=["red"], linewidths=1.5)
        axes[i][3].contour(pred, levels=[0.5], colors=["lime"], linewidths=1.5)

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    _save(fig, cfg.plots_dir, filename, show)


# --------------------------------------------------------------------------- #
#  Training curves                                                              #
# --------------------------------------------------------------------------- #

def plot_training_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_dices: Sequence[float],
    val_dices: Sequence[float],
    cfg: Config,
    filename: str = "training_curves.png",
    show: bool = False,
) -> None:
    """Plot loss and Dice score over training epochs."""
    plt = _plt()
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train Loss", color="steelblue")
    ax1.plot(epochs, val_losses, label="Val Loss", color="coral")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_dices, label="Train Dice", color="steelblue")
    ax2.plot(epochs, val_dices, label="Val Dice", color="coral")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.set_title("Dice Coefficient")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, cfg.plots_dir, filename, show)


# --------------------------------------------------------------------------- #
#  Metric bar chart (for comparing models / quantization levels)               #
# --------------------------------------------------------------------------- #

def plot_metric_comparison(
    labels: List[str],
    metrics: Dict[str, List[float]],
    cfg: Config,
    filename: str = "metric_comparison.png",
    show: bool = False,
) -> None:
    """
    Bar chart comparing multiple metrics across model variants.

    Args:
        labels:  List of model/variant names (x-axis).
        metrics: Dict mapping metric name → list of values aligned with labels.
        cfg:     Config.
    """
    plt = _plt()
    import numpy as np

    n_groups = len(labels)
    n_metrics = len(metrics)
    x = np.arange(n_groups)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(8, 2 * n_groups), 5))

    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric_name)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Metric Comparison")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, cfg.plots_dir, filename, show)


# --------------------------------------------------------------------------- #
#  Internal helper                                                              #
# --------------------------------------------------------------------------- #

def _save(fig, plots_dir: str, filename: str, show: bool) -> None:
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[viz] Saved → {path}")
    if show:
        _plt().show()
    _plt().close(fig)
