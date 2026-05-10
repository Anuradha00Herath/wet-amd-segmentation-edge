"""
visualization.py
----------------
Plotting utilities for 6-class OCT wetAMD segmentation results and training curves.

All functions save figures to cfg.plots_dir and optionally display
inline (useful in Google Colab notebooks).
"""

import os
from typing import Dict, List, Sequence

import numpy as np
import torch

from utils.config import Config
from models.baseline_model import CLASS_NAMES, NUM_CLASSES

# Class colour map — matches the original notebook exactly
# Index → RGB
CLASS_COLORS = np.array([
    [0,   0,   0  ],   # 0 Background   – black
    [255, 0,   255],   # 1 Retinal Layer– magenta
    [255, 255, 0  ],   # 2 PED          – yellow
    [255, 0,   0  ],   # 3 SRF          – red
    [0,   0,   255],   # 4 IRF          – blue
    [0,   255, 0  ],   # 5 RPE          – green
], dtype=np.uint8)


def _plt():
    import matplotlib.pyplot as plt
    return plt


def mask_to_rgb(mask_np: np.ndarray) -> np.ndarray:
    """
    Convert a class index mask (H, W) int to an RGB image (H, W, 3) uint8.

    Args:
        mask_np: Class index array (H, W).

    Returns:
        RGB array (H, W, 3) uint8.
    """
    h, w   = mask_np.shape
    rgb    = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        rgb[mask_np == c] = CLASS_COLORS[c]
    return rgb


def denormalize(tensor: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Reverse normalisation for visualisation (grayscale OCT images)."""
    mean = torch.tensor(cfg.norm_mean, dtype=tensor.dtype, device=tensor.device)
    std  = torch.tensor(cfg.norm_std,  dtype=tensor.dtype, device=tensor.device)

    if tensor.dim() == 4:
        mean = mean[None, :, None, None]
        std  = std[None,  :, None, None]
    else:
        mean = mean[:, None, None]
        std  = std[:,  None, None]

    return (tensor * std + mean).clamp(0, 1)


# --------------------------------------------------------------------------- #
#  Qualitative segmentation grid                                               #
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
    Save a grid of: OCT image | Ground Truth (RGB) | Prediction (RGB).

    Args:
        images:     Normalised OCT images  (B, 1, H, W)  float32.
        gt_masks:   Ground-truth class indices (B, H, W) int64.
        pred_masks: Predicted class indices   (B, H, W) int64.
        cfg:        Config.
        n:          Number of samples to display.
        filename:   Output filename.
        show:       Display inline in Colab.
    """
    plt = _plt()

    n   = min(n, images.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["OCT Image", "Ground Truth", "Prediction"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight="bold")

    images_disp = denormalize(images[:n], cfg).cpu()

    for i in range(n):
        img      = images_disp[i].squeeze().numpy()                  # (H, W)
        gt_rgb   = mask_to_rgb(gt_masks[i].cpu().numpy())            # (H, W, 3)
        pred_rgb = mask_to_rgb(pred_masks[i].cpu().numpy())          # (H, W, 3)

        axes[i][0].imshow(img, cmap="gray")
        axes[i][1].imshow(gt_rgb)
        axes[i][2].imshow(pred_rgb)

        for ax in axes[i]:
            ax.axis("off")

    # Colour legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CLASS_COLORS[c] / 255.0, label=f"{c}: {CLASS_NAMES[c]}")
        for c in range(NUM_CLASSES)
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout()
    _save(fig, cfg.plots_dir, filename, show)


# --------------------------------------------------------------------------- #
#  Training curves                                                             #
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
    """Plot loss and mean Dice score over training epochs."""
    plt = _plt()
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train Loss", color="steelblue")
    ax1.plot(epochs, val_losses,   label="Val Loss",   color="coral")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_dices, label="Train Dice", color="steelblue")
    ax2.plot(epochs, val_dices,   label="Val Dice",   color="coral")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Mean Dice")
    ax2.set_title("Mean Dice (macro)"); ax2.legend()
    ax2.set_ylim(0, 1); ax2.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, cfg.plots_dir, filename, show)


# --------------------------------------------------------------------------- #
#  Per-class Dice bar chart                                                    #
# --------------------------------------------------------------------------- #

def plot_per_class_dice(
    class_dice_scores: List[float],
    cfg: Config,
    title: str = "Per-Class Dice",
    filename: str = "per_class_dice.png",
    show: bool = False,
) -> None:
    """
    Horizontal bar chart of per-class Dice scores.

    Args:
        class_dice_scores: List of 6 Dice scores, one per class.
        cfg:               Config.
        title:             Chart title.
        filename:          Output filename.
        show:              Display inline.
    """
    plt = _plt()

    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = [CLASS_COLORS[c] / 255.0 for c in range(NUM_CLASSES)]
    bars    = ax.barh(CLASS_NAMES, class_dice_scores, color=colors)

    for bar, score in zip(bars, class_dice_scores):
        ax.text(
            score + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}", va="center", fontsize=9
        )

    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Dice Score")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    _save(fig, cfg.plots_dir, filename, show)


# --------------------------------------------------------------------------- #
#  Model comparison bar chart                                                  #
# --------------------------------------------------------------------------- #

def plot_metric_comparison(
    labels: List[str],
    metrics: Dict[str, List[float]],
    cfg: Config,
    filename: str = "metric_comparison.png",
    show: bool = False,
) -> None:
    """
    Grouped bar chart comparing metrics across model variants.

    Args:
        labels:  Model/variant names (x-axis).
        metrics: Dict mapping metric name → list of values per label.
        cfg:     Config.
    """
    plt = _plt()

    n_groups  = len(labels)
    n_metrics = len(metrics)
    x         = np.arange(n_groups)
    width     = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(8, 2 * n_groups), 5))

    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        bars   = ax.bar(x + offset, values, width, label=metric_name)
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
#  Internal helper                                                             #
# --------------------------------------------------------------------------- #

def _save(fig, plots_dir: str, filename: str, show: bool) -> None:
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[viz] Saved → {path}")
    if show:
        _plt().show()
    _plt().close(fig)