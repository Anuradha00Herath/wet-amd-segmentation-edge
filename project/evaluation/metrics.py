"""
metrics.py
----------
Segmentation metrics for 6-class OCT wetAMD segmentation.

Inputs expected:
    pred_mask : (B, H, W) int64  – predicted class indices (from argmax)
    target    : (B, H, W) int64  – ground-truth class indices

Metrics implemented
-------------------
  - Mean Dice coefficient (macro-averaged across 6 classes)
  - Per-class Dice
  - Mean IoU (macro-averaged)
  - Per-class IoU
  - Pixel accuracy

Usage:
    from evaluation.metrics import compute_all_metrics
    scores = compute_all_metrics(pred_mask, target)
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from models.baseline_model import NUM_CLASSES, CLASS_NAMES


# --------------------------------------------------------------------------- #
#  Per-class Dice and IoU                                                      #
# --------------------------------------------------------------------------- #

def per_class_dice(
    pred_mask: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    eps: float = 1e-6,
) -> List[float]:
    """
    Compute Dice coefficient for each class separately.

    Args:
        pred_mask:   Predicted class indices (B, H, W) int64.
        target:      Ground-truth class indices (B, H, W) int64.
        num_classes: Number of classes.
        eps:         Smoothing term.

    Returns:
        List of Dice scores, one per class.
    """
    scores = []
    for c in range(num_classes):
        p     = (pred_mask == c).float()
        t     = (target    == c).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        scores.append(((2 * inter + eps) / (denom + eps)).item())
    return scores


def per_class_iou(
    pred_mask: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    eps: float = 1e-6,
) -> List[float]:
    """
    Compute IoU (Jaccard) for each class separately.

    Args:
        pred_mask:   Predicted class indices (B, H, W) int64.
        target:      Ground-truth class indices (B, H, W) int64.
        num_classes: Number of classes.
        eps:         Smoothing term.

    Returns:
        List of IoU scores, one per class.
    """
    scores = []
    for c in range(num_classes):
        p     = (pred_mask == c).float()
        t     = (target    == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter
        scores.append(((inter + eps) / (union + eps)).item())
    return scores


# --------------------------------------------------------------------------- #
#  Scalar summary metrics                                                      #
# --------------------------------------------------------------------------- #

def mean_dice(
    pred_mask: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    eps: float = 1e-6,
) -> float:
    """Macro-averaged Dice across all classes."""
    return sum(per_class_dice(pred_mask, target, num_classes, eps)) / num_classes


def mean_iou(
    pred_mask: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    eps: float = 1e-6,
) -> float:
    """Macro-averaged IoU across all classes."""
    return sum(per_class_iou(pred_mask, target, num_classes, eps)) / num_classes


def pixel_accuracy(
    pred_mask: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Overall pixel-level accuracy."""
    correct = (pred_mask == target).float().sum()
    total   = target.numel()
    return (correct / total).item()


# --------------------------------------------------------------------------- #
#  Aggregate                                                                   #
# --------------------------------------------------------------------------- #

def compute_all_metrics(
    pred_mask: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = False,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, float]:
    """
    Compute all segmentation metrics and return as a plain Python dict.

    Args:
        pred_mask:   Predicted class indices (B, H, W) int64.
                     If from_logits=True, pass raw logits (B, 6, H, W) instead.
        target:      Ground-truth class indices (B, H, W) int64.
        from_logits: If True, apply argmax to pred_mask first.
        num_classes: Number of classes.

    Returns:
        Dict mapping metric name → float value.
    """
    if from_logits:
        pred_mask = torch.argmax(pred_mask, dim=1)   # (B, 6, H, W) → (B, H, W)

    pred_mask = pred_mask.long()
    target    = target.long()

    class_dice = per_class_dice(pred_mask, target, num_classes)
    class_iou  = per_class_iou(pred_mask, target, num_classes)

    metrics: Dict[str, float] = {
        "mean_dice":  sum(class_dice) / num_classes,
        "mean_iou":   sum(class_iou)  / num_classes,
        "pixel_acc":  pixel_accuracy(pred_mask, target),
    }

    # Per-class Dice and IoU
    for i, name in enumerate(CLASS_NAMES):
        key = name.lower().replace(" ", "_")
        metrics[f"dice_{key}"] = class_dice[i]
        metrics[f"iou_{key}"]  = class_iou[i]

    return metrics


# --------------------------------------------------------------------------- #
#  Running averages                                                            #
# --------------------------------------------------------------------------- #

class MetricAccumulator:
    """
    Accumulate per-batch metrics and compute the epoch mean.

    Usage:
        acc = MetricAccumulator()
        for images, masks in loader:
            pred_mask = model(images).argmax(dim=1)
            metrics   = compute_all_metrics(pred_mask, masks)
            acc.update(metrics)
        epoch_metrics = acc.mean()
        acc.reset()
    """

    def __init__(self) -> None:
        self._sums:   Dict[str, float] = {}
        self._counts: Dict[str, int]   = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self._sums[k]   = self._sums.get(k, 0.0) + v
            self._counts[k] = self._counts.get(k, 0)  + 1

    def mean(self) -> Dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}

    def reset(self) -> None:
        self._sums   = {}
        self._counts = {}