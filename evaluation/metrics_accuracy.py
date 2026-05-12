"""
evaluation/metrics_accuracy.py
-------------------------------
Pixel-level accuracy metrics for multi-class segmentation.

All functions operate on *batched* PyTorch tensors (logits + integer masks)
and return scalar Python floats — easy to accumulate across batches.

Metrics implemented
-------------------
- Dice Score (macro-averaged over classes)
- IoU / Jaccard (macro-averaged over classes)
- Pixel Accuracy
- Sensitivity (Recall, macro-averaged)
- Specificity (macro-averaged)
- Per-class Dice  (returns list for detailed reporting)
- Per-class IoU   (returns list for detailed reporting)
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_EPS = 1e-6


# ── Helpers ────────────────────────────────────────────────────────────────────

def _preds_to_labels(logits: torch.Tensor) -> torch.Tensor:
    """Argmax along class dimension (B,C,H,W) → (B,H,W)."""
    return logits.argmax(dim=1)


def _to_binary(pred_labels: torch.Tensor, target: torch.Tensor, cls: int):
    """Return (pred_binary, target_binary) for a single class."""
    return (pred_labels == cls).float(), (target == cls).float()


# ── Per-class ──────────────────────────────────────────────────────────────────

def per_class_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> List[float]:
    """
    Dice score for every class.

    Parameters
    ----------
    logits      : (B, C, H, W) raw model output.
    targets     : (B, H, W) integer ground-truth class ids.
    num_classes : C.

    Returns
    -------
    list of floats, length = num_classes.
    """
    preds = _preds_to_labels(logits)
    scores = []
    for c in range(num_classes):
        p, t  = _to_binary(preds, targets, c)
        inter = (p * t).sum()
        scores.append(((2 * inter + _EPS) / (p.sum() + t.sum() + _EPS)).item())
    return scores


def per_class_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> List[float]:
    """IoU (Jaccard) score for every class."""
    preds = _preds_to_labels(logits)
    scores = []
    for c in range(num_classes):
        p, t  = _to_binary(preds, targets, c)
        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter
        scores.append(((inter + _EPS) / (union + _EPS)).item())
    return scores


# ── Macro-averaged scalars ──────────────────────────────────────────────────────

def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> float:
    """Macro-averaged Dice score (primary accuracy metric)."""
    return float(torch.tensor(per_class_dice(logits, targets, num_classes)).mean())


def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> float:
    """Macro-averaged Intersection-over-Union."""
    return float(torch.tensor(per_class_iou(logits, targets, num_classes)).mean())


def pixel_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Fraction of correctly classified pixels.

    PixelAcc = sum(correct pixels) / sum(total pixels)
    """
    preds   = _preds_to_labels(logits)
    correct = (preds == targets).sum().item()
    total   = targets.numel()
    return correct / (total + _EPS)


def sensitivity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> float:
    """
    Macro-averaged Sensitivity (Recall / True-Positive Rate).

    Sensitivity_c = TP_c / (TP_c + FN_c)
    """
    preds  = _preds_to_labels(logits)
    scores = []
    for c in range(num_classes):
        p, t = _to_binary(preds, targets, c)
        tp   = (p * t).sum()
        fn   = ((1 - p) * t).sum()
        scores.append((tp / (tp + fn + _EPS)).item())
    return float(torch.tensor(scores).mean())


def specificity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> float:
    """
    Macro-averaged Specificity (True-Negative Rate).

    Specificity_c = TN_c / (TN_c + FP_c)
    """
    preds  = _preds_to_labels(logits)
    scores = []
    for c in range(num_classes):
        p, t = _to_binary(preds, targets, c)
        tn   = ((1 - p) * (1 - t)).sum()
        fp   = (p * (1 - t)).sum()
        scores.append((tn / (tn + fp + _EPS)).item())
    return float(torch.tensor(scores).mean())


# ── Convenience aggregator ─────────────────────────────────────────────────────

def compute_all_accuracy_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> dict:
    """
    Compute all accuracy metrics in a single call.

    Returns
    -------
    dict with keys:
        dice, iou, pixel_accuracy, sensitivity, specificity,
        per_class_dice (list), per_class_iou (list)
    """
    return {
        "dice":           dice_score(logits, targets, num_classes),
        "iou":            iou_score(logits, targets, num_classes),
        "pixel_accuracy": pixel_accuracy(logits, targets),
        "sensitivity":    sensitivity(logits, targets, num_classes),
        "specificity":    specificity(logits, targets, num_classes),
        "per_class_dice": per_class_dice(logits, targets, num_classes),
        "per_class_iou":  per_class_iou(logits, targets, num_classes),
    }
