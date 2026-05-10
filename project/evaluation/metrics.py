"""
metrics.py
----------
Segmentation metrics for binary OCT wetAMD masks.

All functions operate on batched float32 tensors in {0, 1} (after
thresholding). They are numerically stable, handle the all-zero edge
case, and return scalar tensors unless noted otherwise.

Metrics implemented
-------------------
  - Dice coefficient (F1)
  - Intersection over Union (IoU / Jaccard)
  - Pixel accuracy
  - Sensitivity (Recall / True Positive Rate)
  - Specificity (True Negative Rate)
  - Precision

Usage:
    from evaluation.metrics import compute_all_metrics
    scores = compute_all_metrics(pred_mask, gt_mask)
"""

from typing import Dict, Optional

import torch


# --------------------------------------------------------------------------- #
#  Primitive metrics                                                            #
# --------------------------------------------------------------------------- #

def _confusion_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Compute TP, FP, FN, TN summed across the batch.

    Args:
        pred:   Binary prediction tensor (B, 1, H, W) – values in {0, 1}.
        target: Binary ground-truth tensor (B, 1, H, W) – values in {0, 1}.
        eps:    Small value for numerical stability.

    Returns:
        Dict with keys: tp, fp, fn, tn (all scalar float tensors).
    """
    pred = pred.float().view(-1)
    target = target.float().view(-1)

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Sørensen–Dice coefficient.

    Args:
        pred:   Binary prediction (B, 1, H, W).
        target: Binary ground truth (B, 1, H, W).
        eps:    Smoothing term.

    Returns:
        Scalar tensor in [0, 1].
    """
    c = _confusion_components(pred, target)
    return (2 * c["tp"] + eps) / (2 * c["tp"] + c["fp"] + c["fn"] + eps)


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Intersection over Union (Jaccard index).

    Args:
        pred:   Binary prediction (B, 1, H, W).
        target: Binary ground truth (B, 1, H, W).
        eps:    Smoothing term.

    Returns:
        Scalar tensor in [0, 1].
    """
    c = _confusion_components(pred, target)
    return (c["tp"] + eps) / (c["tp"] + c["fp"] + c["fn"] + eps)


def pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Overall pixel-level accuracy.

    Returns:
        Scalar tensor in [0, 1].
    """
    correct = (pred == target).float().sum()
    total = torch.tensor(target.numel(), dtype=torch.float32)
    return correct / total


def sensitivity(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Recall / True Positive Rate."""
    c = _confusion_components(pred, target)
    return (c["tp"] + eps) / (c["tp"] + c["fn"] + eps)


def specificity(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """True Negative Rate."""
    c = _confusion_components(pred, target)
    return (c["tn"] + eps) / (c["tn"] + c["fp"] + eps)


def precision(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Positive Predictive Value."""
    c = _confusion_components(pred, target)
    return (c["tp"] + eps) / (c["tp"] + c["fp"] + eps)


# --------------------------------------------------------------------------- #
#  Aggregate                                                                    #
# --------------------------------------------------------------------------- #

def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = False,
) -> Dict[str, float]:
    """
    Compute all segmentation metrics and return as a plain Python dict.

    Args:
        pred:        Prediction tensor (B, 1, H, W).
                     If from_logits=True, raw logits are expected.
                     Otherwise binary {0, 1} or probability [0, 1] values.
        target:      Binary ground-truth (B, 1, H, W).
        threshold:   Binarisation threshold when pred contains probabilities.
        from_logits: Apply sigmoid to pred before thresholding.

    Returns:
        Dict mapping metric name → float value.
    """
    if from_logits:
        pred = torch.sigmoid(pred)

    # Binarise if not already binary
    if pred.is_floating_point():
        pred = (pred >= threshold).float()

    target = target.float()

    return {
        "dice":      dice_coefficient(pred, target).item(),
        "iou":       iou_score(pred, target).item(),
        "pixel_acc": pixel_accuracy(pred, target).item(),
        "sensitivity": sensitivity(pred, target).item(),
        "specificity": specificity(pred, target).item(),
        "precision": precision(pred, target).item(),
    }


# --------------------------------------------------------------------------- #
#  Running averages                                                             #
# --------------------------------------------------------------------------- #

class MetricAccumulator:
    """
    Accumulate per-batch metrics and compute the epoch mean.

    Usage:
        acc = MetricAccumulator()
        for images, masks in loader:
            metrics = compute_all_metrics(pred, masks)
            acc.update(metrics)
        epoch_metrics = acc.mean()
        acc.reset()
    """

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + v
            self._counts[k] = self._counts.get(k, 0) + 1

    def mean(self) -> Dict[str, float]:
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
        }

    def reset(self) -> None:
        self._sums = {}
        self._counts = {}
