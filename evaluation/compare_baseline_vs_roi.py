"""
evaluation/compare_baseline_vs_roi.py
--------------------------------------
Side-by-side comparison: Phase 1 baseline segmentation vs Phase 3
ROI-guided segmentation.

Metrics compared
----------------
Accuracy  : Dice, IoU, Pixel Accuracy, Sensitivity, Specificity
Compute   : FLOPs (on actual input size), Parameters, Memory
Throughput: FPS, ms/frame (end-to-end)
Energy    : power (W), J/frame

Output
------
- Comparison CSV (one row per method, same schema as Phase 1 MetricsBundle)
- Publication-ready LaTeX table string
- Side-by-side bar chart PNG
- Printed summary table
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


# ── Comparison bundle ─────────────────────────────────────────────────────────

@dataclass
class ComparisonRow:
    """One row in the comparison table (one method)."""
    method:             str   = ""
    timestamp:          str   = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    # Accuracy
    dice:               float = float("nan")
    iou:                float = float("nan")
    pixel_accuracy:     float = float("nan")
    sensitivity:        float = float("nan")
    specificity:        float = float("nan")
    # Compute
    gflops:             float = float("nan")
    params_M:           float = float("nan")
    memory_mb:          float = float("nan")
    # Throughput
    fps:                float = float("nan")
    ms_per_frame:       float = float("nan")
    # Energy
    power_W:            float = float("nan")
    energy_per_frame_J: float = float("nan")
    # Extra
    notes:              str   = ""


def save_comparison_csv(rows: List[ComparisonRow], csv_path: str) -> None:
    """Save all comparison rows to CSV."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    if not rows:
        return
    headers = list(asdict(rows[0]).keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    logger.info("Comparison CSV saved → %s", csv_path)


# ── Evaluator ─────────────────────────────────────────────────────────────────

def run_comparison(
    cfg_seg,          # Phase 1 seg config
    cfg_pipe,         # Phase 3 pipeline config
    test_images: List[np.ndarray],
    test_gts:    List[np.ndarray],
    seg_model    = None,    # SegmentationWrapper (pre-loaded or None)
    pipeline     = None,    # ROIPipeline (pre-loaded or None)
) -> pd.DataFrame:
    """
    Evaluate both methods on the same test set and return a comparison DataFrame.

    Parameters
    ----------
    cfg_seg      : Phase 1 config (for baseline segmentation model loading).
    cfg_pipe     : Phase 3 config (for pipeline).
    test_images  : list of H×W uint8 greyscale arrays.
    test_gts     : list of H×W uint8 integer class maps.
    seg_model    : optional pre-loaded SegmentationWrapper.
    pipeline     : optional pre-loaded ROIPipeline.

    Returns
    -------
    pd.DataFrame  two rows: baseline and roi_guided.
    """
    from evaluation.metrics_accuracy    import compute_all_accuracy_metrics
    from evaluation.metrics_compute     import compute_all_compute_metrics
    from benchmarking.pipeline_profiler import benchmark_pipeline, measure_pipeline_energy
    from segmentation.seg_wrapper       import SegmentationWrapper
    from pipelines.roi_seg_pipeline     import ROIPipeline
    from models.model_loader            import get_model
    from utils.config_loader            import get_class_info

    rows = []

    # ── A: Baseline (full-image) ──────────────────────────────────────────────
    logger.info("=== Evaluating Baseline (full-image segmentation) ===")

    if seg_model is None:
        seg_model = SegmentationWrapper(cfg_pipe)

    baseline_row = _eval_seg_only(
        seg_model, test_images, test_gts,
        cfg_pipe.classes.num_classes,
        cfg_pipe.roi.seg_input_size,
        method_name="baseline_full_image",
    )

    # Compute cost on full image
    try:
        comp = compute_all_compute_metrics(
            seg_model.model,
            input_shape=tuple(cfg_seg.profiling.flops_input_shape),
            device=cfg_pipe.segmentation.device,
        )
        baseline_row.gflops   = comp["gflops"]
        baseline_row.params_M = comp["total_params_M"]
        baseline_row.memory_mb= comp["memory_mb"]
    except Exception as e:
        logger.warning("Baseline compute metrics failed: %s", e)

    rows.append(baseline_row)

    # ── B: ROI-guided ─────────────────────────────────────────────────────────
    logger.info("=== Evaluating ROI-Guided Pipeline ===")

    if pipeline is None:
        pipeline = ROIPipeline(cfg_pipe)

    roi_row = _eval_roi_pipeline(
        pipeline, test_images, test_gts,
        cfg_pipe.classes.num_classes,
        method_name="roi_guided",
    )

    # Compute cost on ROI input (smaller than full image)
    try:
        roi_shape = tuple(cfg_pipe.profiling.flops_input_shape)
        comp = compute_all_compute_metrics(
            seg_model.model, input_shape=roi_shape,
            device=cfg_pipe.segmentation.device,
        )
        roi_row.gflops    = comp["gflops"]
        roi_row.params_M  = comp["total_params_M"]
        roi_row.memory_mb = comp["memory_mb"]
    except Exception as e:
        logger.warning("ROI compute metrics failed: %s", e)

    rows.append(roi_row)

    # ── Save + visualise ──────────────────────────────────────────────────────
    csv_path = cfg_pipe.evaluation.comparison_csv
    save_comparison_csv(rows, csv_path)

    df = pd.DataFrame([asdict(r) for r in rows])
    _print_comparison_table(df)
    _plot_comparison(df, cfg_pipe.evaluation.visualizations_dir)
    _save_latex_table(df, cfg_pipe.evaluation.visualizations_dir)

    return df


# ── Method evaluators ─────────────────────────────────────────────────────────

def _eval_seg_only(
    seg_model,
    images: List[np.ndarray],
    gts:    List[np.ndarray],
    num_classes: int,
    target_size: int,
    method_name: str,
    num_warmup: int = 3,
) -> ComparisonRow:
    """Evaluate segmentation-only (baseline) method."""
    import time, statistics
    from evaluation.metrics_accuracy import (
        dice_score, iou_score, pixel_accuracy, sensitivity, specificity
    )

    # Warm-up
    for img in images[:num_warmup]:
        seg_model.predict(img, target_size=target_size)

    preds, times = [], []
    for img in images[num_warmup:]:
        t0 = time.perf_counter()
        p  = seg_model.predict(img, target_size=target_size)
        t1 = time.perf_counter()
        # Resize back to original if needed
        if p.shape != img.shape[:2]:
            p = __import__("cv2").resize(p, (img.shape[1], img.shape[0]),
                                         interpolation=__import__("cv2").INTER_NEAREST)
        preds.append(p)
        times.append((t1 - t0) * 1000.0)

    acc = _class_map_accuracy(
        np.stack(preds), np.stack(gts[num_warmup:]), num_classes
    )
    mean_ms = statistics.mean(times)

    return ComparisonRow(
        method        = method_name,
        dice          = acc["dice"],
        iou           = acc["iou"],
        pixel_accuracy= acc["pixel_accuracy"],
        sensitivity   = acc["sensitivity"],
        specificity   = acc["specificity"],
        fps           = 1000.0 / mean_ms,
        ms_per_frame  = mean_ms,
    )


def _eval_roi_pipeline(
    pipeline,
    images: List[np.ndarray],
    gts:    List[np.ndarray],
    num_classes: int,
    method_name: str,
    num_warmup: int = 3,
) -> ComparisonRow:
    """Evaluate ROI-guided pipeline."""
    import statistics
    from benchmarking.pipeline_profiler import benchmark_pipeline

    results = [pipeline.run(img) for img in images[num_warmup:]]
    preds   = np.stack([r.full_pred for r in results])
    gt_arr  = np.stack(gts[num_warmup:])

    acc     = _class_map_accuracy(preds, gt_arr, num_classes)
    speed   = benchmark_pipeline(pipeline, images, num_warmup=num_warmup)

    return ComparisonRow(
        method        = method_name,
        dice          = acc["dice"],
        iou           = acc["iou"],
        pixel_accuracy= acc["pixel_accuracy"],
        sensitivity   = acc["sensitivity"],
        specificity   = acc["specificity"],
        fps           = speed.get("fps", float("nan")),
        ms_per_frame  = speed.get("mean_total_ms", float("nan")),
        notes         = f"det_rate={speed.get('detection_rate',0):.2f}",
    )


def _class_map_accuracy(preds: np.ndarray, gts: np.ndarray, num_classes: int) -> dict:
    """Convert numpy class maps to fake logits and compute Phase 1 metrics."""
    from evaluation.metrics_accuracy import (
        dice_score, iou_score, pixel_accuracy, sensitivity, specificity
    )
    N, H, W = preds.shape
    logits  = torch.zeros(N, num_classes, H, W)
    for i in range(N):
        for c in range(num_classes):
            logits[i, c][preds[i] == c] = 1.0
    targets = torch.from_numpy(gts.astype(np.int64))
    return {
        "dice":           dice_score(logits, targets, num_classes),
        "iou":            iou_score(logits, targets, num_classes),
        "pixel_accuracy": pixel_accuracy(logits, targets),
        "sensitivity":    sensitivity(logits, targets, num_classes),
        "specificity":    specificity(logits, targets, num_classes),
    }


# ── Output utilities ──────────────────────────────────────────────────────────

def _print_comparison_table(df: pd.DataFrame) -> None:
    sep = "─" * 70
    print(f"\n{sep}")
    print("  BASELINE vs ROI-GUIDED — COMPARISON SUMMARY")
    print(sep)
    cols = ["method", "dice", "iou", "fps", "ms_per_frame",
            "gflops", "memory_mb", "energy_per_frame_J"]
    print(df[cols].to_string(index=False, float_format="%.4f"))
    print(sep + "\n")


def _plot_comparison(df: pd.DataFrame, viz_dir: str) -> None:
    """Side-by-side grouped bar chart."""
    import matplotlib.pyplot as plt

    os.makedirs(viz_dir, exist_ok=True)
    metrics = ["dice", "iou", "fps", "ms_per_frame", "gflops", "memory_mb"]
    labels  = ["Dice", "IoU", "FPS", "ms/frame", "GFLOPs", "Memory (MB)"]

    n = len(metrics)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Baseline vs ROI-Guided Segmentation", fontsize=13, fontweight="bold")
    colours = ["#4C72B0", "#DD8452"]

    for ax, col, label in zip(axes.flat, metrics, labels):
        vals = df[col].values if col in df.columns else [0, 0]
        methods = df["method"].values
        bars = ax.bar(methods, vals, color=colours[:len(methods)], edgecolor="black",
                      linewidth=0.5, width=0.4)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=10)

    plt.tight_layout()
    path = os.path.join(viz_dir, "baseline_vs_roi_comparison.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison plot saved → %s", path)


def _save_latex_table(df: pd.DataFrame, viz_dir: str) -> None:
    """Export a publication-ready LaTeX table."""
    os.makedirs(viz_dir, exist_ok=True)
    cols    = ["method", "dice", "iou", "pixel_accuracy",
               "fps", "gflops", "memory_mb"]
    present = [c for c in cols if c in df.columns]
    latex   = df[present].to_latex(index=False, float_format="%.4f",
                                   caption="Baseline vs ROI-Guided Segmentation",
                                   label="tab:comparison")
    path = os.path.join(viz_dir, "comparison_table.tex")
    with open(path, "w") as f:
        f.write(latex)
    logger.info("LaTeX table saved → %s", path)
