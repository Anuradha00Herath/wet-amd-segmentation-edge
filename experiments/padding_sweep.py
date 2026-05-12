"""
experiments/padding_sweep.py
-----------------------------
Systematic experiment: test multiple ROI padding sizes and measure the
accuracy vs. speed trade-off.

For each padding value the pipeline:
1. Runs on the test set.
2. Computes Dice, IoU, Pixel Accuracy, Sensitivity, Specificity.
3. Benchmarks FPS, ms/frame, stage breakdown.
4. Measures energy (codecarbon).
5. Saves one CSV row per padding value.
6. Generates comparison plots.

Design decisions
----------------
- The pipeline and models are loaded ONCE; only the ``padding_px``
  argument changes per run — this avoids GPU re-allocation overhead.
- Ground-truth masks are loaded and converted to class maps upfront
  so disk I/O is not included in timing.
- Results are accumulated into a pandas DataFrame for easy plotting.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def run_padding_sweep(
    cfg,
    padding_values: Optional[List[int]] = None,
    max_images: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run the padding sweep experiment.

    Parameters
    ----------
    cfg            : loaded pipeline config namespace.
    padding_values : list of padding sizes in pixels.
                     Defaults to ``cfg.padding_sweep.padding_values``.
    max_images     : cap number of test images (None = all).

    Returns
    -------
    pd.DataFrame  one row per padding value, all metrics as columns.
    """
    from pipelines.roi_seg_pipeline import ROIPipeline
    from benchmarking.pipeline_profiler import benchmark_pipeline, measure_pipeline_energy
    from evaluation.metrics_accuracy    import compute_all_accuracy_metrics
    from utils.config_loader            import get_class_info
    from utils.dataset                  import rgb_mask_to_class
    from utils.reproducibility          import seed_everything

    seed_everything(cfg.project.seed)

    pad_vals = padding_values or cfg.padding_sweep.padding_values
    out_dir  = cfg.padding_sweep.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Load pipeline once ────────────────────────────────────────────────────
    logger.info("Loading pipeline (shared across all padding runs)…")
    pipeline = ROIPipeline(cfg)

    # ── Load test images + GT masks ───────────────────────────────────────────
    class_info   = get_class_info_from_pipeline_cfg(cfg)
    images, gts  = _load_test_data(cfg, class_info, max_images)
    logger.info("Loaded %d test images for padding sweep.", len(images))

    # ── Sweep ─────────────────────────────────────────────────────────────────
    rows = []
    for pad in pad_vals:
        logger.info("─── Padding = %d px ───────────────────────────────", pad)

        # Run pipeline
        results = [pipeline.run(img, padding_px=pad) for img in images]

        # Accuracy metrics (need logits-like tensors or direct class maps)
        full_preds = np.stack([r.full_pred for r in results])  # (N,H,W)
        gt_maps    = np.stack(gts)                             # (N,H,W)
        acc        = _compute_accuracy(full_preds, gt_maps, cfg.classes.num_classes)

        # Speed benchmark (re-run with warmup)
        speed = benchmark_pipeline(pipeline, images, num_warmup=3, padding_px=pad)

        # Energy
        energy = {"energy_per_frame_J": float("nan"), "power_W": float("nan"),
                  "backend": "skipped"}
        if cfg.profiling.measure_energy:
            try:
                energy = measure_pipeline_energy(
                    pipeline, images, backend=cfg.profiling.energy_backend,
                    num_warmup=3, padding_px=pad,
                )
            except Exception as e:
                logger.warning("Energy measurement failed for pad=%d: %s", pad, e)

        row = {
            "padding_px":           pad,
            "dice":                 acc["dice"],
            "iou":                  acc["iou"],
            "pixel_accuracy":       acc["pixel_accuracy"],
            "sensitivity":          acc["sensitivity"],
            "specificity":          acc["specificity"],
            "fps":                  speed.get("fps", float("nan")),
            "mean_total_ms":        speed.get("mean_total_ms", float("nan")),
            "mean_detector_ms":     speed.get("mean_detector_ms", float("nan")),
            "mean_seg_ms":          speed.get("mean_seg_ms", float("nan")),
            "detection_rate":       speed.get("detection_rate", float("nan")),
            "energy_per_frame_J":   energy.get("energy_per_frame_J", float("nan")),
            "power_W":              energy.get("power_W", float("nan")),
        }
        rows.append(row)
        logger.info(
            "pad=%d → Dice=%.4f | FPS=%.2f | ms=%.2f",
            pad, row["dice"], row["fps"], row["mean_total_ms"],
        )

    df = pd.DataFrame(rows)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "padding_sweep_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Padding sweep results saved → %s", csv_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_sweep_results(df, out_dir)

    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_class_info_from_pipeline_cfg(cfg) -> dict:
    """Build class_info dict from pipeline config."""
    colors = vars(cfg.classes.colors_rgb)
    return {
        int(k): {"name": f"class_{k}", "rgb": tuple(colors[k])}
        for k in colors
    }


def _load_test_data(cfg, class_info: dict, max_images: Optional[int]):
    """Load images and GT class maps from Phase 1 test split."""
    from utils.dataset import rgb_mask_to_class

    img_dir  = cfg.data.image_dir
    mask_dir = cfg.data.mask_dir
    exts     = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    fnames   = sorted(f for f in os.listdir(img_dir)
                      if Path(f).suffix.lower() in exts)
    if max_images:
        fnames = fnames[:max_images]

    # Use last 15% as test set (matches Phase 1 split)
    import random
    rng = random.Random(cfg.project.seed)
    shuffled = fnames.copy(); rng.shuffle(shuffled)
    n_test = max(1, int(0.15 * len(shuffled)))
    test_files = shuffled[-n_test:]

    images, gts = [], []
    for fname in test_files:
        img = cv2.imread(os.path.join(img_dir,  fname), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_COLOR)
        if img is None or msk is None:
            continue
        img = cv2.resize(img, (cfg.data.image_size, cfg.data.image_size))
        msk = cv2.resize(msk, (cfg.data.image_size, cfg.data.image_size),
                         interpolation=cv2.INTER_NEAREST)
        gt  = rgb_mask_to_class(msk, class_info)
        images.append(img)
        gts.append(gt)

    return images, gts


def _compute_accuracy(
    preds:       np.ndarray,    # (N,H,W) uint8
    gts:         np.ndarray,    # (N,H,W) uint8
    num_classes: int,
) -> dict:
    """Compute accuracy metrics from numpy class maps."""
    from evaluation.metrics_accuracy import (
        dice_score, iou_score, pixel_accuracy, sensitivity, specificity
    )
    _EPS = 1e-6

    # Convert to one-hot-like logits for reuse of Phase 1 functions
    # Build fake logits: (N, C, H, W) where argmax = pred
    N, H, W = preds.shape
    logits   = torch.zeros(N, num_classes, H, W)
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


def _plot_sweep_results(df: pd.DataFrame, out_dir: str) -> None:
    """Generate accuracy vs. speed trade-off plots for the padding sweep."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("ROI Padding Sweep — Accuracy vs. Speed Trade-off", fontsize=13)

    metrics = [
        ("dice",           "Dice Score",     axes[0, 0]),
        ("iou",            "IoU",            axes[0, 1]),
        ("pixel_accuracy", "Pixel Accuracy", axes[0, 2]),
        ("fps",            "FPS",            axes[1, 0]),
        ("mean_total_ms",  "ms / frame",     axes[1, 1]),
        ("energy_per_frame_J", "J / frame",  axes[1, 2]),
    ]

    for col, label, ax in metrics:
        if col in df.columns:
            ax.plot(df["padding_px"], df[col], marker="o", linewidth=2, color="#4C72B0")
            ax.set_xlabel("Padding (px)")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(alpha=0.3)
            for x, y in zip(df["padding_px"], df[col]):
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                            xytext=(0, 6), fontsize=7, ha="center")

    plt.tight_layout()
    path = os.path.join(out_dir, "padding_sweep_plot.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Padding sweep plot saved → %s", path)
