"""
pipelines/run_phase3.py
------------------------
Main Phase 3 experiment runner.

Execution order
---------------
1.  Load config + set seed.
2.  Load ROIPipeline (detector + seg model, shared for all steps).
3.  Load test images + GT masks.
4.  Run ROI-guided pipeline → collect predictions + timings.
5.  Compute accuracy metrics.
6.  Benchmark speed (FPS, stage breakdown).
7.  Measure energy.
8.  Export metrics CSV.
9.  Save qualitative visualisations.
10. Run padding sweep experiment (if enabled in config).
11. Run baseline vs ROI comparison.
12. Save failure cases.

Usage (Colab)
-------------
    %cd /content/your_project
    !python pipelines/run_phase3.py --config configs/roi_pipeline_config.yaml

Or import ``run_phase3(cfg)`` from a notebook.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


def run_phase3(cfg, experiment_name: str = "roi_guided") -> dict:
    """
    Full Phase 3 evaluation pipeline.

    Parameters
    ----------
    cfg             : loaded pipeline config namespace.
    experiment_name : label written to CSV.

    Returns
    -------
    dict  with keys: results, metrics, benchmark, energy, df_comparison.
    """
    from utils.reproducibility       import seed_everything, save_experiment_metadata, namespace_to_dict
    from utils.dataset               import rgb_mask_to_class
    from pipelines.roi_seg_pipeline  import ROIPipeline
    from pipelines.pipeline_visualizer import (
        visualize_pipeline_result, save_pipeline_grid, save_failure_cases
    )
    from benchmarking.pipeline_profiler import benchmark_pipeline, measure_pipeline_energy, save_timing_breakdown_csv
    from evaluation.metrics_accuracy    import compute_all_accuracy_metrics
    from experiments.padding_sweep      import (
        run_padding_sweep, get_class_info_from_pipeline_cfg, _load_test_data
    )

    seed_everything(cfg.project.seed)
    os.makedirs(cfg.evaluation.visualizations_dir, exist_ok=True)
    os.makedirs(cfg.evaluation.failure_cases_dir,  exist_ok=True)
    os.makedirs(os.path.dirname(cfg.evaluation.metrics_csv) or ".", exist_ok=True)

    class_info   = get_class_info_from_pipeline_cfg(cfg)
    class_names  = [vars(cfg.classes.names)[str(i)]
                    for i in range(cfg.classes.num_classes)]

    # ── 1. Load pipeline ──────────────────────────────────────────────────────
    logger.info("=== Step 1/8: Loading pipeline ===")
    pipeline = ROIPipeline(cfg)

    # ── 2. Load test data ─────────────────────────────────────────────────────
    logger.info("=== Step 2/8: Loading test data ===")
    images, gts = _load_test_data(cfg, class_info, max_images=None)
    logger.info("Test set: %d images", len(images))

    # ── 3. Run pipeline ───────────────────────────────────────────────────────
    logger.info("=== Step 3/8: Running ROI pipeline ===")
    results = pipeline.run_batch(images)
    preds   = np.stack([r.full_pred for r in results])
    gt_arr  = np.stack(gts)
    n_detected = sum(1 for r in results if r.detected)
    logger.info("Detection rate: %d / %d", n_detected, len(results))

    # ── 4. Accuracy metrics ───────────────────────────────────────────────────
    logger.info("=== Step 4/8: Computing accuracy metrics ===")
    import torch
    N, H, W = preds.shape
    num_cls = cfg.classes.num_classes
    logits  = torch.zeros(N, num_cls, H, W)
    for i in range(N):
        for c in range(num_cls):
            logits[i, c][preds[i] == c] = 1.0
    targets = torch.from_numpy(gt_arr.astype(np.int64))
    acc     = compute_all_accuracy_metrics(logits, targets, num_cls)
    logger.info(
        "Dice=%.4f | IoU=%.4f | PixAcc=%.4f | Sens=%.4f | Spec=%.4f",
        acc["dice"], acc["iou"], acc["pixel_accuracy"],
        acc["sensitivity"], acc["specificity"],
    )

    # ── 5. Speed benchmark ────────────────────────────────────────────────────
    logger.info("=== Step 5/8: Benchmarking speed ===")
    speed = benchmark_pipeline(pipeline, images,
                               num_warmup=cfg.inference.num_warmup_batches)
    save_timing_breakdown_csv(
        speed,
        os.path.join("results/pipeline/profiling", "timing_breakdown.csv"),
        experiment_name=experiment_name,
    )

    # ── 6. Energy ─────────────────────────────────────────────────────────────
    energy = {"energy_per_frame_J": float("nan"), "power_W": float("nan"),
              "backend": "skipped"}
    if cfg.profiling.measure_energy:
        logger.info("=== Step 6/8: Measuring energy ===")
        try:
            energy = measure_pipeline_energy(
                pipeline, images,
                backend    = cfg.profiling.energy_backend,
                num_warmup = cfg.inference.num_warmup_batches,
            )
        except Exception as e:
            logger.warning("Energy measurement failed: %s", e)

    # ── 7. Export metrics CSV ─────────────────────────────────────────────────
    logger.info("=== Step 7/8: Exporting metrics ===")
    row = {
        "experiment":         experiment_name,
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_test_images":    len(images),
        "num_detected":       n_detected,
        "detection_rate":     n_detected / max(len(results), 1),
        "dice":               acc["dice"],
        "iou":                acc["iou"],
        "pixel_accuracy":     acc["pixel_accuracy"],
        "sensitivity":        acc["sensitivity"],
        "specificity":        acc["specificity"],
        "fps":                speed.get("fps", float("nan")),
        "mean_total_ms":      speed.get("mean_total_ms", float("nan")),
        "mean_detector_ms":   speed.get("mean_detector_ms", float("nan")),
        "mean_seg_ms":        speed.get("mean_seg_ms", float("nan")),
        "energy_per_frame_J": energy.get("energy_per_frame_J", float("nan")),
        "power_W":            energy.get("power_W", float("nan")),
        "padding_px":         cfg.roi.padding_px,
    }
    _write_csv_row(cfg.evaluation.metrics_csv, row)
    _print_summary(row)

    # ── 8. Visualisations ─────────────────────────────────────────────────────
    logger.info("=== Step 8/8: Saving visualisations ===")
    n_viz = min(cfg.evaluation.num_viz_samples, len(images))
    save_pipeline_grid(
        images[:n_viz], gts[:n_viz], results[:n_viz],
        save_path   = os.path.join(cfg.evaluation.visualizations_dir, "pipeline_grid.png"),
        class_colors= [vars(cfg.classes.colors_rgb)[str(i)]
                       for i in range(cfg.classes.num_classes)],
        max_samples = n_viz,
    )
    for i in range(min(4, len(images))):
        visualize_pipeline_result(
            image       = images[i],
            gt_mask     = gts[i],
            result      = results[i],
            save_path   = os.path.join(cfg.evaluation.visualizations_dir,
                                       f"pipeline_sample_{i:04d}.png"),
            class_names = class_names,
            title       = f"Sample {i} | det={results[i].detected}",
        )

    # Failure cases
    save_failure_cases(
        images=images, gt_masks=gts, results=results,
        gt_boxes=[[] for _ in images],   # GT boxes not required for basic failure analysis
        output_dir=cfg.evaluation.failure_cases_dir,
        max_cases=20,
    )

    # ── Optional: padding sweep ───────────────────────────────────────────────
    output = {"results": results, "metrics": acc, "benchmark": speed, "energy": energy}
    if getattr(cfg, "padding_sweep", None) and cfg.padding_sweep.enabled:
        logger.info("=== Running padding sweep experiment ===")
        df_sweep = run_padding_sweep(cfg)
        output["df_sweep"] = df_sweep

    logger.info("Phase 3 complete.")
    return output


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_csv_row(csv_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    mode = "a" if os.path.isfile(csv_path) else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if mode == "w":
            writer.writeheader()
        writer.writerow(row)
    logger.info("Metrics CSV → %s", csv_path)


def _print_summary(row: dict) -> None:
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  PHASE 3 ROI PIPELINE SUMMARY")
    print(sep)
    for k, v in row.items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k:<28}: {val}")
    print(sep + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 — ROI-Guided Segmentation")
    p.add_argument("--config",     type=str, default="configs/roi_pipeline_config.yaml")
    p.add_argument("--experiment", type=str, default="roi_guided")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from utils.config_loader import load_config
    cfg  = load_config(args.config)
    logging.basicConfig(
        level  = getattr(logging, cfg.logging.level, logging.INFO),
        format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_phase3(cfg, experiment_name=args.experiment)
