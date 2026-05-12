"""
evaluation/evaluate_baseline.py
--------------------------------
Main evaluation entry point for Phase 1.

Execution order
---------------
1.  Load config.
2.  Build model + restore checkpoint.
3.  Build DataLoaders.
4.  Compute FLOPs / params / memory (once, from config input shape).
5.  Measure throughput via benchmark_throughput.
6.  Measure single-image latency.
7.  Run full inference (collect logits + targets for accuracy metrics).
8.  Compute all accuracy metrics.
9.  Measure energy (whole inference pass).
10. Export all metrics to CSV.
11. Save qualitative visualisations.

Usage (in Colab)
----------------
    %cd /content/oct_seg_phase1
    !python evaluation/evaluate_baseline.py --config configs/config.yaml

Or call ``run_evaluation(cfg)`` directly from a notebook.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

# ── Ensure project root is on PYTHONPATH ──────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.config_loader   import load_config, get_class_info
from utils.visualizer      import save_batch_grid, plot_metric_bar
from models.model_loader   import get_model
from utils.dataset         import build_dataloaders, _val_transforms, OCTDataset
from inference.baseline_inference  import run_batch_inference
from evaluation.metrics_accuracy   import compute_all_accuracy_metrics
from evaluation.metrics_compute    import compute_all_compute_metrics
from evaluation.metrics_throughput import benchmark_throughput, single_image_latency
from evaluation.metrics_energy     import measure_inference_energy
from evaluation.metrics_exporter   import MetricsBundle

logger = logging.getLogger(__name__)


def _setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


# ── Main evaluation function ───────────────────────────────────────────────────

def run_evaluation(cfg, experiment_name: str = "baseline") -> MetricsBundle:
    """
    Full Phase 1 evaluation pipeline.

    Parameters
    ----------
    cfg             : loaded config namespace (from ``load_config``).
    experiment_name : label stored in the CSV.

    Returns
    -------
    MetricsBundle  Populated with all metrics (already saved to CSV).
    """
    device     = cfg.inference.device
    class_info = get_class_info(cfg)
    num_classes= cfg.classes.num_classes
    class_names= [vars(cfg.classes.names)[str(i)] for i in range(num_classes)]

    bundle = MetricsBundle(
        experiment_name = experiment_name,
        device          = device,
        checkpoint      = cfg.model.checkpoint,
    )

    # ── 1. Load model ─────────────────────────────────────────────────────────
    logger.info("=== Step 1/9: Loading model ===")
    model = get_model(cfg, device)

    # ── 2. Build DataLoaders ──────────────────────────────────────────────────
    logger.info("=== Step 2/9: Building DataLoaders ===")
    _, _, test_loader = build_dataloaders(cfg, class_info)

    # ── 3. Compute cost metrics (FLOPs / params / memory) ────────────────────
    logger.info("=== Step 3/9: Measuring compute cost ===")
    input_shape = tuple(cfg.profiling.flops_input_shape)
    compute_metrics = compute_all_compute_metrics(model, input_shape, device)
    bundle.set_compute(compute_metrics)

    # ── 4. Throughput benchmark ───────────────────────────────────────────────
    logger.info("=== Step 4/9: Benchmarking throughput ===")
    tput = benchmark_throughput(
        model, test_loader, device,
        num_warmup_batches = cfg.inference.num_warmup_batches,
    )
    # Single-image latency (more stable than per-batch)
    lat = single_image_latency(
        model, input_shape=input_shape, device=device,
        num_warmup=10, num_runs=100,
    )
    bundle.set_throughput(tput, lat)

    # ── 5. Full inference pass ────────────────────────────────────────────────
    logger.info("=== Step 5/9: Running full inference ===")
    inf_result = run_batch_inference(
        model, test_loader, device,
        save_predictions    = cfg.inference.save_predictions,
        output_dir          = cfg.inference.output_dir,
        num_warmup_batches  = cfg.inference.num_warmup_batches,
    )
    all_logits  = inf_result["all_logits"]
    all_targets = inf_result["all_targets"]
    bundle.num_test_images = inf_result["num_images"]

    # ── 6. Accuracy metrics ───────────────────────────────────────────────────
    logger.info("=== Step 6/9: Computing accuracy metrics ===")
    acc = compute_all_accuracy_metrics(all_logits, all_targets, num_classes)
    bundle.set_accuracy(acc)

    # ── 7. Energy measurement ─────────────────────────────────────────────────
    if cfg.profiling.measure_energy:
        logger.info("=== Step 7/9: Measuring energy ===")
        energy = measure_inference_energy(
            model, test_loader, device,
            backend            = cfg.profiling.energy_backend,
            num_warmup_batches = cfg.inference.num_warmup_batches,
        )
        bundle.set_energy(energy)
    else:
        logger.info("=== Step 7/9: Energy measurement skipped (config) ===")

    # ── 8. Export CSV ─────────────────────────────────────────────────────────
    logger.info("=== Step 8/9: Exporting metrics CSV ===")
    bundle.save_csv(cfg.evaluation.metrics_output_csv)
    bundle.print_summary(class_names)

    # ── 9. Visualisations ─────────────────────────────────────────────────────
    logger.info("=== Step 9/9: Saving visualisations ===")
    _save_visualisations(
        cfg, all_logits, all_targets, test_loader, class_names, acc
    )

    logger.info("Evaluation complete.")
    return bundle


def _save_visualisations(cfg, all_logits, all_targets, test_loader,
                          class_names, acc) -> None:
    """Save qualitative and quantitative visualisation figures."""
    viz_dir = cfg.evaluation.visualizations_dir
    os.makedirs(viz_dir, exist_ok=True)

    # ── Batch grid (qualitative) ──────────────────────────────────────────────
    # Grab first batch from the loader for visualisation
    try:
        imgs_vis, masks_vis = next(iter(test_loader))
        # Use the already-computed logits slice for consistency
        n = min(imgs_vis.size(0), cfg.evaluation.num_viz_samples)
        save_batch_grid(
            images      = imgs_vis[:n],
            gt_masks    = masks_vis[:n],
            pred_logits = all_logits[:n],
            save_path   = os.path.join(viz_dir, "baseline_predictions_grid.png"),
            class_names = class_names,
            max_samples = n,
        )
    except Exception as e:
        logger.warning("Batch grid visualisation failed: %s", e)

    # ── Per-class Dice bar chart ──────────────────────────────────────────────
    import json
    pcd = json.loads(
        getattr(acc, "per_class_dice", None) or "[]"
    ) if hasattr(acc, "per_class_dice") else acc.get("per_class_dice", [])

    if pcd:
        plot_metric_bar(
            values      = pcd,
            class_names = class_names,
            metric_name = "Dice Score",
            save_path   = os.path.join(viz_dir, "per_class_dice.png"),
        )

    # ── Per-class IoU bar chart ───────────────────────────────────────────────
    pci = acc.get("per_class_iou", [])
    if pci:
        plot_metric_bar(
            values      = pci,
            class_names = class_names,
            metric_name = "IoU",
            save_path   = os.path.join(viz_dir, "per_class_iou.png"),
        )


# ── CLI entry point ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 — Baseline Segmentation Evaluation"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--experiment", type=str, default="baseline",
        help="Experiment name label written to CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    _setup_logging(
        level    = cfg.logging.level,
        log_file = cfg.logging.log_file,
    )
    run_evaluation(cfg, experiment_name=args.experiment)
