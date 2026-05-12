"""
detectors/prepare_dataset.py
-----------------------------
End-to-end ROI dataset preparation pipeline (Phase 2, Step 1).

Execution order
---------------
1.  Load detector config.
2.  Set random seed.
3.  Generate bounding boxes from all segmentation masks.
4.  Print + save bbox generation statistics.
5.  Build YOLO-format dataset (copy images, write labels, dataset.yaml).
6.  Print dataset summary.
7.  Save sample visualisations.
8.  Save experiment metadata.

Usage (Colab)
-------------
    %cd /content/<project_root>
    !python detectors/prepare_dataset.py --config configs/detector_config.yaml

Or call ``run_dataset_preparation(cfg)`` from a notebook.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.config_loader         import load_config, get_seg_class_info_from_detector_cfg
from utils.reproducibility       import seed_everything, save_experiment_metadata, namespace_to_dict
from detectors.bbox_generator    import generate_all_boxes
from detectors.dataset_builder   import build_yolo_dataset, print_dataset_summary
from detectors.roi_visualizer    import save_dataset_samples

logger = logging.getLogger(__name__)


def run_dataset_preparation(cfg) -> dict:
    """
    Full Phase 2 dataset preparation pipeline.

    Parameters
    ----------
    cfg : loaded detector config namespace.

    Returns
    -------
    dict  with keys: bbox_results, splits, stats.
    """
    # ── Reproducibility ───────────────────────────────────────────────────────
    seed_everything(cfg.project.seed)

    # ── Build seg class info from detector config ──────────────────────────────
    class_info = get_seg_class_info_from_detector_cfg(cfg)

    lesion_ids = cfg.bbox.lesion_class_ids

    logger.info("=== Step 1/4: Generating bounding boxes ===")
    logger.info("Lesion class IDs: %s", lesion_ids)
    logger.info("Padding: %d px | Min area: %d px²", cfg.bbox.padding_px, cfg.bbox.min_area_px)

    bbox_results, stats = generate_all_boxes(
        image_dir        = cfg.data.image_dir,
        mask_dir         = cfg.data.mask_dir,
        class_info       = class_info,
        lesion_class_ids = lesion_ids,
        min_area_px      = cfg.bbox.min_area_px,
        padding_px       = cfg.bbox.padding_px,
        merge_all        = cfg.bbox.merge_all_lesions,
    )

    print(f"\nBBox Generation Stats:\n  {stats.report()}")
    if stats.skipped_filenames:
        print(f"  Skipped: {stats.skipped_filenames[:5]} ...")

    # ── Build YOLO dataset ────────────────────────────────────────────────────
    logger.info("=== Step 2/4: Building YOLO dataset ===")
    det_class_names = [cfg.classes.detector_classes.__dict__[k]
                       for k in sorted(cfg.classes.detector_classes.__dict__)]

    splits = build_yolo_dataset(
        image_dir         = cfg.data.image_dir,
        bbox_results      = bbox_results,
        output_root       = cfg.roi_dataset.root_dir,
        train_ratio       = cfg.roi_dataset.train_split,
        val_ratio         = cfg.roi_dataset.val_split,
        seed              = cfg.project.seed,
        include_negatives = False,
        class_names       = det_class_names,
    )
    print_dataset_summary(splits, bbox_results)

    # ── Visualise samples ─────────────────────────────────────────────────────
    logger.info("=== Step 3/4: Saving visualisations ===")
    viz_dir = "results/detector/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    save_dataset_samples(
        image_dir   = cfg.data.image_dir,
        bbox_results= bbox_results,
        save_path   = os.path.join(viz_dir, "bbox_samples.png"),
        n_samples   = 12,
        seed        = cfg.project.seed,
    )
    logger.info("Sample grid saved → %s/bbox_samples.png", viz_dir)

    # ── Save experiment metadata ──────────────────────────────────────────────
    logger.info("=== Step 4/4: Saving metadata ===")
    meta_path = "results/detector/dataset_preparation_metadata.json"
    save_experiment_metadata(
        output_path = meta_path,
        cfg_dict    = namespace_to_dict(cfg),
        extra       = {
            "bbox_stats": {
                "total":           stats.total,
                "with_lesion":     stats.with_lesion,
                "no_lesion":       stats.no_lesion,
                "multi_component": stats.multi_component,
                "errors":          stats.errors,
            },
            "splits": {k: len(v) for k, v in splits.items()},
        },
    )

    logger.info("Dataset preparation complete.")
    return {"bbox_results": bbox_results, "splits": splits, "stats": stats}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 — ROI Dataset Preparation")
    parser.add_argument("--config", type=str, default="configs/detector_config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    logging.basicConfig(
        level   = getattr(logging, cfg.logging.level, logging.INFO),
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_dataset_preparation(cfg)
