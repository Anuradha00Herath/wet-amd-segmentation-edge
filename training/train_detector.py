"""
training/train_detector.py
---------------------------
YOLOv8n ROI detector training pipeline.

Uses the ultralytics YOLO API, which handles:
  - data loading, augmentation, mixed-precision
  - cosine LR schedule, early stopping
  - checkpoint saving (best.pt, last.pt)
  - TensorBoard + CSV logging under the experiment directory
  - mAP50, mAP50-95, Precision, Recall during training

This module wraps the ultralytics API with:
  - config-driven hyperparameter injection (no hardcoding)
  - resume-training support
  - pre-training sanity checks
  - a post-training summary

Design decisions
----------------
- ``model.train()`` is called with explicit kwargs derived from the config
  rather than a separate YOLO yaml, so Phase 2 config.yaml is the single
  source of truth for training settings.
- ``mosaic=0.0`` is set by default: mosaic augmentation stitches random
  images together, which is harmful for medical images where spatial
  context matters.
- ``hsv_h=0.0`` is set because OCT is greyscale; hue shift is meaningless.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def train_detector(cfg) -> str:
    """
    Train (or resume) a YOLOv8n ROI detector using the ultralytics API.

    Parameters
    ----------
    cfg : loaded detector config namespace (from ``load_config``).

    Returns
    -------
    str  Path to the best checkpoint (``best.pt``).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is not installed. Run: pip install ultralytics"
        )

    t = cfg.training

    # ── Resolve device string ──────────────────────────────────────────────────
    # ultralytics accepts "cpu", "0", "0,1" (GPU ids) — not "cuda"
    device = t.device
    if device == "cuda":
        device = "0"

    # ── Build model ────────────────────────────────────────────────────────────
    if t.resume and os.path.isfile(str(t.resume)):
        logger.info("Resuming training from checkpoint: %s", t.resume)
        model = YOLO(str(t.resume))
    elif t.pretrained:
        model_name = f"yolov8{t.model_size}.pt"   # e.g. "yolov8n.pt"
        logger.info("Loading pretrained YOLOv8%s weights.", t.model_size)
        model = YOLO(model_name)
    else:
        model_name = f"yolov8{t.model_size}.yaml"
        logger.info("Training YOLOv8%s from scratch.", t.model_size)
        model = YOLO(model_name)

    # ── Sanity check: dataset.yaml exists ─────────────────────────────────────
    dataset_yaml = cfg.roi_dataset.dataset_yaml
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(
            f"dataset.yaml not found at {dataset_yaml}. "
            "Run the dataset preparation pipeline first."
        )

    # ── Collect training kwargs ────────────────────────────────────────────────
    train_kwargs = dict(
        data         = str(Path(dataset_yaml).resolve()),
        epochs       = t.epochs,
        imgsz        = t.image_size,
        batch        = t.batch_size,
        lr0          = t.lr0,
        lrf          = t.lrf,
        momentum     = t.momentum,
        weight_decay = t.weight_decay,
        warmup_epochs= t.warmup_epochs,
        patience     = t.patience,
        save_period  = t.save_period,
        workers      = t.workers,
        device       = device,
        amp          = t.amp,
        exist_ok     = t.exist_ok,
        project      = t.project_dir,
        name         = t.experiment_name,
        # ── Augmentation: tuned for greyscale medical images ─────────────────
        mosaic       = t.mosaic,
        mixup        = t.mixup,
        copy_paste   = t.copy_paste,
        hsv_h        = t.hsv_h,
        hsv_s        = t.hsv_s,
        hsv_v        = t.hsv_v,
        degrees      = t.degrees,
        translate    = t.translate,
        scale        = t.scale,
        shear        = t.shear,
        perspective  = t.perspective,
        flipud       = t.flipud,
        fliplr       = t.fliplr,
        # ── Verbosity ─────────────────────────────────────────────────────────
        verbose      = True,
    )

    logger.info("Starting YOLOv8%s training for %d epochs.", t.model_size, t.epochs)
    logger.info("Dataset: %s", dataset_yaml)
    logger.info("Project: %s / %s", t.project_dir, t.experiment_name)

    # ── Train ─────────────────────────────────────────────────────────────────
    results = model.train(**train_kwargs)

    # ── Locate best checkpoint ────────────────────────────────────────────────
    best_pt = Path(t.project_dir) / t.experiment_name / "weights" / "best.pt"
    if not best_pt.exists():
        logger.warning("best.pt not found at expected path: %s", best_pt)
        best_pt = Path(t.project_dir) / t.experiment_name / "weights" / "last.pt"

    logger.info("Training complete. Best checkpoint: %s", best_pt)
    return str(best_pt)


def validate_detector(
    checkpoint_path: str,
    dataset_yaml: str,
    image_size: int = 640,
    device: str = "cpu",
    split: str = "val",
) -> dict:
    """
    Run YOLOv8 validation on a saved checkpoint.

    Parameters
    ----------
    checkpoint_path : path to best.pt or last.pt.
    dataset_yaml    : path to dataset.yaml.
    image_size      : inference image size.
    device          : "cpu" or GPU id string.
    split           : "val" | "test".

    Returns
    -------
    dict  ``{"mAP50": ..., "mAP50-95": ..., "precision": ..., "recall": ...}``
    """
    from ultralytics import YOLO

    if device == "cuda":
        device = "0"

    model   = YOLO(checkpoint_path)
    results = model.val(
        data   = str(Path(dataset_yaml).resolve()),
        imgsz  = image_size,
        device = device,
        split  = split,
        verbose= True,
    )

    metrics = {
        "mAP50":     float(results.box.map50),
        "mAP50-95":  float(results.box.map),
        "precision": float(results.box.mp),
        "recall":    float(results.box.mr),
    }
    logger.info("Validation metrics: %s", metrics)
    return metrics
