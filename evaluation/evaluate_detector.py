"""
evaluation/evaluate_detector.py
--------------------------------
Comprehensive ROI detector evaluation framework.

Computes
--------
Accuracy  : mAP50, mAP50-95, Precision, Recall, F1
Speed     : FPS, ms/frame (via detector.benchmark)
Compute   : FLOPs, Parameters, Memory (via Phase 1 metrics_compute)
Energy    : Power, J/frame (via Phase 1 metrics_energy)

Also produces
-------------
- Per-image detection results CSV
- Summary metrics CSV (appended, one row per experiment)
- Qualitative detection grid PNG
- Metric bar chart PNG
- Failure case visualisations (images with no detection but with GT boxes)
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Metrics bundle ─────────────────────────────────────────────────────────────

@dataclass
class DetectorMetricsBundle:
    """Flat container for all Phase 2 detector metrics."""

    experiment_name: str = "detector_baseline"
    timestamp: str       = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    checkpoint: str      = ""
    device: str          = "cpu"
    num_test_images: int = 0

    # Accuracy
    mAP50:       float = float("nan")
    mAP50_95:    float = float("nan")
    precision:   float = float("nan")
    recall:      float = float("nan")
    f1:          float = float("nan")

    # Speed
    fps:          float = float("nan")
    mean_ms:      float = float("nan")
    std_ms:       float = float("nan")

    # Compute
    gflops:        float = float("nan")
    params_M:      float = float("nan")
    memory_mb:     float = float("nan")

    # Energy
    energy_J:          float = float("nan")
    power_W:           float = float("nan")
    energy_per_frame_J: float = float("nan")
    energy_backend: str = "unknown"

    notes: str = ""

    def set_accuracy(self, m: dict) -> None:
        self.mAP50     = m.get("mAP50",     float("nan"))
        self.mAP50_95  = m.get("mAP50-95",  float("nan"))
        self.precision = m.get("precision",  float("nan"))
        self.recall    = m.get("recall",     float("nan"))
        p = self.precision; r = self.recall
        self.f1 = 2*p*r/(p+r+1e-6) if (p + r) > 0 else float("nan")

    def set_speed(self, m: dict) -> None:
        self.fps     = m.get("fps",      float("nan"))
        self.mean_ms = m.get("mean_ms",  float("nan"))
        self.std_ms  = m.get("std_ms",   float("nan"))

    def set_compute(self, m: dict) -> None:
        self.gflops    = m.get("gflops",        float("nan"))
        self.params_M  = m.get("total_params_M", float("nan"))
        self.memory_mb = m.get("memory_mb",     float("nan"))

    def set_energy(self, m: dict) -> None:
        self.energy_J           = m.get("energy_J",           float("nan"))
        self.power_W            = m.get("power_W",            float("nan"))
        self.energy_per_frame_J = m.get("energy_per_frame_J", float("nan"))
        self.energy_backend     = m.get("backend",            "unknown")

    def save_csv(self, csv_path: str, append: bool = True) -> None:
        row     = asdict(self)
        headers = list(row.keys())
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        mode = "a" if (append and os.path.isfile(csv_path)) else "w"
        with open(csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if mode == "w":
                writer.writeheader()
            writer.writerow(row)
        logger.info("Detector metrics saved → %s", csv_path)

    def print_summary(self) -> None:
        sep = "─" * 55
        print(f"\n{sep}")
        print(f"  DETECTOR EVALUATION  [{self.experiment_name}]  {self.timestamp}")
        print(sep)
        print(f"  Checkpoint    : {self.checkpoint}")
        print(f"  Device        : {self.device}")
        print(f"  Test images   : {self.num_test_images}")
        print(f"\n  ── Accuracy ──────────────────────────────────────")
        print(f"  mAP50         : {self.mAP50:.4f}")
        print(f"  mAP50-95      : {self.mAP50_95:.4f}")
        print(f"  Precision     : {self.precision:.4f}")
        print(f"  Recall        : {self.recall:.4f}")
        print(f"  F1            : {self.f1:.4f}")
        print(f"\n  ── Speed ─────────────────────────────────────────")
        print(f"  FPS           : {self.fps:.2f}")
        print(f"  ms/frame      : {self.mean_ms:.3f} ± {self.std_ms:.3f}")
        print(f"\n  ── Compute ───────────────────────────────────────")
        print(f"  GFLOPs        : {self.gflops:.4f}")
        print(f"  Params (M)    : {self.params_M:.2f}")
        print(f"  Memory (MB)   : {self.memory_mb:.2f}")
        print(f"\n  ── Energy ────────────────────────────────────────")
        print(f"  Power (W)     : {self.power_W:.4f}")
        print(f"  Energy/frame  : {self.energy_per_frame_J:.6f} J")
        print(sep + "\n")


# ── Box IoU ───────────────────────────────────────────────────────────────────

def box_iou(
    pred: tuple,   # (x1,y1,x2,y2)
    gt:   tuple,   # (x1,y1,x2,y2)
) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(pred[0], gt[0]); iy1 = max(pred[1], gt[1])
    ix2 = min(pred[2], gt[2]); iy2 = min(pred[3], gt[3])
    iw  = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_p = (pred[2]-pred[0]) * (pred[3]-pred[1])
    area_g = (gt[2]-gt[0])     * (gt[3]-gt[1])
    union  = area_p + area_g - inter
    return inter / (union + 1e-6)


# ── Failure case analysis ─────────────────────────────────────────────────────

def find_failure_cases(
    image_paths: List[str],
    detections_dict: Dict[str, list],    # {path: List[Detection]}
    gt_boxes_dict:   Dict[str, list],    # {path: List[BoundingBox]}
    iou_threshold:   float = 0.5,
) -> List[str]:
    """
    Identify images where the detector failed:
    - Missed detections (GT exists, no prediction above IoU threshold).
    - False positives (prediction but no GT).

    Returns
    -------
    List[str]  paths of failure-case images.
    """
    failures = []
    for path in image_paths:
        dets = detections_dict.get(path, [])
        gts  = gt_boxes_dict.get(path, [])

        if gts and not dets:
            failures.append(path)   # missed
        elif dets and not gts:
            failures.append(path)   # false positive
        elif dets and gts:
            # Check if best prediction overlaps with any GT
            best = dets[0]
            best_iou = max(
                box_iou((best.x1,best.y1,best.x2,best.y2),
                        (g.x1, g.y1, g.x2, g.y2))
                for g in gts
            )
            if best_iou < iou_threshold:
                failures.append(path)  # poor localisation

    logger.info("Failure cases: %d / %d images", len(failures), len(image_paths))
    return failures


# ── Main evaluation runner ────────────────────────────────────────────────────

def run_detector_evaluation(
    cfg,
    experiment_name: str = "detector_baseline",
) -> DetectorMetricsBundle:
    """
    Full Phase 2 detector evaluation pipeline.

    Steps
    -----
    1. Load detector.
    2. Run YOLOv8 val() for mAP metrics.
    3. Benchmark latency.
    4. Measure compute cost (FLOPs, params, memory).
    5. Measure energy.
    6. Save CSV + visualisations.

    Parameters
    ----------
    cfg             : loaded detector config namespace.
    experiment_name : label written to CSV.

    Returns
    -------
    DetectorMetricsBundle
    """
    import torch
    from inference.roi_inference import ROIDetector
    from training.train_detector import validate_detector
    from evaluation.metrics_compute import compute_all_compute_metrics
    from evaluation.metrics_energy  import measure_inference_energy
    from detectors.roi_visualizer   import plot_detection_metrics

    bundle = DetectorMetricsBundle(
        experiment_name = experiment_name,
        checkpoint      = cfg.detector.checkpoint,
        device          = cfg.detector.device,
    )

    # ── 1. Accuracy via YOLOv8 val() ──────────────────────────────────────────
    logger.info("=== Step 1/5: mAP evaluation ===")
    device_str = "0" if cfg.detector.device == "cuda" else cfg.detector.device
    acc = validate_detector(
        checkpoint_path = cfg.detector.checkpoint,
        dataset_yaml    = cfg.roi_dataset.dataset_yaml,
        image_size      = cfg.detector.image_size,
        device          = device_str,
        split           = "test",
    )
    bundle.set_accuracy(acc)

    # ── 2. Latency benchmark ───────────────────────────────────────────────────
    logger.info("=== Step 2/5: Latency benchmark ===")
    detector = ROIDetector(
        checkpoint_path = cfg.detector.checkpoint,
        conf_threshold  = cfg.detector.conf_threshold,
        iou_threshold   = cfg.detector.iou_threshold,
        image_size      = cfg.detector.image_size,
        device          = cfg.detector.device,
        padding_px      = cfg.detector.padding_px,
    )
    dummy_img = np.zeros(
        (cfg.roi_dataset.image_size, cfg.roi_dataset.image_size), dtype=np.uint8
    )
    speed = detector.benchmark(dummy_img, num_warmup=10, num_runs=100)
    bundle.set_speed(speed)
    bundle.num_test_images = speed.get("num_images", 0)

    # ── 3. Compute cost ────────────────────────────────────────────────────────
    logger.info("=== Step 3/5: Compute metrics ===")
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(cfg.detector.checkpoint)
        pt_model   = yolo_model.model  # underlying nn.Module
        input_shape = tuple(cfg.profiling.flops_input_shape)
        comp = compute_all_compute_metrics(
            pt_model, input_shape=input_shape, device=cfg.detector.device
        )
        bundle.set_compute(comp)
    except Exception as e:
        logger.warning("Compute metrics failed: %s", e)

    # ── 4. Energy ──────────────────────────────────────────────────────────────
    if cfg.profiling.measure_energy:
        logger.info("=== Step 4/5: Energy measurement ===")
        try:
            from evaluation.metrics_energy import EnergyMeter
            meter = EnergyMeter(backend=cfg.profiling.energy_backend)

            n_images = 0
            meter.start()
            test_img_dir = os.path.join(cfg.roi_dataset.root_dir, "images", "test")
            test_imgs    = sorted(os.listdir(test_img_dir))
            for fname in test_imgs:
                img = cv2.imread(os.path.join(test_img_dir, fname), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    detector.predict(img)
                    n_images += 1
            meter.stop()
            energy_result = meter.result()
            energy_result["num_images"] = n_images
            energy_result["energy_per_frame_J"] = (
                energy_result.get("energy_J", float("nan")) / max(n_images, 1)
            )
            bundle.set_energy(energy_result)
        except Exception as e:
            logger.warning("Energy measurement failed: %s", e)

    # ── 5. Export + visualise ──────────────────────────────────────────────────
    logger.info("=== Step 5/5: Saving results ===")
    bundle.save_csv(cfg.evaluation.metrics_csv)
    bundle.print_summary()

    viz_dir = cfg.evaluation.visualizations_dir
    os.makedirs(viz_dir, exist_ok=True)
    plot_detection_metrics(
        {
            "mAP50":     bundle.mAP50,
            "mAP50-95":  bundle.mAP50_95,
            "Precision": bundle.precision,
            "Recall":    bundle.recall,
            "F1":        bundle.f1,
        },
        save_path = os.path.join(viz_dir, "detector_accuracy_metrics.png"),
        title     = f"ROI Detector Metrics — {experiment_name}",
    )

    logger.info("Detector evaluation complete.")
    return bundle
