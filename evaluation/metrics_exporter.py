"""
evaluation/metrics_exporter.py
--------------------------------
Aggregate all metrics from an evaluation run and export to CSV.

Design decisions
----------------
- A single flat dict ``MetricsBundle`` accumulates all metric categories
  (accuracy, compute, throughput, energy) so they can be serialised in one
  row — making it easy to compare multiple experiments (baseline, ROI-guided,
  quantised) in a single CSV.
- Per-class metrics are stored as JSON strings within the CSV so the row
  remains flat but per-class data is still recoverable.
- Timestamp and experiment name are always included for reproducibility.
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

import pandas as pd

logger = logging.getLogger(__name__)


# ── Metrics bundle dataclass ───────────────────────────────────────────────────

@dataclass
class MetricsBundle:
    """
    Container for all Phase 1 metrics.

    Populated incrementally by calling helper setters, then serialised
    with ``to_flat_dict()`` / ``save_csv()``.
    """
    # Experiment metadata
    experiment_name: str = "baseline"
    timestamp: str       = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    device: str          = "cpu"
    checkpoint: str      = ""

    # ── Accuracy ──────────────────────────────────────────────────────────────
    dice: float               = float("nan")
    iou: float                = float("nan")
    pixel_accuracy: float     = float("nan")
    sensitivity: float        = float("nan")
    specificity: float        = float("nan")
    per_class_dice_json: str  = "[]"   # JSON string for CSV compatibility
    per_class_iou_json: str   = "[]"

    # ── Compute cost ──────────────────────────────────────────────────────────
    gflops: float             = float("nan")
    total_params_M: float     = float("nan")
    trainable_params_M: float = float("nan")
    memory_mb: float          = float("nan")

    # ── Throughput ────────────────────────────────────────────────────────────
    fps: float                = float("nan")
    ms_per_frame: float       = float("nan")
    mean_latency_ms: float    = float("nan")
    std_latency_ms: float     = float("nan")

    # ── Energy ────────────────────────────────────────────────────────────────
    energy_J: float           = float("nan")
    power_W: float            = float("nan")
    energy_per_frame_J: float = float("nan")
    energy_backend: str       = "unknown"

    # ── Misc ──────────────────────────────────────────────────────────────────
    num_test_images: int      = 0
    notes: str                = ""

    # ── Setters ───────────────────────────────────────────────────────────────

    def set_accuracy(self, acc: dict) -> None:
        """Populate from ``compute_all_accuracy_metrics`` output."""
        self.dice           = acc.get("dice",           float("nan"))
        self.iou            = acc.get("iou",            float("nan"))
        self.pixel_accuracy = acc.get("pixel_accuracy", float("nan"))
        self.sensitivity    = acc.get("sensitivity",    float("nan"))
        self.specificity    = acc.get("specificity",    float("nan"))
        self.per_class_dice_json = json.dumps(acc.get("per_class_dice", []))
        self.per_class_iou_json  = json.dumps(acc.get("per_class_iou",  []))

    def set_compute(self, comp: dict) -> None:
        """Populate from ``compute_all_compute_metrics`` output."""
        self.gflops             = comp.get("gflops",             float("nan"))
        self.total_params_M     = comp.get("total_params_M",     float("nan"))
        self.trainable_params_M = comp.get("trainable_params_M", float("nan"))
        self.memory_mb          = comp.get("memory_mb",          float("nan"))

    def set_throughput(self, tput: dict, latency: Optional[dict] = None) -> None:
        """Populate from ``benchmark_throughput`` / ``single_image_latency``."""
        self.fps         = tput.get("fps",          float("nan"))
        self.ms_per_frame= tput.get("ms_per_frame", float("nan"))
        if latency:
            self.mean_latency_ms = latency.get("mean_ms", float("nan"))
            self.std_latency_ms  = latency.get("std_ms",  float("nan"))

    def set_energy(self, energy: dict) -> None:
        """Populate from ``measure_inference_energy`` output."""
        self.energy_J           = energy.get("energy_J",           float("nan"))
        self.power_W            = energy.get("power_W",            float("nan"))
        self.energy_per_frame_J = energy.get("energy_per_frame_J", float("nan"))
        self.energy_backend     = energy.get("backend",            "unknown")

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_flat_dict(self) -> dict:
        """Return a flat dict (all values are primitives — CSV-safe)."""
        return asdict(self)

    def save_csv(self, csv_path: str, append: bool = True) -> None:
        """
        Write (or append) one row to a CSV file.

        Parameters
        ----------
        csv_path : output file path.
        append   : if True and file exists, append; otherwise overwrite.
        """
        row     = self.to_flat_dict()
        headers = list(row.keys())
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

        file_exists = os.path.isfile(csv_path)
        mode = "a" if (append and file_exists) else "w"

        with open(csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if mode == "w" or not file_exists:
                writer.writeheader()
            writer.writerow(row)

        logger.info("Metrics saved → %s", csv_path)

    def print_summary(self, class_names: Optional[List[str]] = None) -> None:
        """Print a formatted summary to stdout."""
        sep = "─" * 55
        print(f"\n{sep}")
        print(f"  EVALUATION SUMMARY  [{self.experiment_name}]  {self.timestamp}")
        print(sep)
        print(f"  Device        : {self.device}")
        print(f"  Test images   : {self.num_test_images}")
        print(f"\n  ── Accuracy ──────────────────────────────────────")
        print(f"  Dice          : {self.dice:.4f}")
        print(f"  IoU           : {self.iou:.4f}")
        print(f"  Pixel Acc     : {self.pixel_accuracy:.4f}")
        print(f"  Sensitivity   : {self.sensitivity:.4f}")
        print(f"  Specificity   : {self.specificity:.4f}")
        print(f"\n  ── Compute ───────────────────────────────────────")
        print(f"  GFLOPs        : {self.gflops:.4f}")
        print(f"  Params (M)    : {self.total_params_M:.2f}")
        print(f"  Memory (MB)   : {self.memory_mb:.2f}")
        print(f"\n  ── Throughput ────────────────────────────────────")
        print(f"  FPS           : {self.fps:.2f}")
        print(f"  ms/frame      : {self.ms_per_frame:.3f}")
        print(f"  Latency (ms)  : {self.mean_latency_ms:.3f} ± {self.std_latency_ms:.3f}")
        print(f"\n  ── Energy ────────────────────────────────────────")
        print(f"  Power (W)     : {self.power_W:.4f}")
        print(f"  Energy/frame  : {self.energy_per_frame_J:.6f} J")
        print(f"  Backend       : {self.energy_backend}")

        if class_names:
            pcd = json.loads(self.per_class_dice_json)
            pci = json.loads(self.per_class_iou_json)
            print(f"\n  ── Per-Class Dice ────────────────────────────────")
            for name, score in zip(class_names, pcd):
                bar = "█" * int(score * 30)
                print(f"  {name:<18} {score:.4f}  {bar}")
            print(f"\n  ── Per-Class IoU ─────────────────────────────────")
            for name, score in zip(class_names, pci):
                bar = "█" * int(score * 30)
                print(f"  {name:<18} {score:.4f}  {bar}")

        print(sep + "\n")
