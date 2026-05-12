"""
benchmarking/pipeline_profiler.py
----------------------------------
End-to-end pipeline benchmarking for Phase 3.

Measures
--------
- Per-stage latency: detector / crop / segmentation / map-back
- Total end-to-end latency and FPS
- Memory usage (CPU RSS or GPU peak)
- CPU utilisation during inference
- Energy via Phase 1 EnergyMeter (codecarbon fallback)

Design decisions
----------------
- Uses ``PipelineResult.t_*_ms`` fields accumulated across N images so
  each stage is timed by the pipeline itself (no double-timing overhead).
- Warm-up runs are discarded before stats accumulation.
- Results are returned as a flat dict compatible with MetricsBundle / CSV.
- Reuses Phase 1 ``utils/profiler.py`` for memory + system snapshots.
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from typing import Dict, List, Optional

import numpy as np

from pipelines.roi_seg_pipeline import ROIPipeline, PipelineResult

logger = logging.getLogger(__name__)


# ── Core benchmarking function ────────────────────────────────────────────────

def benchmark_pipeline(
    pipeline:           ROIPipeline,
    images:             List[np.ndarray],
    num_warmup:         int = 3,
    padding_px:         Optional[int] = None,
) -> Dict[str, float]:
    """
    Benchmark the ROI pipeline over a list of images.

    Parameters
    ----------
    pipeline    : initialised ROIPipeline.
    images      : list of H×W uint8 greyscale arrays.
    num_warmup  : images to discard before timing.
    padding_px  : padding override for this run.

    Returns
    -------
    dict with keys:
        fps, mean_total_ms, mean_detector_ms, mean_crop_ms,
        mean_seg_ms, mean_map_ms, std_total_ms,
        num_images, num_detected, detection_rate,
        peak_memory_mb.
    """
    from utils.profiler import CPUMemorySnapshot, GPUMemorySnapshot

    results: List[PipelineResult] = []
    device  = pipeline.cfg.segmentation.device

    # ── Warm-up ───────────────────────────────────────────────────────────────
    for img in images[:num_warmup]:
        pipeline.run(img, padding_px)

    # ── Memory snapshot start ─────────────────────────────────────────────────
    if device.startswith("cuda"):
        mem_snap = GPUMemorySnapshot(device)
        mem_snap.before()
    else:
        from utils.profiler import CPUMemorySnapshot
        mem_snap = CPUMemorySnapshot()
        mem_snap.before()

    # ── Timed runs ────────────────────────────────────────────────────────────
    for img in images[num_warmup:]:
        r = pipeline.run(img, padding_px)
        results.append(r)

    # ── Memory snapshot end ───────────────────────────────────────────────────
    mem_snap.after()
    if device.startswith("cuda"):
        peak_mb = mem_snap.peak_mb
    else:
        peak_mb = mem_snap.delta_mb

    if not results:
        logger.warning("No timed images (dataset smaller than warmup).")
        return {}

    # ── Accumulate timings ────────────────────────────────────────────────────
    total_ms = [r.t_total_ms    for r in results]
    det_ms   = [r.t_detector_ms for r in results]
    crop_ms  = [r.t_crop_ms     for r in results]
    seg_ms   = [r.t_seg_ms      for r in results]
    map_ms   = [r.t_map_ms      for r in results]

    n_detected = sum(1 for r in results if r.detected)
    n_total    = len(results)

    mean_total = statistics.mean(total_ms)
    fps        = 1000.0 / mean_total if mean_total > 0 else float("nan")

    stats = {
        "fps":              fps,
        "mean_total_ms":    mean_total,
        "std_total_ms":     statistics.stdev(total_ms) if len(total_ms) > 1 else 0.0,
        "mean_detector_ms": statistics.mean(det_ms),
        "mean_crop_ms":     statistics.mean(crop_ms),
        "mean_seg_ms":      statistics.mean(seg_ms),
        "mean_map_ms":      statistics.mean(map_ms),
        "num_images":       n_total,
        "num_detected":     n_detected,
        "detection_rate":   n_detected / n_total if n_total else 0.0,
        "peak_memory_mb":   peak_mb,
    }

    logger.info(
        "Pipeline benchmark → FPS: %.2f | total: %.2f ms "
        "| det: %.2f | crop: %.2f | seg: %.2f | map: %.2f",
        fps, mean_total,
        stats["mean_detector_ms"], stats["mean_crop_ms"],
        stats["mean_seg_ms"],      stats["mean_map_ms"],
    )
    return stats


# ── Energy measurement ────────────────────────────────────────────────────────

def measure_pipeline_energy(
    pipeline:   ROIPipeline,
    images:     List[np.ndarray],
    backend:    str = "codecarbon",
    num_warmup: int = 3,
    padding_px: Optional[int] = None,
) -> Dict[str, float]:
    """
    Measure energy consumption of the full ROI pipeline.

    Reuses Phase 1 ``evaluation/metrics_energy.EnergyMeter``.

    Returns
    -------
    dict  energy_J, power_W, energy_per_frame_J, backend, num_images.
    """
    from evaluation.metrics_energy import EnergyMeter

    # Warm-up
    for img in images[:num_warmup]:
        pipeline.run(img, padding_px)

    meter     = EnergyMeter(backend=backend)
    n_images  = 0

    meter.start()
    for img in images[num_warmup:]:
        pipeline.run(img, padding_px)
        n_images += 1
    meter.stop()

    result = meter.result()
    result["num_images"]         = n_images
    energy_J                     = result.get("energy_J", float("nan"))
    result["energy_per_frame_J"] = energy_J / max(n_images, 1)

    logger.info(
        "Pipeline energy → %.4f J | %.6f J/frame | backend=%s",
        energy_J, result["energy_per_frame_J"], result["backend"],
    )
    return result


# ── Stage-breakdown CSV export ────────────────────────────────────────────────

def save_timing_breakdown_csv(
    benchmark_result: dict,
    csv_path: str,
    experiment_name: str = "roi_pipeline",
) -> None:
    """
    Save a stage-breakdown timing CSV row.

    Parameters
    ----------
    benchmark_result : dict from ``benchmark_pipeline``.
    csv_path         : output CSV path.
    experiment_name  : label column.
    """
    import csv, os
    from datetime import datetime

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    row = {"experiment": experiment_name,
           "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           **benchmark_result}

    mode = "a" if os.path.isfile(csv_path) else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if mode == "w":
            writer.writeheader()
        writer.writerow(row)
    logger.info("Timing breakdown saved → %s", csv_path)
