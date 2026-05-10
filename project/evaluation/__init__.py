# evaluation/__init__.py

from evaluation.metrics import (
    per_class_dice,
    per_class_iou,
    mean_dice,
    mean_iou,
    pixel_accuracy,
    compute_all_metrics,
    MetricAccumulator,
)
from evaluation.benchmark import Benchmarker, time_inference
from evaluation.energy_monitor import EnergyMonitor
from evaluation.profiler import profile_model, estimate_memory_footprint
from evaluation.visualization import (
    plot_predictions,
    plot_training_curves,
    plot_per_class_dice,
    plot_metric_comparison,
    mask_to_rgb,
)

__all__ = [
    # Metrics
    "per_class_dice", "per_class_iou",
    "mean_dice", "mean_iou",
    "pixel_accuracy",
    "compute_all_metrics", "MetricAccumulator",
    # Benchmark
    "Benchmarker", "time_inference",
    # Energy
    "EnergyMonitor",
    # Profiler
    "profile_model", "estimate_memory_footprint",
    # Visualization
    "plot_predictions", "plot_training_curves",
    "plot_per_class_dice", "plot_metric_comparison",
    "mask_to_rgb",
]