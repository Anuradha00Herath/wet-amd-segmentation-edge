# evaluation/__init__.py
from evaluation.metrics import (
    dice_coefficient, iou_score, pixel_accuracy,
    sensitivity, specificity, precision,
    compute_all_metrics, MetricAccumulator,
)
from evaluation.benchmark import Benchmarker, time_inference
from evaluation.energy_monitor import EnergyMonitor
from evaluation.profiler import profile_model, estimate_memory_footprint
from evaluation.visualization import (
    plot_predictions, plot_training_curves, plot_metric_comparison,
)

__all__ = [
    "dice_coefficient", "iou_score", "pixel_accuracy",
    "sensitivity", "specificity", "precision",
    "compute_all_metrics", "MetricAccumulator",
    "Benchmarker", "time_inference",
    "EnergyMonitor",
    "profile_model", "estimate_memory_footprint",
    "plot_predictions", "plot_training_curves", "plot_metric_comparison",
]
