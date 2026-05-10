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
from evaluation.energy_monitor import EnergyMonitor, print_energy_results
from evaluation.profiler import (
    ModelProfiler,
    profile_model,
    estimate_memory_footprint,
    count_flops,
    count_parameters,
    model_size_mb,
)
from evaluation.visualization import (
    plot_predictions,
    plot_training_curves,
    plot_per_class_dice,
    plot_metric_comparison,
    mask_to_rgb,
)
from evaluation.report_generator import (
    generate_markdown_report,
    save_full_results_csv,
    print_summary_table,
    print_comparison_table,
)

__all__ = [
    # Metrics
    "per_class_dice", "per_class_iou", "mean_dice", "mean_iou",
    "pixel_accuracy", "compute_all_metrics", "MetricAccumulator",
    # Benchmark
    "Benchmarker", "time_inference",
    # Energy
    "EnergyMonitor", "print_energy_results",
    # Profiler
    "ModelProfiler", "profile_model", "estimate_memory_footprint",
    "count_flops", "count_parameters", "model_size_mb",
    # Visualization
    "plot_predictions", "plot_training_curves",
    "plot_per_class_dice", "plot_metric_comparison", "mask_to_rgb",
    # Reports
    "generate_markdown_report", "save_full_results_csv",
    "print_summary_table", "print_comparison_table",
]