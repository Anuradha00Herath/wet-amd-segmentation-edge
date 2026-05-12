"""
utils/__init__.py
-----------------
Convenience re-exports so callers can write:
    from utils import load_config, get_class_info
"""

from utils.config_loader import load_config, get_class_info
from utils.dataset       import (
    OCTDataset, build_dataloaders,
    rgb_mask_to_class, mask_to_rgb,
    train_transforms, val_transforms, test_transforms,
)
from utils.visualizer    import (
    save_prediction_overlay, save_batch_grid,
    plot_metric_curves, plot_metric_bar, plot_confusion_matrix,
)
from utils.profiler      import (
    Timer, CPUMemorySnapshot, GPUMemorySnapshot,
    system_snapshot, profile_forward,
)

__all__ = [
    "load_config", "get_class_info",
    "OCTDataset", "build_dataloaders",
    "rgb_mask_to_class", "mask_to_rgb",
    "train_transforms", "val_transforms", "test_transforms",
    "save_prediction_overlay", "save_batch_grid",
    "plot_metric_curves", "plot_metric_bar", "plot_confusion_matrix",
    "Timer", "CPUMemorySnapshot", "GPUMemorySnapshot",
    "system_snapshot", "profile_forward",
]
