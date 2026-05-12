"""detectors/__init__.py"""
from detectors.bbox_generator  import (
    BoundingBox, BboxGenerationStats,
    boxes_from_mask, generate_all_boxes,
    write_yolo_label,
)
from detectors.dataset_builder import (
    build_yolo_dataset, split_filenames, print_dataset_summary,
)
from detectors.roi_visualizer  import (
    visualize_bbox_on_image, visualize_mask_vs_bbox,
    save_dataset_samples, save_detection_grid, plot_detection_metrics,
)
