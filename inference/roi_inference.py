"""
inference/roi_inference.py
---------------------------
YOLOv8n ROI detector inference pipeline.

Responsibilities
----------------
1. Load YOLOv8n checkpoint.
2. Run inference on a single image or a batch DataLoader.
3. Apply confidence + NMS thresholds.
4. Return ROI bounding boxes in pixel coordinates.
5. Optionally save cropped ROI patches (used in Phase 3).

All output is in the form of plain dicts and numpy arrays so
downstream code (Phase 3 segmentation) has no dependency on
ultralytics internal objects.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Detection result dataclass ────────────────────────────────────────────────

@dataclass
class Detection:
    """
    Single bounding-box detection result.

    Attributes
    ----------
    x1, y1, x2, y2 : pixel coordinates (original image space).
    conf            : confidence score [0, 1].
    cls             : class id.
    """
    x1:   int
    y1:   int
    x2:   int
    y2:   int
    conf: float
    cls:  int = 0

    @property
    def width(self)  -> int: return self.x2 - self.x1
    @property
    def height(self) -> int: return self.y2 - self.y1
    @property
    def area(self)   -> int: return self.width * self.height

    def to_dict(self) -> dict:
        return dict(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2,
                    conf=self.conf, cls=self.cls)

    def pad(self, px: int, img_w: int, img_h: int) -> "Detection":
        return Detection(
            x1=max(0, self.x1 - px), y1=max(0, self.y1 - px),
            x2=min(img_w, self.x2 + px), y2=min(img_h, self.y2 + px),
            conf=self.conf, cls=self.cls,
        )


# ── Model loader ──────────────────────────────────────────────────────────────

class ROIDetector:
    """
    Thin wrapper around a YOLOv8 model for ROI inference.

    Parameters
    ----------
    checkpoint_path : path to best.pt.
    conf_threshold  : minimum confidence to keep a detection.
    iou_threshold   : NMS IoU threshold.
    image_size      : inference image size.
    device          : "cpu" or "0" (GPU index).
    padding_px      : extra padding added to each detected box.
    max_detections  : maximum boxes returned per image.
    """

    def __init__(
        self,
        checkpoint_path: str,
        conf_threshold:  float = 0.25,
        iou_threshold:   float = 0.45,
        image_size:      int   = 640,
        device:          str   = "cpu",
        padding_px:      int   = 15,
        max_detections:  int   = 10,
    ) -> None:
        from ultralytics import YOLO

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # ultralytics uses "0" not "cuda"
        self._device = "0" if device == "cuda" else device
        self._model  = YOLO(checkpoint_path)
        self.conf_threshold  = conf_threshold
        self.iou_threshold   = iou_threshold
        self.image_size      = image_size
        self.padding_px      = padding_px
        self.max_detections  = max_detections
        logger.info(
            "ROIDetector loaded: %s | conf=%.2f | iou=%.2f | device=%s",
            checkpoint_path, conf_threshold, iou_threshold, device,
        )

    def predict(
        self,
        image: np.ndarray,
        apply_padding: bool = True,
    ) -> List[Detection]:
        """
        Run detection on a single image.

        Parameters
        ----------
        image         : H×W greyscale or H×W×3 BGR numpy array.
        apply_padding : if True, expand detected boxes by ``padding_px``.

        Returns
        -------
        List[Detection]  Sorted by confidence (highest first).
        """
        # Convert greyscale → pseudo-RGB if needed
        if image.ndim == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        results = self._model.predict(
            source  = img_rgb,
            imgsz   = self.image_size,
            conf    = self.conf_threshold,
            iou     = self.iou_threshold,
            device  = self._device,
            max_det = self.max_detections,
            verbose = False,
        )

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()   # (N, 4)
            confs      = r.boxes.conf.cpu().numpy()   # (N,)
            classes    = r.boxes.cls.cpu().numpy()    # (N,)

            for (x1, y1, x2, y2), conf, cls in zip(boxes_xyxy, confs, classes):
                det = Detection(
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                    conf=float(conf), cls=int(cls),
                )
                if apply_padding:
                    det = det.pad(self.padding_px, img_w=w, img_h=h)
                detections.append(det)

        detections.sort(key=lambda d: d.conf, reverse=True)
        return detections

    def predict_batch(
        self,
        image_paths: List[str],
        apply_padding: bool = True,
    ) -> Dict[str, List[Detection]]:
        """
        Run detection on a list of image file paths.

        Parameters
        ----------
        image_paths   : list of image file paths.
        apply_padding : expand boxes by padding_px.

        Returns
        -------
        dict ``{image_path: List[Detection]}``.
        """
        results: Dict[str, List[Detection]] = {}
        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning("Cannot read: %s", path)
                results[path] = []
                continue
            results[path] = self.predict(img, apply_padding)
        return results

    def benchmark(
        self,
        image: np.ndarray,
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> dict:
        """
        Measure single-image inference latency statistics.

        Returns
        -------
        dict  mean_ms, std_ms, fps.
        """
        import statistics

        if image.ndim == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for _ in range(num_warmup):
            self._model.predict(source=img_rgb, imgsz=self.image_size,
                                device=self._device, verbose=False)

        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            self._model.predict(source=img_rgb, imgsz=self.image_size,
                                device=self._device, verbose=False)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        result = {
            "mean_ms":   statistics.mean(times),
            "std_ms":    statistics.stdev(times),
            "min_ms":    min(times),
            "max_ms":    max(times),
            "fps":       1000.0 / statistics.mean(times),
        }
        logger.info("Detector benchmark → %.2f FPS | %.3f ms/img",
                    result["fps"], result["mean_ms"])
        return result


# ── Convenience loader ─────────────────────────────────────────────────────────

def load_detector(cfg) -> ROIDetector:
    """
    Build a ROIDetector from config.

    Parameters
    ----------
    cfg : loaded detector config namespace.

    Returns
    -------
    ROIDetector  Ready for inference.
    """
    d = cfg.detector
    return ROIDetector(
        checkpoint_path = d.checkpoint,
        conf_threshold  = d.conf_threshold,
        iou_threshold   = d.iou_threshold,
        image_size      = d.image_size,
        device          = d.device,
        padding_px      = d.padding_px,
        max_detections  = d.max_detections,
    )
