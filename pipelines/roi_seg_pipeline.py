"""
pipelines/roi_seg_pipeline.py
------------------------------
Core ROI-guided segmentation pipeline (Phase 3).

Full pipeline per image
-----------------------
1.  YOLOv8n detects lesion ROI -> bounding box.
2.  Asymmetric padding applied (padding_px horizontal, padding_py vertical)
    at inference time — no retraining needed.
3.  Image cropped to padded rectangular ROI.
4.  Crop resized to segmentation model input size (must be divisible by 32).
5.  Segmentation model runs on crop only.
6.  Prediction resized back to original crop pixel dimensions.
7.  Prediction mapped into full-image canvas via coord_mapper.
8.  Timing recorded at each stage (detector / crop / seg / map).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pipelines.coord_mapper import map_roi_pred_to_full, compute_effective_padding

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of one ROI-guided segmentation inference."""
    full_pred:      np.ndarray
    roi_pred:       Optional[np.ndarray]     = None
    roi_coords:     Optional[Tuple[int,...]] = None
    detected:       bool                     = False
    fallback_used:  bool                     = False
    t_detector_ms:  float                    = 0.0
    t_crop_ms:      float                    = 0.0
    t_seg_ms:       float                    = 0.0
    t_map_ms:       float                    = 0.0

    @property
    def t_total_ms(self) -> float:
        return self.t_detector_ms + self.t_crop_ms + self.t_seg_ms + self.t_map_ms


class ROIPipeline:
    """End-to-end ROI-guided segmentation pipeline."""

    def __init__(self, cfg) -> None:
        from inference.roi_inference  import load_detector
        from segmentation.seg_wrapper import SegmentationWrapper

        self.cfg         = cfg
        self.detector    = load_detector_from_pipeline_cfg(cfg)
        self.seg_model   = SegmentationWrapper(cfg)
        self.num_classes = cfg.classes.num_classes

        logger.info(
            "ROIPipeline ready | pad_x=%dpx pad_y=%dpx | seg_input=%s | fallback=%s",
            cfg.roi.padding_px,
            getattr(cfg.roi, 'padding_py', cfg.roi.padding_px),
            cfg.roi.seg_input_size,
            cfg.roi.fallback_to_full,
        )

    def run(
        self,
        image: np.ndarray,
        padding_px: Optional[int] = None,
    ) -> PipelineResult:
        """
        Run the full ROI-guided pipeline on one greyscale OCT image.

        Parameters
        ----------
        image      : H x W uint8 greyscale numpy array.
        padding_px : horizontal padding override (vertical uses cfg.roi.padding_py).
                     If None, uses cfg.roi.padding_px.

        Returns
        -------
        PipelineResult
        """
        cfg            = self.cfg
        orig_h, orig_w = image.shape[:2]

        # Resolve padding — asymmetric: pad_px for left/right, pad_py for top/bottom
        pad_px = padding_px if padding_px is not None else cfg.roi.padding_px
        pad_py = getattr(cfg.roi, 'padding_py', pad_px)

        # ── Stage 1: Detection ────────────────────────────────────────────────
        t0         = time.perf_counter()
        detections = self.detector.predict(image, apply_padding=False)
        t_det      = (time.perf_counter() - t0) * 1000.0

        if not detections:
            return self._handle_no_detection(image, t_det, orig_h, orig_w)

        best = detections[0]

        # ── Stage 2: Asymmetric crop ──────────────────────────────────────────
        # More horizontal padding to capture wide retinal structures,
        # less vertical padding to avoid wasted vitreous/choroid area.
        t0  = time.perf_counter()
        cx1 = max(0,      best.x1 - pad_px)
        cy1 = max(0,      best.y1 - pad_py)
        cx2 = min(orig_w, best.x2 + pad_px)
        cy2 = min(orig_h, best.y2 + pad_py)

        # Crop the rectangular ROI from the input image (not a global variable)
        raw_crop = image[cy1:cy2, cx1:cx2]

        # Resize to seg model input size (must be divisible by 32)
        seg_size = cfg.roi.seg_input_size
        crop     = cv2.resize(raw_crop, (seg_size, seg_size),
                              interpolation=cv2.INTER_LINEAR)
        t_crop   = (time.perf_counter() - t0) * 1000.0

        # ── Stage 3: Segmentation on resized crop ─────────────────────────────
        t0       = time.perf_counter()
        roi_pred = self.seg_model.predict(crop, target_size=None)
        t_seg    = (time.perf_counter() - t0) * 1000.0

        # ── Stage 4: Resize prediction back → map to full image ───────────────
        t0          = time.perf_counter()
        orig_crop_h = cy2 - cy1
        orig_crop_w = cx2 - cx1

        # Resize from seg output size back to original crop pixel dimensions
        roi_pred_resized = cv2.resize(
            roi_pred.astype(np.uint8),
            (orig_crop_w, orig_crop_h),
            interpolation=cv2.INTER_NEAREST,   # preserve class ids — no blending
        )

        # Place into full-image canvas
        full_pred = map_roi_pred_to_full(
            roi_pred_resized, cx1, cy1, cx2, cy2, orig_h, orig_w
        )
        t_map = (time.perf_counter() - t0) * 1000.0

        return PipelineResult(
            full_pred     = full_pred,
            roi_pred      = roi_pred,
            roi_coords    = (cx1, cy1, cx2, cy2),
            detected      = True,
            fallback_used = False,
            t_detector_ms = t_det,
            t_crop_ms     = t_crop,
            t_seg_ms      = t_seg,
            t_map_ms      = t_map,
        )

    def run_batch(
        self,
        images: List[np.ndarray],
        padding_px: Optional[int] = None,
    ) -> List[PipelineResult]:
        """Run the pipeline on a list of images."""
        return [self.run(img, padding_px) for img in images]

    def _handle_no_detection(
        self,
        image: np.ndarray,
        t_det: float,
        orig_h: int,
        orig_w: int,
    ) -> PipelineResult:
        """Handle the no-detection case — fallback or return background mask."""
        if self.cfg.roi.fallback_to_full:
            logger.debug("No detection -> fallback to full-image segmentation.")
            t0    = time.perf_counter()
            pred  = self.seg_model.predict(image,
                                           target_size=self.cfg.roi.seg_input_size)
            t_seg = (time.perf_counter() - t0) * 1000.0
            if pred.shape != (orig_h, orig_w):
                pred = cv2.resize(pred, (orig_w, orig_h),
                                  interpolation=cv2.INTER_NEAREST)
            return PipelineResult(
                full_pred=pred, detected=False, fallback_used=True,
                t_detector_ms=t_det, t_seg_ms=t_seg,
            )
        else:
            logger.debug("No detection -> returning background mask.")
            return PipelineResult(
                full_pred=np.zeros((orig_h, orig_w), dtype=np.uint8),
                detected=False, fallback_used=False,
                t_detector_ms=t_det,
            )


def load_detector_from_pipeline_cfg(cfg):
    """Build ROIDetector from pipeline config. Padding applied in pipeline."""
    from inference.roi_inference import ROIDetector
    d = cfg.detector
    return ROIDetector(
        checkpoint_path = d.checkpoint,
        conf_threshold  = d.conf_threshold,
        iou_threshold   = d.iou_threshold,
        image_size      = d.image_size,
        device          = d.device,
        padding_px      = 0,            # padding applied in pipeline, not here
        max_detections  = d.max_detections,
    )