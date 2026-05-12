"""
pipelines/roi_seg_pipeline.py
------------------------------
Core ROI-guided segmentation pipeline (Phase 3).

Full pipeline per image
-----------------------
1.  YOLOv8n detects lesion ROI → bounding box.
2.  Padding applied (fixed px or % of bbox size) — at inference time,
    no retraining needed.
3.  Image cropped to padded ROI.
4.  Crop resized to segmentation model input size (e.g. 256×256).
5.  Segmentation model runs on crop only.
6.  Prediction resized back to crop pixel dimensions.
7.  Prediction mapped into full-image canvas via coord_mapper.
8.  Timing recorded at each stage (detector / crop / seg / map).

Fallback
--------
If no ROI is detected (or detector confidence < threshold), the pipeline
can either:
  a) Return an all-background mask (strict mode).
  b) Fall back to full-image segmentation (cfg.roi.fallback_to_full=true).

Design decisions
----------------
- The pipeline is stateless per call — ``ROIPipeline`` holds model
  references but no per-image state, so it is safe to call from loops.
- Stage timings are returned in the result dict so the benchmarking
  module can accumulate them without re-running.
- Phase 1 ``inference/roi_cropper.py`` is reused for all crop/pad logic.
- Phase 2 ``inference/roi_inference.py`` ROIDetector is reused as-is.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from inference.roi_cropper  import safe_pad_coords, crop_roi
from pipelines.coord_mapper import (
    map_roi_pred_to_full, compute_effective_padding,
    validate_coords, merge_roi_predictions,
)

logger = logging.getLogger(__name__)


# ── Per-image result ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Result of one ROI-guided segmentation inference.

    Attributes
    ----------
    full_pred        : H×W uint8 class map in original image space.
    roi_pred         : H'×W' uint8 class map in crop space (before mapping).
    roi_coords       : (x1,y1,x2,y2) padded crop coordinates.
    detected         : True if detector found at least one box.
    fallback_used    : True if full-image seg was used as fallback.
    t_detector_ms    : detector stage wall time (ms).
    t_crop_ms        : crop + resize stage wall time (ms).
    t_seg_ms         : segmentation forward pass wall time (ms).
    t_map_ms         : coordinate mapping stage wall time (ms).
    t_total_ms       : sum of all stages (ms).
    """
    full_pred:      np.ndarray
    roi_pred:       Optional[np.ndarray]        = None
    roi_coords:     Optional[Tuple[int,...]]    = None
    detected:       bool                        = False
    fallback_used:  bool                        = False
    t_detector_ms:  float                       = 0.0
    t_crop_ms:      float                       = 0.0
    t_seg_ms:       float                       = 0.0
    t_map_ms:       float                       = 0.0

    @property
    def t_total_ms(self) -> float:
        return self.t_detector_ms + self.t_crop_ms + self.t_seg_ms + self.t_map_ms


# ── Pipeline class ─────────────────────────────────────────────────────────────

class ROIPipeline:
    """
    End-to-end ROI-guided segmentation pipeline.

    Parameters
    ----------
    cfg : loaded pipeline config namespace (``roi_pipeline_config.yaml``).
    """

    def __init__(self, cfg) -> None:
        from inference.roi_inference   import load_detector
        from segmentation.seg_wrapper  import SegmentationWrapper

        self.cfg        = cfg
        self.detector   = load_detector_from_pipeline_cfg(cfg)
        self.seg_model  = SegmentationWrapper(cfg)
        self.num_classes= cfg.classes.num_classes

        logger.info(
            "ROIPipeline ready | padding=%dpx | seg_input=%d | fallback=%s",
            cfg.roi.padding_px, cfg.roi.seg_input_size, cfg.roi.fallback_to_full,
        )

    # ── Single image ───────────────────────────────────────────────────────────

    def run(
        self,
        image: np.ndarray,
        padding_px: Optional[int] = None,
    ) -> PipelineResult:
        """
        Run the full ROI-guided pipeline on one greyscale OCT image.

        Parameters
        ----------
        image      : H×W uint8 greyscale numpy array.
        padding_px : override padding for this call (for sweep experiments).
                     If None, uses cfg.roi.padding_px.

        Returns
        -------
        PipelineResult
        """
        cfg    = self.cfg
        orig_h, orig_w = image.shape[:2]
        pad_px = padding_px if padding_px is not None else cfg.roi.padding_px

        # ── Stage 1: Detection ────────────────────────────────────────────────
        t0 = time.perf_counter()
        detections = self.detector.predict(image, apply_padding=False)
        t_det = (time.perf_counter() - t0) * 1000.0

        if not detections:
            return self._handle_no_detection(image, t_det, orig_h, orig_w)

        # Use highest-confidence detection
        best = detections[0]

        # ── Stage 2: Crop ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        eff_pad = compute_effective_padding(
            best.x1, best.y1, best.x2, best.y2,
            padding_mode    = cfg.roi.padding_mode,
            padding_px      = pad_px,
            padding_percent = cfg.roi.padding_percent,
            img_w=orig_w, img_h=orig_h,
        )
        cx1, cy1, cx2, cy2 = safe_pad_coords(
            best.x1, best.y1, best.x2, best.y2, eff_pad, orig_w, orig_h
        )
        crop, _ = crop_roi(
            image, cx1, cy1, cx2, cy2,
            target_size       = cfg.roi.seg_input_size,
            keep_aspect_ratio = cfg.roi.keep_aspect_ratio,
            padding_px        = 0,   # already applied above
        )
        t_crop = (time.perf_counter() - t0) * 1000.0

        # ── Stage 3: Segmentation ─────────────────────────────────────────────
        t0 = time.perf_counter()
        roi_pred = self.seg_model.predict(crop, target_size=None)
        t_seg = (time.perf_counter() - t0) * 1000.0

        # ── Stage 4: Map back ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        full_pred = map_roi_pred_to_full(
            roi_pred, cx1, cy1, cx2, cy2, orig_h, orig_w
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

    # ── Batch ──────────────────────────────────────────────────────────────────

    def run_batch(
        self,
        images: List[np.ndarray],
        padding_px: Optional[int] = None,
    ) -> List[PipelineResult]:
        """
        Run the pipeline on a list of images.

        Parameters
        ----------
        images     : list of H×W uint8 greyscale arrays.
        padding_px : optional padding override.

        Returns
        -------
        List[PipelineResult]
        """
        return [self.run(img, padding_px) for img in images]

    # ── Fallback ───────────────────────────────────────────────────────────────

    def _handle_no_detection(
        self,
        image: np.ndarray,
        t_det: float,
        orig_h: int, orig_w: int,
    ) -> PipelineResult:
        """Handle the no-detection case."""
        if self.cfg.roi.fallback_to_full:
            logger.debug("No detection → fallback to full-image segmentation.")
            t0   = time.perf_counter()
            pred = self.seg_model.predict(image, target_size=self.cfg.roi.seg_input_size)
            t_seg= (time.perf_counter() - t0) * 1000.0
            # Resize back if seg model used different size
            if pred.shape != (orig_h, orig_w):
                pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            return PipelineResult(
                full_pred=pred, detected=False, fallback_used=True,
                t_detector_ms=t_det, t_seg_ms=t_seg,
            )
        else:
            logger.debug("No detection → returning background mask.")
            return PipelineResult(
                full_pred=np.zeros((orig_h, orig_w), dtype=np.uint8),
                detected=False, fallback_used=False,
                t_detector_ms=t_det,
            )


# ── Detector adaptor ──────────────────────────────────────────────────────────

def load_detector_from_pipeline_cfg(cfg):
    """
    Build a ROIDetector from the pipeline config's ``detector`` section.
    Adapts the pipeline cfg format to ROIDetector's constructor.
    """
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
