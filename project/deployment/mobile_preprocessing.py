"""
mobile_preprocessing.py
-----------------------
Preprocessing utilities matching the Android app's ImageProcessor.kt.

All preprocessing steps here must exactly mirror the Android Kotlin
implementation so Colab validation and device inference produce
identical inputs.

Steps:
  1. Load image (grayscale)
  2. Resize to model input size
  3. Normalise: (pixel/255 - mean) / std
  4. Add batch dimension
  5. Convert to float32 numpy array

Also provides:
  - Postprocessing: logits → class index mask → RGB overlay
  - Batch preprocessing for Colab validation runs
"""

from typing import Tuple

import cv2
import numpy as np

from utils.config import Config


# Colour map — must match ImageProcessor.kt CLASS_COLORS exactly
CLASS_COLORS_RGB = np.array([
    [0,   0,   0  ],  # 0 Background
    [255, 0,   255],  # 1 Retinal Layer
    [255, 255, 0  ],  # 2 PED
    [255, 0,   0  ],  # 3 SRF
    [0,   0,   255],  # 4 IRF
    [0,   255, 0  ],  # 5 RPE
], dtype=np.uint8)


def preprocess_image(
    image_path: str,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess an OCT image for ONNX inference.

    Mirrors ImageProcessor.kt::preprocess() exactly.

    Args:
        image_path: Path to OCT image file.
        cfg:        Config (image_size, norm_mean, norm_std).

    Returns:
        (input_tensor, original_image)
        input_tensor:   float32 (1, 1, H, W) normalised array
        original_image: uint8 (H, W) grayscale resized image
    """
    h, w = cfg.image_size

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalise: (pixel/255 - mean) / std
    mean = cfg.norm_mean[0]
    std  = cfg.norm_std[0]
    img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std

    # (H, W) → (1, 1, H, W)
    input_tensor = img_norm[np.newaxis, np.newaxis, :, :]

    return input_tensor, img_resized


def postprocess_output(
    logits: np.ndarray,
    original_image: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert ONNX logits to class mask and RGB overlay.

    Mirrors SegmentationVisualizer.kt::generateOverlay() exactly.

    Args:
        logits:         float32 (1, 6, H, W) raw model output.
        original_image: uint8 (H, W) grayscale image.
        alpha:          Overlay transparency (0=mask only, 1=image only).

    Returns:
        (class_mask, overlay_rgb)
        class_mask:  uint8 (H, W) class index map [0..5]
        overlay_rgb: uint8 (H, W, 3) blended overlay
    """
    # Argmax over class dimension
    class_mask = np.argmax(logits[0], axis=0).astype(np.uint8)  # (H, W)

    # Map class indices to RGB colours
    mask_rgb = CLASS_COLORS_RGB[class_mask]  # (H, W, 3)

    # Blend with original image
    img_rgb  = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    overlay  = cv2.addWeighted(img_rgb, alpha, mask_rgb, 1 - alpha, 0)

    return class_mask, overlay


def preprocess_from_array(
    image_array: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess from a numpy array (for Colab/validation use).

    Args:
        image_array: uint8 (H, W) or (H, W, C) grayscale image array.
        cfg:         Config.

    Returns:
        (input_tensor, resized_image) same as preprocess_image().
    """
    h, w = cfg.image_size

    if image_array.ndim == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    img_resized = cv2.resize(image_array, (w, h), interpolation=cv2.INTER_LINEAR)

    mean     = cfg.norm_mean[0]
    std      = cfg.norm_std[0]
    img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std

    return img_norm[np.newaxis, np.newaxis, :, :], img_resized
