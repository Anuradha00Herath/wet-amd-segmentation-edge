"""
export_mobile_onnx.py
---------------------
Export and prepare ONNX models for ONNX Runtime Mobile deployment.

Pipeline:
  1. Validate existing ONNX files (from Phase 3/4)
  2. Optimize ONNX graph for mobile (constant folding, node fusion)
  3. Convert to ORT format (.ort) for ONNX Runtime Mobile
  4. Generate model metadata JSON for Android app
  5. Package deployment bundle

Reuses:
  - quantization.quant_utils  (create_ort_session, onnx_size_mb)
  - utils.config              (Config)
  - utils.logger              (get_logger)
"""

import json
import os
import shutil
from typing import Dict, List, Optional

import numpy as np

from utils.config import Config
from utils.logger import get_logger
from quantization.quant_utils import create_ort_session, onnx_size_mb

_log = get_logger(__name__)

# Models to deploy
DEPLOY_MODELS = {
    "fp32":       "baseline_fp32.onnx",
    "ptq_int8":   "baseline_int8_static.onnx",
    "qat_int8":   "qat_int8_static.onnx",
}


# --------------------------------------------------------------------------- #
#  ONNX graph optimisation for mobile                                          #
# --------------------------------------------------------------------------- #

def optimise_for_mobile(
    onnx_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Apply mobile-targeted ONNX graph optimisations:
      - Constant folding
      - Operator fusion
      - Dead node elimination

    Args:
        onnx_path:   Input ONNX path.
        output_path: Output path. Defaults to <name>_mobile_opt.onnx.

    Returns:
        Path to optimised ONNX.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("pip install onnxruntime")

    if output_path is None:
        base, ext = os.path.splitext(onnx_path)
        output_path = f"{base}_mobile_opt{ext}"

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath  = output_path

    _ = ort.InferenceSession(
        onnx_path, sess_options,
        providers=["CPUExecutionProvider"]
    )

    size_mb = onnx_size_mb(output_path)
    _log.info(f"Optimised → {output_path}  ({size_mb:.2f} MB)")
    return output_path


# --------------------------------------------------------------------------- #
#  Convert to ORT format for ONNX Runtime Mobile                              #
# --------------------------------------------------------------------------- #

def convert_to_ort_format(
    onnx_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Convert ONNX model to ORT flatbuffer format (.ort).

    The .ort format is required by ONNX Runtime Mobile (Android/iOS)
    as it avoids ONNX protobuf parsing overhead at runtime.

    Requires: pip install onnxruntime-tools  (or ort-nightly)

    Args:
        onnx_path:   Optimised ONNX path.
        output_path: Output .ort path.

    Returns:
        Path to .ort file.
    """
    if output_path is None:
        output_path = onnx_path.replace(".onnx", ".ort")

    try:
        from onnxruntime.tools import convert_onnx_models_to_ort  # type: ignore
        convert_onnx_models_to_ort.convert_onnx_models_to_ort(
            onnx_path,
            output_dir=os.path.dirname(output_path),
            optimization_style="Fixed",
        )
        _log.info(f"ORT format → {output_path}")
    except ImportError:
        _log.warning(
            "onnxruntime-tools not available. "
            "Copying .onnx as fallback (ORT Mobile can load .onnx too)."
        )
        shutil.copy(onnx_path, output_path.replace(".ort", ".onnx"))
        output_path = output_path.replace(".ort", ".onnx")

    return output_path


# --------------------------------------------------------------------------- #
#  Validate ONNX output shape                                                  #
# --------------------------------------------------------------------------- #

def validate_onnx_output(
    onnx_path: str,
    cfg: Config,
    num_classes: int = 6,
) -> bool:
    """
    Run a dummy inference and validate output shape and dtype.

    Args:
        onnx_path:   ONNX model path.
        cfg:         Config (for input shape).
        num_classes: Expected number of output classes.

    Returns:
        True if validation passes.
    """
    session    = create_ort_session(onnx_path)
    input_name = session.get_inputs()[0].name
    dummy      = np.random.randn(1, cfg.in_channels, *cfg.image_size).astype(np.float32)

    outputs = session.run(None, {input_name: dummy})
    logits  = outputs[0]

    expected_shape = (1, num_classes, *cfg.image_size)
    ok = logits.shape == expected_shape

    if ok:
        _log.info(f"✓ {os.path.basename(onnx_path)} output shape: {logits.shape}")
    else:
        _log.error(
            f"✗ {os.path.basename(onnx_path)} "
            f"expected {expected_shape}, got {logits.shape}"
        )
    return ok


# --------------------------------------------------------------------------- #
#  Model metadata for Android app                                              #
# --------------------------------------------------------------------------- #

def generate_model_metadata(
    cfg: Config,
    deploy_dir: str,
    num_classes: int = 6,
) -> str:
    """
    Generate model_metadata.json consumed by the Android app.

    Contains input/output specs, class names, colour map, and
    normalisation parameters.

    Args:
        cfg:         Config.
        deploy_dir:  Where to save the JSON.
        num_classes: Number of segmentation classes.

    Returns:
        Path to saved JSON file.
    """
    from models.baseline_model import CLASS_NAMES

    CLASS_COLORS_HEX = [
        "#000000",   # Background
        "#FF00FF",   # Retinal Layer
        "#FFFF00",   # PED
        "#FF0000",   # SRF
        "#0000FF",   # IRF
        "#00FF00",   # RPE
    ]

    metadata = {
        "model_info": {
            "task":        "segmentation",
            "num_classes": num_classes,
            "class_names": CLASS_NAMES,
            "class_colors_hex": CLASS_COLORS_HEX,
        },
        "input": {
            "name":        "input",
            "shape":       [1, cfg.in_channels, *cfg.image_size],
            "dtype":       "float32",
            "norm_mean":   list(cfg.norm_mean),
            "norm_std":    list(cfg.norm_std),
            "grayscale":   cfg.in_channels == 1,
        },
        "output": {
            "name":        "output",
            "shape":       [1, num_classes, *cfg.image_size],
            "dtype":       "float32",
            "postprocess": "argmax",
        },
        "models": {
            name: {
                "filename": fname,
                "size_mb":  round(
                    onnx_size_mb(os.path.join(cfg.checkpoints_dir, fname)), 2
                ) if os.path.exists(
                    os.path.join(cfg.checkpoints_dir, fname)
                ) else None,
            }
            for name, fname in DEPLOY_MODELS.items()
        },
    }

    os.makedirs(deploy_dir, exist_ok=True)
    path = os.path.join(deploy_dir, "model_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

    _log.info(f"Metadata JSON → {path}")
    return path


# --------------------------------------------------------------------------- #
#  Full deployment bundle                                                       #
# --------------------------------------------------------------------------- #

def build_deployment_bundle(
    cfg: Config,
    deploy_dir: str,
) -> Dict[str, str]:
    """
    Build a complete deployment bundle:
      - Optimised ONNX models
      - model_metadata.json
      - README_DEPLOYMENT.txt

    Returns:
        Dict mapping model_name → bundle file path.
    """
    os.makedirs(deploy_dir, exist_ok=True)
    bundle: Dict[str, str] = {}

    for model_name, filename in DEPLOY_MODELS.items():
        src = os.path.join(cfg.checkpoints_dir, filename)
        if not os.path.exists(src):
            _log.warning(f"Skipping {model_name}: {src} not found.")
            continue

        # Optimise
        opt_name = filename.replace(".onnx", "_mobile_opt.onnx")
        opt_path = os.path.join(deploy_dir, opt_name)
        optimise_for_mobile(src, opt_path)

        # Validate
        validate_onnx_output(opt_path, cfg)

        bundle[model_name] = opt_path

    # Metadata
    generate_model_metadata(cfg, deploy_dir)

    # README
    readme_path = os.path.join(deploy_dir, "README_DEPLOYMENT.txt")
    with open(readme_path, "w") as f:
        f.write(
            "OCT wetAMD Segmentation — Mobile Deployment Bundle\n"
            "====================================================\n\n"
            "Models included:\n"
            + "\n".join(
                f"  {k}: {os.path.basename(v)}" for k, v in bundle.items()
            )
            + "\n\nAndroid integration:\n"
            "  1. Copy *.onnx files to Android/app/src/main/assets/\n"
            "  2. Copy model_metadata.json to assets/\n"
            "  3. Build and run the Android app\n"
        )

    _log.info(f"Deployment bundle ready → {deploy_dir}")
    return bundle
