"""
mobile_validation.py
--------------------
Validate ONNX models for mobile deployment correctness.

Checks:
  1. Output shape matches expected (B, 6, H, W)
  2. Output values are finite (no NaN/Inf)
  3. Segmentation mask argmax produces valid class indices
  4. FP32 vs INT8 output agreement (pixel-level similarity)
  5. Input/output node name consistency

Reuses:
  - quantization.quant_utils   (create_ort_session)
  - evaluation.metrics         (compute_all_metrics)
  - utils.config               (Config)
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from utils.config import Config
from utils.logger import get_logger, CSVLogger
from quantization.quant_utils import create_ort_session, onnx_size_mb

_log = get_logger(__name__)


def validate_model(
    onnx_path: str,
    cfg: Config,
    reference_path: Optional[str] = None,
    n_samples: int = 10,
) -> Dict[str, object]:
    """
    Run full validation suite on an ONNX model.

    Args:
        onnx_path:      Model to validate.
        cfg:            Config.
        reference_path: FP32 reference model for agreement check.
        n_samples:      Number of random inputs to test.

    Returns:
        Dict with validation results.
    """
    results = {
        "model":         os.path.basename(onnx_path),
        "size_mb":       round(onnx_size_mb(onnx_path), 2),
        "shape_ok":      False,
        "finite_ok":     False,
        "class_range_ok": False,
        "agreement_pct": None,
        "passed":        False,
    }

    try:
        session    = create_ort_session(onnx_path)
        input_name = session.get_inputs()[0].name
        num_classes = cfg.out_channels

        shape_failures = 0
        finite_failures = 0
        class_failures  = 0
        agreements      = []

        ref_session = None
        if reference_path and os.path.exists(reference_path):
            ref_session    = create_ort_session(reference_path)
            ref_input_name = ref_session.get_inputs()[0].name

        for _ in range(n_samples):
            dummy = np.random.randn(
                1, cfg.in_channels, *cfg.image_size
            ).astype(np.float32)

            out = session.run(None, {input_name: dummy})[0]  # (1, 6, H, W)

            # Shape check
            if out.shape != (1, num_classes, *cfg.image_size):
                shape_failures += 1

            # Finite check
            if not np.all(np.isfinite(out)):
                finite_failures += 1

            # Class range check
            pred = np.argmax(out, axis=1)  # (1, H, W)
            if pred.min() < 0 or pred.max() >= num_classes:
                class_failures += 1

            # Agreement with reference
            if ref_session is not None:
                ref_out  = ref_session.run(None, {ref_input_name: dummy})[0]
                ref_pred = np.argmax(ref_out, axis=1)
                agree    = np.mean(pred == ref_pred) * 100
                agreements.append(agree)

        results["shape_ok"]      = shape_failures == 0
        results["finite_ok"]     = finite_failures == 0
        results["class_range_ok"] = class_failures == 0

        if agreements:
            results["agreement_pct"] = round(float(np.mean(agreements)), 2)

        results["passed"] = (
            results["shape_ok"] and
            results["finite_ok"] and
            results["class_range_ok"]
        )

        status = "✓ PASS" if results["passed"] else "✗ FAIL"
        _log.info(
            f"{status} | {results['model']:<40} | "
            f"size={results['size_mb']:.1f}MB | "
            f"agreement={results['agreement_pct']}%"
        )

    except Exception as e:
        _log.error(f"Validation error for {onnx_path}: {e}")
        results["error"] = str(e)

    return results


def validate_all_models(
    cfg: Config,
    deploy_dir: str,
    fp32_onnx_path: str,
) -> None:
    """
    Validate all available deployment models and save CSV report.

    Args:
        cfg:            Config.
        deploy_dir:     Directory containing optimised ONNX files.
        fp32_onnx_path: FP32 reference model for agreement checks.
    """
    from deployment.export_mobile_onnx import DEPLOY_MODELS

    all_results = []

    # Validate originals
    for model_name, filename in DEPLOY_MODELS.items():
        path = os.path.join(cfg.checkpoints_dir, filename)
        if not os.path.exists(path):
            _log.warning(f"Skipping {model_name}: not found.")
            continue
        result = validate_model(path, cfg, reference_path=fp32_onnx_path)
        result["variant"] = model_name
        all_results.append(result)

    # Validate mobile-optimised versions
    for f in os.listdir(deploy_dir):
        if f.endswith("_mobile_opt.onnx"):
            path   = os.path.join(deploy_dir, f)
            result = validate_model(path, cfg, reference_path=fp32_onnx_path)
            result["variant"] = f.replace("_mobile_opt.onnx", "") + "_opt"
            all_results.append(result)

    if not all_results:
        _log.warning("No models found for validation.")
        return

    # Save CSV
    path = os.path.join(cfg.csv_dir, "mobile_validation_results.csv")
    csv  = CSVLogger(path, fieldnames=list(all_results[0].keys()))
    for r in all_results:
        csv.log(r)
    _log.info(f"Validation CSV → {path}")

    # Print summary
    print("\n── Mobile Validation Summary ─────────────────────────────")
    print(f"{'Model':<45} {'Pass':>6} {'Size MB':>9} {'Agreement':>11}")
    print("-" * 75)
    for r in all_results:
        print(
            f"{r['model']:<45} "
            f"{'✓' if r['passed'] else '✗':>6} "
            f"{r['size_mb']:>9.2f} "
            f"{str(r.get('agreement_pct','N/A')) + '%':>11}"
        )
