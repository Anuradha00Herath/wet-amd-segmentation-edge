"""
quant_config.py
---------------
Quantization configuration for Phase 3 PTQ via ONNX Runtime.

Supports:
  - Dynamic INT8 quantization (ONNXRuntime)
  - Static INT8 quantization (ONNXRuntime)
  - Backend selection: CPUExecutionProvider (edge / Colab)

Why ONNX Runtime instead of torch.quantization:
  - EfficientNet-B4 uses SiLU activations incompatible with PyTorch
    static quantization observers.
  - smp UNet++ has deeply nested modules that cannot be auto-fused.
  - ONNX Runtime's quantization tool handles arbitrary op graphs
    and has been validated to work with this model architecture.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QuantConfig:
    # ── ONNX export settings ─────────────────────────────────────────────────
    opset_version: int = 17          # ONNX opset (17 = PyTorch 2.0+ recommended)
    onnx_filename: str = "baseline_fp32.onnx"
    quant_filename: str = "baseline_int8.onnx"

    # ── Quantization type ────────────────────────────────────────────────────
    # "static"  – requires calibration data (better accuracy)
    # "dynamic" – no calibration needed (faster to apply)
    quant_type: str = "static"

    # ── Calibration settings (static only) ───────────────────────────────────
    calib_batches: int = 20          # number of batches for calibration
    calib_batch_size: int = 4

    # ── ONNX Runtime execution provider ─────────────────────────────────────
    # CPUExecutionProvider = edge / Colab CPU inference
    execution_providers: List[str] = field(
        default_factory=lambda: ["CPUExecutionProvider"]
    )

    # ── Quantization parameters ──────────────────────────────────────────────
    per_channel: bool = True         # per-channel weight quantization (more accurate)
    reduce_range: bool = False       # True for VNNI-capable CPUs

    # ── Experiment tracking ──────────────────────────────────────────────────
    experiment_name: str = "ptq_int8"
    compare_backends: List[str] = field(
        default_factory=lambda: ["static", "dynamic"]
    )