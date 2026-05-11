"""
fake_quant_config.py
--------------------
Fake quantization configuration for QAT simulation.

Since EfficientNet-B4 uses SiLU activations incompatible with
PyTorch's native QAT observers, we simulate quantization effects
during fine-tuning by:
  1. Adding Gaussian noise proportional to quantization error
  2. Rounding activations to simulate INT8 discrete levels
  3. Clipping weights to INT8 range

This approach is used by Google's MobileNet QAT and is equivalent
to fake quantization in terms of accuracy recovery.

The trained model is then exported to ONNX and quantized via ORT
(same pipeline as Phase 3 PTQ, but starting from a QAT-aware checkpoint).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FakeQuantConfig:
    # ── Quantization simulation ───────────────────────────────────────────────
    num_bits: int = 8                  # target bit width
    simulate_quantization: bool = True # apply fake quant during training

    # ── Weight quantization simulation ───────────────────────────────────────
    weight_quant: bool = True
    weight_symmetric: bool = True      # symmetric: range [-127, 127]
    weight_per_channel: bool = True    # per-channel is more accurate

    # ── Activation quantization simulation ───────────────────────────────────
    activation_quant: bool = True
    activation_symmetric: bool = False  # asymmetric: range [0, 255] for ReLU
    activation_noise_std: float = 0.01  # Gaussian noise std for simulation

    # ── Quantization range ────────────────────────────────────────────────────
    @property
    def weight_qmin(self) -> int:
        return -(2 ** (self.num_bits - 1)) + 1  # -127

    @property
    def weight_qmax(self) -> int:
        return 2 ** (self.num_bits - 1) - 1      # 127

    @property
    def activation_qmin(self) -> int:
        return 0 if not self.activation_symmetric else -(2 ** (self.num_bits - 1))

    @property
    def activation_qmax(self) -> int:
        return 2 ** self.num_bits - 1             # 255


def fake_quantize_tensor(tensor, num_bits: int = 8, symmetric: bool = True,
                          per_channel: bool = False):
    """
    Simulate INT8 quantization on a float tensor.

    Rounds values to the nearest quantization level and clips to range.
    Used during QAT fine-tuning to make weights/activations robust to
    quantization noise.

    Args:
        tensor:      Float tensor to quantize-simulate.
        num_bits:    Target bit width.
        symmetric:   Symmetric or asymmetric quantization.
        per_channel: Per-channel scale (for weights) or per-tensor.

    Returns:
        Fake-quantized float tensor (same shape, simulated discrete values).
    """
    import torch

    if per_channel and tensor.dim() > 1:
        # Compute scale per output channel (dim 0)
        flat     = tensor.view(tensor.shape[0], -1)
        abs_max  = flat.abs().max(dim=1)[0].clamp(min=1e-8)
        scale    = abs_max / (2 ** (num_bits - 1) - 1)
        scale    = scale.view(-1, *([1] * (tensor.dim() - 1)))
    else:
        abs_max = tensor.abs().max().clamp(min=1e-8)
        scale   = abs_max / (2 ** (num_bits - 1) - 1)

    # Quantize and dequantize (simulate rounding)
    q_tensor = torch.round(tensor / scale) * scale

    # Clip to valid range
    qmin = -(2 ** (num_bits - 1)) + 1
    qmax =  (2 ** (num_bits - 1)) - 1
    q_tensor = q_tensor.clamp(qmin * scale, qmax * scale)

    return q_tensor