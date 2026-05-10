"""
static_quant.py
---------------
Post-Training Static Quantization (PTQ) – Phase 2 placeholder.

Workflow (to be completed in Phase 2):
  1. Load float32 baseline model.
  2. Fuse eligible layers (Conv-BN-ReLU).
  3. Insert quantization observers.
  4. Run calibration (calibration.py::run_calibration).
  5. Convert to quantized model.
  6. Evaluate and save.

References:
  https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
"""

import torch
import torch.nn as nn

from utils.config import Config
from utils.logger import get_logger, save_checkpoint

_log = get_logger(__name__)

# Layers to fuse (Conv-BN-ReLU patterns) – update when architecture changes
_FUSE_MODULES = [
    ["block.0", "block.1", "block.2"],   # DoubleConv first conv
    ["block.3", "block.4", "block.5"],   # DoubleConv second conv
]


def fuse_model(model: nn.Module) -> nn.Module:
    """
    Fuse Conv-BN-ReLU sequences for quantization efficiency.

    Args:
        model: Float32 model.

    Returns:
        Fused model (in-place modification).

    Note:
        Fully implement in Phase 2 after confirming layer names
        with `print(dict(model.named_modules()))`.
    """
    _log.warning("fuse_model: placeholder – implement layer names in Phase 2.")
    # torch.quantization.fuse_modules(model, _FUSE_MODULES, inplace=True)
    return model


def apply_ptq(
    model: nn.Module,
    cfg: Config,
    calibration_loader,
    backend: str = "qnnpack",   # 'qnnpack' for ARM/mobile, 'fbgemm' for x86
    save_path: str = None,
) -> nn.Module:
    """
    Full PTQ pipeline: fuse → prepare → calibrate → convert → save.

    Args:
        model:              Float32 model.
        cfg:                Project Config.
        calibration_loader: DataLoader for calibration data.
        backend:            Quantization backend.
        save_path:          Where to save the quantized model.

    Returns:
        INT8 quantized model.

    Note:
        Placeholder – fully implement in Phase 2.
    """
    raise NotImplementedError(
        "PTQ implementation is scheduled for Phase 2. "
        "Scaffolding is ready in static_quant.py."
    )

    # --- Phase 2 implementation outline ---
    # torch.backends.quantized.engine = backend
    # model = fuse_model(model)
    # model.qconfig = torch.quantization.get_default_qconfig(backend)
    # torch.quantization.prepare(model, inplace=True)
    # run_calibration(model, calibration_loader, cfg)
    # torch.quantization.convert(model, inplace=True)
    # if save_path:
    #     save_checkpoint({"model_state_dict": model.state_dict()}, save_path)
    # return model
