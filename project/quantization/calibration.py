"""
calibration.py
--------------
Calibration data pipeline for ONNX Runtime static INT8 quantization.

The calibration reader feeds representative images to the ONNX Runtime
quantization tool so it can compute activation ranges (min/max or
percentile) for each op in the graph.

Reuses:
  - utils.dataset.build_dataloaders  (no duplicate data loading code)
  - utils.config.Config
"""

import numpy as np
from typing import Iterator, List, Optional

from utils.config import Config
from utils.dataset import build_inference_loader
from utils.logger import get_logger

_log = get_logger(__name__)


class OCTCalibrationReader:
    """
    ONNX Runtime CalibrationDataReader for OCT images.

    Feeds numpy arrays (not torch tensors) to the ONNX Runtime
    quantization calibration API.

    Args:
        cfg:          Project Config.
        n_batches:    Number of batches to use for calibration.
        input_name:   ONNX model input node name (default 'input').
    """

    def __init__(
        self,
        cfg: Config,
        n_batches: Optional[int] = None,
        input_name: str = "input",
        batch_size: int = 1,
    ) -> None:
        self.cfg        = cfg
        self.n_batches  = n_batches or cfg.ptq_calib_batches
        self.input_name = input_name
        self.batch_size = batch_size
        self._data      = self._collect_calibration_data()
        self._index     = 0

        _log.info(
            f"CalibrationReader: {len(self._data)} batches, "
            f"input_name='{input_name}'"
        )

    def _collect_calibration_data(self) -> List[np.ndarray]:
        """Load calibration batches from the inference DataLoader."""
        loader = build_inference_loader(self.cfg, batch_size=self.batch_size)
        data   = []

        for i, (images, _) in enumerate(loader):
            if i >= self.n_batches:
                break
            # images: (B, 1, H, W) float32 tensor → numpy
            data.append(images.numpy().astype(np.float32))

        _log.info(f"Collected {len(data)} calibration batches.")
        return data

    def get_next(self) -> Optional[dict]:
        """Return next calibration batch or None when exhausted."""
        if self._index >= len(self._data):
            return None
        batch = {self.input_name: self._data[self._index]}
        self._index += 1
        return batch

    def rewind(self) -> None:
        """Reset iterator to the beginning."""
        self._index = 0


def get_calibration_loader(cfg: Config, batch_size: int = 4):
    """
    Return a standard DataLoader for calibration preview / debugging.
    The ONNX Runtime uses OCTCalibrationReader directly.
    """
    return build_inference_loader(cfg, batch_size=batch_size)