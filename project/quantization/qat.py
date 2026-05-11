"""
qat.py
------
End-to-end QAT pipeline for OCT wetAMD segmentation.

Why not torch.quantization.prepare_qat():
  EfficientNet-B4 uses SiLU activations which are incompatible with
  PyTorch's native QAT observers. Instead we use:
    1. Fake-quantization hooks on Conv2d weights during fine-tuning
    2. Export QAT-trained FP32 model to ONNX
    3. Apply ORT INT8 static quantization (same as Phase 3 PTQ)

  This produces a QAT INT8 model that can be directly compared with
  the PTQ INT8 model from Phase 3.

Pipeline:
  pretrained FP32 (baseline.pth)
      ↓  QATTrainer.train()      ← fake quant hooks
  QAT FP32 (qat_best.pth)
      ↓  export_qat_to_onnx()
  QAT FP32 ONNX (qat_fp32.onnx)
      ↓  quantize_static_onnx()  ← ORT calibration
  QAT INT8 ONNX (qat_int8.onnx)
"""

import os
from typing import Any, Dict, Optional

import torch

from utils.config import Config
from utils.logger import get_logger, CSVLogger
from utils.dataset import build_dataloaders
from models.model_loader import load_model

from quantization.fake_quant_config import FakeQuantConfig
from quantization.qat_scheduler import QATScheduleConfig
from quantization.qat_trainer import QATTrainer
from quantization.qat_utils import load_qat_checkpoint, export_qat_to_onnx
from quantization.quant_config import QuantConfig
from quantization.quant_utils import create_ort_session, onnx_size_mb, compression_ratio
from quantization.static_quant import quantize_static_onnx, benchmark_ort_session

_log = get_logger(__name__)


class QATPipeline:
    """
    Full QAT pipeline:
      1. Load pretrained FP32 baseline
      2. QAT fine-tune with fake-quant hooks
      3. Export QAT model to ONNX
      4. Quantize QAT ONNX to INT8 via ORT
      5. Benchmark and evaluate
      6. Save CSV comparison report

    Usage:
        pipeline = QATPipeline(cfg, fq_cfg, schedule_cfg, qcfg, device)
        results  = pipeline.run()
    """

    def __init__(
        self,
        cfg: Config,
        fq_cfg: FakeQuantConfig,
        schedule_cfg: QATScheduleConfig,
        qcfg: QuantConfig,
        device: torch.device,
    ) -> None:
        self.cfg          = cfg
        self.fq_cfg       = fq_cfg
        self.schedule_cfg = schedule_cfg
        self.qcfg         = qcfg
        self.device       = device

        self.qat_onnx_path  = os.path.join(cfg.checkpoints_dir, "qat_fp32.onnx")
        self.qat_int8_path  = os.path.join(cfg.checkpoints_dir, "qat_int8_static.onnx")

    def run(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        # ── Step 1: Load pretrained FP32 model ──────────────────────────────
        _log.info("Step 1: Loading pretrained FP32 baseline ...")
        model = load_model(self.cfg, device=self.device)

        # ── Step 2: Build data loaders ───────────────────────────────────────
        _log.info("Step 2: Building data loaders ...")
        train_loader, val_loader, test_loader = build_dataloaders(self.cfg)

        # ── Step 3: QAT fine-tuning ──────────────────────────────────────────
        _log.info("Step 3: QAT fine-tuning ...")
        trainer = QATTrainer(
            model, self.cfg, self.fq_cfg,
            self.schedule_cfg, self.device,
        )
        model, history = trainer.train(train_loader, val_loader, label="qat")
        results["history"] = history

        # ── Step 4: Load best QAT checkpoint ────────────────────────────────
        _log.info("Step 4: Loading best QAT checkpoint ...")
        model = load_qat_checkpoint(model, self.cfg, label="qat", device=self.device)

        # ── Step 5: Export QAT model to ONNX ────────────────────────────────
        _log.info("Step 5: Exporting QAT model to ONNX ...")
        self.qat_onnx_path = export_qat_to_onnx(
            model, self.cfg, filename="qat_fp32.onnx"
        )
        results["qat_onnx_size_mb"] = onnx_size_mb(self.qat_onnx_path)

        # ── Step 6: INT8 quantization via ORT ───────────────────────────────
        _log.info("Step 6: Quantizing QAT ONNX to INT8 ...")
        self.qat_int8_path = quantize_static_onnx(
            self.qat_onnx_path, self.qat_int8_path,
            self.cfg, self.qcfg,
        )
        results["qat_int8_size_mb"] = onnx_size_mb(self.qat_int8_path)
        results["qat_compression"]  = compression_ratio(
            results["qat_onnx_size_mb"], results["qat_int8_size_mb"]
        )

        # ── Step 7: Benchmark QAT INT8 ──────────────────────────────────────
        _log.info("Step 7: Benchmarking QAT INT8 model ...")
        qat_session   = create_ort_session(self.qat_int8_path)
        qat_input     = qat_session.get_inputs()[0].name
        results["qat_latency"] = benchmark_ort_session(
            qat_session, self.cfg,
            n_runs=self.cfg.benchmark_runs,
            warmup=self.cfg.benchmark_warmup,
            input_name=qat_input,
        )

        # ── Step 8: Save CSV summary ─────────────────────────────────────────
        self._save_csv(results)
        return results

    def _save_csv(self, results: Dict[str, Any]) -> None:
        row = {
            "model":       "qat_int8",
            "size_mb":     results.get("qat_int8_size_mb", 0),
            "compression": results.get("qat_compression",  0),
            "mean_ms":     results["qat_latency"].get("mean_ms", 0),
            "fps":         results["qat_latency"].get("fps",      0),
        }
        path    = os.path.join(self.cfg.csv_dir, "qat_results.csv")
        csv_log = CSVLogger(path, fieldnames=list(row.keys()))
        csv_log.log(row)
        _log.info(f"QAT results CSV → {path}")