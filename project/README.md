# OCT wetAMD Segmentation – Lightweight Research Framework

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A clean, modular, reproducible research framework for lightweight
OCT (Optical Coherence Tomography) wetAMD binary segmentation,
optimised for low-resource edge devices.

Designed for progressive quantization experiments:
**Float32 → PTQ (INT8) → QAT → Mixed Precision → ONNX deployment**

---

## Project Overview

| Property | Value |
|---|---|
| Task | Binary semantic segmentation (wetAMD lesion) |
| Input | Greyscale OCT images |
| Output | Binary segmentation mask |
| Baseline | Lightweight U-Net (`baseline.pth`) |
| Hardware target | CPU-first (edge devices / Google Colab) |
| Framework | PyTorch 2.0+ |

---

## Folder Structure

```
project/
│
├── data/
│   ├── images/          ← OCT images (PNG/JPG/TIF)
│   └── mask/            ← Binary masks (same filenames)
│
├── models/
│   ├── baseline_model.py   ← Lightweight U-Net definition
│   ├── model_loader.py     ← Registry, checkpoint I/O, inference wrapper
│   └── checkpoints/        ← Saved .pth files
│
├── quantization/           ← Phase 2–3 (scaffolded, not yet active)
│   ├── calibration.py      ← PTQ calibration pass
│   ├── static_quant.py     ← Post-Training Static Quantization
│   ├── qat.py              ← Quantization-Aware Training
│   ├── mixed_precision.py  ← Per-layer bit-width assignment
│   └── sensitivity_analysis.py ← Layer sensitivity scoring
│
├── evaluation/
│   ├── metrics.py          ← Dice, IoU, pixel acc, sensitivity, specificity
│   ├── benchmark.py        ← Latency, FPS, memory benchmarking
│   ├── energy_monitor.py   ← GPU/CPU power & energy monitoring
│   ├── profiler.py         ← torch.profiler wrapper
│   └── visualization.py    ← Prediction grids, training curves, comparisons
│
├── utils/
│   ├── config.py           ← Centralised Config dataclass
│   ├── dataset.py          ← OCTSegmentationDataset + DataLoader factory
│   ├── transforms.py       ← Train/val pipelines, optional Albumentations
│   ├── logger.py           ← Console/file logger, CSVLogger, checkpointing
│   ├── helpers.py          ← Device, FLOPs, parameters, timing
│   └── seed.py             ← set_seed(), reproducible DataLoader generator
│
├── notebooks/
│   ├── 00_colab_setup.ipynb        ← First-time Colab environment setup
│   └── 01_baseline_evaluation.ipynb ← Baseline evaluation walkthrough
│
├── results/
│   ├── csv/               ← Per-epoch and evaluation CSV logs
│   ├── plots/             ← Saved figures
│   └── logs/              ← .log files
│
├── main.py                ← Unified CLI entry point
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Local

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your data
#    data/images/  → OCT images
#    data/mask/    → Corresponding binary masks

# 5. Place the baseline checkpoint
cp /path/to/baseline.pth models/checkpoints/baseline.pth
```

### Google Colab

Open `notebooks/00_colab_setup.ipynb` and run all cells. It will:
- Mount Google Drive (optional)
- Clone the repository
- Install dependencies
- Verify the environment
- Help you copy `baseline.pth`

---

## Usage

### Train

```bash
python main.py train
python main.py train --epochs 100 --lr 0.0005 --batch_size 16
python main.py train --seed 123 --cpu     # force CPU
```

### Evaluate

```bash
python main.py evaluate
python main.py evaluate --checkpoint models/checkpoints/best_baseline.pth
```

### Benchmark

```bash
python main.py benchmark
```

### Profile (torch.profiler)

```bash
python main.py profile
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Dice | Segmentation overlap quality (primary metric) |
| IoU | Intersection over Union (Jaccard index) |
| Pixel Accuracy | Overall pixel-level correctness |
| Sensitivity | True Positive Rate (lesion detection) |
| Specificity | True Negative Rate (background rejection) |
| Precision | Positive Predictive Value |
| FPS | Frames per second at inference |
| Mean latency | Average inference time per image (ms) |
| FLOPs | Floating-point operations per forward pass |
| Parameters | Total model parameter count |
| Peak memory | Maximum memory during inference (MB) |
| Energy/inf | Millijoules per inference (GPU, via pynvml) |

---

## Training Workflow

```
set_seed(42)
    ↓
build_dataloaders()   [train 80% / val 20%, seeded split]
    ↓
get_model("baseline")
    ↓
Adam + ReduceLROnPlateau + BCEWithLogitsLoss
    ↓
Training loop → MetricAccumulator → CSVLogger → save_checkpoint
    ↓
plot_training_curves()
```

---

## Quantization Roadmap

| Phase | Method | Status |
|---|---|---|
| 1 | Float32 baseline training + evaluation | ✅ Active |
| 2 | PTQ (INT8, qnnpack / fbgemm) | 🔲 Scaffolded |
| 2 | QAT (INT8 fine-tuning) | 🔲 Scaffolded |
| 2 | ONNX export | 🔲 Scaffolded |
| 3 | Mixed precision (INT8 + INT4) | 🔲 Scaffolded |
| 3 | Layer sensitivity analysis | 🔲 Scaffolded |
| 4 | Energy profiling on target hardware | 🔲 Scaffolded |

---

## Experiment Reproducibility

Every experiment is fully reproducible via:

1. **Fixed seed** – `set_seed(cfg.seed)` sets Python, NumPy, PyTorch,
   CUDA, and cuDNN seeds. Change `cfg.seed` to reproduce different runs.
2. **Seeded DataLoader** – `get_generator(seed)` passed to `DataLoader`
   ensures batch ordering is deterministic.
3. **Seeded train/val split** – `_split_indices(n, n_val, seed)` in
   `dataset.py` uses Python's `random.Random(seed)` for reproducibility.
4. **cuDNN deterministic mode** – `torch.backends.cudnn.deterministic = True`.
5. **CSV logs** – every epoch's metrics are appended to `results/csv/`.
6. **Config dataclass** – saved inside every checkpoint under `"cfg"`.

To reproduce a run exactly:
```bash
python main.py train --seed 42 --epochs 50 --lr 0.001
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Config dataclass | Single source of truth for all hyper-parameters; avoids magic numbers scattered across files |
| Model registry (`_MODEL_REGISTRY`) | Add future quantized variants without changing caller code |
| `MetricAccumulator` | Correct batch-weighted running averages without storing all predictions in memory |
| `CSVLogger` | Simple, dependency-free experiment tracking; readable in Excel/pandas |
| Transforms separated from Dataset | Allows swapping Albumentations ↔ torchvision without touching Dataset code |
| BCEWithLogitsLoss | Numerically stable (fused sigmoid); model outputs raw logits |
| `ReduceLROnPlateau(mode="max")` | Monitors Dice (not loss) to reduce LR when the metric plateaus |
| Bilinear upsampling in decoder | More portable to ONNX/TFLite than transposed convolutions |

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{oct_wetamd_framework_2024,
  title  = {Lightweight OCT wetAMD Segmentation Research Framework},
  author = {Your Name},
  year   = {2024},
  url    = {https://github.com/YOUR_USERNAME/YOUR_REPO}
}
```
