# OCT Lesion Segmentation — Phase 1: Baseline Benchmarking

Research project: lightweight OCT lesion segmentation for low-resource edge devices.

**Phase 1 goal:** Establish a complete benchmarking baseline for the full-image
UNet++ EfficientNet-B4 segmentation model before any ROI-guided or quantised variants
are introduced.

---

## Project Structure

```
oct_seg_phase1/
│
├── configs/
│   └── config.yaml                   ← All hyperparameters & paths
│
├── data/                             ← Dataset root (populated from Drive)
│   ├── images/
│   └── masks/
│
├── models/
│   └── model_loader.py               ← Build model, load checkpoint
│
├── utils/
│   ├── __init__.py                   ← Convenience re-exports
│   ├── config_loader.py              ← YAML config → SimpleNamespace
│   ├── dataset.py                    ← OCTDataset, transforms, DataLoaders
│   ├── visualizer.py                 ← Prediction overlays, plots
│   └── profiler.py                   ← Timer, memory, CPU/GPU snapshots
│
├── inference/
│   └── baseline_inference.py         ← Batch & single-image inference
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate_baseline.py          ← ⭐ Main entry point
│   ├── metrics_accuracy.py           ← Dice, IoU, PixelAcc, Sens, Spec
│   ├── metrics_compute.py            ← FLOPs, Params, Memory
│   ├── metrics_throughput.py         ← FPS, ms/frame, latency stats
│   ├── metrics_energy.py             ← pyRAPL / codecarbon wrappers
│   └── metrics_exporter.py           ← MetricsBundle → CSV
│
├── results/                          ← Auto-created at runtime
│   ├── metrics/
│   │   └── baseline_metrics.csv
│   ├── visualizations/
│   ├── predictions/
│   └── run.log
│
├── notebooks/
│   └── phase1_evaluation.ipynb       ← Colab-ready end-to-end notebook
│
└── install_dependencies.py           ← One-shot Colab installer
```

---

## Quick Start (Google Colab)

```python
# 1. Mount Drive and navigate to project
from google.colab import drive
drive.mount('/content/drive')
%cd /content/oct_seg_phase1

# 2. Install dependencies
!python install_dependencies.py

# 3. Add project to Python path
import sys; sys.path.insert(0, '/content/oct_seg_phase1')

# 4. Run evaluation
from utils.config_loader import load_config
from evaluation.evaluate_baseline import run_evaluation

cfg    = load_config('configs/config.yaml')
bundle = run_evaluation(cfg, experiment_name='baseline_full_image')
```

Or use the notebook: `notebooks/phase1_evaluation.ipynb`.

---

## Metrics Measured

| Category    | Metric                     | Module                    |
|-------------|----------------------------|---------------------------|
| Accuracy    | Dice Score (macro + per-class) | metrics_accuracy.py   |
| Accuracy    | IoU (macro + per-class)    | metrics_accuracy.py       |
| Accuracy    | Pixel Accuracy             | metrics_accuracy.py       |
| Accuracy    | Sensitivity / Specificity  | metrics_accuracy.py       |
| Compute     | GFLOPs                     | metrics_compute.py        |
| Compute     | Parameters (M)             | metrics_compute.py        |
| Compute     | Peak Memory (MB)           | metrics_compute.py        |
| Throughput  | FPS                        | metrics_throughput.py     |
| Throughput  | ms/frame (mean ± std)      | metrics_throughput.py     |
| Energy      | Power (W)                  | metrics_energy.py         |
| Energy      | Energy per frame (J/frame) | metrics_energy.py         |

---

## Energy Measurement in Colab

| Backend      | Access required        | Colab compatible? |
|--------------|------------------------|-------------------|
| `pyRAPL`     | /dev/cpu/*/msr (root)  | ❌ (VM restriction) |
| `codecarbon` | None (CPU util + TDP)  | ✅ (automatic fallback) |

Set `energy_backend: codecarbon` in `config.yaml` for Colab.
For publication results, run on bare-metal Linux with root access and set `energy_backend: rapl`.

---

## Recommended Implementation Order

1. Upload project to Drive: `MyDrive/FYP/oct_seg_phase1/`
2. Open `notebooks/phase1_evaluation.ipynb` in Colab
3. Run cells 1–6 (install + evaluate) — approx. 5–10 min
4. Results appear in `results/metrics/baseline_metrics.csv`

---

## Design Decisions

- **Separation of concerns:** Each metric category lives in its own module.
  Adding a new metric never touches other files.
- **Config-driven:** Swapping the model architecture, image size, or dataset
  path requires only a YAML edit — no code changes.
- **Graceful degradation:** Energy and profiling tools degrade to NaN rather
  than crash when hardware access is unavailable (Colab).
- **Phase-ready CSV schema:** `MetricsBundle` appends one row per experiment,
  so Phase 2 (ROI-guided) and Phase 3 (quantised) results land in the same
  CSV for direct comparison.
- **No code duplication:** `rgb_mask_to_class` and `mask_to_rgb` exist once in
  `utils/dataset.py` and are imported everywhere — consistent with the baseline
  notebook's logic.

---

## Extending to Phase 2 (ROI-Guided)

Phase 2 will introduce a YOLOv8n detector before the segmentation model:

```
OCT Image → YOLOv8n → Crop ROI → UNet++ (on ROI) → Expand mask
```

The evaluation pipeline is already prepared:
- `run_batch_inference` accepts any DataLoader → swap in ROI-cropped batches.
- `MetricsBundle.save_csv` appends rows → one CSV, multiple phases.
- `config.yaml` has an `inference.output_dir` that can be overridden per phase.
