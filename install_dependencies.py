"""
install_dependencies.py
------------------------
One-shot dependency installer for Google Colab (Phase 1).

Run this once at the top of a Colab session before importing project modules.

    !python install_dependencies.py

Design notes
------------
- pyRAPL is included but its RAPL interface requires /dev/cpu/*/msr access
  which is NOT available in Colab's VM. Installation succeeds; runtime use
  gracefully falls back to codecarbon (see evaluation/metrics_energy.py).
- pynvml is listed for GPU memory profiling; no-op if no GPU is present.
- Packages are installed with --quiet to keep Colab output clean.
"""

import subprocess
import sys

PACKAGES = [
    # Deep learning core
    "torch",
    "torchvision",
    # Segmentation model zoo (UNet++, etc.)
    "segmentation-models-pytorch",
    # Data augmentation
    "albumentations",
    # YOLO (needed later; installed now to avoid session restart in Phase 2)
    "ultralytics",
    # Metrics
    "torchmetrics",
    # FLOPs counting
    "ptflops",
    "thop",
    # System profiling
    "psutil",
    "pynvml",
    # Data / IO
    "pandas",
    "pyyaml",
    "tqdm",
    # Visualisation
    "matplotlib",
    "opencv-python",
    # Energy measurement
    "pyRAPL",
    "codecarbon",
]

def install(packages: list) -> None:
    """Install a list of pip packages, logging success / failure per package."""
    for pkg in packages:
        print(f"  Installing {pkg} ...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", pkg],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("OK")
        else:
            print(f"FAILED\n    stderr: {result.stderr.strip()[:200]}")

if __name__ == "__main__":
    print("=" * 55)
    print("  OCT Lesion Seg — Phase 1 Dependency Installer")
    print("=" * 55)
    install(PACKAGES)
    print("\nAll packages processed. Restart runtime if prompted by Colab.")
