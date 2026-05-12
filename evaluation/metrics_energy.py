"""
evaluation/metrics_energy.py
-----------------------------
Energy and power consumption measurement for inference benchmarking.

Two backends are supported:
  1. ``pyRAPL``     — Linux RAPL hardware counters (needs root / MSR access).
  2. ``codecarbon`` — Software estimation; works everywhere including Colab.

Colab limitations
-----------------
- Google Colab runs in a VM without MSR register access, so pyRAPL
  will raise a PermissionError on ``/dev/cpu/*/msr``.
- ``codecarbon`` uses CPU utilisation × TDP estimates, which is an
  approximation but is cross-platform and Colab-safe.
- GPU energy cannot be measured reliably without NVML root access in Colab;
  codecarbon provides a software estimate from GPU utilisation.
- For publication-quality energy numbers, run on bare-metal Linux with root.

Design decisions
----------------
- Both backends are wrapped in try/except so unavailability never crashes
  the evaluation pipeline — it simply logs a warning and returns NaN.
- The ``EnergyMeter`` context-manager interface is backend-agnostic:
  ``with EnergyMeter(backend) as meter: ...`` then ``meter.result()``.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Dict, Generator, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── pyRAPL backend ────────────────────────────────────────────────────────────

class RAPLMeter:
    """Thin wrapper around pyRAPL for CPU package energy."""

    def __init__(self) -> None:
        try:
            import pyRAPL
            pyRAPL.setup()
            self._meter = pyRAPL.Measurement("inference")
            self._available = True
            logger.info("pyRAPL initialised successfully.")
        except Exception as e:
            self._available = False
            logger.warning("pyRAPL unavailable (%s). Energy will be NaN.", e)

    def start(self) -> None:
        if self._available:
            self._meter.begin()

    def stop(self) -> None:
        if self._available:
            self._meter.end()

    def result(self) -> Dict[str, float]:
        if not self._available:
            return {"energy_J": float("nan"), "power_W": float("nan"),
                    "duration_s": float("nan"), "backend": "rapl_unavailable"}
        try:
            # pyRAPL reports energy in µJ; duration in µs
            energy_uJ  = sum(self._meter.result.pkg or [0])
            dur_us     = self._meter.result.time or 1
            energy_J   = energy_uJ * 1e-6
            duration_s = dur_us    * 1e-6
            power_W    = energy_J  / duration_s if duration_s > 0 else float("nan")
            return {
                "energy_J":   energy_J,
                "power_W":    power_W,
                "duration_s": duration_s,
                "backend":    "rapl",
            }
        except Exception as e:
            logger.error("pyRAPL result extraction failed: %s", e)
            return {"energy_J": float("nan"), "power_W": float("nan"),
                    "duration_s": float("nan"), "backend": "rapl_error"}


# ── codecarbon backend ────────────────────────────────────────────────────────

class CodeCarbonMeter:
    """Wrapper around codecarbon's EmissionsTracker for energy estimation."""

    def __init__(self) -> None:
        try:
            from codecarbon import EmissionsTracker
            self._tracker   = EmissionsTracker(
                log_level="error",          # suppress verbose output
                save_to_file=False,         # no CSV side-effect
                tracking_mode="process",    # process-level CPU tracking
            )
            self._available = True
            self._t0        = None
            logger.info("codecarbon tracker initialised.")
        except Exception as e:
            self._available = False
            logger.warning("codecarbon unavailable (%s). Energy will be NaN.", e)

    def start(self) -> None:
        if self._available:
            self._tracker.start()
            self._t0 = time.perf_counter()

    def stop(self) -> None:
        if self._available:
            self._tracker.stop()

    def result(self) -> Dict[str, float]:
        if not self._available:
            return {"energy_J": float("nan"), "power_W": float("nan"),
                    "duration_s": float("nan"), "backend": "codecarbon_unavailable"}
        try:
            emissions   = self._tracker._total_energy          # kWh
            energy_J    = emissions.kWh * 3_600_000            # kWh → J
            duration_s  = time.perf_counter() - (self._t0 or 0)
            power_W     = energy_J / duration_s if duration_s > 0 else float("nan")
            return {
                "energy_J":   energy_J,
                "power_W":    power_W,
                "duration_s": duration_s,
                "backend":    "codecarbon",
            }
        except Exception as e:
            logger.error("codecarbon result extraction failed: %s", e)
            return {"energy_J": float("nan"), "power_W": float("nan"),
                    "duration_s": float("nan"), "backend": "codecarbon_error"}


# ── Unified EnergyMeter ───────────────────────────────────────────────────────

_BACKENDS = {"rapl": RAPLMeter, "codecarbon": CodeCarbonMeter}


class EnergyMeter:
    """
    Backend-agnostic energy meter.

    Usage::

        meter = EnergyMeter(backend="codecarbon")
        meter.start()
        run_inference(...)
        meter.stop()
        result = meter.result()   # dict: energy_J, power_W, duration_s, backend

    Parameters
    ----------
    backend : "rapl" | "codecarbon" | "auto"
        "auto" tries rapl first, falls back to codecarbon.
    """

    def __init__(self, backend: str = "auto") -> None:
        if backend == "auto":
            rapl = RAPLMeter()
            self._impl = rapl if rapl._available else CodeCarbonMeter()
        elif backend in _BACKENDS:
            self._impl = _BACKENDS[backend]()
        else:
            raise ValueError(f"Unknown energy backend: {backend!r}. "
                             f"Choose from {list(_BACKENDS)} or 'auto'.")

    def start(self)  -> None: self._impl.start()
    def stop(self)   -> None: self._impl.stop()
    def result(self) -> dict: return self._impl.result()


# ── Convenience function ──────────────────────────────────────────────────────

def measure_inference_energy(
    model: nn.Module,
    data_loader,
    device: str = "cpu",
    backend: str = "auto",
    num_warmup_batches: int = 3,
) -> Dict[str, float]:
    """
    Run full inference on data_loader and measure energy consumption.

    Returns
    -------
    dict with keys: energy_J, power_W, duration_s, energy_per_frame_J,
                    backend, num_images.
    """
    model.eval()
    meter        = EnergyMeter(backend=backend)
    num_images   = 0

    # Warm-up (not measured)
    with torch.no_grad():
        for i, (imgs, _) in enumerate(data_loader):
            if i >= num_warmup_batches:
                break
            _ = model(imgs.to(device))

    # Measured pass
    meter.start()
    with torch.no_grad():
        for imgs, _ in data_loader:
            imgs = imgs.to(device)
            _    = model(imgs)
            num_images += imgs.size(0)
    meter.stop()

    result = meter.result()
    result["num_images"] = num_images
    energy_J = result.get("energy_J", float("nan"))
    result["energy_per_frame_J"] = (
        energy_J / num_images if num_images > 0 else float("nan")
    )
    logger.info(
        "Energy measurement → %.4f J total | %.6f J/frame | backend=%s",
        energy_J, result["energy_per_frame_J"], result["backend"],
    )
    return result
