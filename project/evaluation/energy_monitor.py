"""
energy_monitor.py
-----------------
Energy and power profiling for Phase 2 benchmarking.

Backends (auto-detected in priority order):
  1. CodeCarbon  – cross-platform, Colab-compatible, CO₂ + energy
  2. pynvml      – NVIDIA GPU power (Watts) via polling thread
  3. pyRAPL      – Intel CPU RAPL (Linux only)
  4. psutil      – CPU % proxy (always available as fallback)

All backends share the same context-manager interface so the rest of
the codebase never needs to know which backend is active.
"""

import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from utils.logger import get_logger

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Backend detection                                                            #
# --------------------------------------------------------------------------- #

def _has_codecarbon() -> bool:
    try:
        import codecarbon  # type: ignore
        return True
    except ImportError:
        return False


def _has_nvml() -> bool:
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        return True
    except Exception:
        return False


def _has_rapl() -> bool:
    try:
        import pyrapl  # type: ignore
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
#  CodeCarbon backend                                                          #
# --------------------------------------------------------------------------- #

class CodeCarbonMonitor:
    """
    Energy monitor using CodeCarbon (recommended for Colab).

    Install: pip install codecarbon

    Tracks CPU + GPU + DRAM energy in kWh and converts to Joules.
    Also reports estimated CO₂ equivalent.
    """

    @contextmanager
    def track(self, label: str = "inference") -> Generator[Dict[str, Any], None, None]:
        result: Dict[str, Any] = {}
        try:
            from codecarbon import EmissionsTracker  # type: ignore
            tracker = EmissionsTracker(
                project_name=label,
                log_level="error",
                save_to_file=False,
                allow_multiple_runs=True,
            )
            t0 = time.perf_counter()
            tracker.start()
            yield result
            emissions = tracker.stop()           # kgCO₂eq
            duration  = time.perf_counter() - t0

            # codecarbon stores energy in kWh on the tracker object
            energy_kwh = getattr(tracker, "_total_energy", None)
            energy_j   = (energy_kwh * 3_600_000) if energy_kwh else 0.0

            result.update({
                "backend":        "codecarbon",
                "duration_s":     duration,
                "energy_j":       energy_j,
                "energy_kwh":     energy_kwh or 0.0,
                "co2_kg":         emissions or 0.0,
                "mean_power_w":   energy_j / duration if duration > 0 else 0.0,
                "energy_mj_per_inference": (energy_j * 1000) if duration > 0 else 0.0,
            })
        except Exception as exc:
            _log.warning(f"CodeCarbonMonitor error: {exc}")
            yield result
            result.setdefault("backend", "codecarbon_failed")
            result.setdefault("energy_j", 0.0)


# --------------------------------------------------------------------------- #
#  NVML polling backend                                                        #
# --------------------------------------------------------------------------- #

def _sample_gpu_power_w(device_index: int = 0) -> Optional[float]:
    try:
        import pynvml
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0   # mW → W
    except Exception:
        return None


class NVMLMonitor:
    """
    Background-thread GPU power sampler using pynvml.
    Install: pip install pynvml
    """

    def __init__(self, interval_s: float = 0.05, device_index: int = 0) -> None:
        self.interval_s   = interval_s
        self.device_index = device_index
        self._samples: List[float] = []
        self._stop = threading.Event()

    def _poll(self) -> None:
        while not self._stop.is_set():
            w = _sample_gpu_power_w(self.device_index)
            if w is not None:
                self._samples.append(w)
            time.sleep(self.interval_s)

    @contextmanager
    def track(self, label: str = "inference") -> Generator[Dict[str, Any], None, None]:
        self._samples = []
        self._stop.clear()
        result: Dict[str, Any] = {}

        thread = threading.Thread(target=self._poll, daemon=True)
        t0 = time.perf_counter()
        thread.start()

        try:
            yield result
        finally:
            duration = time.perf_counter() - t0
            self._stop.set()
            thread.join(timeout=2.0)

            mean_power = (sum(self._samples) / len(self._samples)) if self._samples else 0.0
            energy_j   = mean_power * duration

            result.update({
                "backend":        "nvml",
                "duration_s":     duration,
                "mean_power_w":   mean_power,
                "energy_j":       energy_j,
                "energy_mj_per_inference": energy_j * 1000,
                "n_samples":      len(self._samples),
            })


# --------------------------------------------------------------------------- #
#  psutil CPU proxy (fallback)                                                 #
# --------------------------------------------------------------------------- #

class PSUtilMonitor:
    """
    CPU utilisation proxy using psutil.
    Always available — no extra install required beyond psutil.

    Note: Does NOT measure true power/energy; estimates from CPU %.
    """

    @contextmanager
    def track(self, label: str = "inference") -> Generator[Dict[str, Any], None, None]:
        result: Dict[str, Any] = {}
        try:
            import psutil
            process = psutil.Process(os.getpid())
            process.cpu_percent(interval=None)   # first call baseline
            t0 = time.perf_counter()
            yield result
            duration = time.perf_counter() - t0
            cpu_pct  = process.cpu_percent(interval=None)

            result.update({
                "backend":        "psutil",
                "duration_s":     duration,
                "cpu_pct":        cpu_pct,
                "mean_power_w":   0.0,    # not measurable via psutil alone
                "energy_j":       0.0,
                "energy_mj_per_inference": 0.0,
                "note": "Install codecarbon or pynvml for real energy measurements.",
            })
        except ImportError:
            t0 = time.perf_counter()
            yield result
            result.update({
                "backend": "none",
                "duration_s": time.perf_counter() - t0,
                "energy_j": 0.0,
            })


# --------------------------------------------------------------------------- #
#  Auto-selecting EnergyMonitor                                                #
# --------------------------------------------------------------------------- #

class EnergyMonitor:
    """
    Auto-selects the best available energy monitoring backend.

    Priority: CodeCarbon > NVML > psutil

    Usage:
        monitor = EnergyMonitor()
        with monitor.track("baseline_inference") as record:
            run_inference(model, data)
        print(record)
    """

    def __init__(self, prefer_codecarbon: bool = True) -> None:
        if prefer_codecarbon and _has_codecarbon():
            self._backend = CodeCarbonMonitor()
            self._backend_name = "codecarbon"
        elif _has_nvml():
            self._backend = NVMLMonitor()
            self._backend_name = "nvml"
        else:
            self._backend = PSUtilMonitor()
            self._backend_name = "psutil"

        _log.info(f"EnergyMonitor backend: {self._backend_name}")

    @contextmanager
    def track(self, label: str = "inference") -> Generator[Dict[str, Any], None, None]:
        with self._backend.track(label=label) as result:
            yield result

    @property
    def backend_name(self) -> str:
        return self._backend_name


def print_energy_results(result: Dict[str, Any]) -> None:
    """Pretty-print energy monitoring results."""
    print("\n--- Energy Profile ---")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:.6f}")
        else:
            print(f"  {k:<35} {v}")
    print()