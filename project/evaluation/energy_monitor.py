"""
energy_monitor.py
-----------------
Energy and power profiling utilities.

Phase 1 provides the monitoring scaffolding. Full integration
with hardware power sensors (NVIDIA NVML, Intel RAPL, Jetson INA)
is added in later phases.

Backends supported (auto-detected):
  - NVML  : NVIDIA GPUs via pynvml
  - RAPL  : Intel CPUs via pyrapl (Linux only)
  - psutil: CPU utilisation proxy (always available as fallback)
"""

import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from utils.logger import get_logger

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  NVML backend (NVIDIA GPU power)                                             #
# --------------------------------------------------------------------------- #

def _nvml_available() -> bool:
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        return True
    except Exception:
        return False


def sample_gpu_power_w(device_index: int = 0) -> Optional[float]:
    """Return instantaneous GPU power in Watts, or None if unavailable."""
    try:
        import pynvml
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
        return mw / 1000.0
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  Polling energy monitor                                                       #
# --------------------------------------------------------------------------- #

class EnergyMonitor:
    """
    Background-thread power sampler.

    Usage (context manager):
        monitor = EnergyMonitor(interval_s=0.05)
        with monitor.track() as record:
            run_inference(model, data)
        print(record)  # {'duration_s': ..., 'energy_j': ..., 'mean_power_w': ...}
    """

    def __init__(self, interval_s: float = 0.05, device_index: int = 0) -> None:
        self.interval_s = interval_s
        self.device_index = device_index
        self._samples: list = []
        self._stop_event = threading.Event()

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            power = sample_gpu_power_w(self.device_index)
            if power is not None:
                self._samples.append(power)
            time.sleep(self.interval_s)

    @contextmanager
    def track(self) -> Generator[Dict[str, Any], None, None]:
        """Context manager that returns a result dict populated on exit."""
        self._samples = []
        self._stop_event.clear()

        result: Dict[str, Any] = {}

        thread = threading.Thread(target=self._poll, daemon=True)
        t0 = time.perf_counter()
        thread.start()

        try:
            yield result
        finally:
            duration = time.perf_counter() - t0
            self._stop_event.set()
            thread.join(timeout=2.0)

            if self._samples:
                mean_power = sum(self._samples) / len(self._samples)
                energy_j = mean_power * duration
            else:
                mean_power = 0.0
                energy_j = 0.0
                _log.warning(
                    "EnergyMonitor: no power samples collected. "
                    "Install pynvml for NVIDIA GPU power measurement."
                )

            result.update({
                "duration_s": duration,
                "mean_power_w": mean_power,
                "energy_j": energy_j,
                "energy_mj_per_inference": (energy_j * 1000) if duration > 0 else 0.0,
                "n_samples": len(self._samples),
            })


# --------------------------------------------------------------------------- #
#  RAPL backend (Intel CPU energy – Linux only)                                #
# --------------------------------------------------------------------------- #

def _rapl_available() -> bool:
    try:
        import pyrapl  # type: ignore
        return True
    except ImportError:
        return False


class RAPLMonitor:
    """
    Intel RAPL energy monitor for CPU-only benchmarks (Linux only).

    Placeholder – full implementation in Phase 2.
    Requires: pip install pyrapl
    """

    def __init__(self) -> None:
        if not _rapl_available():
            _log.warning("pyrapl not installed. RAPLMonitor will return zeros.")

    @contextmanager
    def track(self) -> Generator[Dict[str, Any], None, None]:
        result: Dict[str, Any] = {}
        try:
            import pyrapl
            meter = pyrapl.measurement.Measurement("rapl")
            meter.begin()
            yield result
            meter.end()
            result.update({
                "pkg_energy_j":  meter.result.pkg[0].energy if meter.result else 0.0,
                "dram_energy_j": meter.result.dram[0].energy if meter.result else 0.0,
            })
        except Exception as exc:
            _log.warning(f"RAPLMonitor error: {exc}")
            yield result
            result.update({"pkg_energy_j": 0.0, "dram_energy_j": 0.0})


# --------------------------------------------------------------------------- #
#  Summary helper                                                               #
# --------------------------------------------------------------------------- #

def print_energy_results(result: Dict[str, Any]) -> None:
    print("\n--- Energy Profile ---")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:.4f}")
        else:
            print(f"  {k:<35} {v}")
    print()
