"""
GPU Telemetry Probe

Reads GPU load and temperature through NVIDIA's own NVML bindings
(``nvidia-ml-py``), replacing GPUtil — whose last release was December 2018
and which shelled out to ``nvidia-smi`` to parse its text output.

Three call sites used to import GPUtil independently, each inside its own
try/except, and each with a slightly different idea of what to do when there
was no GPU. They share this module now, so "no NVIDIA GPU", "driver not
loaded" and "library not installed" are one answer in one place: ``None``.

Every function here is safe to call on a machine with no GPU at all.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pynvml  # provided by the nvidia-ml-py package

    _NVML_IMPORTED = True
except ImportError:
    _NVML_IMPORTED = False
    logger.debug("nvidia-ml-py not installed; GPU telemetry unavailable")

# NVML has to be initialised once per process, and it fails on machines with no
# NVIDIA driver — which is not an error worth logging on every sample.
_nvml_ready: Optional[bool] = None


def _handle(index: int = 0):
    """Returns an NVML device handle, or None when there is nothing to read."""
    global _nvml_ready

    if not _NVML_IMPORTED:
        return None

    if _nvml_ready is None:
        try:
            pynvml.nvmlInit()
            _nvml_ready = True
        except Exception as e:
            _nvml_ready = False
            logger.debug(f"NVML unavailable: {e}")

    if not _nvml_ready:
        return None

    try:
        if pynvml.nvmlDeviceGetCount() <= index:
            return None
        return pynvml.nvmlDeviceGetHandleByIndex(index)
    except Exception as e:
        logger.debug(f"Could not open GPU {index}: {e}")
        return None


def gpu_load_percent(index: int = 0) -> Optional[float]:
    """GPU utilisation as 0-100, or None when it cannot be read.

    GPUtil reported this as a 0-1 ``load`` that every call site multiplied by
    100. NVML reports whole percent directly, so the multiplication is gone —
    check the call sites if you are porting more of them.
    """
    handle = _handle(index)
    if handle is None:
        return None
    try:
        return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception as e:
        logger.debug(f"Could not read GPU utilisation: {e}")
        return None


def gpu_temperature_c(index: int = 0) -> Optional[float]:
    """GPU core temperature in Celsius, or None when it cannot be read."""
    handle = _handle(index)
    if handle is None:
        return None
    try:
        return float(
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        )
    except Exception as e:
        logger.debug(f"Could not read GPU temperature: {e}")
        return None


def gpu_memory_percent(index: int = 0) -> Optional[float]:
    """GPU memory in use as 0-100, or None when it cannot be read.

    GPUtil's ``memoryUtil`` was a 0-1 fraction; this is whole percent, for the
    same reason [gpu_load_percent] is.
    """
    handle = _handle(index)
    if handle is None:
        return None
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if not info.total:
            return None
        return info.used / info.total * 100.0
    except Exception as e:
        logger.debug(f"Could not read GPU memory: {e}")
        return None


def gpu_available() -> bool:
    """Whether GPU telemetry can be read at all."""
    return _handle() is not None
