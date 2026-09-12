"""
GPU Telemetry Probe

Reads GPU load, memory and temperature through NVIDIA's NVML bindings
(``nvidia-ml-py``). Every function returns ``None`` on a machine with no
NVIDIA GPU, no driver, or no bindings installed, so callers never need to
guard the import themselves.
"""

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pynvml  # provided by the nvidia-ml-py package

    _NVML_IMPORTED = True
except ImportError:
    _NVML_IMPORTED = False
    logger.debug("nvidia-ml-py not installed; GPU telemetry unavailable")


@lru_cache(maxsize=1)
def _nvml_ready() -> bool:
    """Initialise NVML once per process; False when there is no driver."""
    if not _NVML_IMPORTED:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception as e:
        logger.debug(f"NVML unavailable: {e}")
        return False


def _handle(index: int = 0):
    """Return the NVML handle for GPU ``index``, or None."""
    if not _nvml_ready():
        return None
    try:
        if pynvml.nvmlDeviceGetCount() <= index:
            return None
        return pynvml.nvmlDeviceGetHandleByIndex(index)
    except Exception as e:
        logger.debug(f"Could not open GPU {index}: {e}")
        return None


def gpu_load_percent(index: int = 0) -> Optional[float]:
    """
    Read GPU utilisation.

    Args:
        index: GPU index, 0 for the first device

    Returns:
        Utilisation as 0-100, or None when it cannot be read
    """
    handle = _handle(index)
    if handle is None:
        return None
    try:
        return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception as e:
        logger.debug(f"Could not read GPU utilisation: {e}")
        return None


def gpu_memory_percent(index: int = 0) -> Optional[float]:
    """
    Read GPU memory in use.

    Args:
        index: GPU index, 0 for the first device

    Returns:
        Memory in use as 0-100, or None when it cannot be read
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


def gpu_temperature_c(index: int = 0) -> Optional[float]:
    """
    Read GPU core temperature.

    Args:
        index: GPU index, 0 for the first device

    Returns:
        Temperature in Celsius, or None when it cannot be read
    """
    handle = _handle(index)
    if handle is None:
        return None
    try:
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return float(temp)
    except Exception as e:
        logger.debug(f"Could not read GPU temperature: {e}")
        return None


def gpu_available() -> bool:
    """
    Whether GPU telemetry can be read at all.

    Returns:
        True when an NVIDIA GPU with a working driver is present
    """
    return _handle() is not None
