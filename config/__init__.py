"""
Configuration management for the Intruder Detection System.

This package contains configuration modules:
- settings.py: System settings and thresholds
- camera_config.py: IP camera configurations
- detection_config.py: Detection parameters and models
"""

__version__ = "1.0.0"
__author__ = "Intruder Detection System Team"

from .settings import Settings
from .camera_config import CameraConfig
from .detection_config import DetectionConfig

__all__ = ['Settings', 'CameraConfig', 'DetectionConfig']
