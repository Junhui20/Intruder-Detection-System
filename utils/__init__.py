"""
Utility modules for the Intruder Detection System.

This package contains utility functions and helpers:
- image_processing.py: Image utilities and color detection
- gpu_optimization.py: CUDA/TensorRT setup and optimization
- performance_tracker.py: Metrics collection and analysis
- logger.py: Comprehensive logging system
"""

__version__ = "1.0.0"
__author__ = "Intruder Detection System Team"

from .image_processing import ImageProcessor
from .gpu_optimization import GPUOptimizer
from .performance_tracker import PerformanceTracker
from .logger import setup_logging, get_logger

__all__ = [
    'ImageProcessor',
    'GPUOptimizer', 
    'PerformanceTracker',
    'setup_logging',
    'get_logger'
]
