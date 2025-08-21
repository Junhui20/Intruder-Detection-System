"""
Core detection modules for the Intruder Detection System.

This package contains the main detection engines and processing components:
- detection_engine.py: YOLO11n integration with person tracking
- face_recognition.py: Unified face recognition system with multiple backends (face_recognition, OpenCV DNN+LBPH, basic OpenCV)
- animal_recognition.py: Individual pet identification with hybrid approach
- camera_manager.py: IP camera and local camera handling
- notification_system.py: Bidirectional Telegram bot integration
- model_optimization.py: TensorRT and quantization optimization
- multi_camera_manager.py: Simultaneous multi-camera handling
- performance_optimizer.py: System performance optimization
- error_recovery.py: Error handling and recovery system
"""

__version__ = "1.0.0"
__author__ = "Intruder Detection System Team"

# Import main classes for easy access
from .detection_engine import DetectionEngine
from .face_recognition import FaceRecognitionSystem
from .animal_recognition import AnimalRecognitionSystem
from .camera_manager import CameraManager
from .notification_system import NotificationSystem

__all__ = [
    'DetectionEngine',
    'FaceRecognitionSystem', 
    'AnimalRecognitionSystem',
    'CameraManager',
    'NotificationSystem'
]
