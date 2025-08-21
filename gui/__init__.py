"""
GUI modules for the Intruder Detection System.

This package contains the modern GUI interface components:
- main_window.py: Main dashboard with 5 modules
- detection_view.py: Real-time detection with controls
- ip_camera_manager.py: Network camera configuration
- entity_management.py: Human and animal registration
- notification_center.py: Telegram user management
- performance_monitor.py: Real-time metrics display
"""

__version__ = "1.0.0"
__author__ = "Intruder Detection System Team"

# Import main classes for easy access
from .main_window import MainWindow
from .detection_view import DetectionView
from .ip_camera_manager import IPCameraManager
from .entity_management import EntityManagement
from .notification_center import NotificationCenter
from .performance_monitor import PerformanceMonitor

__all__ = [
    'MainWindow',
    'DetectionView',
    'IPCameraManager',
    'EntityManagement',
    'NotificationCenter',
    'PerformanceMonitor'
]
