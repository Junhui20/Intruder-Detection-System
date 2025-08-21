"""
Database layer for the Intruder Detection System.

This package contains the SQLite database management components:
- models.py: SQLite schemas and data models
- database_manager.py: Database operations and CRUD functionality
- migrations/: Schema updates and data migration scripts
"""

__version__ = "1.0.0"
__author__ = "Intruder Detection System Team"

# Import main classes for easy access
from .database_manager import DatabaseManager
from .models import (
    Device, 
    WhitelistEntry, 
    NotificationSettings, 
    DetectionLog, 
    SystemMetrics, 
    SystemConfig
)

__all__ = [
    'DatabaseManager',
    'Device',
    'WhitelistEntry', 
    'NotificationSettings',
    'DetectionLog',
    'SystemMetrics',
    'SystemConfig'
]
