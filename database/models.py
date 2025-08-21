"""
SQLite Data Models for the Intruder Detection System

This module defines the data models and schemas for the SQLite database,
migrated from MariaDB for better performance and simplicity.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
import json


@dataclass
class Device:
    """
    IP Camera device model.
    
    Represents network cameras with HTTP/HTTPS support and configuration.
    """
    id: Optional[int] = None
    ip_address: str = ""
    port: int = 8080
    use_https: bool = False
    end_with_video: bool = False
    status: str = "active"  # 'active' or 'inactive'
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations."""
        return {
            'id': self.id,
            'ip_address': self.ip_address,
            'port': self.port,
            'use_https': self.use_https,
            'end_with_video': self.end_with_video,
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Device':
        """Create instance from dictionary."""
        return cls(
            id=data.get('id'),
            ip_address=data.get('ip_address', ''),
            port=data.get('port', 8080),
            use_https=bool(data.get('use_https', False)),
            end_with_video=bool(data.get('end_with_video', False)),
            status=data.get('status', 'active'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )


@dataclass
class WhitelistEntry:
    """
    Whitelist entry for known humans and animals.
    
    Supports individual pet identification with hybrid recognition methods.
    """
    id: Optional[int] = None
    name: str = ""
    entity_type: str = "human"  # 'human' or 'animal'
    familiar: str = "familiar"  # 'familiar' or 'unfamiliar'
    color: Optional[str] = None
    coco_class_id: Optional[int] = None
    image_path: str = ""
    confidence_threshold: float = 0.6
    pet_breed: Optional[str] = None
    individual_id: Optional[str] = None  # For specific pet identification (e.g., 'jacky')
    face_encodings: Optional[bytes] = None  # Stored face encodings
    multiple_photos: Optional[str] = None  # JSON array of additional photo paths
    identification_method: str = "color"  # 'color', 'face', or 'hybrid'
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations."""
        return {
            'id': self.id,
            'name': self.name,
            'entity_type': self.entity_type,
            'familiar': self.familiar,
            'color': self.color,
            'coco_class_id': self.coco_class_id,
            'image_path': self.image_path,
            'confidence_threshold': self.confidence_threshold,
            'pet_breed': self.pet_breed,
            'individual_id': self.individual_id,
            'face_encodings': self.face_encodings,
            'multiple_photos': self.multiple_photos,
            'identification_method': self.identification_method,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WhitelistEntry':
        """Create instance from dictionary."""
        from datetime import datetime

        # Parse datetime strings from database
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                created_at = None

        updated_at = data.get('updated_at')
        if isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                updated_at = None

        return cls(
            id=data.get('id'),
            name=data.get('name', ''),
            entity_type=data.get('entity_type', 'human'),
            familiar=data.get('familiar', 'familiar'),
            color=data.get('color'),
            coco_class_id=data.get('coco_class_id'),
            image_path=data.get('image_path', ''),
            confidence_threshold=data.get('confidence_threshold', 0.6),
            pet_breed=data.get('pet_breed'),
            individual_id=data.get('individual_id'),
            face_encodings=data.get('face_encodings'),
            multiple_photos=data.get('multiple_photos'),
            identification_method=data.get('identification_method', 'color'),
            created_at=created_at,
            updated_at=updated_at
        )
    
    def get_additional_photos(self) -> List[str]:
        """Get list of additional photo paths."""
        if self.multiple_photos:
            try:
                return json.loads(self.multiple_photos)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_additional_photos(self, photo_paths: List[str]):
        """Set additional photo paths."""
        self.multiple_photos = json.dumps(photo_paths) if photo_paths else None


@dataclass
class NotificationSettings:
    """
    Telegram notification settings for users.
    
    Manages individual user preferences and permissions.
    """
    id: Optional[int] = None
    chat_id: int = 0
    telegram_username: str = ""
    notify_human_detection: bool = True
    notify_animal_detection: bool = True
    sendstatus: str = "open"  # 'open' or 'close'
    last_notification: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations."""
        return {
            'id': self.id,
            'chat_id': self.chat_id,
            'telegram_username': self.telegram_username,
            'notify_human_detection': self.notify_human_detection,
            'notify_animal_detection': self.notify_animal_detection,
            'sendstatus': self.sendstatus,
            'last_notification': self.last_notification,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NotificationSettings':
        """Create instance from dictionary."""
        return cls(
            id=data.get('id'),
            chat_id=data.get('chat_id', 0),
            telegram_username=data.get('telegram_username', ''),
            notify_human_detection=bool(data.get('notify_human_detection', True)),
            notify_animal_detection=bool(data.get('notify_animal_detection', True)),
            sendstatus=data.get('sendstatus', 'open'),
            last_notification=data.get('last_notification'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )


@dataclass
class DetectionLog:
    """
    Detection log entry for tracking system activity.
    
    Records all detection events for analysis and monitoring.
    """
    id: Optional[int] = None
    detection_type: str = "human"  # 'human' or 'animal'
    entity_name: Optional[str] = None
    confidence: Optional[float] = None
    camera_id: Optional[int] = None
    image_path: Optional[str] = None
    notification_sent: bool = False
    detected_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations."""
        return {
            'id': self.id,
            'detection_type': self.detection_type,
            'entity_name': self.entity_name,
            'confidence': self.confidence,
            'camera_id': self.camera_id,
            'image_path': self.image_path,
            'notification_sent': self.notification_sent,
            'detected_at': self.detected_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionLog':
        """Create instance from dictionary."""
        return cls(
            id=data.get('id'),
            detection_type=data.get('detection_type', 'human'),
            entity_name=data.get('entity_name'),
            confidence=data.get('confidence'),
            camera_id=data.get('camera_id'),
            image_path=data.get('image_path'),
            notification_sent=bool(data.get('notification_sent', False)),
            detected_at=data.get('detected_at')
        )


@dataclass
class SystemMetrics:
    """
    System performance metrics for monitoring.
    
    Tracks performance data for analysis and optimization.
    """
    id: Optional[int] = None
    metric_type: str = ""
    metric_value: float = 0.0
    unit: Optional[str] = None
    recorded_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations."""
        return {
            'id': self.id,
            'metric_type': self.metric_type,
            'metric_value': self.metric_value,
            'unit': self.unit,
            'recorded_at': self.recorded_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemMetrics':
        """Create instance from dictionary."""
        return cls(
            id=data.get('id'),
            metric_type=data.get('metric_type', ''),
            metric_value=data.get('metric_value', 0.0),
            unit=data.get('unit'),
            recorded_at=data.get('recorded_at')
        )


@dataclass
class SystemConfig:
    """
    System configuration settings.
    
    Stores configurable system parameters and user preferences.
    """
    id: Optional[int] = None
    config_key: str = ""
    config_value: str = ""
    config_type: str = "string"  # 'string', 'integer', 'float', 'boolean'
    description: Optional[str] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations."""
        return {
            'id': self.id,
            'config_key': self.config_key,
            'config_value': self.config_value,
            'config_type': self.config_type,
            'description': self.description,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Create instance from dictionary."""
        return cls(
            id=data.get('id'),
            config_key=data.get('config_key', ''),
            config_value=data.get('config_value', ''),
            config_type=data.get('config_type', 'string'),
            description=data.get('description'),
            updated_at=data.get('updated_at')
        )
    
    def get_typed_value(self) -> Any:
        """Get config value converted to appropriate type."""
        if self.config_type == 'integer':
            try:
                return int(self.config_value)
            except ValueError:
                return 0
        elif self.config_type == 'float':
            try:
                return float(self.config_value)
            except ValueError:
                return 0.0
        elif self.config_type == 'boolean':
            return self.config_value.lower() in ('true', '1', 'yes', 'on')
        else:
            return self.config_value
    
    def set_typed_value(self, value: Any):
        """Set config value from typed value."""
        if self.config_type == 'boolean':
            self.config_value = '1' if value else '0'
        else:
            self.config_value = str(value)


# Database schema constants
DATABASE_SCHEMA = """
-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Create the devices table for IP camera management
CREATE TABLE IF NOT EXISTS devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ip_address TEXT NOT NULL,
    port INTEGER NOT NULL,
    use_https BOOLEAN DEFAULT 0,
    end_with_video BOOLEAN DEFAULT 0,
    status TEXT DEFAULT 'active' CHECK(status IN ('active', 'inactive')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Whitelist table for authorized individuals and recognized animals
CREATE TABLE IF NOT EXISTS whitelist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL CHECK(entity_type IN ('human', 'animal')),
    familiar TEXT DEFAULT 'familiar' CHECK(familiar IN ('familiar', 'unfamiliar')),
    color TEXT,
    coco_class_id INTEGER,
    image_path TEXT NOT NULL,
    confidence_threshold REAL DEFAULT 0.6,
    pet_breed TEXT,
    individual_id TEXT,
    face_encodings BLOB,
    multiple_photos TEXT,
    identification_method TEXT DEFAULT 'color' CHECK(identification_method IN ('color', 'face', 'hybrid')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create the notification_settings table for Telegram users
CREATE TABLE IF NOT EXISTS notification_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL UNIQUE,
    telegram_username TEXT NOT NULL,
    notify_human_detection BOOLEAN NOT NULL DEFAULT 1,
    notify_animal_detection BOOLEAN NOT NULL DEFAULT 1,
    sendstatus TEXT DEFAULT 'open' CHECK(sendstatus IN ('open', 'close')),
    last_notification DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create detection_logs table for tracking detections
CREATE TABLE IF NOT EXISTS detection_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_type TEXT NOT NULL CHECK(detection_type IN ('human', 'animal')),
    entity_name TEXT,
    confidence REAL,
    camera_id INTEGER,
    image_path TEXT,
    notification_sent BOOLEAN DEFAULT 0,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (camera_id) REFERENCES devices(id)
);

-- Create system_metrics table for performance monitoring
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_type TEXT NOT NULL,
    metric_value REAL NOT NULL,
    unit TEXT,
    recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create configuration table for system settings
CREATE TABLE IF NOT EXISTS system_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_key TEXT NOT NULL UNIQUE,
    config_value TEXT NOT NULL,
    config_type TEXT DEFAULT 'string' CHECK(config_type IN ('string', 'integer', 'float', 'boolean')),
    description TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# Default configuration values
DEFAULT_CONFIG = [
    ('yolo_model', 'yolo11n.pt', 'string', 'YOLO11n model file name'),
    ('yolo_confidence', '0.5', 'float', 'YOLO detection confidence threshold'),
    ('human_confidence_threshold', '0.6', 'float', 'Configurable confidence threshold for human face recognition'),
    ('animal_confidence_threshold', '0.6', 'float', 'Configurable confidence threshold for animal detection'),
    ('pet_identification_threshold', '0.7', 'float', 'Confidence threshold for individual pet identification'),
    ('multi_face_detection', '1', 'boolean', 'Enable simultaneous multi-face detection'),
    ('max_faces_per_frame', '10', 'integer', 'Maximum faces to process per frame'),
    ('unknown_person_timer', '5', 'integer', 'Seconds before unknown person alert'),
    ('unfamiliar_animal_timer', '5', 'integer', 'Seconds before unfamiliar animal alert'),
    ('notification_cooldown', '20', 'integer', 'Seconds between notifications'),
    ('frame_width', '640', 'integer', 'Video frame processing width'),
    ('frame_height', '480', 'integer', 'Video frame processing height'),
    ('target_fps', '30', 'integer', 'Target processing FPS'),
    ('enable_gpu', '1', 'boolean', 'Enable GPU acceleration'),
    ('enable_performance_monitoring', '1', 'boolean', 'Enable performance metrics collection'),
    ('pet_identification_method', 'hybrid', 'string', 'Method for pet identification: color, face, or hybrid')
]
