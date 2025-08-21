"""
SQLite Database Manager

This module provides comprehensive database operations for the Intruder Detection System,
including CRUD operations, migrations, and connection management.
"""

import sqlite3
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import os
import threading
from contextlib import contextmanager

from .models import (
    Device, WhitelistEntry, NotificationSettings, DetectionLog, 
    SystemMetrics, SystemConfig, DATABASE_SCHEMA, DEFAULT_CONFIG
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Comprehensive SQLite database manager for the Intruder Detection System.
    
    Features:
    - Thread-safe database operations
    - Automatic schema creation and migration
    - CRUD operations for all entities
    - Connection pooling and management
    - Error handling and recovery
    """
    
    def __init__(self, db_path: str = "detection_system.db"):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.connection_lock = threading.Lock()
        self._initialize_database()
        
        logger.info(f"Database Manager initialized with database: {db_path}")
    
    def _initialize_database(self):
        """Initialize the database with schema and default data."""
        try:
            with self.get_connection() as conn:
                # Execute schema creation
                conn.executescript(DATABASE_SCHEMA)
                
                # Insert default configuration if not exists
                self._insert_default_config(conn)
                
                # Create indexes for better performance
                self._create_indexes(conn)
                
                # Create triggers for timestamp updates
                self._create_triggers(conn)

                # Create views for easier data access
                self._create_views(conn)

                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _insert_default_config(self, conn: sqlite3.Connection):
        """Insert default configuration values."""
        for config_key, config_value, config_type, description in DEFAULT_CONFIG:
            conn.execute("""
                INSERT OR IGNORE INTO system_config (config_key, config_value, config_type, description)
                VALUES (?, ?, ?, ?)
            """, (config_key, config_value, config_type, description))
    
    def _create_indexes(self, conn: sqlite3.Connection):
        """Create database indexes for better performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_devices_status ON devices(status)",
            "CREATE INDEX IF NOT EXISTS idx_devices_ip_port ON devices(ip_address, port)",
            "CREATE INDEX IF NOT EXISTS idx_whitelist_entity_type ON whitelist(entity_type)",
            "CREATE INDEX IF NOT EXISTS idx_whitelist_familiar ON whitelist(familiar)",
            "CREATE INDEX IF NOT EXISTS idx_whitelist_coco_class ON whitelist(coco_class_id)",
            "CREATE INDEX IF NOT EXISTS idx_notification_chat_id ON notification_settings(chat_id)",
            "CREATE INDEX IF NOT EXISTS idx_notification_status ON notification_settings(sendstatus)",
            "CREATE INDEX IF NOT EXISTS idx_detection_logs_type ON detection_logs(detection_type)",
            "CREATE INDEX IF NOT EXISTS idx_detection_logs_date ON detection_logs(detected_at)",
            "CREATE INDEX IF NOT EXISTS idx_detection_logs_camera ON detection_logs(camera_id)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_type_date ON system_metrics(metric_type, recorded_at)"
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)
    
    def _create_triggers(self, conn: sqlite3.Connection):
        """Create triggers for automatic timestamp updates."""
        triggers = [
            """
            CREATE TRIGGER IF NOT EXISTS update_devices_timestamp 
                AFTER UPDATE ON devices
                BEGIN
                    UPDATE devices SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS update_whitelist_timestamp 
                AFTER UPDATE ON whitelist
                BEGIN
                    UPDATE whitelist SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS update_notification_timestamp 
                AFTER UPDATE ON notification_settings
                BEGIN
                    UPDATE notification_settings SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS update_config_timestamp 
                AFTER UPDATE ON system_config
                BEGIN
                    UPDATE system_config SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END
            """
        ]
        
        for trigger_sql in triggers:
            conn.execute(trigger_sql)

    def _create_views(self, conn):
        """Create database views for easier data access."""
        views = [
            # Active cameras view
            """
            CREATE VIEW IF NOT EXISTS active_cameras AS
            SELECT * FROM devices WHERE status = 'active'
            """,

            # Known humans view
            """
            CREATE VIEW IF NOT EXISTS known_humans AS
            SELECT * FROM whitelist WHERE entity_type = 'human' AND familiar = 'familiar'
            """,

            # Familiar animals view
            """
            CREATE VIEW IF NOT EXISTS familiar_animals AS
            SELECT * FROM whitelist WHERE entity_type = 'animal' AND familiar = 'familiar'
            """,

            # Active telegram users view
            """
            CREATE VIEW IF NOT EXISTS active_telegram_users AS
            SELECT * FROM notification_settings WHERE sendstatus = 'open'
            """
        ]

        for view_sql in views:
            conn.execute(view_sql)
    
    @contextmanager
    def get_connection(self):
        """Get a thread-safe database connection."""
        with self.connection_lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
            try:
                yield conn
            finally:
                conn.close()
    
    # Device CRUD operations
    def create_device(self, device: Device) -> int:
        """Create a new device and return its ID."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO devices (ip_address, port, use_https, end_with_video, status)
                    VALUES (?, ?, ?, ?, ?)
                """, (device.ip_address, device.port, device.use_https, 
                     device.end_with_video, device.status))
                
                device_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Created device with ID: {device_id}")
                return device_id
                
        except Exception as e:
            logger.error(f"Failed to create device: {e}")
            raise
    
    def get_device(self, device_id: int) -> Optional[Device]:
        """Get a device by ID."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM devices WHERE id = ?", (device_id,))
                row = cursor.fetchone()
                
                if row:
                    return Device.from_dict(dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Failed to get device {device_id}: {e}")
            return None
    
    def get_all_devices(self, status: Optional[str] = None) -> List[Device]:
        """Get all devices, optionally filtered by status."""
        try:
            with self.get_connection() as conn:
                if status:
                    cursor = conn.execute("SELECT * FROM devices WHERE status = ? ORDER BY id", (status,))
                else:
                    cursor = conn.execute("SELECT * FROM devices ORDER BY id")
                
                return [Device.from_dict(dict(row)) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get devices: {e}")
            return []
    
    def update_device(self, device: Device) -> bool:
        """Update an existing device."""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    UPDATE devices 
                    SET ip_address = ?, port = ?, use_https = ?, end_with_video = ?, status = ?
                    WHERE id = ?
                """, (device.ip_address, device.port, device.use_https, 
                     device.end_with_video, device.status, device.id))
                
                conn.commit()
                logger.info(f"Updated device {device.id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update device {device.id}: {e}")
            return False
    
    def delete_device(self, device_id: int) -> bool:
        """Delete a device by ID."""
        try:
            with self.get_connection() as conn:
                conn.execute("DELETE FROM devices WHERE id = ?", (device_id,))
                conn.commit()
                logger.info(f"Deleted device {device_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete device {device_id}: {e}")
            return False

    def reorganize_device_ids(self) -> bool:
        """
        Reorganize device IDs to be sequential (1, 2, 3, ...).
        This removes gaps in ID numbering caused by deletions.
        Uses a safer approach that doesn't drop the main table.

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                # Get all devices ordered by current ID
                cursor = conn.execute("SELECT * FROM devices ORDER BY id")
                devices = [Device.from_dict(dict(row)) for row in cursor.fetchall()]

                if not devices:
                    logger.info("No devices to reorganize")
                    return True

                # Check if IDs are already sequential
                expected_ids = list(range(1, len(devices) + 1))
                actual_ids = [device.id for device in devices]

                if actual_ids == expected_ids:
                    logger.info("Device IDs are already sequential")
                    return True

                logger.info(f"Reorganizing {len(devices)} device IDs from {actual_ids} to {expected_ids}")

                # Use a safer approach: create mapping and update in batches
                # First, create a temporary mapping table
                conn.execute("DROP TABLE IF EXISTS id_mapping")
                conn.execute("""
                    CREATE TEMPORARY TABLE id_mapping (
                        old_id INTEGER,
                        new_id INTEGER,
                        ip_address TEXT,
                        port INTEGER,
                        use_https BOOLEAN,
                        end_with_video BOOLEAN,
                        status TEXT,
                        created_at DATETIME,
                        updated_at DATETIME
                    )
                """)

                # Insert mapping data
                for new_id, device in enumerate(devices, 1):
                    conn.execute("""
                        INSERT INTO id_mapping (old_id, new_id, ip_address, port, use_https, end_with_video, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (device.id, new_id, device.ip_address, device.port, device.use_https,
                         device.end_with_video, device.status, device.created_at, device.updated_at))

                # Clear the devices table and reinsert with new IDs
                conn.execute("DELETE FROM devices")

                # Insert devices with new sequential IDs
                cursor = conn.execute("SELECT * FROM id_mapping ORDER BY new_id")
                for row in cursor.fetchall():
                    conn.execute("""
                        INSERT INTO devices (id, ip_address, port, use_https, end_with_video, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]))  # new_id and other fields

                # Reset the autoincrement counter
                conn.execute(f"UPDATE sqlite_sequence SET seq = {len(devices)} WHERE name = 'devices'")

                # Clean up temporary table
                conn.execute("DROP TABLE id_mapping")

                conn.commit()
                logger.info(f"Successfully reorganized {len(devices)} device IDs to be sequential (1-{len(devices)})")
                return True

        except Exception as e:
            logger.error(f"Failed to reorganize device IDs: {e}")
            return False
    
    # Whitelist CRUD operations
    def create_whitelist_entry(self, entry: WhitelistEntry) -> int:
        """Create a new whitelist entry and return its ID."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO whitelist (name, entity_type, familiar, color, coco_class_id, 
                                         image_path, confidence_threshold, pet_breed, individual_id,
                                         face_encodings, multiple_photos, identification_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (entry.name, entry.entity_type, entry.familiar, entry.color, 
                     entry.coco_class_id, entry.image_path, entry.confidence_threshold,
                     entry.pet_breed, entry.individual_id, entry.face_encodings,
                     entry.multiple_photos, entry.identification_method))
                
                entry_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Created whitelist entry with ID: {entry_id}")
                return entry_id
                
        except Exception as e:
            logger.error(f"Failed to create whitelist entry: {e}")
            raise
    
    def get_whitelist_entry(self, entry_id: int) -> Optional[WhitelistEntry]:
        """Get a whitelist entry by ID."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM whitelist WHERE id = ?", (entry_id,))
                row = cursor.fetchone()
                
                if row:
                    return WhitelistEntry.from_dict(dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Failed to get whitelist entry {entry_id}: {e}")
            return None
    
    def get_whitelist_entries(self, entity_type: Optional[str] = None, 
                            familiar: Optional[str] = None) -> List[WhitelistEntry]:
        """Get whitelist entries with optional filtering."""
        try:
            with self.get_connection() as conn:
                query = "SELECT * FROM whitelist WHERE 1=1"
                params = []
                
                if entity_type:
                    query += " AND entity_type = ?"
                    params.append(entity_type)
                
                if familiar:
                    query += " AND familiar = ?"
                    params.append(familiar)
                
                query += " ORDER BY id"
                
                cursor = conn.execute(query, params)
                return [WhitelistEntry.from_dict(dict(row)) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get whitelist entries: {e}")
            return []
    
    def update_whitelist_entry(self, entry: WhitelistEntry) -> bool:
        """Update an existing whitelist entry."""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    UPDATE whitelist 
                    SET name = ?, entity_type = ?, familiar = ?, color = ?, coco_class_id = ?,
                        image_path = ?, confidence_threshold = ?, pet_breed = ?, individual_id = ?,
                        face_encodings = ?, multiple_photos = ?, identification_method = ?
                    WHERE id = ?
                """, (entry.name, entry.entity_type, entry.familiar, entry.color,
                     entry.coco_class_id, entry.image_path, entry.confidence_threshold,
                     entry.pet_breed, entry.individual_id, entry.face_encodings,
                     entry.multiple_photos, entry.identification_method, entry.id))
                
                conn.commit()
                logger.info(f"Updated whitelist entry {entry.id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update whitelist entry {entry.id}: {e}")
            return False
    
    def delete_whitelist_entry(self, entry_id: int) -> bool:
        """Delete a whitelist entry by ID."""
        try:
            with self.get_connection() as conn:
                conn.execute("DELETE FROM whitelist WHERE id = ?", (entry_id,))
                conn.commit()
                logger.info(f"Deleted whitelist entry {entry_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete whitelist entry {entry_id}: {e}")
            return False
    
    # Notification settings CRUD operations
    def create_notification_settings(self, settings: NotificationSettings) -> int:
        """Create new notification settings and return the ID."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO notification_settings (chat_id, telegram_username, 
                                                     notify_human_detection, notify_animal_detection, sendstatus)
                    VALUES (?, ?, ?, ?, ?)
                """, (settings.chat_id, settings.telegram_username, 
                     settings.notify_human_detection, settings.notify_animal_detection, settings.sendstatus))
                
                settings_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Created notification settings with ID: {settings_id}")
                return settings_id
                
        except Exception as e:
            logger.error(f"Failed to create notification settings: {e}")
            raise
    
    def get_notification_settings(self, chat_id: int) -> Optional[NotificationSettings]:
        """Get notification settings by chat ID."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM notification_settings WHERE chat_id = ?", (chat_id,))
                row = cursor.fetchone()
                
                if row:
                    return NotificationSettings.from_dict(dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Failed to get notification settings for {chat_id}: {e}")
            return None
    
    def get_all_notification_settings(self, status: Optional[str] = None) -> List[NotificationSettings]:
        """Get all notification settings, optionally filtered by status."""
        try:
            with self.get_connection() as conn:
                if status:
                    cursor = conn.execute("SELECT * FROM notification_settings WHERE sendstatus = ? ORDER BY id", (status,))
                else:
                    cursor = conn.execute("SELECT * FROM notification_settings ORDER BY id")
                
                return [NotificationSettings.from_dict(dict(row)) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get notification settings: {e}")
            return []
    
    def update_notification_settings(self, settings: NotificationSettings) -> bool:
        """Update existing notification settings."""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    UPDATE notification_settings 
                    SET telegram_username = ?, notify_human_detection = ?, 
                        notify_animal_detection = ?, sendstatus = ?, last_notification = ?
                    WHERE chat_id = ?
                """, (settings.telegram_username, settings.notify_human_detection,
                     settings.notify_animal_detection, settings.sendstatus, 
                     settings.last_notification, settings.chat_id))
                
                conn.commit()
                logger.info(f"Updated notification settings for {settings.chat_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update notification settings for {settings.chat_id}: {e}")
            return False
    
    def delete_notification_settings(self, chat_id: int) -> bool:
        """Delete notification settings by chat ID."""
        try:
            with self.get_connection() as conn:
                conn.execute("DELETE FROM notification_settings WHERE chat_id = ?", (chat_id,))
                conn.commit()
                logger.info(f"Deleted notification settings for {chat_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete notification settings for {chat_id}: {e}")
            return False
    
    # System configuration operations
    def get_config(self, key: str) -> Optional[SystemConfig]:
        """Get a configuration value by key."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM system_config WHERE config_key = ?", (key,))
                row = cursor.fetchone()
                
                if row:
                    return SystemConfig.from_dict(dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Failed to get config {key}: {e}")
            return None
    
    def set_config(self, key: str, value: Any, config_type: str = "string", description: str = "") -> bool:
        """Set a configuration value."""
        try:
            with self.get_connection() as conn:
                # Convert value to string based on type
                if config_type == 'boolean':
                    str_value = '1' if value else '0'
                else:
                    str_value = str(value)
                
                conn.execute("""
                    INSERT OR REPLACE INTO system_config (config_key, config_value, config_type, description)
                    VALUES (?, ?, ?, ?)
                """, (key, str_value, config_type, description))
                
                conn.commit()
                logger.info(f"Set config {key} = {value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to set config {key}: {e}")
            return False
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration values as a dictionary."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM system_config ORDER BY config_key")
                
                config_dict = {}
                for row in cursor.fetchall():
                    config = SystemConfig.from_dict(dict(row))
                    config_dict[config.config_key] = config.get_typed_value()
                
                return config_dict
                
        except Exception as e:
            logger.error(f"Failed to get all config: {e}")
            return {}
    
    # Utility methods
    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        try:
            with self.get_connection() as conn:
                stats = {}
                
                tables = ['devices', 'whitelist', 'notification_settings', 
                         'detection_logs', 'system_metrics', 'system_config']
                
                for table in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

    # Convenience methods using database views
    def get_active_cameras(self) -> List[Device]:
        """Get all active cameras using the active_cameras view."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM active_cameras ORDER BY id")
                return [Device.from_dict(dict(row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get active cameras: {e}")
            return []

    def get_known_humans(self) -> List[WhitelistEntry]:
        """Get all known humans using the known_humans view."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM known_humans ORDER BY id")
                return [WhitelistEntry.from_dict(dict(row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get known humans: {e}")
            return []

    def get_familiar_animals(self) -> List[WhitelistEntry]:
        """Get all familiar animals using the familiar_animals view."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM familiar_animals ORDER BY id")
                return [WhitelistEntry.from_dict(dict(row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get familiar animals: {e}")
            return []

    def get_active_telegram_users(self) -> List[NotificationSettings]:
        """Get all active telegram users using the active_telegram_users view."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM active_telegram_users ORDER BY id")
                return [NotificationSettings.from_dict(dict(row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get active telegram users: {e}")
            return []

    # Detection logging methods
    def log_detection(self, detection_type: str, entity_name: str = None,
                     confidence: float = None, camera_id: int = None,
                     image_path: str = None, notification_sent: bool = False) -> bool:
        """
        Log a detection event to the database.

        Args:
            detection_type: 'human' or 'animal'
            entity_name: Name of detected entity (e.g., 'John', 'Unknown', 'Jacky')
            confidence: Detection confidence (0.0-1.0)
            camera_id: ID of camera that made the detection
            image_path: Path to saved detection image
            notification_sent: Whether notification was sent

        Returns:
            True if logged successfully
        """
        try:
            detection_log = DetectionLog(
                detection_type=detection_type,
                entity_name=entity_name,
                confidence=confidence,
                camera_id=camera_id,
                image_path=image_path,
                notification_sent=notification_sent
            )

            with self.get_connection() as conn:
                # Only include camera_id if it exists in devices table
                if camera_id:
                    # Check if camera exists
                    camera_check = conn.execute("SELECT id FROM devices WHERE id = ?", (camera_id,))
                    if not camera_check.fetchone():
                        camera_id = None  # Set to None if camera doesn't exist

                cursor = conn.execute(
                    """INSERT INTO detection_logs
                       (detection_type, entity_name, confidence, camera_id, image_path, notification_sent)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (detection_log.detection_type, detection_log.entity_name,
                     detection_log.confidence, camera_id,
                     detection_log.image_path, detection_log.notification_sent)
                )
                conn.commit()

                logger.debug(f"Logged {detection_type} detection: {entity_name} (confidence: {confidence})")
                return True

        except Exception as e:
            logger.error(f"Failed to log detection: {e}")
            return False

    def get_recent_detections(self, limit: int = 100, detection_type: str = None) -> List[DetectionLog]:
        """Get recent detection logs."""
        try:
            with self.get_connection() as conn:
                if detection_type:
                    cursor = conn.execute(
                        """SELECT * FROM detection_logs WHERE detection_type = ?
                           ORDER BY detected_at DESC LIMIT ?""",
                        (detection_type, limit)
                    )
                else:
                    cursor = conn.execute(
                        "SELECT * FROM detection_logs ORDER BY detected_at DESC LIMIT ?",
                        (limit,)
                    )

                return [DetectionLog.from_dict(dict(row)) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get recent detections: {e}")
            return []

    def get_detection_stats(self, days: int = 7) -> Dict[str, int]:
        """Get detection statistics for the last N days."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """SELECT detection_type, COUNT(*) as count
                       FROM detection_logs
                       WHERE detected_at >= datetime('now', '-{} days')
                       GROUP BY detection_type""".format(days)
                )

                stats = {}
                for row in cursor.fetchall():
                    stats[row[0]] = row[1]

                return stats

        except Exception as e:
            logger.error(f"Failed to get detection stats: {e}")
            return {}

    # System metrics methods
    def log_system_metric(self, metric_type: str, metric_value: float, unit: str = None) -> bool:
        """
        Log a system performance metric to the database.

        Args:
            metric_type: Type of metric (e.g., 'fps', 'cpu_usage', 'memory_usage')
            metric_value: Numeric value of the metric
            unit: Unit of measurement (e.g., 'fps', '%', 'MB')

        Returns:
            True if logged successfully
        """
        try:
            system_metric = SystemMetrics(
                metric_type=metric_type,
                metric_value=metric_value,
                unit=unit
            )

            with self.get_connection() as conn:
                cursor = conn.execute(
                    """INSERT INTO system_metrics (metric_type, metric_value, unit)
                       VALUES (?, ?, ?)""",
                    (system_metric.metric_type, system_metric.metric_value, system_metric.unit)
                )
                conn.commit()

                logger.debug(f"Logged system metric: {metric_type} = {metric_value} {unit or ''}")
                return True

        except Exception as e:
            logger.error(f"Failed to log system metric: {e}")
            return False

    def log_performance_metrics(self, metrics: Dict[str, float]) -> bool:
        """
        Log multiple performance metrics at once.

        Args:
            metrics: Dictionary of metric_type -> value pairs

        Returns:
            True if all metrics logged successfully
        """
        try:
            success_count = 0

            # Common metric units
            metric_units = {
                'fps': 'fps',
                'cpu_usage': '%',
                'memory_usage': '%',
                'gpu_usage': '%',
                'detection_time': 'ms',
                'face_recognition_time': 'ms',
                'animal_recognition_time': 'ms'
            }

            for metric_type, value in metrics.items():
                unit = metric_units.get(metric_type, None)
                if self.log_system_metric(metric_type, value, unit):
                    success_count += 1

            return success_count == len(metrics)

        except Exception as e:
            logger.error(f"Failed to log performance metrics: {e}")
            return False

    def get_recent_metrics(self, metric_type: str = None, limit: int = 100) -> List[SystemMetrics]:
        """Get recent system metrics."""
        try:
            with self.get_connection() as conn:
                if metric_type:
                    cursor = conn.execute(
                        """SELECT * FROM system_metrics WHERE metric_type = ?
                           ORDER BY recorded_at DESC LIMIT ?""",
                        (metric_type, limit)
                    )
                else:
                    cursor = conn.execute(
                        "SELECT * FROM system_metrics ORDER BY recorded_at DESC LIMIT ?",
                        (limit,)
                    )

                return [SystemMetrics.from_dict(dict(row)) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get recent metrics: {e}")
            return []

    def get_metric_averages(self, hours: int = 24) -> Dict[str, float]:
        """Get average metrics for the last N hours."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """SELECT metric_type, AVG(metric_value) as avg_value
                       FROM system_metrics
                       WHERE recorded_at >= datetime('now', '-{} hours')
                       GROUP BY metric_type""".format(hours)
                )

                averages = {}
                for row in cursor.fetchall():
                    averages[row[0]] = round(row[1], 2)

                return averages

        except Exception as e:
            logger.error(f"Failed to get metric averages: {e}")
            return {}

    def get_detection_stats(self, days: int = 7) -> Dict[str, int]:
        """Get detection statistics for the last N days."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """SELECT detection_type, COUNT(*) as count
                       FROM detection_logs
                       WHERE detected_at >= datetime('now', '-{} days')
                       GROUP BY detection_type""".format(days)
                )

                stats = {}
                for row in cursor.fetchall():
                    stats[row[0]] = row[1]

                return stats

        except Exception as e:
            logger.error(f"Failed to get detection stats: {e}")
            return {}

    def get_database_stats(self) -> Dict[str, int]:
        """Get database table statistics."""
        try:
            stats = {}
            with self.get_connection() as conn:
                # Get table counts
                tables = ['devices', 'whitelist', 'notification_settings', 'detection_logs', 'system_metrics']

                for table in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    stats[f"{table}_count"] = count

                return stats

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

    def get_recent_detections(self, limit: int = 50) -> List[DetectionLog]:
        """Get recent detection logs."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """SELECT * FROM detection_logs
                       ORDER BY detected_at DESC LIMIT ?""",
                    (limit,)
                )

                detections = []
                for row in cursor.fetchall():
                    detection = DetectionLog.from_dict(dict(row))
                    detections.append(detection)

                return detections

        except Exception as e:
            logger.error(f"Failed to get recent detections: {e}")
            return []

    def get_system_uptime_stats(self) -> Dict[str, Any]:
        """Get system uptime and operational statistics."""
        try:
            with self.get_connection() as conn:
                # Get first detection time as system start approximation
                cursor = conn.execute(
                    "SELECT MIN(detected_at) FROM detection_logs"
                )
                first_detection = cursor.fetchone()[0]

                # Get latest detection
                cursor = conn.execute(
                    "SELECT MAX(detected_at) FROM detection_logs"
                )
                latest_detection = cursor.fetchone()[0]

                # Get total detections today
                cursor = conn.execute(
                    """SELECT COUNT(*) FROM detection_logs
                       WHERE DATE(detected_at) = DATE('now')"""
                )
                today_detections = cursor.fetchone()[0]

                # Get average confidence
                cursor = conn.execute(
                    """SELECT AVG(confidence) FROM detection_logs
                       WHERE confidence IS NOT NULL
                       AND detected_at >= datetime('now', '-7 days')"""
                )
                avg_confidence = cursor.fetchone()[0]

                # Get notification count
                cursor = conn.execute(
                    """SELECT COUNT(*) FROM detection_logs
                       WHERE notification_sent = 1
                       AND detected_at >= datetime('now', '-7 days')"""
                )
                notifications_sent = cursor.fetchone()[0]

                return {
                    'first_detection': first_detection,
                    'latest_detection': latest_detection,
                    'today_detections': today_detections,
                    'avg_confidence': round(avg_confidence, 1) if avg_confidence else 0.0,
                    'notifications_sent': notifications_sent
                }

        except Exception as e:
            logger.error(f"Failed to get uptime stats: {e}")
            return {}

    def get_pet_identification_stats(self, days: int = 7) -> Dict[str, int]:
        """Get pet identification statistics."""
        try:
            with self.get_connection() as conn:
                # Get known pets from whitelist
                cursor = conn.execute(
                    """SELECT COUNT(*) FROM whitelist
                       WHERE entity_type = 'animal' AND familiar = 'familiar'"""
                )
                known_pets = cursor.fetchone()[0]

                # Get pet identifications in detection logs
                cursor = conn.execute(
                    """SELECT COUNT(*) FROM detection_logs
                       WHERE detection_type = 'animal'
                       AND entity_name IS NOT NULL
                       AND detected_at >= datetime('now', '-{} days')""".format(days)
                )
                pet_identifications = cursor.fetchone()[0]

                # Get unknown animals
                cursor = conn.execute(
                    """SELECT COUNT(*) FROM detection_logs
                       WHERE detection_type = 'animal'
                       AND (entity_name IS NULL OR entity_name = 'Unknown')
                       AND detected_at >= datetime('now', '-{} days')""".format(days)
                )
                unknown_animals = cursor.fetchone()[0]

                return {
                    'known_pets': known_pets,
                    'pet_identifications': pet_identifications,
                    'unknown_animals': unknown_animals
                }

        except Exception as e:
            logger.error(f"Failed to get pet identification stats: {e}")
            return {}

    def get_recent_metrics(self, metric_type: str = None, limit: int = 100) -> List[Dict]:
        """Get recent system metrics from database."""
        try:
            with self.get_connection() as conn:
                if metric_type:
                    cursor = conn.execute(
                        """SELECT * FROM system_metrics
                           WHERE metric_type = ?
                           ORDER BY recorded_at DESC LIMIT ?""",
                        (metric_type, limit)
                    )
                else:
                    cursor = conn.execute(
                        """SELECT * FROM system_metrics
                           ORDER BY recorded_at DESC LIMIT ?""",
                        (limit,)
                    )

                metrics = []
                for row in cursor.fetchall():
                    metric = SystemMetrics.from_dict(dict(row))
                    metrics.append(metric.to_dict())

                return metrics

        except Exception as e:
            logger.error(f"Failed to get recent metrics: {e}")
            return []

    def log_performance_metrics(self, metrics: Dict[str, float]) -> bool:
        """Log multiple performance metrics to database."""
        try:
            with self.get_connection() as conn:
                for metric_type, value in metrics.items():
                    if isinstance(value, (int, float)):
                        conn.execute(
                            """INSERT INTO system_metrics (metric_type, metric_value)
                               VALUES (?, ?)""",
                            (metric_type, float(value))
                        )

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to log performance metrics: {e}")
            return False

    def cleanup_old_metrics(self, days: int = 7) -> bool:
        """Clean up metrics older than N days."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM system_metrics WHERE recorded_at < datetime('now', '-{} days')".format(days)
                )
                deleted_count = cursor.rowcount
                conn.commit()

                logger.info(f"Cleaned up {deleted_count} old metrics (older than {days} days)")
                return True

        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")
            return False
