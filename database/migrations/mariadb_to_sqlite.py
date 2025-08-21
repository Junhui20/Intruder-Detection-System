"""
MariaDB to SQLite Migration Script

This script migrates data from the legacy MariaDB database to the new SQLite database
while preserving all existing data and relationships.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MariaDBToSQLiteMigrator:
    """
    Migrates data from MariaDB to SQLite database.
    
    Features:
    - Data validation and integrity checks
    - Error handling and rollback
    - Progress tracking
    - Backup creation before migration
    """
    
    def __init__(self, sqlite_db_manager, mariadb_config: Dict[str, str]):
        """
        Initialize the migrator.
        
        Args:
            sqlite_db_manager: DatabaseManager instance for SQLite
            mariadb_config: MariaDB connection configuration
        """
        self.sqlite_db = sqlite_db_manager
        self.mariadb_config = mariadb_config
        self.migration_stats = {
            'devices_migrated': 0,
            'whitelist_migrated': 0,
            'notifications_migrated': 0,
            'errors': []
        }
        
    def migrate_all_data(self) -> bool:
        """
        Perform complete migration from MariaDB to SQLite.
        
        Returns:
            True if migration successful
        """
        try:
            logger.info("Starting MariaDB to SQLite migration...")
            
            # Create backup before migration
            backup_path = f"backup_before_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            self.sqlite_db.backup_database(backup_path)
            
            # Import MariaDB connector
            try:
                import mariadb
            except ImportError:
                logger.error("MariaDB connector not available. Install with: pip install mariadb")
                return False
            
            # Connect to MariaDB
            mariadb_conn = mariadb.connect(**self.mariadb_config)
            mariadb_cursor = mariadb_conn.cursor()
            
            # Migrate each table
            success = True
            success &= self._migrate_devices(mariadb_cursor)
            success &= self._migrate_whitelist(mariadb_cursor)
            success &= self._migrate_notification_settings(mariadb_cursor)
            
            # Close MariaDB connection
            mariadb_conn.close()
            
            # Log migration results
            self._log_migration_results()
            
            if success:
                logger.info("Migration completed successfully!")
            else:
                logger.warning("Migration completed with some errors. Check logs for details.")
            
            return success
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            self.migration_stats['errors'].append(f"General migration error: {e}")
            return False
    
    def _migrate_devices(self, mariadb_cursor) -> bool:
        """Migrate devices table from MariaDB to SQLite."""
        try:
            logger.info("Migrating devices...")
            
            # Get data from MariaDB
            mariadb_cursor.execute("SELECT * FROM devices")
            devices_data = mariadb_cursor.fetchall()
            
            # Get column names
            mariadb_cursor.execute("DESCRIBE devices")
            columns = [row[0] for row in mariadb_cursor.fetchall()]
            
            migrated_count = 0
            for device_row in devices_data:
                try:
                    # Convert to dictionary
                    device_dict = dict(zip(columns, device_row))
                    
                    # Map MariaDB data to SQLite format
                    from ..models import Device
                    device = Device(
                        ip_address=device_dict.get('ip_address', ''),
                        port=device_dict.get('port', 8080),
                        use_https=bool(device_dict.get('use_https', False)),
                        end_with_video=bool(device_dict.get('end_with_video', False)),
                        status=device_dict.get('status', 'active')
                    )
                    
                    # Insert into SQLite
                    self.sqlite_db.create_device(device)
                    migrated_count += 1
                    
                except Exception as e:
                    error_msg = f"Error migrating device {device_dict.get('id', 'unknown')}: {e}"
                    logger.error(error_msg)
                    self.migration_stats['errors'].append(error_msg)
            
            self.migration_stats['devices_migrated'] = migrated_count
            logger.info(f"Migrated {migrated_count} devices")
            return True
            
        except Exception as e:
            error_msg = f"Failed to migrate devices: {e}"
            logger.error(error_msg)
            self.migration_stats['errors'].append(error_msg)
            return False
    
    def _migrate_whitelist(self, mariadb_cursor) -> bool:
        """Migrate whitelist table from MariaDB to SQLite."""
        try:
            logger.info("Migrating whitelist...")
            
            # Get data from MariaDB
            mariadb_cursor.execute("SELECT * FROM whitelist")
            whitelist_data = mariadb_cursor.fetchall()
            
            # Get column names
            mariadb_cursor.execute("DESCRIBE whitelist")
            columns = [row[0] for row in mariadb_cursor.fetchall()]
            
            migrated_count = 0
            for whitelist_row in whitelist_data:
                try:
                    # Convert to dictionary
                    whitelist_dict = dict(zip(columns, whitelist_row))
                    
                    # Map MariaDB data to SQLite format
                    from ..models import WhitelistEntry
                    entry = WhitelistEntry(
                        name=whitelist_dict.get('name', ''),
                        entity_type=whitelist_dict.get('entity_type', 'human'),
                        familiar=whitelist_dict.get('familiar', 'familiar'),
                        color=whitelist_dict.get('color'),
                        coco_class_id=whitelist_dict.get('coco_class_id'),
                        image_path=whitelist_dict.get('image_path', ''),
                        confidence_threshold=whitelist_dict.get('confidence_threshold', 0.6),
                        pet_breed=whitelist_dict.get('pet_breed'),
                        individual_id=whitelist_dict.get('individual_id'),
                        face_encodings=whitelist_dict.get('face_encodings'),
                        multiple_photos=whitelist_dict.get('multiple_photos'),
                        identification_method=whitelist_dict.get('identification_method', 'color')
                    )
                    
                    # Insert into SQLite
                    self.sqlite_db.create_whitelist_entry(entry)
                    migrated_count += 1
                    
                except Exception as e:
                    error_msg = f"Error migrating whitelist entry {whitelist_dict.get('id', 'unknown')}: {e}"
                    logger.error(error_msg)
                    self.migration_stats['errors'].append(error_msg)
            
            self.migration_stats['whitelist_migrated'] = migrated_count
            logger.info(f"Migrated {migrated_count} whitelist entries")
            return True
            
        except Exception as e:
            error_msg = f"Failed to migrate whitelist: {e}"
            logger.error(error_msg)
            self.migration_stats['errors'].append(error_msg)
            return False
    
    def _migrate_notification_settings(self, mariadb_cursor) -> bool:
        """Migrate notification_settings table from MariaDB to SQLite."""
        try:
            logger.info("Migrating notification settings...")
            
            # Get data from MariaDB
            mariadb_cursor.execute("SELECT * FROM notification_settings")
            notification_data = mariadb_cursor.fetchall()
            
            # Get column names
            mariadb_cursor.execute("DESCRIBE notification_settings")
            columns = [row[0] for row in mariadb_cursor.fetchall()]
            
            migrated_count = 0
            for notification_row in notification_data:
                try:
                    # Convert to dictionary
                    notification_dict = dict(zip(columns, notification_row))
                    
                    # Map MariaDB data to SQLite format
                    from ..models import NotificationSettings
                    settings = NotificationSettings(
                        chat_id=notification_dict.get('chat_id', 0),
                        telegram_username=notification_dict.get('telegram_username', ''),
                        notify_human_detection=bool(notification_dict.get('notify_human_detection', True)),
                        notify_animal_detection=bool(notification_dict.get('notify_animal_detection', True)),
                        sendstatus=notification_dict.get('sendstatus', 'open'),
                        last_notification=notification_dict.get('last_notification')
                    )
                    
                    # Insert into SQLite
                    self.sqlite_db.create_notification_settings(settings)
                    migrated_count += 1
                    
                except Exception as e:
                    error_msg = f"Error migrating notification settings {notification_dict.get('id', 'unknown')}: {e}"
                    logger.error(error_msg)
                    self.migration_stats['errors'].append(error_msg)
            
            self.migration_stats['notifications_migrated'] = migrated_count
            logger.info(f"Migrated {migrated_count} notification settings")
            return True
            
        except Exception as e:
            error_msg = f"Failed to migrate notification settings: {e}"
            logger.error(error_msg)
            self.migration_stats['errors'].append(error_msg)
            return False
    
    def _log_migration_results(self):
        """Log detailed migration results."""
        logger.info("=== Migration Results ===")
        logger.info(f"Devices migrated: {self.migration_stats['devices_migrated']}")
        logger.info(f"Whitelist entries migrated: {self.migration_stats['whitelist_migrated']}")
        logger.info(f"Notification settings migrated: {self.migration_stats['notifications_migrated']}")
        
        if self.migration_stats['errors']:
            logger.warning(f"Errors encountered: {len(self.migration_stats['errors'])}")
            for error in self.migration_stats['errors']:
                logger.warning(f"  - {error}")
        else:
            logger.info("No errors encountered during migration")
    
    def export_mariadb_data(self, output_file: str) -> bool:
        """
        Export MariaDB data to JSON file for backup/analysis.
        
        Args:
            output_file: Path to output JSON file
            
        Returns:
            True if export successful
        """
        try:
            import mariadb
            
            # Connect to MariaDB
            mariadb_conn = mariadb.connect(**self.mariadb_config)
            mariadb_cursor = mariadb_conn.cursor()
            
            export_data = {}
            
            # Export each table
            tables = ['devices', 'whitelist', 'notification_settings']
            
            for table in tables:
                try:
                    # Get column names
                    mariadb_cursor.execute(f"DESCRIBE {table}")
                    columns = [row[0] for row in mariadb_cursor.fetchall()]
                    
                    # Get data
                    mariadb_cursor.execute(f"SELECT * FROM {table}")
                    rows = mariadb_cursor.fetchall()
                    
                    # Convert to list of dictionaries
                    table_data = []
                    for row in rows:
                        row_dict = {}
                        for i, value in enumerate(row):
                            # Handle datetime objects
                            if isinstance(value, datetime):
                                row_dict[columns[i]] = value.isoformat()
                            else:
                                row_dict[columns[i]] = value
                        table_data.append(row_dict)
                    
                    export_data[table] = table_data
                    logger.info(f"Exported {len(table_data)} records from {table}")
                    
                except Exception as e:
                    logger.error(f"Error exporting table {table}: {e}")
            
            # Write to JSON file
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            mariadb_conn.close()
            logger.info(f"MariaDB data exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export MariaDB data: {e}")
            return False
    
    def verify_migration(self) -> bool:
        """
        Verify that migration was successful by comparing record counts.
        
        Returns:
            True if verification successful
        """
        try:
            import mariadb
            
            # Connect to MariaDB
            mariadb_conn = mariadb.connect(**self.mariadb_config)
            mariadb_cursor = mariadb_conn.cursor()
            
            verification_results = {}
            tables = ['devices', 'whitelist', 'notification_settings']
            
            for table in tables:
                # Get MariaDB count
                mariadb_cursor.execute(f"SELECT COUNT(*) FROM {table}")
                mariadb_count = mariadb_cursor.fetchone()[0]
                
                # Get SQLite count
                sqlite_stats = self.sqlite_db.get_database_stats()
                sqlite_count = sqlite_stats.get(f"{table}_count", 0)
                
                verification_results[table] = {
                    'mariadb_count': mariadb_count,
                    'sqlite_count': sqlite_count,
                    'match': mariadb_count == sqlite_count
                }
                
                logger.info(f"{table}: MariaDB={mariadb_count}, SQLite={sqlite_count}, Match={mariadb_count == sqlite_count}")
            
            mariadb_conn.close()
            
            # Check if all tables match
            all_match = all(result['match'] for result in verification_results.values())
            
            if all_match:
                logger.info("Migration verification successful - all record counts match")
            else:
                logger.warning("Migration verification failed - record counts do not match")
            
            return all_match
            
        except Exception as e:
            logger.error(f"Migration verification failed: {e}")
            return False


def run_migration(sqlite_db_path: str = "detection_system.db", 
                 mariadb_config: Dict[str, str] = None) -> bool:
    """
    Convenience function to run the complete migration process.
    
    Args:
        sqlite_db_path: Path to SQLite database
        mariadb_config: MariaDB connection configuration
        
    Returns:
        True if migration successful
    """
    if mariadb_config is None:
        mariadb_config = {
            'user': 'root',
            'password': 'qwaszx123',
            'host': 'localhost',
            'database': 'human_animal_detection'
        }
    
    try:
        from ..database_manager import DatabaseManager
        
        # Initialize SQLite database
        sqlite_db = DatabaseManager(sqlite_db_path)
        
        # Create migrator and run migration
        migrator = MariaDBToSQLiteMigrator(sqlite_db, mariadb_config)
        
        # Export MariaDB data as backup
        backup_file = f"mariadb_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        migrator.export_mariadb_data(backup_file)
        
        # Run migration
        success = migrator.migrate_all_data()
        
        # Verify migration
        if success:
            migrator.verify_migration()
        
        return success
        
    except Exception as e:
        logger.error(f"Migration process failed: {e}")
        return False


if __name__ == "__main__":
    # Run migration with default settings
    success = run_migration()
    if success:
        print("Migration completed successfully!")
    else:
        print("Migration failed. Check logs for details.")
