#!/usr/bin/env python3
"""
MariaDB to SQLite Migration Script

This script migrates data from a MariaDB backup JSON file to SQLite database.
"""

import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from database.database_manager import DatabaseManager
from database.models import Device, WhitelistEntry, NotificationSettings

logger = logging.getLogger(__name__)

class SQLiteMigration:
    """SQLite migration utility."""
    
    def __init__(self, sqlite_db_path: str = "detection_system.db"):
        """
        Initialize SQLite migration.
        
        Args:
            sqlite_db_path: Path to SQLite database
        """
        self.sqlite_db_path = sqlite_db_path
        self.migration_results = []
    
    def load_backup_data(self, backup_file: str) -> Dict[str, Any]:
        """Load backup data from JSON file."""
        print(f"📂 Loading backup data from {backup_file}")
        
        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            print("✅ Backup data loaded successfully")
            
            # Print backup info
            backup_info = backup_data.get('backup_info', {})
            print(f"📅 Backup timestamp: {backup_info.get('timestamp', 'Unknown')}")
            print(f"🗄️ Source database: {backup_info.get('source_database', 'Unknown')}")
            
            return backup_data
            
        except Exception as e:
            print(f"❌ Failed to load backup data: {e}")
            return {}
    
    def migrate_devices(self, db_manager: DatabaseManager, devices_data: List[Dict]) -> bool:
        """Migrate devices table."""
        print(f"📱 Migrating {len(devices_data)} devices...")
        
        success_count = 0
        
        for device_data in devices_data:
            try:
                device = Device(
                    ip_address=device_data['ip_address'],
                    port=device_data['port'],
                    use_https=bool(device_data.get('use_https', False)),
                    end_with_video=bool(device_data.get('end_with_video', False)),
                    status=device_data.get('status', 'active')
                )
                
                device_id = db_manager.add_device(device)
                if device_id:
                    success_count += 1
                    
            except Exception as e:
                self.migration_results.append(f"❌ Failed to migrate device {device_data.get('ip_address', 'Unknown')}: {e}")
        
        self.migration_results.append(f"✅ Migrated {success_count}/{len(devices_data)} devices")
        return success_count == len(devices_data)
    
    def migrate_whitelist(self, db_manager: DatabaseManager, whitelist_data: List[Dict]) -> bool:
        """Migrate whitelist table."""
        print(f"👥 Migrating {len(whitelist_data)} whitelist entries...")
        
        success_count = 0
        
        for entry_data in whitelist_data:
            try:
                entry = WhitelistEntry(
                    name=entry_data['name'],
                    entity_type=entry_data['entity_type'],
                    familiar=entry_data.get('familiar', 'familiar'),
                    color=entry_data.get('color'),
                    coco_class_id=entry_data.get('coco_class_id'),
                    image_path=entry_data['image_path'],
                    confidence_threshold=entry_data.get('confidence_threshold', 0.6),
                    pet_breed=entry_data.get('pet_breed'),
                    individual_id=entry_data.get('individual_id'),
                    identification_method=entry_data.get('identification_method', 'color')
                )
                
                entry_id = db_manager.add_whitelist_entry(entry)
                if entry_id:
                    success_count += 1
                    
            except Exception as e:
                self.migration_results.append(f"❌ Failed to migrate whitelist entry {entry_data.get('name', 'Unknown')}: {e}")
        
        self.migration_results.append(f"✅ Migrated {success_count}/{len(whitelist_data)} whitelist entries")
        return success_count == len(whitelist_data)
    
    def migrate_notifications(self, db_manager: DatabaseManager, notifications_data: List[Dict]) -> bool:
        """Migrate notification_settings table."""
        print(f"📱 Migrating {len(notifications_data)} notification settings...")
        
        success_count = 0
        
        for notification_data in notifications_data:
            try:
                notification = NotificationSettings(
                    chat_id=notification_data['chat_id'],
                    telegram_username=notification_data['telegram_username'],
                    notify_human_detection=bool(notification_data.get('notify_human_detection', True)),
                    notify_animal_detection=bool(notification_data.get('notify_animal_detection', True)),
                    sendstatus=notification_data.get('sendstatus', 'open')
                )
                
                notification_id = db_manager.add_notification_settings(notification)
                if notification_id:
                    success_count += 1
                    
            except Exception as e:
                self.migration_results.append(f"❌ Failed to migrate notification for {notification_data.get('telegram_username', 'Unknown')}: {e}")
        
        self.migration_results.append(f"✅ Migrated {success_count}/{len(notifications_data)} notification settings")
        return success_count == len(notifications_data)
    
    def migrate_data(self, backup_file: str) -> bool:
        """
        Migrate data from backup file to SQLite.
        
        Args:
            backup_file: Path to backup JSON file
            
        Returns:
            True if migration successful
        """
        print(f"🔄 Starting migration from {backup_file} to {self.sqlite_db_path}")
        
        # Load backup data
        backup_data = self.load_backup_data(backup_file)
        if not backup_data:
            return False
        
        try:
            # Initialize SQLite database
            db_manager = DatabaseManager(self.sqlite_db_path)
            
            tables_data = backup_data.get('tables', {})
            
            # Migrate each table
            migration_success = True
            
            # Migrate devices
            if 'devices' in tables_data:
                if not self.migrate_devices(db_manager, tables_data['devices']):
                    migration_success = False
            
            # Migrate whitelist
            if 'whitelist' in tables_data:
                if not self.migrate_whitelist(db_manager, tables_data['whitelist']):
                    migration_success = False
            
            # Migrate notification settings
            if 'notification_settings' in tables_data:
                if not self.migrate_notifications(db_manager, tables_data['notification_settings']):
                    migration_success = False
            
            return migration_success
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate MariaDB backup to SQLite")
    parser.add_argument("--input", required=True, help="Input backup JSON file")
    parser.add_argument("--output", default="detection_system.db", help="Output SQLite database")
    parser.add_argument("--force", action="store_true", help="Overwrite existing SQLite database")
    
    args = parser.parse_args()
    
    print("🚀 MariaDB to SQLite Migration Tool")
    print("=" * 50)
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Check if output database exists
    if Path(args.output).exists() and not args.force:
        print(f"❌ Output database already exists: {args.output}")
        print("💡 Use --force to overwrite existing database")
        sys.exit(1)
    
    migration = SQLiteMigration(args.output)
    
    if migration.migrate_data(args.input):
        print("\n" + "=" * 50)
        print("📋 MIGRATION RESULTS")
        print("=" * 50)
        
        for result in migration.migration_results:
            print(result)
        
        print("\n" + "=" * 50)
        print("✅ Migration completed successfully!")
        print("💡 Next steps:")
        print("1. Run verify_migration.py to verify the migration")
        print("2. Test the system with the new SQLite database")
        print("3. Update configuration to use SQLite")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
