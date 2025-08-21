#!/usr/bin/env python3
"""
Migration Verification Script

This script verifies that the migration from MariaDB to SQLite was successful
by comparing record counts and data integrity.
"""

import json
import sys
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class MigrationVerifier:
    """Migration verification utility."""
    
    def __init__(self, sqlite_db_path: str = "detection_system.db"):
        """
        Initialize migration verifier.
        
        Args:
            sqlite_db_path: Path to SQLite database
        """
        self.sqlite_db_path = sqlite_db_path
        self.verification_results = []
    
    def load_backup_data(self, backup_file: str) -> Dict[str, Any]:
        """Load original backup data for comparison."""
        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load backup data: {e}")
            return {}
    
    def verify_record_counts(self, backup_data: Dict[str, Any]) -> bool:
        """Verify that record counts match between backup and SQLite."""
        print("📊 Verifying record counts...")
        
        try:
            db_manager = DatabaseManager(self.sqlite_db_path)
            sqlite_stats = db_manager.get_database_stats()
            
            tables_data = backup_data.get('tables', {})
            verification_success = True
            
            for table_name, table_data in tables_data.items():
                backup_count = len(table_data)
                sqlite_count = sqlite_stats.get(f"{table_name}_count", 0)
                
                if backup_count == sqlite_count:
                    self.verification_results.append(f"✅ {table_name}: {backup_count} records (match)")
                else:
                    self.verification_results.append(f"❌ {table_name}: backup={backup_count}, sqlite={sqlite_count} (mismatch)")
                    verification_success = False
            
            return verification_success
            
        except Exception as e:
            print(f"❌ Record count verification failed: {e}")
            return False
    
    def verify_data_integrity(self, backup_data: Dict[str, Any]) -> bool:
        """Verify data integrity by sampling records."""
        print("🔍 Verifying data integrity...")
        
        try:
            db_manager = DatabaseManager(self.sqlite_db_path)
            tables_data = backup_data.get('tables', {})
            verification_success = True
            
            # Verify devices
            if 'devices' in tables_data:
                devices_backup = tables_data['devices']
                devices_sqlite = db_manager.get_all_devices()
                
                if len(devices_backup) == len(devices_sqlite):
                    # Sample verification - check first device if exists
                    if devices_backup and devices_sqlite:
                        backup_device = devices_backup[0]
                        sqlite_device = devices_sqlite[0]
                        
                        if (backup_device['ip_address'] == sqlite_device.ip_address and
                            backup_device['port'] == sqlite_device.port):
                            self.verification_results.append("✅ Devices data integrity verified")
                        else:
                            self.verification_results.append("❌ Devices data integrity failed")
                            verification_success = False
                    else:
                        self.verification_results.append("ℹ️ No devices to verify")
                else:
                    self.verification_results.append("❌ Devices count mismatch")
                    verification_success = False
            
            # Verify whitelist
            if 'whitelist' in tables_data:
                whitelist_backup = tables_data['whitelist']
                whitelist_sqlite = db_manager.get_whitelist_entries()
                
                if len(whitelist_backup) == len(whitelist_sqlite):
                    # Sample verification - check first entry if exists
                    if whitelist_backup and whitelist_sqlite:
                        backup_entry = whitelist_backup[0]
                        sqlite_entry = whitelist_sqlite[0]
                        
                        if (backup_entry['name'] == sqlite_entry.name and
                            backup_entry['entity_type'] == sqlite_entry.entity_type):
                            self.verification_results.append("✅ Whitelist data integrity verified")
                        else:
                            self.verification_results.append("❌ Whitelist data integrity failed")
                            verification_success = False
                    else:
                        self.verification_results.append("ℹ️ No whitelist entries to verify")
                else:
                    self.verification_results.append("❌ Whitelist count mismatch")
                    verification_success = False
            
            # Verify notification settings
            if 'notification_settings' in tables_data:
                notifications_backup = tables_data['notification_settings']
                notifications_sqlite = db_manager.get_all_notification_settings()
                
                if len(notifications_backup) == len(notifications_sqlite):
                    # Sample verification - check first notification if exists
                    if notifications_backup and notifications_sqlite:
                        backup_notification = notifications_backup[0]
                        sqlite_notification = notifications_sqlite[0]
                        
                        if (backup_notification['chat_id'] == sqlite_notification.chat_id and
                            backup_notification['telegram_username'] == sqlite_notification.telegram_username):
                            self.verification_results.append("✅ Notification settings data integrity verified")
                        else:
                            self.verification_results.append("❌ Notification settings data integrity failed")
                            verification_success = False
                    else:
                        self.verification_results.append("ℹ️ No notification settings to verify")
                else:
                    self.verification_results.append("❌ Notification settings count mismatch")
                    verification_success = False
            
            return verification_success
            
        except Exception as e:
            print(f"❌ Data integrity verification failed: {e}")
            return False
    
    def verify_database_structure(self) -> bool:
        """Verify that the SQLite database has the correct structure."""
        print("🏗️ Verifying database structure...")
        
        try:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                cursor = conn.cursor()
                
                # Check tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                expected_tables = [
                    'devices', 'whitelist', 'notification_settings',
                    'detection_logs', 'system_metrics', 'system_config'
                ]
                
                missing_tables = set(expected_tables) - set(tables)
                if missing_tables:
                    self.verification_results.append(f"❌ Missing tables: {', '.join(missing_tables)}")
                    return False
                else:
                    self.verification_results.append("✅ All required tables present")
                
                # Check views
                cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
                views = [row[0] for row in cursor.fetchall()]
                
                expected_views = [
                    'active_cameras', 'known_humans', 
                    'familiar_animals', 'active_telegram_users'
                ]
                
                missing_views = set(expected_views) - set(views)
                if missing_views:
                    self.verification_results.append(f"❌ Missing views: {', '.join(missing_views)}")
                    return False
                else:
                    self.verification_results.append("✅ All required views present")
                
                # Check indexes
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
                indexes = [row[0] for row in cursor.fetchall()]
                
                if len(indexes) >= 8:  # We expect at least 8 indexes
                    self.verification_results.append(f"✅ Database indexes present ({len(indexes)} found)")
                else:
                    self.verification_results.append(f"⚠️ Few database indexes found ({len(indexes)})")
                
                return True
                
        except Exception as e:
            print(f"❌ Database structure verification failed: {e}")
            return False
    
    def verify_migration(self, backup_file: str = None) -> bool:
        """
        Perform complete migration verification.
        
        Args:
            backup_file: Original backup file for comparison
            
        Returns:
            True if verification successful
        """
        print(f"🔍 Starting migration verification for {self.sqlite_db_path}")
        
        # Check if SQLite database exists
        if not Path(self.sqlite_db_path).exists():
            print(f"❌ SQLite database not found: {self.sqlite_db_path}")
            return False
        
        verification_success = True
        
        # Verify database structure
        if not self.verify_database_structure():
            verification_success = False
        
        # If backup file provided, verify data migration
        if backup_file:
            backup_data = self.load_backup_data(backup_file)
            if backup_data:
                if not self.verify_record_counts(backup_data):
                    verification_success = False
                
                if not self.verify_data_integrity(backup_data):
                    verification_success = False
            else:
                self.verification_results.append("⚠️ Could not load backup data for comparison")
        
        return verification_success

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify MariaDB to SQLite migration")
    parser.add_argument("--sqlite-db", default="detection_system.db", help="SQLite database path")
    parser.add_argument("--backup-file", help="Original backup JSON file for comparison")
    
    args = parser.parse_args()
    
    print("🚀 Migration Verification Tool")
    print("=" * 50)
    
    verifier = MigrationVerifier(args.sqlite_db)
    
    if verifier.verify_migration(args.backup_file):
        print("\n" + "=" * 50)
        print("📋 VERIFICATION RESULTS")
        print("=" * 50)
        
        for result in verifier.verification_results:
            print(result)
        
        print("\n" + "=" * 50)
        print("✅ Migration verification successful!")
        print("💡 Your SQLite database is ready to use")
        print("🚀 You can now start the detection system with: python main.py")
    else:
        print("\n" + "=" * 50)
        print("📋 VERIFICATION RESULTS")
        print("=" * 50)
        
        for result in verifier.verification_results:
            print(result)
        
        print("\n❌ Migration verification failed!")
        print("💡 Please check the migration process and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
