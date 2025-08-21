#!/usr/bin/env python3
"""
SQLite Database Creation Script

This script creates a new SQLite database with the complete schema
for the Intruder Detection System.
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class SQLiteDBCreator:
    """SQLite database creation utility."""
    
    def __init__(self, db_path: str = "detection_system.db"):
        self.db_path = db_path
        self.project_root = Path(__file__).parent.parent
    
    def create_database(self, force: bool = False):
        """
        Create a new SQLite database.
        
        Args:
            force: If True, overwrite existing database
        """
        print(f"🗄️ Creating SQLite database: {self.db_path}")
        
        # Check if database already exists
        if os.path.exists(self.db_path) and not force:
            print(f"❌ Database {self.db_path} already exists!")
            print("💡 Use --force to overwrite existing database")
            return False
        
        try:
            # Remove existing database if force is True
            if force and os.path.exists(self.db_path):
                os.remove(self.db_path)
                print(f"🗑️ Removed existing database: {self.db_path}")
            
            # Create new database using DatabaseManager
            db_manager = DatabaseManager(self.db_path)
            
            print("✅ Database created successfully!")
            
            # Verify database creation
            self.verify_database()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create database: {e}")
            return False
    
    def verify_database(self):
        """Verify that the database was created correctly."""
        print("🔍 Verifying database structure...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                expected_tables = [
                    'devices', 'whitelist', 'notification_settings',
                    'detection_logs', 'system_metrics', 'system_config'
                ]
                
                print(f"📊 Found {len(tables)} tables:")
                for table in tables:
                    if table != 'sqlite_sequence':  # Skip internal table
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        status = "✅" if table in expected_tables else "⚠️"
                        print(f"  {status} {table}: {count} records")
                
                # Check for missing tables
                missing_tables = set(expected_tables) - set(tables)
                if missing_tables:
                    print(f"❌ Missing tables: {', '.join(missing_tables)}")
                else:
                    print("✅ All expected tables found")
                
                # Get all views
                cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
                views = [row[0] for row in cursor.fetchall()]
                
                expected_views = [
                    'active_cameras', 'known_humans', 
                    'familiar_animals', 'active_telegram_users'
                ]
                
                print(f"👁️ Found {len(views)} views:")
                for view in views:
                    status = "✅" if view in expected_views else "⚠️"
                    print(f"  {status} {view}")
                
                # Check for missing views
                missing_views = set(expected_views) - set(views)
                if missing_views:
                    print(f"❌ Missing views: {', '.join(missing_views)}")
                else:
                    print("✅ All expected views found")
                
                # Get all indexes
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
                indexes = [row[0] for row in cursor.fetchall()]
                
                print(f"📇 Found {len(indexes)} indexes:")
                for index in indexes:
                    print(f"  ✅ {index}")
                
                # Get all triggers
                cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
                triggers = [row[0] for row in cursor.fetchall()]
                
                print(f"⚡ Found {len(triggers)} triggers:")
                for trigger in triggers:
                    print(f"  ✅ {trigger}")
                
        except Exception as e:
            print(f"❌ Database verification failed: {e}")
    
    def add_sample_data(self):
        """Add sample data to the database for testing."""
        print("📝 Adding sample data...")
        
        try:
            db_manager = DatabaseManager(self.db_path)
            
            # Add sample device
            from database.models import Device
            sample_device = Device(
                ip_address="192.168.1.100",
                port=8080,
                use_https=False,
                end_with_video=True,
                status="active"
            )
            
            device_id = db_manager.add_device(sample_device)
            if device_id:
                print("✅ Added sample IP camera device")
            
            # Add sample human
            from database.models import WhitelistEntry
            sample_human = WhitelistEntry(
                name="John Doe",
                entity_type="human",
                familiar="familiar",
                image_path="data/faces/john_doe.jpg"
            )
            
            human_id = db_manager.add_whitelist_entry(sample_human)
            if human_id:
                print("✅ Added sample human entry")
            
            # Add sample animal
            sample_animal = WhitelistEntry(
                name="Fluffy",
                entity_type="animal",
                familiar="familiar",
                color="white",
                coco_class_id=15,  # cat
                image_path="data/animals/fluffy_cat.jpg",
                individual_id="fluffy",
                pet_breed="persian",
                identification_method="hybrid"
            )
            
            animal_id = db_manager.add_whitelist_entry(sample_animal)
            if animal_id:
                print("✅ Added sample animal entry")
            
            # Add sample notification settings
            from database.models import NotificationSettings
            sample_notification = NotificationSettings(
                chat_id=123456789,
                telegram_username="john_doe_telegram",
                notify_human_detection=True,
                notify_animal_detection=True,
                sendstatus="open"
            )
            
            notification_id = db_manager.add_notification_settings(sample_notification)
            if notification_id:
                print("✅ Added sample notification settings")
            
            print("✅ Sample data added successfully!")
            
        except Exception as e:
            print(f"❌ Failed to add sample data: {e}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create SQLite database for Intruder Detection System")
    parser.add_argument("--db-path", default="detection_system.db", help="Database file path")
    parser.add_argument("--force", action="store_true", help="Overwrite existing database")
    parser.add_argument("--sample-data", action="store_true", help="Add sample data")
    
    args = parser.parse_args()
    
    print("🚀 SQLite Database Creator")
    print("=" * 50)
    
    creator = SQLiteDBCreator(args.db_path)
    
    if creator.create_database(force=args.force):
        if args.sample_data:
            creator.add_sample_data()
        
        print("\n" + "=" * 50)
        print("✅ Database creation completed!")
        print(f"📁 Database file: {args.db_path}")
        print("💡 Next steps:")
        print("1. Run setup_environment.py to complete setup")
        print("2. Add your own data using the GUI")
        print("3. Start the detection system with main.py")
    else:
        print("\n❌ Database creation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
