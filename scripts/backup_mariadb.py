#!/usr/bin/env python3
"""
MariaDB Backup Script

This script backs up data from a MariaDB database to JSON format
for migration to SQLite.
"""

import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class MariaDBBackup:
    """MariaDB backup utility."""
    
    def __init__(self, host: str = "localhost", port: int = 3306, 
                 user: str = "root", password: str = "", 
                 database: str = "human_animal_detection"):
        """
        Initialize MariaDB backup.
        
        Args:
            host: MariaDB host
            port: MariaDB port
            user: Database user
            password: Database password
            database: Database name
        """
        self.config = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database
        }
    
    def test_connection(self) -> bool:
        """Test MariaDB connection."""
        try:
            import mariadb
            
            conn = mariadb.connect(**self.config)
            conn.close()
            print("✅ MariaDB connection successful")
            return True
            
        except ImportError:
            print("❌ mariadb package not installed")
            print("💡 Install with: pip install mariadb")
            return False
        except Exception as e:
            print(f"❌ MariaDB connection failed: {e}")
            return False
    
    def backup_table(self, conn, table_name: str) -> List[Dict[str, Any]]:
        """Backup a single table."""
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name}")
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Get all rows
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            table_data = []
            for row in rows:
                row_dict = {}
                for i, value in enumerate(row):
                    # Convert datetime objects to strings
                    if hasattr(value, 'isoformat'):
                        value = value.isoformat()
                    row_dict[columns[i]] = value
                table_data.append(row_dict)
            
            print(f"✅ Backed up {len(table_data)} records from {table_name}")
            return table_data
            
        except Exception as e:
            print(f"❌ Failed to backup table {table_name}: {e}")
            return []
    
    def backup_database(self, output_file: str) -> bool:
        """
        Backup the entire MariaDB database to JSON.
        
        Args:
            output_file: Output JSON file path
            
        Returns:
            True if backup successful
        """
        print(f"🗄️ Starting MariaDB backup to {output_file}")
        
        try:
            import mariadb
            
            # Connect to MariaDB
            conn = mariadb.connect(**self.config)
            
            backup_data = {
                'backup_info': {
                    'timestamp': datetime.now().isoformat(),
                    'source_database': self.config['database'],
                    'source_host': self.config['host'],
                    'backup_version': '1.0'
                },
                'tables': {}
            }
            
            # Tables to backup
            tables = ['devices', 'whitelist', 'notification_settings']
            
            for table in tables:
                print(f"📊 Backing up table: {table}")
                table_data = self.backup_table(conn, table)
                backup_data['tables'][table] = table_data
            
            # Close connection
            conn.close()
            
            # Write to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✅ Backup completed successfully!")
            print(f"📁 Backup file: {output_file}")
            
            # Print backup summary
            total_records = sum(len(data) for data in backup_data['tables'].values())
            print(f"📊 Total records backed up: {total_records}")
            
            for table, data in backup_data['tables'].items():
                print(f"  • {table}: {len(data)} records")
            
            return True
            
        except ImportError:
            print("❌ mariadb package not installed")
            print("💡 Install with: pip install mariadb")
            return False
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    
    def verify_backup(self, backup_file: str) -> bool:
        """Verify the backup file."""
        print(f"🔍 Verifying backup file: {backup_file}")
        
        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # Check backup structure
            required_keys = ['backup_info', 'tables']
            for key in required_keys:
                if key not in backup_data:
                    print(f"❌ Missing key in backup: {key}")
                    return False
            
            # Check tables
            expected_tables = ['devices', 'whitelist', 'notification_settings']
            for table in expected_tables:
                if table not in backup_data['tables']:
                    print(f"⚠️ Missing table in backup: {table}")
                else:
                    count = len(backup_data['tables'][table])
                    print(f"✅ {table}: {count} records")
            
            # Check backup info
            backup_info = backup_data['backup_info']
            print(f"📅 Backup timestamp: {backup_info.get('timestamp', 'Unknown')}")
            print(f"🗄️ Source database: {backup_info.get('source_database', 'Unknown')}")
            
            print("✅ Backup verification completed")
            return True
            
        except Exception as e:
            print(f"❌ Backup verification failed: {e}")
            return False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backup MariaDB database to JSON")
    parser.add_argument("--host", default="localhost", help="MariaDB host")
    parser.add_argument("--port", type=int, default=3306, help="MariaDB port")
    parser.add_argument("--user", default="root", help="Database user")
    parser.add_argument("--password", default="", help="Database password")
    parser.add_argument("--database", default="human_animal_detection", help="Database name")
    parser.add_argument("--output", default="mariadb_backup.json", help="Output JSON file")
    parser.add_argument("--test-connection", action="store_true", help="Test connection only")
    parser.add_argument("--verify", help="Verify existing backup file")
    
    args = parser.parse_args()
    
    print("🚀 MariaDB Backup Tool")
    print("=" * 50)
    
    backup = MariaDBBackup(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database
    )
    
    if args.verify:
        # Verify existing backup
        if backup.verify_backup(args.verify):
            print("✅ Backup verification successful")
        else:
            print("❌ Backup verification failed")
            sys.exit(1)
    elif args.test_connection:
        # Test connection only
        if not backup.test_connection():
            sys.exit(1)
    else:
        # Perform backup
        if not backup.test_connection():
            sys.exit(1)
        
        if backup.backup_database(args.output):
            print("\n" + "=" * 50)
            print("✅ MariaDB backup completed!")
            print("💡 Next steps:")
            print("1. Verify backup with --verify option")
            print("2. Run migrate_to_sqlite.py to import to SQLite")
            print("3. Run verify_migration.py to verify migration")
        else:
            print("\n❌ MariaDB backup failed!")
            sys.exit(1)

if __name__ == "__main__":
    main()
