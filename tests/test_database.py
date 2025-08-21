#!/usr/bin/env python3
"""
Database System Tests

Comprehensive tests for the database manager and models.
"""

import unittest
import sys
import os
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database.database_manager import DatabaseManager
from database.models import Device, WhitelistEntry, NotificationSettings, DetectionLog, SystemMetrics

class TestDatabaseManager(unittest.TestCase):
    """Test cases for the database manager."""
    
    def setUp(self):
        """Set up test database."""
        # Create temporary database file
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.db_manager = DatabaseManager(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test database."""
        # Remove temporary database file
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test database initialization."""
        self.assertIsNotNone(self.db_manager)
        
        # Check that tables exist
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['devices', 'whitelist', 'notification_settings', 
                             'detection_logs', 'system_metrics', 'system_config']
            
            for table in expected_tables:
                self.assertIn(table, tables)
    
    def test_database_views(self):
        """Test database views creation."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='view'")
            views = [row[0] for row in cursor.fetchall()]
            
            expected_views = ['active_cameras', 'known_humans', 
                            'familiar_animals', 'active_telegram_users']
            
            for view in expected_views:
                self.assertIn(view, views)
    
    def test_add_device(self):
        """Test adding a device."""
        device = Device(
            ip_address="192.168.1.100",
            port=8080,
            use_https=False,
            end_with_video=True,
            status="active"
        )
        
        device_id = self.db_manager.add_device(device)
        self.assertIsNotNone(device_id)
        self.assertIsInstance(device_id, int)
    
    def test_get_device(self):
        """Test retrieving a device."""
        # Add device first
        device = Device(
            ip_address="192.168.1.101",
            port=8080,
            use_https=False,
            end_with_video=True,
            status="active"
        )
        
        device_id = self.db_manager.add_device(device)
        
        # Retrieve device
        retrieved_device = self.db_manager.get_device(device_id)
        self.assertIsNotNone(retrieved_device)
        self.assertEqual(retrieved_device.ip_address, "192.168.1.101")
        self.assertEqual(retrieved_device.port, 8080)
    
    def test_add_whitelist_entry(self):
        """Test adding a whitelist entry."""
        entry = WhitelistEntry(
            name="Test Person",
            entity_type="human",
            familiar="familiar",
            image_path="test/path.jpg"
        )
        
        entry_id = self.db_manager.add_whitelist_entry(entry)
        self.assertIsNotNone(entry_id)
        self.assertIsInstance(entry_id, int)
    
    def test_add_notification_settings(self):
        """Test adding notification settings."""
        settings = NotificationSettings(
            chat_id=123456789,
            telegram_username="test_user",
            notify_human_detection=True,
            notify_animal_detection=True,
            sendstatus="open"
        )
        
        settings_id = self.db_manager.add_notification_settings(settings)
        self.assertIsNotNone(settings_id)
        self.assertIsInstance(settings_id, int)
    
    def test_log_detection(self):
        """Test logging a detection."""
        success = self.db_manager.log_detection(
            detection_type="human",
            entity_name="Test Person",
            confidence=0.85,
            camera_id=None,
            notification_sent=True
        )
        
        self.assertTrue(success)
    
    def test_get_recent_detections(self):
        """Test retrieving recent detections."""
        # Log some detections first
        self.db_manager.log_detection("human", "Person 1", 0.9, None, True)
        self.db_manager.log_detection("animal", "Dog 1", 0.8, None, False)
        
        detections = self.db_manager.get_recent_detections(10)
        self.assertIsInstance(detections, list)
        self.assertGreaterEqual(len(detections), 2)
    
    def test_log_system_metric(self):
        """Test logging system metrics."""
        success = self.db_manager.log_system_metric("fps", 25.5, "fps")
        self.assertTrue(success)
        
        success = self.db_manager.log_system_metric("cpu_usage", 45.2, "%")
        self.assertTrue(success)
    
    def test_get_recent_metrics(self):
        """Test retrieving recent metrics."""
        # Log some metrics first
        self.db_manager.log_system_metric("fps", 30.0, "fps")
        self.db_manager.log_system_metric("fps", 28.5, "fps")
        
        metrics = self.db_manager.get_recent_metrics("fps", 10)
        self.assertIsInstance(metrics, list)
        self.assertGreaterEqual(len(metrics), 2)
    
    def test_database_stats(self):
        """Test getting database statistics."""
        stats = self.db_manager.get_database_stats()
        self.assertIsInstance(stats, dict)
        
        # Should have count keys for all tables
        expected_keys = ['devices_count', 'whitelist_count', 'notification_settings_count',
                        'detection_logs_count', 'system_metrics_count', 'system_config_count']
        
        for key in expected_keys:
            self.assertIn(key, stats)

class TestDatabaseModels(unittest.TestCase):
    """Test cases for database models."""
    
    def test_device_model(self):
        """Test Device model."""
        device = Device(
            ip_address="192.168.1.100",
            port=8080,
            use_https=True,
            end_with_video=False,
            status="active"
        )
        
        self.assertEqual(device.ip_address, "192.168.1.100")
        self.assertEqual(device.port, 8080)
        self.assertTrue(device.use_https)
        self.assertFalse(device.end_with_video)
        self.assertEqual(device.status, "active")
    
    def test_whitelist_entry_model(self):
        """Test WhitelistEntry model."""
        entry = WhitelistEntry(
            name="John Doe",
            entity_type="human",
            familiar="familiar",
            image_path="faces/john_doe.jpg",
            confidence_threshold=0.7
        )
        
        self.assertEqual(entry.name, "John Doe")
        self.assertEqual(entry.entity_type, "human")
        self.assertEqual(entry.familiar, "familiar")
        self.assertEqual(entry.confidence_threshold, 0.7)
    
    def test_notification_settings_model(self):
        """Test NotificationSettings model."""
        settings = NotificationSettings(
            chat_id=123456789,
            telegram_username="john_doe",
            notify_human_detection=True,
            notify_animal_detection=False,
            sendstatus="open"
        )
        
        self.assertEqual(settings.chat_id, 123456789)
        self.assertEqual(settings.telegram_username, "john_doe")
        self.assertTrue(settings.notify_human_detection)
        self.assertFalse(settings.notify_animal_detection)
    
    def test_detection_log_model(self):
        """Test DetectionLog model."""
        log = DetectionLog(
            detection_type="human",
            entity_name="Unknown Person",
            confidence=0.85,
            camera_id=1,
            notification_sent=True
        )
        
        self.assertEqual(log.detection_type, "human")
        self.assertEqual(log.entity_name, "Unknown Person")
        self.assertEqual(log.confidence, 0.85)
        self.assertEqual(log.camera_id, 1)
        self.assertTrue(log.notification_sent)
    
    def test_system_metrics_model(self):
        """Test SystemMetrics model."""
        metric = SystemMetrics(
            metric_type="fps",
            metric_value=30.5,
            unit="fps"
        )
        
        self.assertEqual(metric.metric_type, "fps")
        self.assertEqual(metric.metric_value, 30.5)
        self.assertEqual(metric.unit, "fps")

class TestDatabasePerformance(unittest.TestCase):
    """Test cases for database performance."""
    
    def setUp(self):
        """Set up test database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_manager = DatabaseManager(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test database."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_bulk_detection_logging(self):
        """Test bulk detection logging performance."""
        import time
        
        start_time = time.time()
        
        # Log 100 detections
        for i in range(100):
            self.db_manager.log_detection(
                detection_type="human" if i % 2 == 0 else "animal",
                entity_name=f"Entity {i}",
                confidence=0.5 + (i % 50) / 100,
                camera_id=None,
                notification_sent=i % 3 == 0
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(total_time, 5.0)
        
        print(f"Bulk logging time for 100 records: {total_time:.3f} seconds")
    
    def test_query_performance(self):
        """Test query performance."""
        import time
        
        # Add some test data
        for i in range(50):
            self.db_manager.log_detection(
                detection_type="human",
                entity_name=f"Person {i}",
                confidence=0.8,
                camera_id=None,
                notification_sent=True
            )
        
        start_time = time.time()
        
        # Query recent detections
        detections = self.db_manager.get_recent_detections(25)
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # Query should be fast
        self.assertLess(query_time, 1.0)
        self.assertEqual(len(detections), 25)
        
        print(f"Query time for 25 records: {query_time:.3f} seconds")

def run_database_tests():
    """Run all database tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestDatabaseManager))
    suite.addTest(unittest.makeSuite(TestDatabaseModels))
    suite.addTest(unittest.makeSuite(TestDatabasePerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("🗄️ Running Database System Tests")
    print("=" * 50)
    
    success = run_database_tests()
    
    if success:
        print("\n✅ All database tests passed!")
    else:
        print("\n❌ Some database tests failed!")
        sys.exit(1)
