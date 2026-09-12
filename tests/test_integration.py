#!/usr/bin/env python3
"""
Integration Tests for Intruder Detection System

This module contains comprehensive integration tests that test the complete
detection pipeline from camera input to notification output.
"""

import sys
import os
import time
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.detection_engine import DetectionEngine
from core.face_recognition import FaceRecognitionSystem
from core.animal_recognition import AnimalRecognitionSystem
from database.database_manager import DatabaseManager
from database.models import WhitelistEntry
from config.settings import Settings
from config.detection_config import DetectionConfig


class IntegrationTestCase(unittest.TestCase):
    """Base class for integration tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = Path(tempfile.mkdtemp(prefix="ids_test_"))
        cls.test_db_path = cls.test_dir / "test_detection.db"
        cls.test_config_path = cls.test_dir / "test_config.yaml"
        
        # Create test configuration
        cls._create_test_config()
        
        # Initialize test database
        cls.db_manager = DatabaseManager(str(cls.test_db_path))
        
        # Initialize settings and configs
        cls.settings = Settings()
        cls.settings.database_path = str(cls.test_db_path)
        cls.detection_config = DetectionConfig()
        
        print(f"Integration test environment set up in {cls.test_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        try:
            if cls.test_dir.exists():
                shutil.rmtree(cls.test_dir)
            print("Integration test environment cleaned up")
        except Exception as e:
            print(f"Warning: Failed to clean up test environment: {e}")
    
    @classmethod
    def _create_test_config(cls):
        """Create test configuration file."""
        test_config = {
            'video': {
                'frame_width': 640,
                'frame_height': 480,
                'target_fps': 30
            },
            'detection': {
                'yolo_confidence': 0.5,
                'human_confidence_threshold': 0.6,
                'animal_confidence_threshold': 0.6
            },
            'database': {
                'path': str(cls.test_db_path)
            }
        }
        
        import yaml
        with open(cls.test_config_path, 'w') as f:
            yaml.dump(test_config, f)
    
    def create_test_image(self, width: int = 640, height: int = 480, 
                         add_person: bool = False, add_animal: bool = False) -> np.ndarray:
        """Create a test image with optional person/animal shapes."""
        # Create base image
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        if add_person:
            # Add a simple person-like shape (rectangle for body, circle for head)
            cv2.rectangle(image, (width//2-30, height//2-50), (width//2+30, height//2+50), (255, 255, 255), -1)
            cv2.circle(image, (width//2, height//2-70), 20, (255, 255, 255), -1)
        
        if add_animal:
            # Add a simple animal-like shape
            cv2.ellipse(image, (width//4, height//2), (40, 20), 0, 0, 360, (128, 128, 128), -1)
        
        return image


class TestDetectionPipeline(IntegrationTestCase):
    """Test the complete detection pipeline."""
    
    def setUp(self):
        """Set up detection components."""
        self.detection_engine = DetectionEngine.from_config(self.detection_config)
        self.face_recognition = FaceRecognitionSystem()
        self.animal_recognition = AnimalRecognitionSystem()
    
    def test_detection_engine_initialization(self):
        """Test detection engine initialization."""
        self.assertIsNotNone(self.detection_engine)
        self.assertIsNotNone(self.detection_engine.model)
        self.assertEqual(self.detection_engine.confidence, self.detection_config.yolo_confidence)
    
    def test_basic_detection_pipeline(self):
        """Test basic detection without face/animal recognition."""
        # Create test image with person
        test_image = self.create_test_image(add_person=True)
        
        # Run detection
        results = self.detection_engine.detect_objects(test_image)
        
        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertIn('humans', results)
        self.assertIn('animals', results)
        self.assertIsInstance(results['humans'], list)
        self.assertIsInstance(results['animals'], list)
    
    def test_face_recognition_integration(self):
        """Test face recognition integration with detection."""
        # Create test image
        test_image = self.create_test_image(add_person=True)
        
        # Run detection
        detection_results = self.detection_engine.detect_objects(test_image)
        
        # Test face recognition on detected humans
        if detection_results['humans']:
            face_results = self.face_recognition.recognize_faces(test_image, detection_results['humans'])
            
            # Verify face recognition results
            self.assertIsInstance(face_results, list)
            for result in face_results:
                self.assertIn('bbox', result)
                self.assertIn('confidence', result)
    
    def test_animal_recognition_integration(self):
        """Test animal recognition integration with detection."""
        # Create test image with animal
        test_image = self.create_test_image(add_animal=True)
        
        # Run detection
        detection_results = self.detection_engine.detect_objects(test_image)
        
        # Test animal recognition on detected animals
        if detection_results['animals']:
            animal_results = self.animal_recognition.recognize_animals(test_image, detection_results['animals'])
            
            # Verify animal recognition results
            self.assertIsInstance(animal_results, list)
            for result in animal_results:
                self.assertIn('bbox', result)
                self.assertIn('confidence', result)
    
    def test_complete_detection_pipeline(self):
        """Test the complete detection pipeline with all components."""
        # Create test image with both person and animal
        test_image = self.create_test_image(add_person=True, add_animal=True)
        
        # Step 1: Object detection
        detection_results = self.detection_engine.detect_objects(test_image)
        
        # Step 2: Face recognition on humans
        if detection_results['humans']:
            face_results = self.face_recognition.recognize_faces(test_image, detection_results['humans'])
            detection_results['humans'] = face_results
        
        # Step 3: Animal recognition on animals
        if detection_results['animals']:
            animal_results = self.animal_recognition.recognize_animals(test_image, detection_results['animals'])
            detection_results['animals'] = animal_results
        
        # Verify complete pipeline results
        self.assertIsInstance(detection_results, dict)
        self.assertIn('humans', detection_results)
        self.assertIn('animals', detection_results)
        
        # Verify processing metadata. `processing_time` is nested inside
        # frame_info alongside the timestamp and model format — the test used
        # to look for it at the top level, where it has never been.
        self.assertIn('frame_info', detection_results)
        self.assertIn('processing_time', detection_results['frame_info'])
        self.assertIsInstance(
            detection_results['frame_info']['processing_time'], float
        )


class TestDatabaseIntegration(IntegrationTestCase):
    """Test database integration with detection system."""
    
    def test_detection_logging(self):
        """Test logging detection results to database."""
        # Create test detection data
        detection_data = {
            'timestamp': time.time(),
            'camera_id': 'test_camera',
            'detection_type': 'human',
            'confidence': 0.85,
            'bbox': [100, 100, 200, 200],
            'image_path': 'test_image.jpg'
        }
        
        # Log detection to database
        # Keywords, because the positional order here was wrong on every
        # argument: log_detection's first parameter is `detection_type`, so
        # the camera id was being logged as the type, the type as the entity
        # name, and so on. It also has no `bbox` parameter at all.
        success = self.db_manager.log_detection(
            detection_type=detection_data['detection_type'],
            entity_name='Test Person',
            confidence=detection_data['confidence'],
            image_path=detection_data['image_path'],
        )

        self.assertTrue(success)

        # And it can be read back as what it was logged as.
        logged = self.db_manager.get_recent_detections(limit=1)
        self.assertEqual(len(logged), 1)
        self.assertEqual(logged[0].detection_type, 'human')
        self.assertEqual(logged[0].entity_name, 'Test Person')
        # DetectionLog is a dataclass, not a dict — the old assertion indexed
        # it with ['camera_id'] and raised TypeError. camera_id is an int FK
        # to devices, never the free-text 'test_camera' this test invented.
    
    def test_face_data_management(self):
        """A known human is stored in the whitelist and read back."""
        # `add_known_face` and `get_known_faces` never existed on
        # DatabaseManager. Known people are whitelist rows with
        # entity_type='human', and get_known_humans() reads them back.
        entry = WhitelistEntry(
            name='Test Person',
            entity_type='human',
            familiar='familiar',
            image_path='test_face.jpg',
        )

        entry_id = self.db_manager.create_whitelist_entry(entry)
        self.assertIsInstance(entry_id, int)

        humans = self.db_manager.get_known_humans()
        self.assertGreater(len(humans), 0)

        stored = next((h for h in humans if h.name == 'Test Person'), None)
        self.assertIsNotNone(stored)
        self.assertEqual(stored.image_path, 'test_face.jpg')
        self.assertEqual(stored.entity_type, 'human')


class TestSystemConfiguration(IntegrationTestCase):
    """Test system configuration and settings."""
    
    def test_settings_loading(self):
        """Test settings loading from configuration."""
        # Test loading settings with environment support
        settings = Settings.load_with_env_support(str(self.test_config_path))
        
        self.assertIsNotNone(settings)
        self.assertEqual(settings.database_path, str(self.test_db_path))
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test with valid configuration
        self.assertTrue(self.detection_config.yolo_confidence > 0)
        self.assertTrue(self.detection_config.yolo_confidence <= 1)
        
        # Test settings validation
        self.assertIsInstance(self.settings.frame_width, int)
        self.assertIsInstance(self.settings.frame_height, int)
        self.assertGreater(self.settings.frame_width, 0)
        self.assertGreater(self.settings.frame_height, 0)


def run_integration_tests(verbose: bool = True) -> bool:
    """
    Run all integration tests.
    
    Args:
        verbose: Whether to print detailed output
        
    Returns:
        True if all tests passed
    """
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestDetectionPipeline))
    suite.addTest(unittest.makeSuite(TestDatabaseIntegration))
    suite.addTest(unittest.makeSuite(TestSystemConfiguration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running Integration Tests for Intruder Detection System")
    print("=" * 60)
    
    success = run_integration_tests()
    
    if success:
        print("\n✅ All integration tests passed!")
    else:
        print("\n❌ Some integration tests failed!")
        sys.exit(1)
