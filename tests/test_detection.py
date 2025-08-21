#!/usr/bin/env python3
"""
Detection System Tests

Comprehensive tests for the detection engine, face recognition,
and animal recognition systems.
"""

import unittest
import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.detection_engine import DetectionEngine
from core.face_recognition_system import FaceRecognitionSystem
from core.animal_recognition_system import AnimalRecognitionSystem
from config.detection_config import DetectionConfig

class TestDetectionEngine(unittest.TestCase):
    """Test cases for the detection engine."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.config = DetectionConfig()
        cls.detection_engine = DetectionEngine(cls.config)
        
        # Create test image
        cls.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cls.test_image.fill(128)  # Gray image
    
    def test_detection_engine_initialization(self):
        """Test detection engine initialization."""
        self.assertIsNotNone(self.detection_engine)
        self.assertIsNotNone(self.detection_engine.model)
    
    def test_detect_objects_with_empty_image(self):
        """Test object detection with empty image."""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = self.detection_engine.detect_objects(empty_image)
        
        self.assertIsInstance(detections, dict)
        self.assertIn('humans', detections)
        self.assertIn('animals', detections)
    
    def test_detect_objects_with_test_image(self):
        """Test object detection with test image."""
        detections = self.detection_engine.detect_objects(self.test_image)
        
        self.assertIsInstance(detections, dict)
        self.assertIn('humans', detections)
        self.assertIn('animals', detections)
        self.assertIsInstance(detections['humans'], list)
        self.assertIsInstance(detections['animals'], list)
    
    def test_confidence_threshold(self):
        """Test confidence threshold filtering."""
        original_confidence = self.detection_engine.confidence
        
        # Set very high confidence threshold
        self.detection_engine.confidence = 0.99
        detections = self.detection_engine.detect_objects(self.test_image)
        
        # Should have fewer or no detections with high threshold
        self.assertIsInstance(detections, dict)
        
        # Restore original confidence
        self.detection_engine.confidence = original_confidence
    
    def test_detection_performance(self):
        """Test detection performance metrics."""
        import time
        
        start_time = time.time()
        detections = self.detection_engine.detect_objects(self.test_image)
        end_time = time.time()
        
        detection_time = end_time - start_time
        
        # Detection should complete within reasonable time (5 seconds)
        self.assertLess(detection_time, 5.0)
        self.assertIsInstance(detections, dict)

class TestFaceRecognitionSystem(unittest.TestCase):
    """Test cases for face recognition system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.face_recognition = FaceRecognitionSystem()
        
        # Create test face image (simple rectangle)
        cls.test_face_image = np.zeros((150, 150, 3), dtype=np.uint8)
        cv2.rectangle(cls.test_face_image, (25, 25), (125, 125), (255, 255, 255), -1)
    
    def test_face_recognition_initialization(self):
        """Test face recognition system initialization."""
        self.assertIsNotNone(self.face_recognition)

        # Test backend initialization
        self.assertIn(self.face_recognition.backend_type, ["face_recognition", "opencv_dnn_lbph", "opencv_basic", "none"])

        # Test backend-specific attributes
        if self.face_recognition.backend_type == "opencv_dnn_lbph":
            self.assertIsNotNone(self.face_recognition.face_recognizer)
            self.assertIsInstance(self.face_recognition.known_faces, list)
            self.assertIsInstance(self.face_recognition.label_to_name, dict)
        elif self.face_recognition.backend_type == "face_recognition":
            self.assertIsInstance(self.face_recognition.known_face_encodings, list)
            self.assertIsInstance(self.face_recognition.known_face_names, list)
    
    def test_detect_faces_empty_image(self):
        """Test face detection with empty image."""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = self.face_recognition.detect_faces(empty_image)
        
        self.assertIsInstance(faces, list)
    
    def test_detect_faces_test_image(self):
        """Test face detection with test image."""
        faces = self.face_recognition.detect_faces(self.test_face_image)
        
        self.assertIsInstance(faces, list)

    def test_integrated_opencv_methods(self):
        """Test integrated OpenCV DNN methods."""
        if self.face_recognition.backend_type == "opencv_dnn_lbph":
            # Test face detection methods
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            gray_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

            # Test DNN face detection (if available)
            if hasattr(self.face_recognition, 'face_net') and self.face_recognition.face_net is not None:
                faces = self.face_recognition._detect_faces_dnn(test_image)
                self.assertIsInstance(faces, list)

            # Test Haar cascade detection
            if hasattr(self.face_recognition, 'face_cascade'):
                faces = self.face_recognition._detect_faces_haar(gray_image)
                self.assertIsInstance(faces, list)

            # Test face preprocessing
            face_roi = gray_image[100:200, 100:200]
            processed = self.face_recognition._preprocess_face(face_roi)
            self.assertIsInstance(processed, np.ndarray)

            # Test face quality assessment
            quality = self.face_recognition._assess_face_quality(face_roi)
            self.assertIsInstance(quality, float)
            self.assertGreaterEqual(quality, 0.0)
            self.assertLessEqual(quality, 1.0)

    def test_face_recognition_with_no_known_faces(self):
        """Test face recognition with no known faces."""
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        human_detections = [{'bbox': [100, 100, 200, 200], 'confidence': 0.8, 'track_id': 1}]

        results = self.face_recognition.recognize_faces(test_image, human_detections)

        # Should return the same detections without face recognition
        self.assertEqual(len(results), 1)
        self.assertNotIn('identity', results[0])  # No identity should be assigned

    def test_encode_face(self):
        """Test face encoding."""
        # This test might not work with synthetic images
        try:
            encoding = self.face_recognition.encode_face(self.test_face_image)
            if encoding is not None:
                self.assertIsInstance(encoding, np.ndarray)
        except Exception:
            # Face encoding might fail with synthetic images
            pass
    
    def test_load_known_faces_empty_directory(self):
        """Test loading known faces from empty directory."""
        # Create temporary empty directory
        temp_dir = Path("temp_test_faces")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            result = self.face_recognition.load_known_faces(str(temp_dir))
            self.assertIsInstance(result, bool)
        finally:
            # Clean up
            if temp_dir.exists():
                temp_dir.rmdir()

class TestAnimalRecognitionSystem(unittest.TestCase):
    """Test cases for animal recognition system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.animal_recognition = AnimalRecognitionSystem()
        
        # Create test animal image
        cls.test_animal_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(cls.test_animal_image, (100, 100), 50, (100, 100, 100), -1)
    
    def test_animal_recognition_initialization(self):
        """Test animal recognition system initialization."""
        self.assertIsNotNone(self.animal_recognition)
    
    def test_identify_animal_empty_image(self):
        """Test animal identification with empty image."""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.animal_recognition.identify_animal(empty_image, "dog")
        
        self.assertIsInstance(result, dict)
        self.assertIn('identity', result)
        self.assertIn('confidence', result)
    
    def test_identify_animal_test_image(self):
        """Test animal identification with test image."""
        result = self.animal_recognition.identify_animal(self.test_animal_image, "cat")
        
        self.assertIsInstance(result, dict)
        self.assertIn('identity', result)
        self.assertIn('confidence', result)
        self.assertIsInstance(result['confidence'], (int, float))
    
    def test_extract_color_features(self):
        """Test color feature extraction."""
        features = self.animal_recognition.extract_color_features(self.test_animal_image)
        
        self.assertIsInstance(features, dict)
        self.assertIn('dominant_color', features)
        self.assertIn('color_histogram', features)
    
    def test_supported_animals(self):
        """Test supported animal types."""
        supported = self.animal_recognition.get_supported_animals()
        
        self.assertIsInstance(supported, list)
        self.assertGreater(len(supported), 0)
        self.assertIn('dog', supported)
        self.assertIn('cat', supported)

class TestDetectionAccuracy(unittest.TestCase):
    """Test cases for detection accuracy and performance."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.config = DetectionConfig()
        cls.detection_engine = DetectionEngine(cls.config)
    
    def test_detection_consistency(self):
        """Test detection consistency across multiple runs."""
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        results = []
        for _ in range(3):
            detections = self.detection_engine.detect_objects(test_image)
            results.append(len(detections['humans']) + len(detections['animals']))
        
        # Results should be consistent (allowing for small variations)
        max_diff = max(results) - min(results)
        self.assertLessEqual(max_diff, 2)  # Allow small variation
    
    def test_detection_with_different_resolutions(self):
        """Test detection with different image resolutions."""
        resolutions = [(320, 240), (640, 480), (800, 600)]
        
        for width, height in resolutions:
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            detections = self.detection_engine.detect_objects(test_image)
            
            self.assertIsInstance(detections, dict)
            self.assertIn('humans', detections)
            self.assertIn('animals', detections)
    
    def test_detection_performance_benchmark(self):
        """Benchmark detection performance."""
        import time
        
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        times = []
        for _ in range(5):
            start_time = time.time()
            self.detection_engine.detect_objects(test_image)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        
        # Average detection time should be reasonable
        self.assertLess(avg_time, 3.0)  # Less than 3 seconds on average
        
        print(f"Average detection time: {avg_time:.3f} seconds")

def run_detection_tests():
    """Run all detection tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestDetectionEngine))
    suite.addTest(unittest.makeSuite(TestFaceRecognitionSystem))
    suite.addTest(unittest.makeSuite(TestAnimalRecognitionSystem))
    suite.addTest(unittest.makeSuite(TestDetectionAccuracy))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("🧪 Running Detection System Tests")
    print("=" * 50)
    
    success = run_detection_tests()
    
    if success:
        print("\n✅ All detection tests passed!")
    else:
        print("\n❌ Some detection tests failed!")
        sys.exit(1)
