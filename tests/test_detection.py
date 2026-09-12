#!/usr/bin/env python3
"""
Detection System Tests

Comprehensive tests for the detection engine, face recognition,
and animal recognition systems.
"""

import unittest
import tempfile
import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.detection_engine import DetectionEngine
from core.face_recognition import FaceRecognitionSystem
from core.animal_recognition import AnimalRecognitionSystem
from config.detection_config import DetectionConfig

class TestDetectionEngine(unittest.TestCase):
    """Test cases for the detection engine."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.config = DetectionConfig()
        cls.detection_engine = DetectionEngine.from_config(cls.config)
        
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
    """Face recognition against the sample images InsightFace ships."""

    @classmethod
    def setUpClass(cls):
        cls.face_recognition = FaceRecognitionSystem(use_gpu=False)
        if cls.face_recognition.backend_type != "insightface":
            raise unittest.SkipTest("insightface not installed or its model pack unavailable")
        import insightface.data
        from insightface.data import get_image

        # Tom_Hanks_54745 is a tight 112x112 crop — enrolling it exercises the
        # pad-and-retry path. t1 is a group photo of six other people.
        cls.hanks_path = os.path.join(os.path.dirname(insightface.data.__file__), "images", "Tom_Hanks_54745.png")
        cls.hanks = get_image("Tom_Hanks_54745")
        cls.strangers = get_image("t1")

    def setUp(self):
        self.face_recognition.load_known_faces([])
        self.face_recognition.track_identities.clear()

    def in_frame(self, face):
        """A 480x640 frame with ``face`` pasted at (100, 100), and its person box."""
        frame = np.full((480, 640, 3), 90, dtype=np.uint8)
        h, w = face.shape[:2]
        frame[100:100 + h, 100:100 + w] = face
        return frame, [{"bbox": (80, 80, 120 + w, 120 + h), "track_id": 1}]

    def test_recognize_faces_with_no_detections(self):
        self.assertEqual(self.face_recognition.recognize_faces(self.strangers, []), [])

    def test_nobody_enrolled_leaves_detections_untouched(self):
        detections = [{"bbox": (10, 10, 90, 90), "track_id": 1, "confidence": 0.9}]
        results = self.face_recognition.recognize_faces(self.strangers, detections)
        self.assertEqual(results, [{"bbox": (10, 10, 90, 90), "track_id": 1, "confidence": 0.9}])

    def test_load_known_faces_skips_an_unreadable_image(self):
        self.face_recognition.load_known_faces([{"name": "Nobody", "image_path": "does_not_exist.jpg"}])
        self.assertEqual(self.face_recognition.known_face_names, [])

    def test_enrolled_person_is_named_and_strangers_are_not(self):
        self.assertTrue(self.face_recognition.add_known_face("Tom Hanks", self.hanks_path))

        frame, detections = self.in_frame(self.hanks)
        result = self.face_recognition.recognize_faces(frame, detections)[0]
        self.assertEqual(result["identity"], "Tom Hanks")
        self.assertGreater(result["face_confidence"], self.face_recognition.confidence_threshold)

        faces = self.face_recognition.app.get(self.strangers)
        detections = [{"bbox": tuple(int(v) for v in f.bbox), "track_id": i + 10} for i, f in enumerate(faces)]
        for result in self.face_recognition.recognize_faces(self.strangers, detections):
            self.assertEqual(result["identity"], "Unknown")

    def test_identity_survives_frames_without_a_face_then_expires(self):
        self.face_recognition.add_known_face("Tom Hanks", self.hanks_path)
        frame, detections = self.in_frame(self.hanks)
        self.face_recognition.recognize_faces(frame, detections)

        blank = np.zeros_like(frame)
        for _ in range(self.face_recognition.max_misses):
            held = self.face_recognition.recognize_faces(blank, [{"bbox": (0, 0, 50, 50), "track_id": 1}])[0]
            self.assertEqual(held["identity"], "Tom Hanks")
        gone = self.face_recognition.recognize_faces(blank, [{"bbox": (0, 0, 50, 50), "track_id": 1}])[0]
        self.assertEqual(gone["identity"], "Unknown")

class TestAnimalRecognitionSystem(unittest.TestCase):
    """Pet re-ID: an enrolled animal is named, any other is not."""

    @classmethod
    def setUpClass(cls):
        cls.pets = AnimalRecognitionSystem(use_gpu=False)
        if cls.pets.model is None:
            raise unittest.SkipTest("transformers or the DINOv2 checkpoint unavailable")
        from insightface.data import get_image

        # Any two unrelated photos will do: the model does not care that they are people.
        cls.jacky = get_image("t1")
        cls.other = cv2.copyMakeBorder(get_image("Tom_Hanks_54745"), 56, 56, 56, 56, cv2.BORDER_REPLICATE)
        cls.jacky_path = os.path.join(tempfile.mkdtemp(), "jacky.jpg")
        cv2.imwrite(cls.jacky_path, cls.jacky)

    def setUp(self):
        self.pets.load_known_pets([])

    def box(self, image, class_id=16):
        h, w = image.shape[:2]
        return [{"bbox": (0, 0, w, h), "class_id": class_id, "confidence": 0.9}]

    def test_identify_animals_with_no_detections(self):
        self.assertEqual(self.pets.identify_animals(self.jacky, []), [])

    def test_nobody_enrolled_still_annotates(self):
        result = self.pets.identify_animals(self.jacky, self.box(self.jacky))[0]
        self.assertEqual(result["bbox"], (0, 0, self.jacky.shape[1], self.jacky.shape[0]))
        self.assertEqual(result["recognition_status"], "unknown_animal")
        self.assertEqual(result["pet_identity"], "Unknown dog")

    def test_enrolled_pet_is_named_and_a_stranger_is_not(self):
        self.pets.load_known_pets([{"name": "Jacky", "individual_id": None, "coco_class_id": 16,
                                    "image_path": self.jacky_path, "multiple_photos": None}])
        seen_again = self.pets.identify_animals(cv2.flip(self.jacky, 1), self.box(self.jacky))[0]
        self.assertEqual(seen_again["pet_identity"], "Jacky")
        self.assertGreater(seen_again["identification_confidence"], self.pets.pet_identification_threshold)
        stranger = self.pets.identify_animals(self.other, self.box(self.other))[0]
        self.assertEqual(stranger["recognition_status"], "unknown_animal")

    def test_a_cat_is_never_matched_against_an_enrolled_dog(self):
        self.pets.add_known_pet("Jacky", 16, [self.jacky_path])
        result = self.pets.identify_animals(self.jacky, self.box(self.jacky, class_id=15))[0]
        self.assertEqual(result["pet_identity"], "Unknown cat")

    def test_unreadable_photo_enrols_nothing(self):
        self.assertFalse(self.pets.add_known_pet("Ghost", 16, ["does_not_exist.jpg"]))
        self.assertEqual(self.pets.known_pets, {})


class TestDetectionAccuracy(unittest.TestCase):
    """Test cases for detection accuracy and performance."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.config = DetectionConfig()
        cls.detection_engine = DetectionEngine.from_config(cls.config)
    
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
