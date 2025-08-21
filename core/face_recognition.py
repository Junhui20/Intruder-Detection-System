"""
Consolidated Face Recognition System

This module implements a unified face recognition system with multiple backends:
- face_recognition library (primary)
- OpenCV DNN + LBPH (improved fallback)
- Basic OpenCV Haar cascades (basic fallback)

Features:
- Single-threaded multi-face processing (as per user preference)
- Configurable confidence thresholds
- DNN-based face detection with quality assessment
- Temporal smoothing and track-based identity persistence
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import logging
import pickle
import os
from pathlib import Path

# Try to import face_recognition, fall back to OpenCV if not available
try:
    # Suppress the pkg_resources deprecation warning
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
        import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logging.warning("face_recognition library not available, using OpenCV fallback")

# Try to import MediaPipe as alternative
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Import the OpenCV alternative face recognition
try:
    from alternative_face_recognition import OpenCVFaceRecognition
    OPENCV_FACE_AVAILABLE = True
except ImportError:
    OPENCV_FACE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceRecognitionSystem:
    """
    Unified face recognition system with multiple backends and advanced features.

    Features:
    - Single-threaded multi-face processing (as per user preference)
    - Multiple backends: face_recognition, OpenCV DNN+LBPH, basic OpenCV
    - Configurable confidence thresholds
    - Identity assignment prevention for duplicates
    - Performance monitoring and face encoding caching
    - DNN-based face detection with quality assessment
    - Temporal smoothing and track-based identity persistence
    """

    def __init__(self, confidence_threshold: float = 0.6, max_faces_per_frame: int = 10):
        """
        Initialize the unified face recognition system.

        Args:
            confidence_threshold: Minimum confidence for face recognition (user configurable)
            max_faces_per_frame: Maximum faces to process per frame
        """
        self.confidence_threshold = confidence_threshold
        self.max_faces_per_frame = max_faces_per_frame

        # face_recognition library storage
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_encoding_cache = {}

        # OpenCV DNN + LBPH storage (integrated from improved_face_recognition)
        self.face_net = None
        self.face_recognizer = None
        self.known_faces = []
        self.known_labels = []
        self.label_to_name = {}
        self.name_to_label = {}
        self.next_label = 0
        self.is_trained = False
        self.recognition_threshold = 100  # Maximum distance for OpenCV LBPH recognition

        # Track-based identity persistence for temporal smoothing
        self.track_identities = {}  # track_id -> {'identity': str, 'confidence': float, 'last_updated': int, 'consecutive_failures': int}
        self.frame_count = 0
        self.confidence_decay_rate = 0.02  # Confidence decay per frame when no recognition
        self.min_confidence_for_change = 0.7  # Higher confidence required to change identity
        self.max_consecutive_failures = 15  # Frames before reverting to Unknown
        self.identity_persistence_frames = 30  # Frames to keep identity after last recognition

        # Performance tracking
        self.recognition_stats = {
            'total_faces_processed': 0,
            'successful_recognitions': 0,
            'unknown_faces': 0,
            'processing_times': [],
            'multi_face_frames': 0,
            'identity_changes': 0,
            'temporal_smoothing_applied': 0
        }
        
        # Identity tracking to prevent duplicates
        self.current_frame_identities = set()

        # Backend selection
        self.opencv_face_system = None
        self.use_opencv_fallback = False
        self.backend_type = "unknown"

        # Initialize the best available backend
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the best available face recognition backend."""
        # Try face_recognition library first
        if FACE_RECOGNITION_AVAILABLE:
            try:
                # Test with a simple image
                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                face_recognition.face_locations(test_image)
                self.backend_type = "face_recognition"
                logger.info("Using face_recognition library backend")
                return
            except Exception as e:
                logger.warning(f"face_recognition library has issues: {e}")

        # Fall back to integrated OpenCV DNN + LBPH system
        self.use_opencv_fallback = True
        try:
            self._initialize_opencv_dnn_models()
            self.backend_type = "opencv_dnn_lbph"
            logger.info("Using integrated OpenCV DNN + LBPH backend")
            return
        except Exception as e:
            logger.warning(f"OpenCV DNN initialization failed: {e}")

        # Fall back to basic OpenCV if available
        if OPENCV_FACE_AVAILABLE:
            try:
                self.opencv_face_system = OpenCVFaceRecognition()
                self.backend_type = "opencv_basic"
                logger.info("Using basic OpenCV backend")
                return
            except Exception as e:
                logger.warning(f"Basic OpenCV initialization failed: {e}")

        # No working backend available
        self.backend_type = "none"
        logger.error("No working face recognition backend available")

    def _initialize_opencv_dnn_models(self):
        """Initialize OpenCV DNN models for face detection (integrated from improved_face_recognition)."""
        try:
            # Paths for DNN models
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)

            prototxt_path = models_dir / "deploy.prototxt"
            model_path = models_dir / "res10_300x300_ssd_iter_140000.caffemodel"

            # Load DNN model if available
            if prototxt_path.exists() and model_path.exists():
                self.face_net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
                logger.info("DNN face detection model loaded successfully")
            else:
                logger.warning("DNN models not available, falling back to Haar cascades")
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Initialize face recognizer with optimized parameters
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1,
                neighbors=8,
                grid_x=8,
                grid_y=8,
                threshold=100.0
            )

        except Exception as e:
            logger.error(f"Error initializing OpenCV DNN models: {e}")
            # Fallback to Haar cascades
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    def load_known_faces(self, faces_data: List[Dict]):
        """
        Load known faces from database data.
        
        Args:
            faces_data: List of dictionaries containing face information
                       Each dict should have: name, image_path, face_encodings (optional)
        """
        self.known_face_encodings = []
        self.known_face_names = []
        
        for face_data in faces_data:
            name = face_data['name']
            image_path = face_data['image_path']
            
            # Check if pre-computed encodings exist
            if 'face_encodings' in face_data and face_data['face_encodings']:
                try:
                    # Load pre-computed encodings
                    encodings = pickle.loads(face_data['face_encodings'])
                    for encoding in encodings:
                        self.known_face_encodings.append(encoding)
                        self.known_face_names.append(name)
                    logger.info(f"Loaded {len(encodings)} pre-computed encodings for {name}")
                except Exception as e:
                    logger.warning(f"Failed to load pre-computed encodings for {name}: {e}")
                    # Fall back to computing from image
                    self._load_face_from_image(name, image_path)
            else:
                # Compute encodings from image
                self._load_face_from_image(name, image_path)
        
        # Train OpenCV systems if using fallback
        if self.use_opencv_fallback and self.backend_type == "opencv_dnn_lbph":
            self._train_opencv_system()
            logger.info("Integrated OpenCV DNN + LBPH system trained")
        elif self.use_opencv_fallback and self.opencv_face_system:
            self.opencv_face_system.train()
            logger.info("Basic OpenCV face recognition system trained")

        logger.info(f"Loaded {len(self.known_face_encodings)} known face encodings")
    
    def _load_face_from_image(self, name: str, image_path: str):
        """Load and encode face from image file."""
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}")
                return

            if self.use_opencv_fallback and self.backend_type == "opencv_dnn_lbph":
                # Use integrated OpenCV DNN + LBPH system
                success = self._add_person_opencv_dnn(name, image_path)
                if success:
                    logger.info(f"Added {name} to integrated OpenCV DNN + LBPH system")
                else:
                    logger.warning(f"Failed to add {name} to integrated system")
            elif self.use_opencv_fallback and self.opencv_face_system:
                # Use basic OpenCV fallback
                success = self.opencv_face_system.add_person(name, image_path)
                if success:
                    logger.info(f"Added {name} to basic OpenCV face recognition system")
                else:
                    logger.warning(f"Failed to add {name} to basic OpenCV system")
            else:
                # Use original face_recognition library
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    for encoding in encodings:
                        self.known_face_encodings.append(encoding)
                        self.known_face_names.append(name)
                    logger.info(f"Loaded {len(encodings)} face encodings for {name}")
                else:
                    logger.warning(f"No faces found in image: {image_path}")

        except Exception as e:
            logger.error(f"Error loading face from {image_path}: {e}")
            # Try OpenCV fallback if original method failed
            if not self.use_opencv_fallback and self.opencv_face_system:
                logger.info(f"Trying OpenCV fallback for {name}")
                try:
                    success = self.opencv_face_system.add_person(name, image_path)
                    if success:
                        logger.info(f"Successfully added {name} using OpenCV fallback")
                        self.use_opencv_fallback = True
                    else:
                        logger.warning(f"OpenCV fallback also failed for {name}")
                except Exception as e2:
                    logger.error(f"OpenCV fallback also failed: {e2}")

    def _add_person_opencv_dnn(self, name: str, image_path: str) -> bool:
        """Add a person to the integrated OpenCV DNN + LBPH system."""
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}")
                return False

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return False

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces using the best available method
            if hasattr(self, 'face_net') and self.face_net is not None:
                faces = self._detect_faces_dnn(image)
            else:
                faces = self._detect_faces_haar(gray)

            if len(faces) == 0:
                logger.warning(f"No faces detected in image: {image_path}")
                return False

            # Process each detected face
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]

                # Preprocess face
                face_roi = self._preprocess_face(face_roi)

                # Add to training data
                if name not in self.name_to_label:
                    self.name_to_label[name] = self.next_label
                    self.label_to_name[self.next_label] = name
                    self.next_label += 1

                label = self.name_to_label[name]
                self.known_faces.append(face_roi)
                self.known_labels.append(label)

            logger.info(f"Added {len(faces)} face samples for {name}")
            return True

        except Exception as e:
            logger.error(f"Error adding person {name}: {e}")
            return False

    def _train_opencv_system(self) -> bool:
        """Train the integrated OpenCV LBPH recognizer."""
        try:
            if len(self.known_faces) == 0:
                logger.warning("No training data available")
                return False

            # Train the recognizer
            self.face_recognizer.train(self.known_faces, np.array(self.known_labels))
            self.is_trained = True

            logger.info(f"Trained OpenCV system with {len(self.known_faces)} face samples")
            return True

        except Exception as e:
            logger.error(f"Error training OpenCV system: {e}")
            return False

    def _detect_faces_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using DNN model."""
        try:
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                       (300, 300), (104.0, 177.0, 123.0))

            self.face_net.setInput(blob)
            detections = self.face_net.forward()

            faces = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")

                    # Ensure coordinates are within image bounds
                    x = max(0, x)
                    y = max(0, y)
                    x1 = min(w, x1)
                    y1 = min(h, y1)

                    faces.append((x, y, x1-x, y1-y))

            return faces

        except Exception as e:
            logger.error(f"Error in DNN face detection: {e}")
            return []

    def _detect_faces_haar(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar cascades."""
        try:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            return [(x, y, w, h) for (x, y, w, h) in faces]

        except Exception as e:
            logger.error(f"Error in Haar face detection: {e}")
            return []

    def _preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        """Preprocess face region for better recognition."""
        try:
            # Resize to standard size
            face_roi = cv2.resize(face_roi, (100, 100))

            # Apply histogram equalization
            face_roi = cv2.equalizeHist(face_roi)

            # Apply Gaussian blur to reduce noise
            face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)

            return face_roi

        except Exception as e:
            logger.error(f"Error preprocessing face: {e}")
            return face_roi

    def _assess_face_quality(self, face_roi: np.ndarray) -> float:
        """Assess the quality of a face region."""
        try:
            # Calculate Laplacian variance (focus measure)
            laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()

            # Normalize to 0-1 range (higher is better)
            quality = min(1.0, laplacian_var / 500.0)

            return quality

        except Exception as e:
            logger.error(f"Error assessing face quality: {e}")
            return 0.5  # Default medium quality

    def _recognize_face_opencv_dnn(self, face_region: np.ndarray) -> List[Dict]:
        """Recognize faces using the integrated OpenCV DNN + LBPH system."""
        results = []

        try:
            if not self.is_trained:
                return results

            # Convert to grayscale if needed
            if len(face_region.shape) == 3:
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_region

            # Detect faces using the best available method
            if hasattr(self, 'face_net') and self.face_net is not None:
                faces = self._detect_faces_dnn(face_region)
            else:
                faces = self._detect_faces_haar(gray)

            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]

                # Assess face quality
                quality = self._assess_face_quality(face_roi)

                # Preprocess face
                face_roi = self._preprocess_face(face_roi)

                # Recognize face
                label, confidence = self.face_recognizer.predict(face_roi)

                # Convert LBPH confidence to similarity score
                # For LBPH, lower confidence = better match
                if confidence < 50:  # Excellent match
                    similarity = 0.95
                elif confidence < 70:  # Very good match
                    similarity = 0.85
                elif confidence < 90:  # Good match
                    similarity = 0.75
                elif confidence < 110:  # Moderate match
                    similarity = 0.65
                elif confidence < 130:  # Weak match
                    similarity = 0.55
                else:  # Very weak match
                    similarity = 0.45

                # Adjust similarity based on face quality
                similarity = similarity * (0.7 + 0.3 * quality)

                # Only accept if above threshold
                if confidence <= self.recognition_threshold:
                    name = self.label_to_name.get(label, "Unknown")
                else:
                    name = "Unknown"
                    similarity = 0.0

                results.append({
                    'name': name,
                    'confidence': similarity,
                    'bbox': (x, y, x+w, y+h),
                    'method': 'opencv_dnn_lbph',
                    'quality': quality,
                    'raw_confidence': confidence
                })

        except Exception as e:
            logger.error(f"Error recognizing faces with OpenCV DNN: {e}")

        return results

    def recognize_faces(self, frame: np.ndarray, human_detections: List[Dict]) -> List[Dict]:
        """
        Recognize faces in detected human regions with temporal smoothing and track-based identity persistence.

        Args:
            frame: Input image frame
            human_detections: List of human detection dictionaries with bounding boxes and track_ids

        Returns:
            Updated human detections with face recognition results
        """
        start_time = time.time()
        self.current_frame_identities = set()  # Reset for new frame
        self.frame_count += 1

        if not human_detections:
            # Clean up old track identities
            self._cleanup_old_tracks()
            return human_detections

        # Check if we have any recognition system available
        has_face_recognition = (
            (not self.use_opencv_fallback and len(self.known_face_encodings) > 0) or
            (self.use_opencv_fallback and self.backend_type == "opencv_dnn_lbph" and self.is_trained) or
            (self.use_opencv_fallback and self.opencv_face_system and self.opencv_face_system.is_trained)
        )

        if not has_face_recognition:
            return human_detections
        
        # Limit number of faces to process for performance
        faces_to_process = min(len(human_detections), self.max_faces_per_frame)
        
        try:
            # Process each human detection for face recognition
            for i, detection in enumerate(human_detections[:faces_to_process]):
                x1, y1, x2, y2 = detection['bbox']
                
                # Extract face region with some padding
                padding = 20
                face_region = frame[max(0, y1-padding):min(frame.shape[0], y2+padding),
                                  max(0, x1-padding):min(frame.shape[1], x2+padding)]
                
                if face_region.size == 0:
                    continue
                
                # Use appropriate face recognition system
                best_match = None
                best_confidence = 0

                if self.use_opencv_fallback and self.backend_type == "opencv_dnn_lbph":
                    # Use integrated OpenCV DNN + LBPH system
                    opencv_results = self._recognize_face_opencv_dnn(face_region)

                    if opencv_results:
                        # Find the best match from integrated OpenCV results
                        for result in opencv_results:
                            if result['confidence'] > best_confidence and result['confidence'] >= self.confidence_threshold:
                                candidate_name = result['name']

                                # Prevent duplicate identity assignment in same frame
                                if candidate_name not in self.current_frame_identities:
                                    best_match = candidate_name
                                    best_confidence = result['confidence']

                elif self.use_opencv_fallback and self.opencv_face_system:
                    # Use basic OpenCV face recognition
                    opencv_results = self.opencv_face_system.recognize_face(face_region)

                    if opencv_results:
                        # Find the best match from OpenCV results
                        for result in opencv_results:
                            if result['confidence'] > best_confidence and result['confidence'] >= self.confidence_threshold:
                                candidate_name = result['name']

                                # Prevent duplicate identity assignment in same frame
                                if candidate_name not in self.current_frame_identities:
                                    best_match = candidate_name
                                    best_confidence = result['confidence']

                else:
                    # Use original face_recognition library
                    face_locations = face_recognition.face_locations(face_region)

                    if face_locations:
                        # Get face encodings for all detected faces in this region
                        face_encodings = face_recognition.face_encodings(face_region, face_locations)

                        # Process each face in the region
                        for face_encoding in face_encodings:
                            # Compare with known faces
                            matches = face_recognition.compare_faces(
                                self.known_face_encodings,
                                face_encoding,
                                tolerance=1.0 - self.confidence_threshold
                            )

                            face_distances = face_recognition.face_distance(
                                self.known_face_encodings,
                                face_encoding
                            )

                            if len(face_distances) > 0:
                                best_match_index = np.argmin(face_distances)
                                confidence = 1 - face_distances[best_match_index]

                                if (matches[best_match_index] and
                                    confidence >= self.confidence_threshold and
                                    confidence > best_confidence):

                                    candidate_name = self.known_face_names[best_match_index]

                                    # Prevent duplicate identity assignment in same frame
                                    if candidate_name not in self.current_frame_identities:
                                        best_match = candidate_name
                                        best_confidence = confidence

                # Apply temporal smoothing and track-based identity persistence
                track_id = detection.get('track_id')
                if track_id is not None:
                    # Check for track identity conflicts before updating
                    if self._should_create_new_track(track_id, best_match, best_confidence):
                        # Force new track creation by removing track_id
                        detection.pop('track_id', None)
                        track_id = None

                    if track_id is not None:
                        # Use track-based identity management
                        identity, confidence, status = self._update_track_identity(
                            track_id, best_match, best_confidence
                        )
                        detection['identity'] = identity
                        detection['face_confidence'] = confidence
                        detection['recognition_status'] = status

                        if identity != 'Unknown':
                            self.current_frame_identities.add(identity)
                else:
                    # Fallback to old binary logic for detections without track_id
                    if best_match:
                        detection['identity'] = best_match
                        detection['face_confidence'] = best_confidence
                        detection['recognition_status'] = 'known'
                        self.current_frame_identities.add(best_match)
                        self.recognition_stats['successful_recognitions'] += 1
                    else:
                        detection['identity'] = 'Unknown'
                        detection['face_confidence'] = 0.0
                        detection['recognition_status'] = 'unknown'
                        self.recognition_stats['unknown_faces'] += 1

                self.recognition_stats['total_faces_processed'] += 1
            
            # Track multi-face frames
            if len(human_detections) > 1:
                self.recognition_stats['multi_face_frames'] += 1
            
            # Update performance statistics
            processing_time = time.time() - start_time
            self.recognition_stats['processing_times'].append(processing_time)
            
            # Keep only last 100 processing times
            if len(self.recognition_stats['processing_times']) > 100:
                self.recognition_stats['processing_times'] = self.recognition_stats['processing_times'][-100:]
            
            return human_detections
            
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            # Return original detections with error status
            for detection in human_detections:
                detection['identity'] = 'Recognition Error'
                detection['face_confidence'] = 0.0
                detection['recognition_status'] = 'error'
            return human_detections
    
    def add_known_face(self, name: str, image_path: str) -> bool:
        """
        Add a new known face to the system.

        Args:
            name: Name of the person
            image_path: Path to the face image

        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}")
                return False

            # Use integrated OpenCV DNN + LBPH system if available
            if self.use_opencv_fallback and self.backend_type == "opencv_dnn_lbph":
                success = self._add_person_opencv_dnn(name, image_path)
                if success:
                    # Train the integrated system
                    self._train_opencv_system()
                    logger.info(f"Added {name} to integrated OpenCV DNN + LBPH system")
                return success

            # Use basic OpenCV system if available
            elif self.use_opencv_fallback and self.opencv_face_system:
                success = self.opencv_face_system.add_person(name, image_path)
                if success:
                    # Train the basic system
                    self.opencv_face_system.train()
                    logger.info(f"Added {name} to OpenCV face recognition system")
                return success

            # Use face_recognition library
            else:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    for encoding in encodings:
                        self.known_face_encodings.append(encoding)
                        self.known_face_names.append(name)
                    logger.info(f"Added {len(encodings)} face encodings for {name}")
                    return True
                else:
                    logger.warning(f"No faces found in image: {image_path}")
                    return False

        except Exception as e:
            logger.error(f"Error adding face for {name}: {e}")
            return False
    
    def remove_known_face(self, name: str):
        """Remove all encodings for a specific person."""
        indices_to_remove = [i for i, face_name in enumerate(self.known_face_names) if face_name == name]
        
        for index in reversed(indices_to_remove):  # Remove in reverse order to maintain indices
            del self.known_face_encodings[index]
            del self.known_face_names[index]
        
        logger.info(f"Removed {len(indices_to_remove)} face encodings for {name}")
    
    def update_confidence_threshold(self, new_threshold: float):
        """Update the confidence threshold (user configurable)."""
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            logger.info(f"Face recognition confidence threshold updated to {new_threshold}")
        else:
            logger.warning(f"Invalid confidence threshold: {new_threshold}")
    
    def get_performance_stats(self) -> Dict:
        """Get face recognition performance statistics."""
        if not self.recognition_stats['processing_times']:
            return {'error': 'No processing data available'}
        
        processing_times = self.recognition_stats['processing_times']
        total_processed = self.recognition_stats['total_faces_processed']
        
        return {
            'total_faces_processed': total_processed,
            'successful_recognitions': self.recognition_stats['successful_recognitions'],
            'unknown_faces': self.recognition_stats['unknown_faces'],
            'multi_face_frames': self.recognition_stats['multi_face_frames'],
            'recognition_accuracy': (self.recognition_stats['successful_recognitions'] / total_processed * 100) if total_processed > 0 else 0,
            'avg_processing_time': np.mean(processing_times),
            'max_processing_time': np.max(processing_times),
            'min_processing_time': np.min(processing_times),
            'confidence_threshold': self.confidence_threshold,
            'known_faces_count': len(set(self.known_face_names)),
            'total_encodings': len(self.known_face_encodings),
            'identity_changes': self.recognition_stats.get('identity_changes', 0),
            'temporal_smoothing_applied': self.recognition_stats.get('temporal_smoothing_applied', 0),
            'active_tracks': len(self.track_identities)
        }
    
    def export_face_encodings(self, name: str) -> Optional[bytes]:
        """
        Export face encodings for a specific person for database storage.
        
        Args:
            name: Name of the person
            
        Returns:
            Pickled face encodings or None if not found
        """
        encodings = [encoding for i, encoding in enumerate(self.known_face_encodings) 
                    if self.known_face_names[i] == name]
        
        if encodings:
            try:
                return pickle.dumps(encodings)
            except Exception as e:
                logger.error(f"Error exporting encodings for {name}: {e}")
                return None
        return None

    def _update_track_identity(self, track_id: int, new_match: str, new_confidence: float) -> tuple:
        """
        Update identity for a tracked person with temporal smoothing.

        Args:
            track_id: Track ID from PersonTracker
            new_match: New face recognition result (None if no match)
            new_confidence: Confidence of new recognition

        Returns:
            Tuple of (identity, confidence, status)
        """
        current_track = self.track_identities.get(track_id, {
            'identity': 'Unknown',
            'confidence': 0.0,
            'last_updated': self.frame_count,
            'consecutive_failures': 0
        })

        if new_match and new_confidence >= self.confidence_threshold:
            # Successful recognition
            if current_track['identity'] == 'Unknown':
                # First recognition for this track
                self.track_identities[track_id] = {
                    'identity': new_match,
                    'confidence': new_confidence,
                    'last_updated': self.frame_count,
                    'consecutive_failures': 0
                }
                self.recognition_stats['successful_recognitions'] += 1
                return new_match, new_confidence, 'known'

            elif current_track['identity'] == new_match:
                # Same identity confirmed - boost confidence
                boosted_confidence = min(1.0, current_track['confidence'] * 0.8 + new_confidence * 0.2)
                self.track_identities[track_id].update({
                    'confidence': boosted_confidence,
                    'last_updated': self.frame_count,
                    'consecutive_failures': 0
                })
                self.recognition_stats['successful_recognitions'] += 1
                return new_match, boosted_confidence, 'known'

            elif new_confidence > current_track['confidence'] + 0.1:
                # Different identity with significantly higher confidence
                self.track_identities[track_id] = {
                    'identity': new_match,
                    'confidence': new_confidence,
                    'last_updated': self.frame_count,
                    'consecutive_failures': 0
                }
                self.recognition_stats['identity_changes'] += 1
                self.recognition_stats['successful_recognitions'] += 1
                return new_match, new_confidence, 'known'
            else:
                # Different identity but not confident enough to change
                # Keep current identity but decay confidence slightly
                decayed_confidence = max(0.0, current_track['confidence'] - self.confidence_decay_rate)
                self.track_identities[track_id].update({
                    'confidence': decayed_confidence,
                    'consecutive_failures': current_track['consecutive_failures'] + 1
                })
                self.recognition_stats['temporal_smoothing_applied'] += 1

                # Check if we should revert to Unknown
                if (decayed_confidence < 0.3 or
                    current_track['consecutive_failures'] >= self.max_consecutive_failures):
                    self.track_identities[track_id].update({
                        'identity': 'Unknown',
                        'confidence': 0.0
                    })
                    self.recognition_stats['unknown_faces'] += 1
                    return 'Unknown', 0.0, 'unknown'

                return current_track['identity'], decayed_confidence, 'known'
        else:
            # No recognition or low confidence
            if current_track['identity'] != 'Unknown':
                # Decay confidence for existing identity
                decayed_confidence = max(0.0, current_track['confidence'] - self.confidence_decay_rate)
                consecutive_failures = current_track['consecutive_failures'] + 1

                self.track_identities[track_id].update({
                    'confidence': decayed_confidence,
                    'consecutive_failures': consecutive_failures
                })
                self.recognition_stats['temporal_smoothing_applied'] += 1

                # Check if we should revert to Unknown
                if (decayed_confidence < 0.3 or
                    consecutive_failures >= self.max_consecutive_failures):
                    self.track_identities[track_id].update({
                        'identity': 'Unknown',
                        'confidence': 0.0
                    })
                    self.recognition_stats['unknown_faces'] += 1
                    return 'Unknown', 0.0, 'unknown'

                return current_track['identity'], decayed_confidence, 'known'
            else:
                # Already unknown, keep as unknown
                self.recognition_stats['unknown_faces'] += 1
                return 'Unknown', 0.0, 'unknown'

    def _cleanup_old_tracks(self):
        """Remove old track identities that haven't been updated recently."""
        current_frame = self.frame_count
        tracks_to_remove = []

        for track_id, track_data in self.track_identities.items():
            frames_since_update = current_frame - track_data['last_updated']
            if frames_since_update > self.identity_persistence_frames:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.track_identities[track_id]

    def _should_create_new_track(self, track_id: int, new_match: str, new_confidence: float) -> bool:
        """
        Determine if a new track should be created due to identity conflict.

        Args:
            track_id: Current track ID
            new_match: New face recognition result
            new_confidence: Confidence of new recognition

        Returns:
            True if a new track should be created
        """
        if track_id not in self.track_identities:
            return False

        current_track = self.track_identities[track_id]
        current_identity = current_track['identity']

        # If current track is Unknown, no conflict
        if current_identity == 'Unknown':
            return False

        # If no new recognition, no conflict
        if not new_match or new_confidence < self.confidence_threshold:
            return False

        # If same identity, no conflict
        if new_match == current_identity:
            return False

        # Different identity with high confidence - potential conflict
        if new_confidence >= 0.7:  # High confidence threshold for new track creation
            logger.info(f"Identity conflict detected: track {track_id} was '{current_identity}' but now recognizing '{new_match}' with {new_confidence:.2f} confidence. Creating new track.")
            return True

        return False
