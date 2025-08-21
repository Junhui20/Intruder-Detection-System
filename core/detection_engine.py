"""
YOLO11n Detection Engine with Person Tracking

This module implements the core object detection system using YOLO11n model
with IoU-based person tracking and configurable confidence thresholds.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionEngine:
    """
    Advanced detection engine using YOLO11n with person tracking capabilities.
    
    Features:
    - YOLO11n model for fast, accurate detection
    - IoU-based person tracking across frames
    - Configurable confidence thresholds
    - Performance monitoring
    - Support for both human and animal detection
    """
    
    def __init__(self, model_path: str = 'yolo11n.pt', confidence: float = 0.5,
                 use_optimized_engine: bool = False, optimized_model_dir: str = "models"):
        """
        Initialize the detection engine.

        Args:
            model_path: Path to YOLO11n model file
            confidence: Detection confidence threshold (0.0-1.0)
            use_optimized_engine: Whether to use optimized models
            optimized_model_dir: Directory containing optimized models
        """
        self.model_path = model_path
        self.confidence = confidence
        self.use_optimized_engine = use_optimized_engine
        self.optimized_model_dir = optimized_model_dir
        self.model = None
        self.model_format = "pytorch"  # Track which model format is being used
        self.person_tracker = PersonTracker()
        self.detection_stats = {
            'total_detections': 0,
            'human_detections': 0,
            'animal_detections': 0,
            'processing_times': []
        }

        # Detection toggles and confidence thresholds
        self.human_detection_enabled = True
        self.animal_detection_enabled = True
        self.face_recognition_enabled = True
        self.pet_identification_enabled = True
        self.human_confidence = confidence
        self.animal_confidence = confidence
        
        # COCO class definitions
        self.human_classes = [0]  # person
        self.animal_classes = {15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 
                              19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra'}
        
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO11n model (optimized or standard)."""
        try:
            if self.use_optimized_engine:
                # Try to use optimized models
                from core.model_optimization import ModelOptimizer

                logger.info("Loading optimized YOLO model...")
                optimizer = ModelOptimizer(self.model_path)
                self.model_format, optimal_model_path = optimizer.get_optimal_model()

                logger.info(f"Loading {self.model_format} model from {optimal_model_path}")
                self.model = YOLO(optimal_model_path)
                logger.info(f"Optimized YOLO model loaded successfully ({self.model_format})")
            else:
                # Use standard model
                logger.info(f"Loading standard YOLO11n model from {self.model_path}")
                self.model = YOLO(self.model_path)
                self.model_format = "pytorch"
                logger.info("Standard YOLO11n model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            # Fallback to standard model if optimized loading fails
            if self.use_optimized_engine:
                logger.warning("Falling back to standard model...")
                try:
                    self.model = YOLO(self.model_path)
                    self.model_format = "pytorch"
                    logger.info("Fallback to standard model successful")
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise
            else:
                raise
    
    def detect_objects(self, frame: np.ndarray) -> Dict:
        """
        Detect objects in a frame using YOLO11n.
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing detection results
        """
        start_time = time.time()
        
        try:
            # Run YOLO detection
            results = self.model(frame, conf=self.confidence, verbose=False)

            detections = {
                'humans': [],
                'animals': [],
                'frame_info': {
                    'timestamp': time.time(),
                    'processing_time': 0,
                    'total_detections': 0,
                    'model_format': self.model_format,
                    'optimized': self.use_optimized_engine
                }
            }

            # Process detection results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        detection_info = {
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'class_id': cls_id,
                            'area': (x2 - x1) * (y2 - y1)
                        }

                        # Classify detection type with toggle checks
                        if cls_id in self.human_classes and self.human_detection_enabled:
                            # Check human confidence threshold
                            if confidence >= self.human_confidence:
                                detections['humans'].append(detection_info)
                                self.detection_stats['human_detections'] += 1
                        elif cls_id in self.animal_classes and self.animal_detection_enabled:
                            # Check animal confidence threshold
                            if confidence >= self.animal_confidence:
                                detection_info['animal_type'] = self.animal_classes[cls_id]
                                detections['animals'].append(detection_info)
                                self.detection_stats['animal_detections'] += 1
            
            # Update tracking for humans
            if detections['humans']:
                detections['humans'] = self.person_tracker.update_tracks(detections['humans'])
            
            # Update statistics
            processing_time = time.time() - start_time
            detections['frame_info']['processing_time'] = processing_time
            detections['frame_info']['total_detections'] = len(detections['humans']) + len(detections['animals'])
            
            self.detection_stats['total_detections'] += detections['frame_info']['total_detections']
            self.detection_stats['processing_times'].append(processing_time)
            
            # Keep only last 100 processing times for performance monitoring
            if len(self.detection_stats['processing_times']) > 100:
                self.detection_stats['processing_times'] = self.detection_stats['processing_times'][-100:]
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {
                'humans': [],
                'animals': [],
                'frame_info': {
                    'timestamp': time.time(),
                    'processing_time': time.time() - start_time,
                    'total_detections': 0,
                    'error': str(e)
                }
            }
    
    def get_performance_stats(self) -> Dict:
        """Get detection performance statistics."""
        if not self.detection_stats['processing_times']:
            return {'error': 'No processing data available'}
        
        processing_times = self.detection_stats['processing_times']
        return {
            'total_detections': self.detection_stats['total_detections'],
            'human_detections': self.detection_stats['human_detections'],
            'animal_detections': self.detection_stats['animal_detections'],
            'avg_processing_time': np.mean(processing_times),
            'max_processing_time': np.max(processing_times),
            'min_processing_time': np.min(processing_times),
            'current_fps': 1.0 / np.mean(processing_times[-10:]) if len(processing_times) >= 10 else 0,
            'model_confidence': self.confidence
        }
    
    def update_confidence(self, new_confidence: float):
        """Update detection confidence threshold."""
        if 0.0 <= new_confidence <= 1.0:
            self.confidence = new_confidence
            logger.info(f"Detection confidence updated to {new_confidence}")
        else:
            logger.warning(f"Invalid confidence value: {new_confidence}")


class PersonTracker:
    """
    IoU-based person tracking system to maintain identity across frames.
    """
    
    def __init__(self, iou_threshold: float = 0.5, max_disappeared: int = 10):
        """
        Initialize person tracker.
        
        Args:
            iou_threshold: Minimum IoU for track association
            max_disappeared: Maximum frames a person can disappear before removal
        """
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.next_id = 0
        self.tracks = {}
        self.disappeared = {}
    
    def update_tracks(self, detections: List[Dict]) -> List[Dict]:
        """
        Update person tracks with new detections.
        
        Args:
            detections: List of human detection dictionaries
            
        Returns:
            Updated detections with track IDs
        """
        if not detections:
            # Mark all existing tracks as disappeared
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self._deregister_track(track_id)
            return []
        
        # If no existing tracks, register all detections as new tracks
        if not self.tracks:
            for detection in detections:
                self._register_track(detection)
            return detections
        
        # Compute IoU matrix between existing tracks and new detections
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(
                    self.tracks[track_id]['bbox'], 
                    detection['bbox']
                )
        
        # Assign detections to tracks using Hungarian algorithm (simplified)
        used_track_indices = set()
        used_detection_indices = set()
        
        # Find best matches above threshold
        for i in range(len(track_ids)):
            for j in range(len(detections)):
                if (i not in used_track_indices and 
                    j not in used_detection_indices and 
                    iou_matrix[i, j] > self.iou_threshold):
                    
                    track_id = track_ids[i]
                    detections[j]['track_id'] = track_id
                    self.tracks[track_id] = detections[j]
                    
                    # Reset disappeared counter
                    if track_id in self.disappeared:
                        del self.disappeared[track_id]
                    
                    used_track_indices.add(i)
                    used_detection_indices.add(j)
                    break
        
        # Handle unmatched tracks (mark as disappeared)
        for i, track_id in enumerate(track_ids):
            if i not in used_track_indices:
                if track_id not in self.disappeared:
                    self.disappeared[track_id] = 1
                else:
                    self.disappeared[track_id] += 1
                
                if self.disappeared[track_id] > self.max_disappeared:
                    self._deregister_track(track_id)
        
        # Handle unmatched detections (register as new tracks)
        for j in range(len(detections)):
            if j not in used_detection_indices:
                self._register_track(detections[j])
        
        return detections
    
    def _register_track(self, detection: Dict):
        """Register a new track."""
        detection['track_id'] = self.next_id
        self.tracks[self.next_id] = detection
        self.next_id += 1
    
    def _deregister_track(self, track_id: int):
        """Remove a track."""
        if track_id in self.tracks:
            del self.tracks[track_id]
        if track_id in self.disappeared:
            del self.disappeared[track_id]
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
