"""
Individual Pet Identification System

This module implements the hybrid approach for individual pet recognition
using face_recognition + YOLO + color identification as preferred by the user.
"""

import cv2
import numpy as np
import face_recognition
import time
from typing import Dict, List, Tuple, Optional
import logging
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnimalRecognitionSystem:
    """
    Advanced animal recognition system with individual pet identification.
    
    Features:
    - Hybrid approach: face_recognition + color analysis
    - Individual pet recognition (e.g., 'Jacky', 'Fluffy')
    - Configurable confidence thresholds
    - Color-based verification
    - Support for 8 animal types from COCO dataset
    """
    
    def __init__(self, confidence_threshold: float = 0.6, pet_identification_threshold: float = 0.7):
        """
        Initialize the animal recognition system.
        
        Args:
            confidence_threshold: General animal detection confidence
            pet_identification_threshold: Individual pet identification confidence
        """
        self.confidence_threshold = confidence_threshold
        self.pet_identification_threshold = pet_identification_threshold
        self.known_pets = {}  # Dictionary storing pet data
        
        # COCO animal classes
        self.animal_classes = {
            15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
            19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra'
        }
        
        # HSV color ranges for pet identification
        self.color_ranges = {
            'white': ([0, 0, 180], [180, 50, 255]),
            'black': ([0, 0, 0], [180, 255, 50]),
            'golden': ([15, 100, 100], [25, 255, 255]),
            'brown': ([10, 100, 20], [20, 255, 200]),
            'gray': ([0, 0, 50], [180, 50, 200]),
            'beige': ([15, 30, 150], [40, 100, 255])
        }
        
        # Color similarity mapping
        self.color_similarities = {
            'golden': ['yellow', 'brown', 'beige'],
            'brown': ['golden', 'beige'],
            'white': ['light_gray'],
            'black': ['dark_gray'],
            'beige': ['golden', 'brown']
        }
        
        # Performance tracking
        self.recognition_stats = {
            'total_animals_processed': 0,
            'successful_pet_identifications': 0,
            'unknown_animals': 0,
            'processing_times': [],
            'color_matches': 0,
            'face_matches': 0,
            'hybrid_matches': 0
        }
        
        logger.info("Animal Recognition System initialized")
    
    def load_known_pets(self, pets_data: List[Dict]):
        """
        Load known pets from database data.
        
        Args:
            pets_data: List of dictionaries containing pet information
        """
        self.known_pets = {}
        
        for pet_data in pets_data:
            pet_name = pet_data['individual_id'] or pet_data['name']
            
            pet_info = {
                'name': pet_data['name'],
                'individual_id': pet_data['individual_id'],
                'animal_class': pet_data['coco_class_id'],
                'color': pet_data['color'],
                'breed': pet_data.get('pet_breed', ''),
                'identification_method': pet_data.get('identification_method', 'hybrid'),
                'face_encodings': [],
                'image_paths': [pet_data['image_path']]
            }
            
            # Load additional photos if available
            if 'multiple_photos' in pet_data and pet_data['multiple_photos']:
                try:
                    import json
                    additional_photos = json.loads(pet_data['multiple_photos'])
                    pet_info['image_paths'].extend(additional_photos)
                except:
                    pass
            
            # Load face encodings
            if 'face_encodings' in pet_data and pet_data['face_encodings']:
                try:
                    pet_info['face_encodings'] = pickle.loads(pet_data['face_encodings'])
                    logger.info(f"Loaded pre-computed encodings for {pet_name}")
                except Exception as e:
                    logger.warning(f"Failed to load encodings for {pet_name}: {e}")
                    # Compute from images
                    self._compute_pet_encodings(pet_info)
            else:
                # Compute face encodings from images
                self._compute_pet_encodings(pet_info)
            
            self.known_pets[pet_name.lower()] = pet_info
        
        logger.info(f"Loaded {len(self.known_pets)} known pets")
    
    def _compute_pet_encodings(self, pet_info: Dict):
        """Compute face encodings for a pet from its images."""
        encodings = []
        
        for image_path in pet_info['image_paths']:
            try:
                if os.path.exists(image_path):
                    image = face_recognition.load_image_file(image_path)
                    # face_recognition works surprisingly well on animal faces
                    face_encodings = face_recognition.face_encodings(image)
                    encodings.extend(face_encodings)
            except Exception as e:
                logger.warning(f"Error processing {image_path}: {e}")
        
        pet_info['face_encodings'] = encodings
        logger.info(f"Computed {len(encodings)} face encodings for {pet_info['individual_id']}")
    
    def identify_animals(self, frame: np.ndarray, animal_detections: List[Dict]) -> List[Dict]:
        """
        Identify individual animals using hybrid approach.
        
        Args:
            frame: Input image frame
            animal_detections: List of animal detection dictionaries
            
        Returns:
            Updated animal detections with identification results
        """
        start_time = time.time()
        
        if not animal_detections:
            return animal_detections
        
        try:
            for detection in animal_detections:
                x1, y1, x2, y2 = detection['bbox']
                animal_class = detection['class_id']
                
                # Extract animal region
                animal_roi = frame[y1:y2, x1:x2]
                
                if animal_roi.size == 0:
                    continue
                
                # Perform hybrid identification
                identification_result = self._hybrid_pet_identification(animal_roi, animal_class)
                
                # Update detection with results
                detection.update(identification_result)
                self.recognition_stats['total_animals_processed'] += 1
            
            # Update performance statistics
            processing_time = time.time() - start_time
            self.recognition_stats['processing_times'].append(processing_time)
            
            # Keep only last 100 processing times
            if len(self.recognition_stats['processing_times']) > 100:
                self.recognition_stats['processing_times'] = self.recognition_stats['processing_times'][-100:]
            
            return animal_detections
            
        except Exception as e:
            logger.error(f"Animal identification failed: {e}")
            # Return original detections with error status
            for detection in animal_detections:
                detection['pet_identity'] = 'Recognition Error'
                detection['identification_confidence'] = 0.0
                detection['identification_method'] = 'error'
            return animal_detections
    
    def _hybrid_pet_identification(self, animal_image: np.ndarray, animal_class: int) -> Dict:
        """
        Perform hybrid pet identification using face + color analysis.
        
        Args:
            animal_image: Cropped animal image
            animal_class: COCO class ID of the animal
            
        Returns:
            Dictionary with identification results
        """
        # Get dominant color
        detected_color = self._get_dominant_color(animal_image)
        
        # Try face recognition first
        face_results = self._recognize_pet_face(animal_image, animal_class)
        
        # Hybrid scoring
        best_match = None
        best_score = 0
        identification_method = 'unknown'
        
        for pet_name, pet_data in self.known_pets.items():
            if pet_data['animal_class'] == animal_class:
                score = 0
                method_used = []
                
                # Face recognition score (70% weight)
                if face_results and pet_name in face_results:
                    face_confidence = face_results[pet_name]
                    score += face_confidence * 0.7
                    method_used.append('face')
                    self.recognition_stats['face_matches'] += 1
                
                # Color matching score (30% weight)
                if self._is_color_match(pet_data['color'], detected_color):
                    score += 0.3
                    method_used.append('color')
                    self.recognition_stats['color_matches'] += 1
                elif self._is_color_similar(pet_data['color'], detected_color):
                    score += 0.15
                    method_used.append('color_similar')
                
                # Check if this is the best match
                if score > best_score and score > (self.pet_identification_threshold * 0.7):  # Adjusted threshold
                    best_score = score
                    best_match = pet_name
                    identification_method = '+'.join(method_used) if method_used else 'unknown'
        
        # Prepare result
        if best_match:
            pet_data = self.known_pets[best_match]
            result = {
                'pet_identity': pet_data['individual_id'] or pet_data['name'],
                'pet_name': pet_data['name'],
                'pet_breed': pet_data['breed'],
                'identification_confidence': best_score,
                'identification_method': identification_method,
                'detected_color': detected_color,
                'recognition_status': 'known_pet'
            }
            self.recognition_stats['successful_pet_identifications'] += 1
            
            if 'face' in identification_method and 'color' in identification_method:
                self.recognition_stats['hybrid_matches'] += 1
        else:
            animal_type = self.animal_classes.get(animal_class, 'unknown')
            result = {
                'pet_identity': f'Unknown {animal_type}',
                'pet_name': f'Unknown {animal_type}',
                'pet_breed': '',
                'identification_confidence': 0.0,
                'identification_method': 'none',
                'detected_color': detected_color,
                'recognition_status': 'unknown_animal'
            }
            self.recognition_stats['unknown_animals'] += 1
        
        return result
    
    def _recognize_pet_face(self, animal_image: np.ndarray, animal_class: int) -> Optional[Dict]:
        """Use face_recognition library for pet face identification."""
        try:
            # face_recognition works surprisingly well on animal faces
            face_encodings = face_recognition.face_encodings(animal_image)
            
            if face_encodings:
                results = {}
                for pet_name, pet_data in self.known_pets.items():
                    if (pet_data['animal_class'] == animal_class and 
                        pet_data['face_encodings']):
                        
                        # Compare with stored pet face encodings
                        distances = face_recognition.face_distance(
                            pet_data['face_encodings'],
                            face_encodings[0]
                        )
                        
                        if len(distances) > 0:
                            confidence = 1 - min(distances)
                            if confidence > 0.4:  # Lower threshold for animals
                                results[pet_name] = confidence
                
                return results if results else None
        except Exception as e:
            logger.debug(f"Face recognition failed: {e}")
        
        return None
    
    def _get_dominant_color(self, image: np.ndarray) -> str:
        """HSV-based color detection."""
        try:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            color_counts = {}
            total_pixels = image.shape[0] * image.shape[1]
            
            for color, (lower, upper) in self.color_ranges.items():
                mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
                color_counts[color] = np.count_nonzero(mask) / total_pixels
            
            dominant_color = max(color_counts, key=color_counts.get)
            max_ratio = color_counts[dominant_color]
            
            return dominant_color if max_ratio > 0.2 else 'unknown'
        except Exception as e:
            logger.debug(f"Color detection failed: {e}")
            return 'unknown'
    
    def _is_color_match(self, expected_color: str, detected_color: str) -> bool:
        """Check for exact color match."""
        return expected_color.lower() == detected_color.lower()
    
    def _is_color_similar(self, expected_color: str, detected_color: str) -> bool:
        """Check if colors are similar."""
        similarities = self.color_similarities.get(expected_color.lower(), [])
        return detected_color.lower() in similarities
    
    def add_known_pet(self, pet_data: Dict) -> bool:
        """
        Add a new known pet to the system.
        
        Args:
            pet_data: Dictionary containing pet information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pet_name = pet_data['individual_id'] or pet_data['name']
            
            # Compute face encodings
            pet_info = {
                'name': pet_data['name'],
                'individual_id': pet_data['individual_id'],
                'animal_class': pet_data['coco_class_id'],
                'color': pet_data['color'],
                'breed': pet_data.get('pet_breed', ''),
                'identification_method': pet_data.get('identification_method', 'hybrid'),
                'face_encodings': [],
                'image_paths': [pet_data['image_path']]
            }
            
            self._compute_pet_encodings(pet_info)
            self.known_pets[pet_name.lower()] = pet_info
            
            logger.info(f"Added known pet: {pet_name}")
            return True
        except Exception as e:
            logger.error(f"Error adding pet: {e}")
            return False
    
    def update_confidence_thresholds(self, general_threshold: float = None, 
                                   pet_threshold: float = None):
        """Update confidence thresholds."""
        if general_threshold is not None and 0.0 <= general_threshold <= 1.0:
            self.confidence_threshold = general_threshold
            logger.info(f"Animal confidence threshold updated to {general_threshold}")
        
        if pet_threshold is not None and 0.0 <= pet_threshold <= 1.0:
            self.pet_identification_threshold = pet_threshold
            logger.info(f"Pet identification threshold updated to {pet_threshold}")
    
    def get_performance_stats(self) -> Dict:
        """Get animal recognition performance statistics."""
        if not self.recognition_stats['processing_times']:
            return {'error': 'No processing data available'}
        
        processing_times = self.recognition_stats['processing_times']
        total_processed = self.recognition_stats['total_animals_processed']
        
        return {
            'total_animals_processed': total_processed,
            'successful_pet_identifications': self.recognition_stats['successful_pet_identifications'],
            'unknown_animals': self.recognition_stats['unknown_animals'],
            'identification_accuracy': (self.recognition_stats['successful_pet_identifications'] / total_processed * 100) if total_processed > 0 else 0,
            'color_matches': self.recognition_stats['color_matches'],
            'face_matches': self.recognition_stats['face_matches'],
            'hybrid_matches': self.recognition_stats['hybrid_matches'],
            'avg_processing_time': np.mean(processing_times),
            'confidence_threshold': self.confidence_threshold,
            'pet_identification_threshold': self.pet_identification_threshold,
            'known_pets_count': len(self.known_pets)
        }
