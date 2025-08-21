"""
Detection Configuration Management

This module handles detection parameters, model configurations, and AI settings.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """
    Detection system configuration with YOLO11n and recognition settings.
    
    Features:
    - YOLO11n model configuration
    - Face recognition parameters
    - Animal recognition settings
    - Performance optimization
    """
    
    # YOLO Model Configuration - OPTIMIZED
    yolo_model_path: str = "yolo11n.pt"  # Fallback model
    optimized_model_dir: str = "models"  # Directory for optimized models
    use_optimized_engine: bool = True  # Enable optimized detection engine
    yolo_confidence: float = 0.5
    yolo_iou_threshold: float = 0.45
    yolo_max_detections: int = 1000
    yolo_device: str = "auto"  # auto, cpu, cuda:0, etc.
    
    # Face Recognition Configuration (User Configurable)
    human_confidence_threshold: float = 0.6
    face_recognition_model: str = "hog"  # hog or cnn
    face_recognition_tolerance: float = 0.6
    max_faces_per_frame: int = 10
    face_detection_method: str = "hog"  # hog, cnn, or auto
    
    # Animal Recognition Configuration (User Configurable)
    animal_confidence_threshold: float = 0.6
    pet_identification_threshold: float = 0.7
    pet_identification_method: str = "hybrid"  # color, face, hybrid
    color_detection_threshold: float = 0.2
    
    # Supported animal classes (COCO dataset)
    supported_animals: Dict[int, str] = None
    
    # Color ranges for animal identification
    color_ranges: Dict[str, List[List[int]]] = None
    
    # Performance Settings - OPTIMIZED
    enable_gpu_acceleration: bool = True
    enable_model_optimization: bool = True
    enable_tensorrt: bool = True  # TensorRT acceleration enabled
    enable_quantization: bool = True  # Model quantization enabled
    batch_processing: bool = False
    
    # Detection Logic Settings
    person_tracking_enabled: bool = True
    person_tracking_iou_threshold: float = 0.3
    person_tracking_max_disappeared: int = 10
    
    # Timer Settings
    unknown_person_alert_timer: int = 5  # seconds
    unfamiliar_animal_alert_timer: int = 5  # seconds
    detection_memory_frames: int = 30  # frames to remember detections
    
    # Notification Settings
    notification_cooldown: int = 20  # seconds
    enable_human_notifications: bool = True
    enable_animal_notifications: bool = True
    
    def __post_init__(self):
        """Initialize default values after creation."""
        if self.supported_animals is None:
            self.supported_animals = {
                15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
                19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra'
            }
        
        if self.color_ranges is None:
            self.color_ranges = {
                'white': [[0, 0, 180], [180, 50, 255]],
                'black': [[0, 0, 0], [180, 255, 50]],
                'golden': [[15, 100, 100], [25, 255, 255]],
                'brown': [[10, 100, 20], [20, 255, 200]],
                'gray': [[0, 0, 50], [180, 50, 200]],
                'beige': [[15, 30, 150], [40, 100, 255]]
            }
    
    def get_yolo_config(self) -> Dict[str, Any]:
        """Get YOLO-specific configuration with optimization settings."""
        return {
            'model_path': self.yolo_model_path,
            'optimized_model_dir': self.optimized_model_dir,
            'use_optimized_engine': self.use_optimized_engine,
            'conf': self.yolo_confidence,
            'iou': self.yolo_iou_threshold,
            'max_det': self.yolo_max_detections,
            'device': self.yolo_device,
            'verbose': False
        }
    
    def get_face_recognition_config(self) -> Dict[str, Any]:
        """Get face recognition configuration."""
        return {
            'confidence_threshold': self.human_confidence_threshold,
            'model': self.face_recognition_model,
            'tolerance': self.face_recognition_tolerance,
            'max_faces': self.max_faces_per_frame,
            'detection_method': self.face_detection_method
        }
    
    def get_animal_recognition_config(self) -> Dict[str, Any]:
        """Get animal recognition configuration."""
        return {
            'confidence_threshold': self.animal_confidence_threshold,
            'pet_threshold': self.pet_identification_threshold,
            'method': self.pet_identification_method,
            'color_threshold': self.color_detection_threshold,
            'supported_animals': self.supported_animals,
            'color_ranges': self.color_ranges
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance optimization configuration."""
        return {
            'gpu_acceleration': self.enable_gpu_acceleration,
            'model_optimization': self.enable_model_optimization,
            'tensorrt': self.enable_tensorrt,
            'quantization': self.enable_quantization,
            'batch_processing': self.batch_processing
        }
    
    def validate(self) -> Dict[str, str]:
        """
        Validate detection configuration.
        
        Returns:
            Dictionary of validation errors
        """
        errors = {}
        
        # Validate confidence thresholds
        if not 0.0 <= self.yolo_confidence <= 1.0:
            errors['yolo_confidence'] = "YOLO confidence must be between 0.0 and 1.0"
        
        if not 0.0 <= self.human_confidence_threshold <= 1.0:
            errors['human_confidence_threshold'] = "Human confidence must be between 0.0 and 1.0"
        
        if not 0.0 <= self.animal_confidence_threshold <= 1.0:
            errors['animal_confidence_threshold'] = "Animal confidence must be between 0.0 and 1.0"
        
        if not 0.0 <= self.pet_identification_threshold <= 1.0:
            errors['pet_identification_threshold'] = "Pet identification threshold must be between 0.0 and 1.0"
        
        # Validate IoU threshold
        if not 0.0 <= self.yolo_iou_threshold <= 1.0:
            errors['yolo_iou_threshold'] = "IoU threshold must be between 0.0 and 1.0"
        
        if not 0.0 <= self.person_tracking_iou_threshold <= 1.0:
            errors['person_tracking_iou_threshold'] = "Person tracking IoU must be between 0.0 and 1.0"
        
        # Validate face recognition settings
        if self.face_recognition_model not in ['hog', 'cnn']:
            errors['face_recognition_model'] = "Face recognition model must be 'hog' or 'cnn'"
        
        if self.face_detection_method not in ['hog', 'cnn', 'auto']:
            errors['face_detection_method'] = "Face detection method must be 'hog', 'cnn', or 'auto'"
        
        # Validate pet identification method
        if self.pet_identification_method not in ['color', 'face', 'hybrid']:
            errors['pet_identification_method'] = "Pet identification method must be 'color', 'face', or 'hybrid'"
        
        # Validate timers
        if self.unknown_person_alert_timer < 0:
            errors['unknown_person_alert_timer'] = "Alert timer cannot be negative"
        
        if self.notification_cooldown < 0:
            errors['notification_cooldown'] = "Notification cooldown cannot be negative"
        
        # Validate frame limits
        if self.max_faces_per_frame <= 0:
            errors['max_faces_per_frame'] = "Max faces per frame must be positive"
        
        if self.detection_memory_frames <= 0:
            errors['detection_memory_frames'] = "Detection memory frames must be positive"
        
        return errors
    
    def optimize_for_hardware(self, gpu_available: bool = True, gpu_memory_gb: float = 4.0) -> None:
        """
        Optimize configuration based on available hardware.
        
        Args:
            gpu_available: Whether GPU is available
            gpu_memory_gb: Available GPU memory in GB
        """
        if not gpu_available:
            # CPU-only optimizations
            self.enable_gpu_acceleration = False
            self.yolo_device = "cpu"
            self.face_recognition_model = "hog"  # Faster on CPU
            self.face_detection_method = "hog"
            self.max_faces_per_frame = 5  # Reduce load
            self.enable_tensorrt = False
            self.batch_processing = False
            logger.info("Optimized configuration for CPU-only processing")
        
        elif gpu_memory_gb < 2.0:
            # Low GPU memory optimizations
            self.enable_tensorrt = False
            self.enable_quantization = True
            self.max_faces_per_frame = 5
            self.batch_processing = False
            logger.info("Optimized configuration for low GPU memory")
        
        elif gpu_memory_gb >= 6.0:
            # High-end GPU optimizations
            self.enable_tensorrt = True
            self.enable_quantization = False
            self.face_recognition_model = "cnn"  # More accurate
            self.face_detection_method = "cnn"
            self.max_faces_per_frame = 15
            self.batch_processing = True
            logger.info("Optimized configuration for high-end GPU")
        
        else:
            # Mid-range GPU optimizations
            self.enable_tensorrt = False
            self.enable_quantization = True
            self.max_faces_per_frame = 10
            self.batch_processing = False
            logger.info("Optimized configuration for mid-range GPU")
    
    def get_user_configurable_settings(self) -> Dict[str, Dict[str, Any]]:
        """
        Get settings that are configurable by users via GUI.
        
        Returns:
            Dictionary of configurable settings with metadata
        """
        return {
            'human_confidence_threshold': {
                'value': self.human_confidence_threshold,
                'type': 'float',
                'min': 0.1,
                'max': 1.0,
                'step': 0.05,
                'description': 'Confidence threshold for human face recognition'
            },
            'animal_confidence_threshold': {
                'value': self.animal_confidence_threshold,
                'type': 'float',
                'min': 0.1,
                'max': 1.0,
                'step': 0.05,
                'description': 'Confidence threshold for animal detection'
            },
            'pet_identification_threshold': {
                'value': self.pet_identification_threshold,
                'type': 'float',
                'min': 0.1,
                'max': 1.0,
                'step': 0.05,
                'description': 'Confidence threshold for individual pet identification (e.g., Jacky)'
            },
            'pet_identification_method': {
                'value': self.pet_identification_method,
                'type': 'choice',
                'choices': ['color', 'face', 'hybrid'],
                'description': 'Method for individual pet identification'
            },
            'unknown_person_alert_timer': {
                'value': self.unknown_person_alert_timer,
                'type': 'int',
                'min': 1,
                'max': 60,
                'step': 1,
                'description': 'Seconds before unknown person alert'
            },
            'notification_cooldown': {
                'value': self.notification_cooldown,
                'type': 'int',
                'min': 5,
                'max': 300,
                'step': 5,
                'description': 'Seconds between notifications'
            },
            'max_faces_per_frame': {
                'value': self.max_faces_per_frame,
                'type': 'int',
                'min': 1,
                'max': 20,
                'step': 1,
                'description': 'Maximum faces to process per frame'
            }
        }
    
    def update_from_database(self, db_manager) -> bool:
        """
        Update configuration from database settings.
        
        Args:
            db_manager: DatabaseManager instance
            
        Returns:
            True if successful
        """
        try:
            config_dict = db_manager.get_all_config()
            
            # Map database keys to config attributes
            key_mapping = {
                'human_confidence_threshold': 'human_confidence_threshold',
                'animal_confidence_threshold': 'animal_confidence_threshold',
                'pet_identification_threshold': 'pet_identification_threshold',
                'pet_identification_method': 'pet_identification_method',
                'unknown_person_timer': 'unknown_person_alert_timer',
                'notification_cooldown': 'notification_cooldown',
                'max_faces_per_frame': 'max_faces_per_frame',
                'yolo_confidence': 'yolo_confidence'
            }
            
            for db_key, config_key in key_mapping.items():
                if db_key in config_dict:
                    setattr(self, config_key, config_dict[db_key])
            
            logger.info("Detection configuration updated from database")
            return True
            
        except Exception as e:
            logger.error(f"Error updating detection config from database: {e}")
            return False
    
    def save_to_database(self, db_manager) -> bool:
        """
        Save configuration to database.
        
        Args:
            db_manager: DatabaseManager instance
            
        Returns:
            True if successful
        """
        try:
            # Map config attributes to database keys
            config_mapping = {
                'human_confidence_threshold': (self.human_confidence_threshold, 'float'),
                'animal_confidence_threshold': (self.animal_confidence_threshold, 'float'),
                'pet_identification_threshold': (self.pet_identification_threshold, 'float'),
                'pet_identification_method': (self.pet_identification_method, 'string'),
                'unknown_person_timer': (self.unknown_person_alert_timer, 'integer'),
                'notification_cooldown': (self.notification_cooldown, 'integer'),
                'max_faces_per_frame': (self.max_faces_per_frame, 'integer'),
                'yolo_confidence': (self.yolo_confidence, 'float')
            }
            
            for key, (value, config_type) in config_mapping.items():
                db_manager.set_config(key, value, config_type)
            
            logger.info("Detection configuration saved to database")
            return True
            
        except Exception as e:
            logger.error(f"Error saving detection config to database: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionConfig':
        """Create instance from dictionary."""
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation."""
        return f"DetectionConfig(human={self.human_confidence_threshold}, animal={self.animal_confidence_threshold}, method={self.pet_identification_method})"
