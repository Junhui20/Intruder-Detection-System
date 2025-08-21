"""
System Settings and Configuration Management

This module handles system-wide settings, user preferences, and configuration persistence.
Settings are loaded from environment variables (preferred) and configuration files.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

# Import environment configuration manager
from config.env_config import get_config, get_secure_config

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """
    System settings with user-configurable parameters.
    
    All settings are configurable via GUI and stored in database.
    """
    
    # Detection settings (user configurable)
    yolo_model: str = "yolo11n.pt"
    yolo_confidence: float = 0.5
    human_confidence_threshold: float = 0.6  # User configurable via GUI
    animal_confidence_threshold: float = 0.6  # User configurable via GUI
    pet_identification_threshold: float = 0.7  # For individual pet recognition
    
    # Face recognition settings
    multi_face_detection: bool = True
    max_faces_per_frame: int = 10
    
    # Timing settings
    unknown_person_timer: int = 5  # Seconds before unknown alert
    unfamiliar_animal_timer: int = 5  # Seconds before unfamiliar alert
    notification_cooldown: int = 20  # Seconds between notifications
    
    # Video processing
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    process_every_n_frames: int = 1  # Frame skip for performance
    
    # Performance settings
    enable_gpu: bool = True
    enable_performance_monitoring: bool = True
    max_cpu_usage: float = 80.0  # Alert threshold
    max_memory_usage: float = 4096  # MB
    max_gpu_usage: float = 90.0  # Alert threshold
    
    # Pet identification method
    pet_identification_method: str = "hybrid"  # color, face, or hybrid
    
    # Database settings
    database_path: str = "detection_system.db"
    backup_interval: int = 3600  # Backup every hour (seconds)
    max_backup_files: int = 10
    
    # Telegram settings (loaded from environment variables for security)
    bot_token: str = ""  # Will be loaded from TELEGRAM_BOT_TOKEN env var
    polling_interval: int = 1  # Seconds between message checks
    max_retries: int = 3
    timeout: int = 30
    send_photos: bool = True
    photo_quality: int = 85  # JPEG quality (1-100)
    
    # GUI settings
    theme: str = "dark"  # dark or light
    window_size: str = "1280x720"
    auto_refresh_interval: int = 100  # Milliseconds
    show_performance_metrics: bool = True
    show_detection_confidence: bool = True
    enable_real_time_controls: bool = True

    # Camera management settings
    auto_fix_camera_ids: bool = False  # Auto-reorganize camera IDs on startup
    auto_reload_cameras: bool = True  # Auto-reload camera configs when changed
    
    # Logging settings
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "logs/detection_system.log"
    max_log_file_size: int = 10  # MB
    max_log_files: int = 5

    @classmethod
    def load_with_env_support(cls, config_path: str = "config.yaml") -> 'Settings':
        """
        Load settings with environment variable support.

        Environment variables take precedence over config file values.
        Sensitive data like bot tokens are loaded from environment variables.

        Args:
            config_path: Path to configuration file

        Returns:
            Settings instance with environment variable overrides
        """
        try:
            # Start with default settings
            settings = cls()

            # Load from config file first
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}

                # Apply config file values
                for section, values in config_data.items():
                    if isinstance(values, dict):
                        for key, value in values.items():
                            if hasattr(settings, key):
                                setattr(settings, key, value)
                    else:
                        if hasattr(settings, section):
                            setattr(settings, section, values)

            # Override with environment variables (secure approach)
            settings._load_from_environment()

            logger.info(f"Settings loaded from {config_path} with environment variable overrides")
            return settings

        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return cls()

    def _load_from_environment(self):
        """Load settings from environment variables with proper type conversion."""
        # Secure settings (prioritize environment variables)
        self.bot_token = get_secure_config("telegram.bot_token", env_var="TELEGRAM_BOT_TOKEN",
                                         config_path="telegram.bot_token", required=False) or self.bot_token

        # Performance settings
        self.max_cpu_usage = get_config("max_cpu_usage", self.max_cpu_usage,
                                      env_var="MAX_CPU_USAGE", data_type=float)
        self.max_memory_usage = get_config("max_memory_usage", self.max_memory_usage,
                                         env_var="MAX_MEMORY_USAGE", data_type=float)
        self.enable_gpu = get_config("enable_gpu", self.enable_gpu,
                                   env_var="ENABLE_GPU", data_type=bool)

        # Detection thresholds
        self.human_confidence_threshold = get_config("human_confidence_threshold",
                                                   self.human_confidence_threshold,
                                                   env_var="HUMAN_CONFIDENCE_THRESHOLD",
                                                   data_type=float)
        self.animal_confidence_threshold = get_config("animal_confidence_threshold",
                                                    self.animal_confidence_threshold,
                                                    env_var="ANIMAL_CONFIDENCE_THRESHOLD",
                                                    data_type=float)
        self.yolo_confidence = get_config("yolo_confidence", self.yolo_confidence,
                                        env_var="YOLO_CONFIDENCE", data_type=float)

        # Camera settings
        self.frame_width = get_config("frame_width", self.frame_width,
                                    env_var="FRAME_WIDTH", data_type=int)
        self.frame_height = get_config("frame_height", self.frame_height,
                                     env_var="FRAME_HEIGHT", data_type=int)

        # Database settings
        self.database_path = get_config("database_path", self.database_path,
                                      env_var="DATABASE_PATH")

        # Notification settings
        self.notification_cooldown = get_config("notification_cooldown", self.notification_cooldown,
                                               env_var="NOTIFICATION_COOLDOWN", data_type=int)
        self.send_photos = get_config("send_photos", self.send_photos,
                                    env_var="SEND_PHOTOS", data_type=bool)
        self.photo_quality = get_config("photo_quality", self.photo_quality,
                                      env_var="PHOTO_QUALITY", data_type=int)

        # Logging settings
        self.log_level = get_config("log_level", self.log_level, env_var="LOG_LEVEL")

        logger.info("Environment variable overrides applied to settings")

    @classmethod
    def load_from_file(cls, config_path: str = "config.yaml") -> 'Settings':
        """
        Load settings from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Settings instance
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Create settings instance with loaded data
                settings = cls()
                for key, value in config_data.items():
                    if hasattr(settings, key):
                        setattr(settings, key, value)
                
                logger.info(f"Settings loaded from {config_path}")
                return settings
            else:
                logger.info(f"Config file {config_path} not found, using defaults")
                return cls()
                
        except Exception as e:
            logger.error(f"Error loading settings from {config_path}: {e}")
            return cls()
    
    def save_to_file(self, config_path: str = "config.yaml") -> bool:
        """
        Save settings to YAML file.
        
        Args:
            config_path: Path to save configuration file
            
        Returns:
            True if successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Convert to dictionary and save
            config_data = asdict(self)
            
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Settings saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving settings to {config_path}: {e}")
            return False
    
    @classmethod
    def load_from_database(cls, db_manager) -> 'Settings':
        """
        Load settings from database.
        
        Args:
            db_manager: DatabaseManager instance
            
        Returns:
            Settings instance
        """
        try:
            config_dict = db_manager.get_all_config()
            
            settings = cls()
            for key, value in config_dict.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)
            
            logger.info("Settings loaded from database")
            return settings
            
        except Exception as e:
            logger.error(f"Error loading settings from database: {e}")
            return cls()
    
    def save_to_database(self, db_manager) -> bool:
        """
        Save settings to database.
        
        Args:
            db_manager: DatabaseManager instance
            
        Returns:
            True if successful
        """
        try:
            config_data = asdict(self)
            
            for key, value in config_data.items():
                # Determine config type
                if isinstance(value, bool):
                    config_type = "boolean"
                elif isinstance(value, int):
                    config_type = "integer"
                elif isinstance(value, float):
                    config_type = "float"
                else:
                    config_type = "string"
                
                # Save to database
                db_manager.set_config(key, value, config_type)
            
            logger.info("Settings saved to database")
            return True
            
        except Exception as e:
            logger.error(f"Error saving settings to database: {e}")
            return False
    
    def update_setting(self, key: str, value: Any) -> bool:
        """
        Update a single setting.
        
        Args:
            key: Setting key
            value: New value
            
        Returns:
            True if successful
        """
        try:
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated setting {key} = {value}")
                return True
            else:
                logger.warning(f"Unknown setting key: {key}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating setting {key}: {e}")
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value.
        
        Args:
            key: Setting key
            default: Default value if key not found
            
        Returns:
            Setting value or default
        """
        return getattr(self, key, default)
    
    def validate_settings(self) -> Dict[str, str]:
        """
        Validate all settings and return any errors.
        
        Returns:
            Dictionary of validation errors (empty if all valid)
        """
        errors = {}
        
        # Validate confidence thresholds
        if not 0.0 <= self.human_confidence_threshold <= 1.0:
            errors['human_confidence_threshold'] = "Must be between 0.0 and 1.0"
        
        if not 0.0 <= self.animal_confidence_threshold <= 1.0:
            errors['animal_confidence_threshold'] = "Must be between 0.0 and 1.0"
        
        if not 0.0 <= self.pet_identification_threshold <= 1.0:
            errors['pet_identification_threshold'] = "Must be between 0.0 and 1.0"
        
        # Validate frame dimensions
        if self.frame_width <= 0 or self.frame_height <= 0:
            errors['frame_dimensions'] = "Frame dimensions must be positive"
        
        # Validate FPS
        if self.target_fps <= 0:
            errors['target_fps'] = "Target FPS must be positive"
        
        # Validate timers
        if self.unknown_person_timer < 0:
            errors['unknown_person_timer'] = "Timer cannot be negative"
        
        if self.notification_cooldown < 0:
            errors['notification_cooldown'] = "Cooldown cannot be negative"
        
        # Validate file paths
        if self.database_path and not self.database_path.endswith('.db'):
            errors['database_path'] = "Database path should end with .db"
        
        # Validate photo quality
        if not 1 <= self.photo_quality <= 100:
            errors['photo_quality'] = "Photo quality must be between 1 and 100"
        
        # Validate pet identification method
        valid_methods = ['color', 'face', 'hybrid']
        if self.pet_identification_method not in valid_methods:
            errors['pet_identification_method'] = f"Must be one of: {valid_methods}"
        
        return errors
    
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
                'min': 0.0,
                'max': 1.0,
                'step': 0.1,
                'description': 'Confidence threshold for human face recognition'
            },
            'animal_confidence_threshold': {
                'value': self.animal_confidence_threshold,
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'step': 0.1,
                'description': 'Confidence threshold for animal detection'
            },
            'pet_identification_threshold': {
                'value': self.pet_identification_threshold,
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'step': 0.1,
                'description': 'Confidence threshold for individual pet identification'
            },
            'unknown_person_timer': {
                'value': self.unknown_person_timer,
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
            'pet_identification_method': {
                'value': self.pet_identification_method,
                'type': 'choice',
                'choices': ['color', 'face', 'hybrid'],
                'description': 'Method for individual pet identification'
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)
    
    def __str__(self) -> str:
        """String representation of settings."""
        return f"Settings(human_conf={self.human_confidence_threshold}, animal_conf={self.animal_confidence_threshold}, pet_method={self.pet_identification_method})"
