#!/usr/bin/env python3
"""
Intruder Detection System 2025 - Main Entry Point

Advanced intruder detection system with YOLO11n, individual pet recognition,
and comprehensive Telegram integration.

Features:
- YOLO11n object detection with person tracking
- Multi-face recognition (no separate threading as preferred)
- Individual pet identification (e.g., 'Jacky') using hybrid approach
- IP camera support with HTTP/HTTPS and local fallback
- Bidirectional Telegram bot with command listening
- SQLite database (migrated from MariaDB)
- Modern GUI with 5 specialized modules
- Real-time performance monitoring

Author: Intruder Detection System Team
Version: 1.0.0
"""

import sys
import os
import argparse
import signal
import threading
import time
import warnings
from typing import Optional, Any, Dict, List

# Suppress known deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from utils.logger import setup_logging, get_logger, system_logger
from config.settings import Settings
from config.detection_config import DetectionConfig
from config.camera_config import CameraConfigManager
from database.database_manager import DatabaseManager
from core.detection_engine import DetectionEngine
from core.face_recognition import FaceRecognitionSystem
from core.animal_recognition import AnimalRecognitionSystem
from core.camera_manager import CameraManager
from core.notification_system import NotificationSystem
from gui.main_window import MainWindow
from utils.performance_tracker import PerformanceTracker
from utils.image_processing import ImageProcessor

# Global logger
logger = get_logger(__name__)


class IntruderDetectionSystem:
    """
    Main application class for the Intruder Detection System.
    
    Coordinates all subsystems and provides unified control interface.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the intruder detection system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.running = False
        self.detection_active = False

        # Detection tracking for deduplication
        self.current_detections = {
            'humans': set(),  # Track currently detected humans
            'animals': set()  # Track currently detected animals
        }
        self.detection_timeout = 10.0  # seconds without detection before considering object "left"
        self.last_detection_time = {}

        # Enhanced tracking for better deduplication
        self.detection_sessions = {}  # Track detection sessions to avoid duplicates
        self.session_timeout = 30.0  # seconds before a new session can start for same identity

        # FPS monitoring (handled by performance_tracker)
        # self.fps_monitor = None  # Removed - using performance_tracker instead
        
        # Core components
        self.settings: Optional[Settings] = None
        self.config_manager = None
        self.detection_config: Optional[DetectionConfig] = None
        self.db_manager: Optional[DatabaseManager] = None
        self.camera_manager: Optional[CameraManager] = None
        self.camera_config_manager: Optional[CameraConfigManager] = None
        self.detection_engine: Optional[DetectionEngine] = None
        self.face_recognition: Optional[FaceRecognitionSystem] = None
        self.animal_recognition: Optional[AnimalRecognitionSystem] = None
        self.notification_system: Optional[NotificationSystem] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        self.gui: Optional[MainWindow] = None
        
        # Threading
        self.detection_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        system_logger.log_startup("IntruderDetectionSystem")
    
    def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing Intruder Detection System...")
            
            # Load configuration
            if not self._load_configuration():
                return False
            
            # Initialize configuration manager
            if not self._initialize_config_manager():
                return False

            # Initialize database
            if not self._initialize_database():
                return False
            
            # Initialize detection components
            if not self._initialize_detection_systems():
                return False
            
            # Initialize camera system
            if not self._initialize_camera_system():
                return False
            
            # Initialize notification system
            if not self._initialize_notification_system():
                return False
            
            # Initialize performance tracking
            if not self._initialize_performance_tracking():
                return False

            # FPS monitoring is handled by performance_tracker
            # No separate FPS monitor needed
            
            # Initialize GUI
            if not self._initialize_gui():
                return False

            # Set main system reference in GUI for database access
            if self.gui:
                self.gui.set_main_system(self)
            
            logger.info("System initialization completed successfully")
            system_logger.log_startup("All components")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            system_logger.log_error("Initialization", str(e))
            return False
    
    def _load_configuration(self) -> bool:
        """Load system configuration with environment variable support."""
        try:
            # Load main settings with environment variable support (secure)
            self.settings = Settings.load_with_env_support(self.config_path)

            # Validate critical settings
            if not self._validate_critical_settings():
                return False

            # Load detection configuration
            self.detection_config = DetectionConfig()

            # Initialize camera config manager
            self.camera_config_manager = CameraConfigManager()

            logger.info("Configuration loaded successfully with environment variable support")
            return True

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

    def _validate_critical_settings(self) -> bool:
        """Validate critical settings like bot token."""
        try:
            # Check if Telegram bot token is available
            if not self.settings.bot_token:
                logger.warning("⚠️ TELEGRAM_BOT_TOKEN not set. Telegram notifications will be disabled.")
                logger.info("💡 Set TELEGRAM_BOT_TOKEN environment variable or copy .env.template to .env")
                # Don't fail initialization, just disable Telegram features

            return True

        except Exception as e:
            logger.error(f"Critical settings validation failed: {e}")
            return False
    
    def _initialize_database(self) -> bool:
        """Initialize database connection."""
        try:
            self.db_manager = DatabaseManager(self.settings.database_path)
            
            # Update configurations from database
            self.settings = Settings.load_from_database(self.db_manager)
            self.detection_config.update_from_database(self.db_manager)
            self.camera_config_manager.load_from_database(self.db_manager)
            
            system_logger.log_database_operation("Initialize", "All tables", True)
            logger.info("Database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            system_logger.log_database_operation("Initialize", "All tables", False)
            return False
    
    def _initialize_detection_systems(self) -> bool:
        """Initialize detection engines."""
        try:
            # Initialize YOLO detection engine with optimization support
            if self.detection_config.use_optimized_engine:
                logger.info("Initializing Enhanced Detection Engine with optimization...")
            else:
                logger.info("Initializing Standard Detection Engine...")

            self.detection_engine = DetectionEngine(
                model_path=self.detection_config.yolo_model_path,
                confidence=self.detection_config.yolo_confidence,
                use_optimized_engine=self.detection_config.use_optimized_engine,
                optimized_model_dir=self.detection_config.optimized_model_dir
            )
            
            # Initialize face recognition system
            self.face_recognition = FaceRecognitionSystem(
                confidence_threshold=self.detection_config.human_confidence_threshold,
                max_faces_per_frame=self.detection_config.max_faces_per_frame
            )
            
            # Load known faces from database
            humans = self.db_manager.get_whitelist_entries(entity_type="human")
            human_data = [entry.to_dict() for entry in humans]
            self.face_recognition.load_known_faces(human_data)
            
            # Initialize animal recognition system
            self.animal_recognition = AnimalRecognitionSystem(
                confidence_threshold=self.detection_config.animal_confidence_threshold,
                pet_identification_threshold=self.detection_config.pet_identification_threshold
            )
            
            # Load known pets from database
            animals = self.db_manager.get_whitelist_entries(entity_type="animal")
            animal_data = [entry.to_dict() for entry in animals]
            self.animal_recognition.load_known_pets(animal_data)
            
            logger.info("Detection systems initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Detection systems initialization failed: {e}")
            return False
    
    def _initialize_camera_system(self) -> bool:
        """Initialize camera management."""
        try:
            self.camera_manager = CameraManager()

            # Load camera configurations
            camera_configs = [config.to_dict() for config in self.camera_config_manager.get_all_cameras()]
            self.camera_manager.load_camera_configs(camera_configs)

            # Auto-fix camera IDs if enabled
            if self.settings.auto_fix_camera_ids:
                self._auto_fix_camera_ids()

            # Set up cameras
            if not self.camera_manager.setup_cameras():
                logger.warning("No cameras connected, but system will continue")

            logger.info("Camera system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Camera system initialization failed: {e}")
            return False
    
    def _initialize_notification_system(self) -> bool:
        """Initialize Telegram notification system."""
        try:
            # Get bot token from config manager (telegram.bot_token)
            bot_token = None
            if self.config_manager:
                bot_token = self.config_manager.get('telegram.bot_token')

            # Fallback to settings if not in config manager
            if not bot_token and hasattr(self.settings, 'bot_token'):
                bot_token = self.settings.bot_token

            if not bot_token:
                logger.warning("No Telegram bot token configured")
                return True

            self.notification_system = NotificationSystem(bot_token, self.db_manager)
            
            # Load users from database
            users = self.db_manager.get_all_notification_settings(status="open")
            user_data = [user.to_dict() for user in users]
            self.notification_system.load_users(user_data)
            
            # Test bot connection
            if self.notification_system.test_connection():
                self.notification_system.start_listening()
                logger.info("Telegram notification system initialized successfully")
            else:
                logger.warning("Telegram bot connection failed")
            
            return True
            
        except Exception as e:
            logger.error(f"Notification system initialization failed: {e}")
            return False
    
    def _initialize_performance_tracking(self) -> bool:
        """Initialize performance monitoring."""
        try:
            self.performance_tracker = PerformanceTracker(db_manager=self.db_manager)
            
            if self.settings.enable_performance_monitoring:
                self.performance_tracker.start_monitoring()
            
            logger.info("Performance tracking initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Performance tracking initialization failed: {e}")
            return False

    def _initialize_config_manager(self) -> bool:
        """Initialize configuration manager."""
        try:
            from config.config_manager import ConfigManager
            self.config_manager = ConfigManager(self.config_path)

            # Register callbacks for runtime configuration changes
            self.config_manager.register_change_callback('video', self._on_video_config_change)
            self.config_manager.register_change_callback('detection', self._on_detection_config_change)
            self.config_manager.register_change_callback('performance', self._on_performance_config_change)

            logger.info("Configuration manager initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize configuration manager: {e}")
            return False

    # FPS monitoring removed - using performance_tracker instead
    # def _initialize_fps_monitoring(self) -> bool:
    #     """FPS monitoring is handled by performance_tracker."""
    #     return True
    
    def _initialize_gui(self) -> bool:
        """Initialize graphical user interface."""
        try:
            self.gui = MainWindow("Intruder Detection System 2025")
            
            # Set up GUI callbacks
            self.gui.set_callback('start_detection', self.start_detection)
            self.gui.set_callback('stop_detection', self.stop_detection)
            self.gui.set_callback('get_system_status', self.get_system_status)
            self.gui.set_callback('get_performance_metrics', self.get_performance_metrics)
            
            logger.info("GUI initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"GUI initialization failed: {e}")
            return False


    
    def start_detection(self) -> bool:
        """Start the detection system."""
        try:
            if self.detection_active:
                logger.warning("Detection system is already running")
                return False
            
            if not self.camera_manager or not self.detection_engine:
                logger.error("Detection components not initialized")
                return False
            
            self.detection_active = True
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()

            # FPS monitoring is handled by performance_tracker
            # (already started in _initialize_performance_tracking)

            logger.info("Detection system started")
            system_logger.log_startup("Detection system")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start detection system: {e}")
            return False
    
    def stop_detection(self) -> bool:
        """Stop the detection system."""
        try:
            if not self.detection_active:
                logger.warning("Detection system is not running")
                return False
            
            self.detection_active = False

            # FPS monitoring is handled by performance_tracker
            # (will be stopped when performance_tracker is stopped)

            if self.detection_thread:
                self.detection_thread.join(timeout=5)

            # Clear video feed in GUI
            self._clear_gui_frame()

            # Clear detection log in GUI
            self._clear_gui_detection_log()

            logger.info("Detection system stopped")
            system_logger.log_shutdown("Detection system")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop detection system: {e}")
            return False
    
    def _detection_loop(self):
        """Main detection processing loop with FPS optimization."""
        logger.info("Detection loop started")

        frame_count = 0
        process_every_n_frames = getattr(self.settings, 'process_every_n_frames', 1)

        while self.detection_active and not self.shutdown_event.is_set():
            try:
                # Capture frame from camera
                frame = self.camera_manager.capture_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                frame_count += 1

                # Update FPS tracker
                if self.performance_tracker:
                    self.performance_tracker.fps_tracker.update()

                # Always update GUI with current frame for smooth video
                annotated_frame = frame.copy()

                # Skip processing for performance if configured
                if frame_count % process_every_n_frames != 0:
                    self._update_gui_frame(annotated_frame)
                    continue

                # Perform object detection only on selected frames
                detections = self.detection_engine.detect_objects(frame)
                
                # Process human detections with face recognition (if enabled)
                if (detections['humans'] and self.face_recognition and
                    hasattr(self.detection_engine, 'face_recognition_enabled') and
                    self.detection_engine.face_recognition_enabled):
                    detections['humans'] = self.face_recognition.recognize_faces(
                        frame, detections['humans']
                    )

                # Process animal detections with pet identification (if enabled)
                if (detections['animals'] and self.animal_recognition and
                    hasattr(self.detection_engine, 'pet_identification_enabled') and
                    self.detection_engine.pet_identification_enabled):
                    detections['animals'] = self.animal_recognition.identify_animals(
                        frame, detections['animals']
                    )
                
                # Handle notifications and alerts
                self._process_detections(detections, frame)

                # Create frame with detection overlays
                annotated_frame = ImageProcessor.create_detection_overlay(frame, detections)

                # Update GUI with annotated frame
                self._update_gui_frame(annotated_frame)

                # Adaptive delay based on FPS
                if self.performance_tracker:
                    current_fps = self.performance_tracker.fps_tracker.get_fps()
                    if current_fps > 25:
                        time.sleep(0.02)  # Longer delay if FPS is good
                    else:
                        time.sleep(0.005)  # Shorter delay if FPS is low
                else:
                    time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(1)  # Wait longer on error
        
        logger.info("Detection loop ended")

    def _update_gui_frame(self, frame):
        """Update the GUI with the current frame."""
        try:
            if self.gui and hasattr(self.gui, 'modules') and 'detection' in self.gui.modules:
                detection_view = self.gui.modules['detection']
                if hasattr(detection_view, 'update_video_feed'):
                    # Schedule GUI update in main thread
                    self.gui.root.after(0, lambda: detection_view.update_video_feed(frame))
        except Exception as e:
            logger.debug(f"Error updating GUI frame: {e}")

    def _clear_gui_frame(self):
        """Clear the video feed in the GUI."""
        try:
            if self.gui and hasattr(self.gui, 'modules') and 'detection' in self.gui.modules:
                detection_view = self.gui.modules['detection']
                if hasattr(detection_view, 'clear_video_feed'):
                    # Schedule GUI update in main thread
                    self.gui.root.after(0, lambda: detection_view.clear_video_feed())
        except Exception as e:
            logger.debug(f"Error clearing GUI frame: {e}")

    def _update_gui_detection_log(self, detection_type, name, confidence=None, additional_info=None):
        """Update the detection log in the GUI."""
        try:
            if self.gui and hasattr(self.gui, 'modules') and 'detection' in self.gui.modules:
                detection_view = self.gui.modules['detection']
                if hasattr(detection_view, 'add_detection_log'):
                    # Schedule GUI update in main thread
                    self.gui.root.after(0, lambda: detection_view.add_detection_log(
                        detection_type, name, confidence, additional_info
                    ))
        except Exception as e:
            logger.debug(f"Error updating GUI detection log: {e}")

    def _clear_gui_detection_log(self):
        """Clear the detection log in the GUI."""
        try:
            if self.gui and hasattr(self.gui, 'modules') and 'detection' in self.gui.modules:
                detection_view = self.gui.modules['detection']
                if hasattr(detection_view, 'clear_detection_log'):
                    # Schedule GUI update in main thread
                    self.gui.root.after(0, lambda: detection_view.clear_detection_log())
        except Exception as e:
            logger.debug(f"Error clearing GUI detection log: {e}")

    def _get_grid_position(self, bbox, grid_size=100):
        """Get grid position for bbox to create stable detection IDs."""
        if not bbox or len(bbox) < 4:
            return "0_0"

        # Calculate center point of bbox
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2

        # Convert to grid coordinates
        grid_x = center_x // grid_size
        grid_y = center_y // grid_size

        return f"{grid_x}_{grid_y}"

    def _is_new_detection_session(self, session_id: str, current_time: float) -> bool:
        """
        Check if this is a new detection session.

        Args:
            session_id: Unique identifier for the detection session
            current_time: Current timestamp

        Returns:
            True if this is a new session that should be logged
        """
        # If we've never seen this identity, it's a new session
        if session_id not in self.detection_sessions:
            return True

        # If currently being tracked, not a new session
        if session_id in self.current_detections['humans'] or session_id in self.current_detections['animals']:
            return False

        # If the object left (not in current_detections) and enough time has passed, it's a new session
        last_seen = self.last_detection_time.get(session_id, 0)
        if current_time - last_seen >= self.detection_timeout:
            return True

        return False

    def _process_detections(self, detections: dict, frame):
        """Process detection results and handle notifications."""
        try:
            import time
            current_time = time.time()

            # Clean up old detections (objects that left the camera)
            self._cleanup_old_detections(current_time)

            # Clean up old detection sessions
            self._cleanup_old_sessions(current_time)

            # Process human detections
            for human in detections['humans']:
                identity = human.get('identity', 'Unknown')
                confidence = human.get('confidence', 0) * 100  # Convert to percentage
                face_confidence = human.get('face_confidence', 0) * 100  # Convert to percentage
                bbox = human.get('bbox', (0, 0, 0, 0))

                # Create session identifier based on identity only (not position)
                session_id = f"human_{identity}"

                # Check if this is a new detection session
                is_new_session = self._is_new_detection_session(session_id, current_time)

                if is_new_session:
                    # Start new detection session
                    self.detection_sessions[session_id] = current_time
                    self.current_detections['humans'].add(session_id)

                    notification_sent = False
                    photo_path = None

                    # Check if this is an unknown person (multiple conditions)
                    is_unknown = (
                        human.get('recognition_status') == 'unknown' or
                        identity == 'Unknown' or
                        identity == 'unknown'
                    )

                    if is_unknown:
                        # Unknown person detected - capture screenshot
                        photo_path = self._capture_detection_screenshot(frame, 'unknown_human', confidence)
                        logger.info(f"📸 Screenshot captured for unknown person: {photo_path}")

                        if self.notification_system:
                            message = f"🚨 Unknown person detected (confidence: {confidence:.1f}%)"
                            self.notification_system.send_notification('human', message, photo_path=photo_path)
                            notification_sent = True
                            logger.info(f"📱 Telegram notification sent with photo")

                    # Log detection to database
                    if self.db_manager:
                        self.db_manager.log_detection(
                            detection_type='human',
                            entity_name=identity,
                            confidence=face_confidence / 100 if face_confidence > 0 else confidence / 100,
                            camera_id=None,  # Will be enhanced when camera management is improved
                            notification_sent=notification_sent
                        )

                    # Legacy logging
                    from utils.logger import detection_logger
                    detection_logger.log_human_detection(
                        identity,
                        face_confidence / 100,  # Convert back to decimal for logging
                        human.get('bbox', (0, 0, 0, 0))
                    )

                    # Update GUI log with percentage
                    display_confidence = face_confidence if face_confidence > 0 else confidence
                    self._update_gui_detection_log(
                        'human',
                        identity,
                        display_confidence
                    )

                # Update last seen time for this session
                self.last_detection_time[session_id] = current_time
            
            # Process animal detections
            for animal in detections['animals']:
                confidence = animal.get('confidence', 0) * 100  # Convert to percentage
                identification_confidence = animal.get('identification_confidence', 0) * 100
                animal_type = animal.get('animal_type', 'animal')
                bbox = animal.get('bbox', (0, 0, 0, 0))

                if animal.get('recognition_status') == 'known_pet':
                    pet_identity = animal.get('pet_identity', 'Unknown')
                    session_id = f"animal_{pet_identity}"

                    # Check if this is a new detection session
                    is_new_session = self._is_new_detection_session(session_id, current_time)

                    if is_new_session:
                        # Start new detection session
                        self.detection_sessions[session_id] = current_time
                        self.current_detections['animals'].add(session_id)

                        # Log detection to database
                        if self.db_manager:
                            self.db_manager.log_detection(
                                detection_type='animal',
                                entity_name=pet_identity,
                                confidence=identification_confidence / 100,
                                camera_id=None,  # Will be enhanced when camera management is improved
                                notification_sent=False  # Known animals don't trigger notifications
                            )

                        # Legacy logging - Known pet identified
                        from utils.logger import detection_logger
                        detection_logger.log_pet_identification(
                            pet_identity,
                            identification_confidence / 100,  # Convert back to decimal
                            animal.get('identification_method', 'unknown')
                        )

                        # Update GUI log for known pet
                        self._update_gui_detection_log(
                            'animal',
                            pet_identity,
                            identification_confidence
                        )

                    # Update last seen time
                    self.last_detection_time[session_id] = current_time

                else:
                    # Unknown animal
                    session_id = f"animal_Unknown_{animal_type}"

                    # Check if this is a new detection session
                    is_new_session = self._is_new_detection_session(session_id, current_time)

                    if is_new_session:
                        # Start new detection session
                        self.detection_sessions[session_id] = current_time
                        self.current_detections['animals'].add(session_id)

                        notification_sent = False
                        photo_path = None

                        # Check if this is an unknown animal (multiple conditions)
                        is_unknown_animal = (
                            animal.get('recognition_status') == 'unknown_animal' or
                            animal_type.lower() in ['unknown', 'unidentified'] or
                            confidence > 0  # Any detected animal for now
                        )

                        if is_unknown_animal:
                            # Unknown animal detected - capture screenshot
                            photo_path = self._capture_detection_screenshot(frame, f'unknown_{animal_type}', confidence)
                            logger.info(f"📸 Screenshot captured for unknown {animal_type}: {photo_path}")

                            if self.notification_system:
                                message = f"🐾 Unknown {animal_type} detected (confidence: {confidence:.1f}%)"
                                self.notification_system.send_notification('animal', message, photo_path=photo_path)
                                notification_sent = True
                                logger.info(f"📱 Telegram notification sent with photo")

                        # Log detection to database
                        if self.db_manager:
                            self.db_manager.log_detection(
                                detection_type='animal',
                                entity_name=f"Unknown {animal_type}",
                                confidence=confidence / 100,
                                camera_id=None,  # Will be enhanced when camera management is improved
                                notification_sent=notification_sent
                            )

                        # Update GUI log for unknown animal
                        self._update_gui_detection_log(
                            'animal',
                            'Unknown',
                            confidence,
                            animal_type
                        )

                    # Update last seen time
                    self.last_detection_time[session_id] = current_time
            
        except Exception as e:
            logger.error(f"Error processing detections: {e}")

    def _cleanup_old_detections(self, current_time):
        """Remove detections that haven't been seen recently."""
        try:
            # Find expired detections
            expired_detections = []
            for session_id, last_seen in self.last_detection_time.items():
                if current_time - last_seen >= self.detection_timeout:
                    expired_detections.append(session_id)

            # Remove expired detections
            for session_id in expired_detections:
                # Remove from tracking sets
                if session_id.startswith('human_'):
                    self.current_detections['humans'].discard(session_id)
                elif session_id.startswith('animal_'):
                    self.current_detections['animals'].discard(session_id)

                # Remove from timing tracking
                del self.last_detection_time[session_id]

                # Keep session record for session_timeout period to prevent immediate re-detection
                # The session will be cleaned up later when session_timeout expires

        except Exception as e:
            logger.debug(f"Error cleaning up old detections: {e}")

    def _cleanup_old_sessions(self, current_time):
        """Remove old detection sessions that have expired."""
        try:
            expired_sessions = []
            for session_id, session_start in self.detection_sessions.items():
                # Clean up sessions that are older than session_timeout
                if current_time - session_start > self.session_timeout:
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                del self.detection_sessions[session_id]

        except Exception as e:
            logger.debug(f"Error cleaning up old sessions: {e}")

    def _capture_detection_screenshot(self, frame, detection_type: str, confidence: float) -> str:
        """
        Capture and save screenshot when unknown detection occurs.

        Args:
            frame: Current camera frame
            detection_type: Type of detection (e.g., 'unknown_human', 'unknown_dog')
            confidence: Detection confidence

        Returns:
            Path to saved screenshot file
        """
        try:
            import os
            from datetime import datetime

            # Create detection photos directory
            photos_dir = "data/detection_photos"
            os.makedirs(photos_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{detection_type}_{confidence:.1f}%_{timestamp}.jpg"
            filepath = os.path.join(photos_dir, filename)

            # Save screenshot
            import cv2
            success = cv2.imwrite(filepath, frame)

            if success:
                logger.info(f"Detection screenshot saved: {filepath}")
                return filepath
            else:
                logger.error(f"Failed to save detection screenshot: {filepath}")
                return None

        except Exception as e:
            logger.error(f"Error capturing detection screenshot: {e}")
            return None

    def get_system_status(self) -> dict:
        """Get current system status."""
        return {
            'detection_active': self.detection_active,
            'camera_connected': self.camera_manager and len(self.camera_manager.cameras) > 0,
            'telegram_connected': self.notification_system and self.notification_system.test_connection(),
            'database_connected': self.db_manager is not None
        }
    
    def get_performance_metrics(self) -> dict:
        """Get current performance metrics."""
        if self.performance_tracker:
            return self.performance_tracker.get_current_metrics()
        return {}

    def reload_camera_configurations(self) -> bool:
        """
        Reload camera configurations from database and reconnect cameras.
        This is called when cameras are added/modified in the Camera Manager.

        Returns:
            True if successful
        """
        try:
            logger.info("Reloading camera configurations...")

            # Reload camera configurations from database
            if not self.camera_config_manager.load_from_database(self.db_manager):
                logger.error("Failed to reload camera configurations from database")
                return False

            # Get updated camera configs
            camera_configs = [config.to_dict() for config in self.camera_config_manager.get_all_cameras()]

            # Auto-fix camera IDs if enabled (before reloading)
            if self.settings.auto_fix_camera_ids:
                self._auto_fix_camera_ids()
                # Reload configs again after ID fix
                camera_configs = [config.to_dict() for config in self.camera_config_manager.get_all_cameras()]

            # Reload cameras using camera manager
            if self.camera_manager:
                success = self.camera_manager.reload_camera_configs(camera_configs)
                return success
            else:
                logger.error("Camera manager not initialized")
                return False

        except Exception as e:
            logger.error(f"Failed to reload camera configurations: {e}")
            return False

    def _auto_fix_camera_ids(self) -> bool:
        """
        Automatically fix camera IDs if enabled in settings.

        Returns:
            True if successful or no action needed
        """
        try:
            if not self.settings.auto_fix_camera_ids:
                return True

            logger.info("Auto-fixing camera IDs...")

            # Check if IDs need fixing
            devices = self.db_manager.get_all_devices()
            if not devices:
                return True

            # Check if IDs are already sequential
            device_ids = [device.id for device in devices]
            expected_ids = list(range(1, len(devices) + 1))

            if device_ids == expected_ids:
                logger.info("Camera IDs are already sequential, no auto-fix needed")
                return True

            # Perform auto-fix
            logger.info(f"Auto-fixing camera IDs from {device_ids} to {expected_ids}")
            success = self.db_manager.reorganize_device_ids()

            if success:
                logger.info("Camera IDs auto-fixed successfully")
                return True
            else:
                logger.error("Failed to auto-fix camera IDs")
                return False

        except Exception as e:
            logger.error(f"Error during auto-fix camera IDs: {e}")
            return False

    def update_detection_settings(self, settings: dict):
        """Update detection settings from GUI controls."""
        try:
            logger.info(f"Updating detection settings: {settings}")

            # Update detection engine settings
            if self.detection_engine:
                # Update confidence thresholds
                if 'human_confidence' in settings:
                    self.detection_engine.human_confidence = settings['human_confidence']
                if 'animal_confidence' in settings:
                    self.detection_engine.animal_confidence = settings['animal_confidence']

                # Update detection toggles
                if 'human_detection' in settings:
                    self.detection_engine.human_detection_enabled = settings['human_detection']
                if 'animal_detection' in settings:
                    self.detection_engine.animal_detection_enabled = settings['animal_detection']
                if 'face_recognition' in settings:
                    self.detection_engine.face_recognition_enabled = settings['face_recognition']
                if 'pet_identification' in settings:
                    self.detection_engine.pet_identification_enabled = settings['pet_identification']

                logger.info("Detection settings updated successfully")
            else:
                logger.warning("Detection engine not available for settings update")

        except Exception as e:
            logger.error(f"Error updating detection settings: {e}")

    def get_recent_detections(self, limit: int = 50):
        """Get recent detection logs from database."""
        if self.db_manager:
            return self.db_manager.get_recent_detections(limit)
        return []

    def get_metric_averages(self, hours: int = 24) -> dict:
        """Get average performance metrics for the last N hours."""
        if self.db_manager:
            return self.db_manager.get_metric_averages(hours)
        return {}

    def get_detection_stats(self, days: int = 7) -> dict:
        """Get detection statistics for the last N days."""
        if self.db_manager:
            return self.db_manager.get_detection_stats(days)
        return {}

    def get_system_metrics(self, metric_type: str = None, limit: int = 100):
        """Get recent system metrics from database."""
        if self.db_manager:
            return self.db_manager.get_recent_metrics(metric_type, limit)
        return []

    def get_database_stats(self) -> dict:
        """Get database statistics."""
        if self.db_manager:
            return self.db_manager.get_database_stats()
        return {}

    def get_uptime_stats(self) -> dict:
        """Get system uptime and operational statistics."""
        if self.db_manager:
            return self.db_manager.get_system_uptime_stats()
        return {}

    def get_pet_identification_stats(self, days: int = 7) -> dict:
        """Get pet identification statistics."""
        if self.db_manager:
            return self.db_manager.get_pet_identification_stats(days)
        return {}

    def get_current_performance_metrics(self) -> dict:
        """Get current real-time performance metrics."""
        metrics = {}

        if self.performance_tracker:
            current_metrics = self.performance_tracker.get_current_metrics()

            # Get FPS
            fps = self.performance_tracker.fps_tracker.get_fps()
            metrics['detection_fps'] = round(fps, 1)

            # Get resource usage
            metrics.update(current_metrics)

            # Calculate processing times from recent metrics
            recent_detection_times = self.performance_tracker.get_recent_values('detection_time', 10)
            if recent_detection_times:
                metrics['processing_time'] = round(sum(recent_detection_times) / len(recent_detection_times), 1)

            recent_face_times = self.performance_tracker.get_recent_values('face_recognition_time', 10)
            if recent_face_times:
                metrics['face_recognition_fps'] = round(1000 / (sum(recent_face_times) / len(recent_face_times)), 1)

            recent_animal_times = self.performance_tracker.get_recent_values('animal_recognition_time', 10)
            if recent_animal_times:
                metrics['animal_id_fps'] = round(1000 / (sum(recent_animal_times) / len(recent_animal_times)), 1)

        return metrics

    def get_system_health_status(self) -> dict:
        """Get comprehensive system health status."""
        status = self.get_system_status()

        # Add more detailed health information
        health_status = {
            'detection_engine': 'Running' if status['detection_active'] else 'Stopped',
            'camera_connection': 'Connected' if status['camera_connected'] else 'Disconnected',
            'telegram_bot': 'Active' if status['telegram_connected'] else 'Inactive',
            'database': 'Connected' if status['database_connected'] else 'Disconnected',
            'gpu_acceleration': 'Enabled' if self.settings.enable_gpu else 'Disabled',
            'performance_monitoring': 'Running' if (self.performance_tracker and self.performance_tracker.monitoring_active) else 'Stopped'
        }

        # Add resource usage warnings
        if self.performance_tracker:
            current_metrics = self.performance_tracker.get_current_metrics()

            # Storage warning
            storage_usage = current_metrics.get('storage_usage', 0)
            if storage_usage > 80:
                health_status['storage_space'] = 'WARNING: Low'
            else:
                health_status['storage_space'] = 'Normal'

            # Temperature warning (if available)
            gpu_temp = current_metrics.get('gpu_temperature', 0)
            if gpu_temp > 80:
                health_status['temperature'] = 'WARNING: High'
            else:
                health_status['temperature'] = 'Normal'

        return health_status

    def get_detection_stats(self, days: int = 7):
        """Get detection statistics."""
        if self.db_manager:
            return self.db_manager.get_detection_stats(days)
        return {}

    def get_system_metrics(self, metric_type: str = None, limit: int = 100):
        """Get recent system metrics from database."""
        if self.db_manager:
            return self.db_manager.get_recent_metrics(metric_type, limit)
        return []

    def get_metric_averages(self, hours: int = 24):
        """Get average metrics for the last N hours."""
        if self.db_manager:
            return self.db_manager.get_metric_averages(hours)
        return {}
    
    def run(self):
        """Run the main application."""
        try:
            self.running = True
            
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info("Starting Intruder Detection System...")
            
            if self.gui:
                # Run GUI main loop
                self.gui.run()
            else:
                # Run in headless mode
                logger.info("Running in headless mode")
                while self.running and not self.shutdown_event.is_set():
                    time.sleep(1)
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        finally:
            self.shutdown()
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown the system."""
        try:
            logger.info("Shutting down Intruder Detection System...")
            self.running = False
            self.shutdown_event.set()
            
            # Stop detection
            if self.detection_active:
                self.stop_detection()
            
            # Stop notification system
            if self.notification_system:
                self.notification_system.stop_listening()
            
            # Stop performance monitoring
            if self.performance_tracker:
                self.performance_tracker.stop_monitoring()
            
            # Release camera resources
            if self.camera_manager:
                self.camera_manager.release_all_cameras()
            
            # Clean up GUI
            if self.gui:
                self.gui.cleanup()
            
            logger.info("System shutdown completed")
            system_logger.log_shutdown("IntruderDetectionSystem")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    # Configuration change callbacks
    def _on_video_config_change(self, key_path: str, new_value: "Any", old_value: "Any"):
        """Handle video configuration changes."""
        try:
            logger.info(f"Video config changed: {key_path} = {new_value}")

            # Apply changes to camera manager if active
            if self.camera_manager and key_path in ['video.frame_width', 'video.frame_height']:
                # Camera resolution changes require restart
                logger.info("Camera resolution changed - restart detection for changes to take effect")

        except Exception as e:
            logger.error(f"Error handling video config change: {e}")

    def _on_detection_config_change(self, key_path: str, new_value: "Any", old_value: "Any"):
        """Handle detection configuration changes."""
        try:
            logger.info(f"Detection config changed: {key_path} = {new_value}")

            # Apply changes to detection engine if active
            if self.detection_engine:
                if key_path == 'detection.yolo_confidence':
                    self.detection_engine.confidence = new_value
                elif key_path == 'detection.yolo_iou_threshold':
                    self.detection_engine.iou_threshold = new_value

        except Exception as e:
            logger.error(f"Error handling detection config change: {e}")

    def _on_performance_config_change(self, key_path: str, new_value: "Any", old_value: "Any"):
        """Handle performance configuration changes."""
        try:
            logger.info(f"Performance config changed: {key_path} = {new_value}")

            # Apply changes to performance tracker if active
            if self.performance_tracker and key_path == 'performance.enable_performance_monitoring':
                if new_value and not self.performance_tracker.monitoring_active:
                    self.performance_tracker.start_monitoring()
                elif not new_value and self.performance_tracker.monitoring_active:
                    self.performance_tracker.stop_monitoring()

        except Exception as e:
            logger.error(f"Error handling performance config change: {e}")

    def update_config(self, updates: "Dict[str, Any]") -> bool:
        """Update configuration at runtime."""
        if self.config_manager:
            return self.config_manager.update_runtime(updates)
        return False

    def get_config(self, key_path: str = None):
        """Get configuration value(s)."""
        if self.config_manager:
            if key_path:
                return self.config_manager.get(key_path)
            else:
                return self.config_manager.config
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Intruder Detection System 2025")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--migrate", action="store_true", help="Migrate from MariaDB to SQLite")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(log_level=args.log_level)
    
    logger.info("=" * 60)
    logger.info("Intruder Detection System 2025 - Starting")
    logger.info("=" * 60)
    
    # Handle migration if requested
    if args.migrate:
        logger.info("Starting database migration from MariaDB to SQLite...")
        from database.migrations.mariadb_to_sqlite import run_migration
        success = run_migration()
        if success:
            logger.info("Migration completed successfully")
        else:
            logger.error("Migration failed")
        return
    
    # Create and initialize system
    system = IntruderDetectionSystem(args.config)
    
    if not system.initialize():
        logger.error("System initialization failed")
        sys.exit(1)
    
    # Run the system
    try:
        system.run()
    except Exception as e:
        logger.error(f"System crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
