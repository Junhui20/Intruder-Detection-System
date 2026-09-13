#!/usr/bin/env python3
"""
Intruder Detection System - Main Entry Point

Advanced intruder detection system with YOLO11n, individual pet recognition,
and comprehensive Telegram integration.

Features:
- YOLO11n object detection with person tracking
- Multi-face recognition (no separate threading as preferred)
- Individual pet identification (e.g., 'Jacky') using hybrid approach
- IP camera support with HTTP/HTTPS and local fallback
- Bidirectional Telegram bot with command listening
- SQLite database
- Web UI (FastAPI + htmx) behind WEB_UI_PASSWORD
- Real-time performance monitoring

Author: Intruder Detection System Team
Version: 1.0.0
"""

import sys
import os
import argparse
import json
import re
import signal
import threading
import time
import warnings

import cv2
import numpy as np
from typing import Optional, Any, Dict, List
from pathlib import Path

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
from database.models import WhitelistEntry
from core.detection_engine import DetectionEngine
from core.event_captions import TIER_MODELS as CAPTION_MODELS, EventCaptioner
from core.face_recognition import TIER_MODELS as FACE_MODELS, FaceRecognitionSystem
from core.animal_recognition import TIER_MODELS as PET_MODELS, AnimalRecognitionSystem
from core.camera_manager import CameraManager
from core.notification_system import NotificationSystem
from web.app import serve as serve_web
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
        self.started_at = time.time()
        self.detection_timeout = 10.0  # seconds out of frame before the same identity is a new visit
        self.last_detection_time: Dict[str, float] = {}  # "kind_name" -> last seen

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
        self.captioner: Optional[EventCaptioner] = None
        self.notification_system: Optional[NotificationSystem] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        self.latest_frame = None  # newest annotated frame, for the web UI's MJPEG stream
        self.latest_raw = None  # the same frame before boxes were drawn: what enrolment crops
        self.latest_detections: dict = {"humans": [], "animals": []}  # of that frame
        
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
            for key, problem in self.settings.validate_settings().items():
                logger.error(f"config.yaml: {key}: {problem}")
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

            engine = DetectionEngine.from_config(self.detection_config)

            # Both recognisers are built and given their rosters before they replace
            # the live ones: the detection thread must never see an empty roster.
            faces = FaceRecognitionSystem(
                confidence_threshold=self.detection_config.human_confidence_threshold,
                max_faces_per_frame=self.detection_config.max_faces_per_frame,
                model=self.settings.face_model or FACE_MODELS[self.settings.tier],
                use_gpu=self.settings.enable_gpu,
            )
            pets = AnimalRecognitionSystem(
                confidence_threshold=self.detection_config.animal_confidence_threshold,
                pet_identification_threshold=self.detection_config.pet_identification_threshold,
                model=self.settings.pet_model or PET_MODELS[self.settings.tier],
                use_gpu=self.settings.enable_gpu,
            )
            if faces.backend_type != "insightface" or pets.model is None:
                logger.error("A recognition model failed to load; see the errors above")
                return False
            faces.load_known_faces([e.to_dict() for e in self.db_manager.get_whitelist_entries(entity_type="human")])
            pets.load_known_pets([e.to_dict() for e in self.db_manager.get_whitelist_entries(entity_type="animal")])
            self.detection_engine, self.face_recognition, self.animal_recognition = engine, faces, pets

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
            bot_token = self.settings.bot_token  # TELEGRAM_BOT_TOKEN only; config.yaml is ignored
            if not bot_token:
                logger.warning("No Telegram bot token configured")
                return True

            self.notification_system = NotificationSystem(bot_token, self.db_manager, system=self)
            self.notification_system.default_cooldown = self.settings.notification_cooldown
            if self.settings.captions_enabled:
                try:  # a caption problem must never cost the alert channel
                    self.captioner = EventCaptioner(
                        self.settings.captions_model or CAPTION_MODELS[self.settings.tier],
                        self.settings.ollama_host,
                    )
                except Exception as e:
                    logger.warning(f"Event captions off: {e}")
            
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
            self.latest_frame = self.latest_raw = None

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

                # Skip processing for performance if configured
                if frame_count % max(1, int(self.settings.process_every_n_frames)) != 0:
                    self.latest_frame = self.latest_raw = frame
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
                
                self._process_detections(detections, frame)
                self.latest_raw = frame
                self.latest_frame = ImageProcessor.create_detection_overlay(frame, detections)
                self.latest_detections = detections

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

    def _process_detections(self, detections: dict, frame) -> None:
        """
        Turn this frame's detections into events.

        A session is one visit by one identity ("human_Hui", "animal_Unknown dog"):
        it opens on first sight, is logged once, and can reopen after the identity
        has been out of frame for ``detection_timeout`` seconds.
        """
        now = time.time()
        for sid, last in list(self.last_detection_time.items()):
            if now - last >= self.detection_timeout:
                del self.last_detection_time[sid]
        seen = []
        for human in detections["humans"]:
            known = human.get("recognition_status") == "known"
            name = human["identity"] if known else "Unknown person"
            seen.append(("human", name, known, human.get("face_confidence") if known else human.get("confidence", 0),
                         human.get("bbox"), None, "person"))
        for animal in detections["animals"]:
            known = animal.get("recognition_status") == "known_pet"
            kind_name = animal.get("animal_type", "animal")
            name = animal["pet_identity"] if known else f"Unknown {kind_name}"
            seen.append(("animal", name, known, animal.get("identification_confidence") if known else animal.get("confidence", 0),
                         animal.get("bbox"), animal.get("class_id"), kind_name))
        for kind, name, known, confidence, bbox, class_id, subject in seen:
            sid = f"{kind}_{name}"
            if sid not in self.last_detection_time:  # a new visit
                self._record(kind, name, known, float(confidence or 0), frame, bbox, class_id, subject)
            self.last_detection_time[sid] = now

    def _record(self, kind, name, known, confidence, frame, bbox, class_id, subject) -> None:
        """
        Log one visit; alert Telegram about strangers, caption the photo when
        the VLM is up. Sending and captioning run on a thread so the detection
        loop is never held up; the photo goes out first, the caption is edited
        in under it — and stored on the event — when it arrives.
        """
        bot = self.notification_system
        muted = not bot or time.time() < bot.muted_until
        photo_path = None if known and not self.settings.notify_family else self._save_photo(frame, f"{kind}_{name}")
        will_alert = not muted and (not known or self.settings.notify_family)
        log_id = self.db_manager.log_detection(
            detection_type=kind, entity_name=name, confidence=confidence, image_path=photo_path,
            notification_sent=False,  # the send thread sets it once a message has gone out
        ) if self.db_manager else None
        logger.info(f"{name} ({kind}, {confidence:.0%}){'' if will_alert else ', no alert'}")
        if not will_alert:
            return
        message = (f"👋 {name} is home" if known else
                   f"🚨 Unknown person detected ({confidence:.0%})" if kind == "human" else
                   f"🐾 Unknown {subject} detected ({confidence:.0%})")
        snapshot = frame.copy()
        context = {"kind": kind, "label": subject, "bbox": bbox, "class_id": class_id}

        def send():
            try:
                sent = bot.send_notification(kind, message, photo_path=photo_path, context=context)
                if sent and log_id:
                    self.db_manager.set_detection_sent(log_id, True)
                if not (sent and photo_path and self.captioner and self.settings.captions_enabled) or known:
                    return
                caption = self.captioner.describe(snapshot, subject)
                if not caption:
                    return
                if log_id:
                    self.db_manager.set_detection_caption(log_id, caption)
                for chat_id, message_id in sent:
                    bot.edit_caption(chat_id, message_id, f"{message}\n💬 {caption}")
            except Exception:
                logger.exception("Alert delivery failed")

        threading.Thread(target=send, daemon=True).start()

    def _save_photo(self, frame, label: str) -> Optional[str]:
        """Write the frame under data/detection_photos; None if that fails."""
        folder = Path("data/detection_photos")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{re.sub(r'[^a-z0-9]+', '_', label.lower())}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        return str(path) if cv2.imwrite(str(path), frame) else None

    def snapshot(self) -> Optional[str]:
        """Save the current frame and send it to every recipient (ignores mute); the path."""
        if self.latest_frame is None:
            return None
        path = self._save_photo(self.latest_frame, "snapshot")
        if self.notification_system and path:
            self.notification_system.send_notification("snapshot", "📷 Snapshot", photo_path=path, force=True)
        return path

    def enrol_from_frame(self, kind: str, name: str, bbox, class_id: Optional[int] = None) -> Optional[int]:
        """Crop the box out of the current frame and enrol it; the new row id."""
        if self.latest_raw is None:
            return None
        x1, y1, x2, y2 = (int(v) for v in bbox)
        pad = 20 if kind == "human" else 0
        crop = self.latest_raw[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        folder = Path("data/faces" if kind == "human" else "data/animals")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{re.sub(r'[^a-z0-9]+', '_', name.lower())}_{int(time.time())}.jpg"
        cv2.imwrite(str(path), crop)
        return self.enrol(kind, name, [str(path)], class_id)

    def test_entry(self, entry_id: int) -> Optional[dict]:
        """
        Run the matching recogniser on the current frame against one whitelist row.

        Returns:
            {score, threshold, match} or None when there is no frame or row.
        """
        entry = self.db_manager.get_whitelist_entry(entry_id)
        frame = self.latest_raw
        if entry is None or frame is None or not (self.face_recognition and self.animal_recognition):
            return None
        if entry.entity_type == "human":
            fr = self.face_recognition
            own = [e for n, e in zip(fr.known_face_names, fr.known_face_encodings) if n == entry.name]
            faces = fr.app.get(frame) if fr.app and own else []
            score = max((float(np.stack(own) @ f.normed_embedding).max() for f in faces), default=0.0)
            return {"score": score, "threshold": fr.confidence_threshold, "match": score >= fr.confidence_threshold}
        pets = self.animal_recognition
        pet = pets.known_pets.get(entry.name)
        boxes = [d["bbox"] for d in self.latest_detections.get("animals", []) if d.get("class_id") == entry.coco_class_id]
        crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes] or [frame]
        score = max((float((pet["embeddings"] @ q).max()) for q in pets.embed(crops)), default=0.0) if pet else 0.0
        return {"score": score, "threshold": pets.pet_identification_threshold,
                "match": score >= pets.pet_identification_threshold}

    def enrol(self, kind: str, name: str, photo_paths: List[str], class_id: Optional[int] = None):
        """
        Add a person (``human``) or pet (``animal``) to the whitelist and the live roster.

        Args:
            kind: 'human' or 'animal'.
            name: Shown in alerts.
            photo_paths: Saved photos; the first is the primary, the rest go in multiple_photos.
            class_id: COCO class for a pet (16 dog, 15 cat); ignored for people.

        Returns:
            The new whitelist row id.
        """
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        entry_id = self.db_manager.create_whitelist_entry(WhitelistEntry(
            name=name, entity_type=kind, image_path=photo_paths[0],
            multiple_photos=json.dumps(photo_paths[1:]) if len(photo_paths) > 1 else None,
            coco_class_id=class_id if kind == "animal" else None,
            individual_id=slug if kind == "animal" else None,
        ))
        self.reload_roster(kind)
        return entry_id

    def forget(self, entry_id: int) -> None:
        """Remove a whitelist row, its photos, and its place in the live roster."""
        entry = self.db_manager.get_whitelist_entry(entry_id)
        if not entry:
            return
        for path in [entry.image_path] + (json.loads(entry.multiple_photos) if entry.multiple_photos else []):
            Path(path).unlink(missing_ok=True)
        self.db_manager.delete_whitelist_entry(entry_id)
        self.reload_roster(entry.entity_type)

    def reload_roster(self, kind: str) -> None:
        """Re-read one kind of whitelist rows into the matching recogniser."""
        rows = [e.to_dict() for e in self.db_manager.get_whitelist_entries(entity_type=kind)]
        if kind == "human" and self.face_recognition:
            self.face_recognition.load_known_faces(rows)
        elif kind == "animal" and self.animal_recognition:
            self.animal_recognition.load_known_pets(rows)

    # what the Settings page may change, with the type system_config stores
    TUNABLE = {
        "human_confidence_threshold": "float", "pet_identification_threshold": "float",
        "yolo_confidence": "float", "process_every_n_frames": "integer",
        "captions_enabled": "boolean", "captions_model": "string",
        "notify_family": "boolean", "notification_cooldown": "integer",
    }

    def apply_settings(self, changes: Dict[str, Any]) -> None:
        """
        Change tunables live and remember them in the database, which
        Settings.load_from_database reads back at the next start.

        Args:
            changes: Subset of TUNABLE keys with their new values (already typed).
        """
        for key, value in changes.items():
            if key not in self.TUNABLE:
                continue
            setattr(self.settings, key, value)
            if self.db_manager:
                self.db_manager.set_config(key, value, self.TUNABLE[key])
        if "human_confidence_threshold" in changes and self.face_recognition:
            self.face_recognition.confidence_threshold = changes["human_confidence_threshold"]
            self.detection_config.human_confidence_threshold = changes["human_confidence_threshold"]
        if "pet_identification_threshold" in changes and self.animal_recognition:
            self.animal_recognition.pet_identification_threshold = changes["pet_identification_threshold"]
            self.detection_config.pet_identification_threshold = changes["pet_identification_threshold"]
        if "yolo_confidence" in changes and self.detection_engine:
            e = self.detection_engine
            e.confidence = e.human_confidence = e.animal_confidence = changes["yolo_confidence"]
        if "notification_cooldown" in changes and self.notification_system:
            self.notification_system.default_cooldown = changes["notification_cooldown"]
        if ("captions_model" in changes or changes.get("captions_enabled")) and self.notification_system:
            try:
                self.captioner = EventCaptioner(
                    self.settings.captions_model or CAPTION_MODELS[self.settings.tier], self.settings.ollama_host
                ) if self.settings.captions_enabled else None
            except Exception as e:
                logger.warning(f"Event captions off: {e}")

    def apply_tier(self, tier: str) -> None:
        """
        Switch the model-size preset live and remember it in config.yaml.

        Face, pet and caption models are rebuilt from the new tier; the
        detection loop picks them up on its next frame.
        """
        previous, self.settings.tier = self.settings.tier, tier
        if not self._initialize_detection_systems():
            self.settings.tier = previous
            logger.error(f"Tier {tier} not applied: its models failed to load; still on {previous}")
            return  # the live recognisers were never replaced
        if self.captioner:
            try:
                self.captioner = EventCaptioner(
                    self.settings.captions_model or CAPTION_MODELS[tier], self.settings.ollama_host
                )
            except Exception as e:
                logger.warning(f"Event captions off after tier switch: {e}")
        path = self.config_path
        text = open(path).read()
        updated = re.sub(r"^tier:.*$", f"tier: {tier}", text, count=1, flags=re.M)
        open(path, "w").write(updated if updated != text else f"tier: {tier}\n{text}")

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
            self.start_detection()
            serve_web(self, self.settings.web_host, self.settings.web_port)  # no-op without WEB_UI_PASSWORD
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
    parser = argparse.ArgumentParser(description="Intruder Detection System")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(log_level=args.log_level)
    
    logger.info("=" * 60)
    logger.info("Intruder Detection System - Starting")
    logger.info("=" * 60)
    
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
