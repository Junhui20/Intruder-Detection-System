# 📚 API Documentation - Intruder Detection System

## 🎯 Overview

This document provides comprehensive API reference for the Intruder Detection System's core modules and classes.

## 🔧 Core Modules

### DetectionEngine

**Location**: `core/detection_engine.py`

#### Class: `DetectionEngine`

Main YOLO11n detection engine with optimization support.

```python
from core.detection_engine import DetectionEngine

# Initialize with optimization
engine = DetectionEngine(
    model_path='yolo11n.pt',
    confidence=0.5,
    use_optimized_engine=True,
    optimized_model_dir="models"
)

# Detect objects in frame
detections = engine.detect_objects(frame)
```

**Methods:**

- `__init__(model_path, confidence, use_optimized_engine, optimized_model_dir)`
- `detect_objects(frame: np.ndarray) -> Dict`
- `get_performance_stats() -> Dict`
- `update_confidence_thresholds(human_conf: float, animal_conf: float)`

**Detection Result Format:**
```python
{
    'humans': [
        {
            'bbox': [x1, y1, x2, y2],
            'confidence': 0.85,
            'track_id': 1,
            'identity': {'name': 'Jun Hui', 'confidence': 0.92}
        }
    ],
    'animals': [
        {
            'bbox': [x1, y1, x2, y2],
            'confidence': 0.78,
            'animal_type': 'dog',
            'pet_identity': {'name': 'Jacky', 'confidence': 0.88}
        }
    ],
    'frame_info': {
        'timestamp': 1692123456.789,
        'processing_time': 0.008,
        'total_detections': 2,
        'model_format': 'tensorrt',
        'optimized': True
    }
}
```

### FaceRecognitionSystem

**Location**: `core/face_recognition.py`

#### Class: `FaceRecognitionSystem`

Multi-face recognition with configurable confidence thresholds.

```python
from core.face_recognition import FaceRecognitionSystem

# Initialize face recognition
face_system = FaceRecognitionSystem(
    confidence_threshold=0.6,
    max_faces_per_frame=5
)

# Load known faces
face_system.load_known_faces(human_data)

# Recognize faces in detections
recognized = face_system.recognize_faces(frame, human_detections)
```

**Methods:**

- `__init__(confidence_threshold, max_faces_per_frame)`
- `load_known_faces(human_data: List[Dict])`
- `recognize_faces(frame: np.ndarray, human_detections: List[Dict]) -> List[Dict]`
- `add_known_face(name: str, image_path: str)`
- `get_performance_stats() -> Dict`

### AnimalRecognitionSystem

**Location**: `core/animal_recognition.py`

#### Class: `AnimalRecognitionSystem`

Individual pet identification with hybrid approach.

```python
from core.animal_recognition import AnimalRecognitionSystem

# Initialize animal recognition
animal_system = AnimalRecognitionSystem(
    confidence_threshold=0.6,
    pet_identification_threshold=0.7
)

# Identify animals
identified = animal_system.identify_animals(frame, animal_detections)
```

**Methods:**

- `__init__(confidence_threshold, pet_identification_threshold)`
- `identify_animals(frame: np.ndarray, animal_detections: List[Dict]) -> List[Dict]`
- `add_known_pet(name: str, animal_class: str, image_paths: List[str])`
- `get_known_pets() -> Dict`

### CameraManager

**Location**: `core/camera_manager.py`

#### Class: `CameraManager`

Advanced camera management with IP camera support.

```python
from core.camera_manager import CameraManager

# Initialize camera manager
camera_manager = CameraManager()

# Capture frame
frame = camera_manager.capture_frame()

# Switch camera
camera_manager.switch_camera(camera_id=1)
```

**Methods:**

- `__init__()`
- `capture_frame() -> np.ndarray`
- `switch_camera(camera_id: int) -> bool`
- `get_camera_status() -> Dict`
- `reload_camera_config()`

### NotificationSystem

**Location**: `core/notification_system.py`

#### Class: `NotificationSystem`

Bidirectional Telegram bot with command listening.

```python
from core.notification_system import NotificationSystem

# Initialize notification system
notification_system = NotificationSystem(bot_token="your_token")

# Send notification
notification_system.send_detection_notification(
    detection_type="human",
    entity_name="Jun Hui",
    confidence=0.92,
    image_path="detection.jpg"
)
```

**Methods:**

- `__init__(bot_token: str)`
- `send_detection_notification(detection_type, entity_name, confidence, image_path)`
- `start_command_listener()`
- `stop_command_listener()`
- `add_user(chat_id: int, username: str)`

## 🗄️ Database API

### DatabaseManager

**Location**: `database/database_manager.py`

#### Class: `DatabaseManager`

SQLite database operations with full CRUD support.

```python
from database.database_manager import DatabaseManager

# Initialize database
db = DatabaseManager()

# Add whitelist entry
db.add_whitelist_entry(
    entity_type="human",
    name="Jun Hui",
    image_path="jun_hui.jpg"
)

# Get recent detections
recent = db.get_recent_detections(limit=10)
```

**Methods:**

- `__init__(db_path: str = "detection_system.db")`
- `add_whitelist_entry(entity_type, name, image_path, **kwargs)`
- `get_whitelist_entries(entity_type: str = None) -> List[WhitelistEntry]`
- `get_recent_detections(limit: int = 50) -> List[Detection]`
- `log_detection(detection_type, entity_name, confidence, **kwargs)`

## 🎨 GUI API

### MainWindow

**Location**: `gui/main_window.py`

#### Class: `MainWindow`

Main application window with 5-module interface.

```python
from gui.main_window import MainWindow

# Initialize GUI
gui = MainWindow()
gui.set_main_system(main_system)
gui.run()
```

**Methods:**

- `__init__()`
- `set_main_system(system)`
- `run()`
- `show_module(module_id: str)`

## 🔧 Utility APIs

### PerformanceTracker

**Location**: `utils/performance_tracker.py`

Real-time performance monitoring.

```python
from utils.performance_tracker import PerformanceTracker

tracker = PerformanceTracker()
tracker.start_monitoring()

# Get current metrics
metrics = tracker.get_current_metrics()
```

### ImageProcessor

**Location**: `utils/image_processing.py`

Image processing utilities.

```python
from utils.image_processing import ImageProcessor

# Create detection overlay
annotated_frame = ImageProcessor.create_detection_overlay(frame, detections)

# Extract dominant color
color = ImageProcessor.extract_dominant_color(image_region)
```

## 📝 Configuration API

### Settings

**Location**: `config/settings.py`

System settings management.

```python
from config.settings import Settings

settings = Settings()
settings.load_from_database()

# Update setting
settings.update_setting("human_confidence_threshold", 0.7)
```

## 🚀 Usage Examples

### Complete Detection Pipeline

```python
import cv2
from core.detection_engine import DetectionEngine
from core.face_recognition import FaceRecognitionSystem
from utils.image_processing import ImageProcessor

# Initialize systems
detection_engine = DetectionEngine(use_optimized_engine=True)
face_recognition = FaceRecognitionSystem()

# Process frame
frame = cv2.imread("test_image.jpg")
detections = detection_engine.detect_objects(frame)

# Add face recognition
if detections['humans']:
    detections['humans'] = face_recognition.recognize_faces(frame, detections['humans'])

# Create annotated frame
result_frame = ImageProcessor.create_detection_overlay(frame, detections)
```

### Database Operations

```python
from database.database_manager import DatabaseManager

db = DatabaseManager()

# Add known person
db.add_whitelist_entry(
    entity_type="human",
    name="Jun Hui",
    image_path="photos/jun_hui.jpg",
    face_encoding=face_encoding_data
)

# Log detection
db.log_detection(
    detection_type="human",
    entity_name="Jun Hui",
    confidence=0.92,
    bbox=[100, 100, 200, 300],
    notification_sent=True
)
```

## 🔗 Integration Examples

### Custom Detection Handler

```python
class CustomDetectionHandler:
    def __init__(self):
        self.detection_engine = DetectionEngine(use_optimized_engine=True)
        self.face_recognition = FaceRecognitionSystem()
    
    def process_frame(self, frame):
        # Run detection
        detections = self.detection_engine.detect_objects(frame)
        
        # Add face recognition
        if detections['humans']:
            detections['humans'] = self.face_recognition.recognize_faces(
                frame, detections['humans']
            )
        
        return detections
```

---

## 📞 Support

For additional API questions or custom integrations, refer to:
- **Implementation Guide**: `IMPLEMENTATION_GUIDE.md`
- **Development Guide**: `docs/DEVELOPMENT.md`
- **Project Summary**: `PROJECT_SUMMARY.md`
