# 🔧 Development Guide - Intruder Detection System

## 🎯 Development Environment Setup

### Prerequisites

- **Python 3.8+** (Recommended: Python 3.12)
- **NVIDIA GPU** with CUDA support (RTX 3050+ recommended)
- **Windows 11** (64-bit)
- **Git** for version control
- **Visual Studio Code** (recommended IDE)

### Development Installation

```bash
# Clone repository
git clone <repository-url>
cd intruder-detection-system

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Install in development mode
pip install -e .
```

## 📁 Project Architecture

### Core Components

```
core/
├── detection_engine.py      # YOLO11n + optimization
├── face_recognition.py      # Multi-face recognition
├── animal_recognition.py    # Pet identification
├── camera_manager.py        # IP camera handling
├── notification_system.py   # Telegram integration
├── model_optimization.py    # TensorRT/quantization
└── performance_optimizer.py # System optimization
```

### GUI Architecture

```
gui/
├── main_window.py          # Main dashboard
├── detection_view.py       # Real-time detection
├── ip_camera_manager.py    # Camera configuration
├── entity_management.py    # Human/animal registration
├── notification_center.py  # Telegram user management
└── performance_monitor.py  # System metrics
```

### Database Layer

```
database/
├── models.py              # SQLAlchemy models
├── database_manager.py    # Database operations
└── migrations/            # Schema migrations
```

## 🛠️ Development Workflow

### 1. Setting Up Development Environment

```bash
# Install development tools
pip install black flake8 pytest pytest-cov

# Set up pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### 2. Code Style Guidelines

**Python Style**: Follow PEP 8 with these specifics:
- **Line length**: 88 characters (Black default)
- **Imports**: Use absolute imports
- **Docstrings**: Google style
- **Type hints**: Required for public methods

**Example:**
```python
def detect_objects(self, frame: np.ndarray) -> Dict[str, Any]:
    """
    Detect objects in a frame using YOLO11n.
    
    Args:
        frame: Input image frame as numpy array
        
    Returns:
        Dictionary containing detection results with humans and animals
        
    Raises:
        ValueError: If frame is invalid
    """
    if frame is None or frame.size == 0:
        raise ValueError("Invalid frame provided")
    
    # Implementation here
    return detection_results
```

### 3. Testing Strategy

#### Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_detection.py

# Run with coverage
python -m pytest --cov=core tests/
```

#### Integration Tests
```bash
# Test database operations
python tests/test_database.py

# Test detection pipeline
python tests/test_detection.py

# Test camera functionality
python tests/test_multi_camera.py
```

### 4. Adding New Features

#### Step 1: Create Feature Branch
```bash
git checkout -b feature/new-detection-algorithm
```

#### Step 2: Implement Feature
```python
# Example: Adding new detection algorithm
class NewDetectionEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load the detection model."""
        # Implementation
        pass
    
    def detect_objects(self, frame: np.ndarray) -> Dict:
        """Detect objects in frame."""
        # Implementation
        return {}
```

#### Step 3: Add Tests
```python
# tests/test_new_detection.py
import pytest
from core.new_detection import NewDetectionEngine

class TestNewDetectionEngine:
    def test_initialization(self):
        engine = NewDetectionEngine("model.pt")
        assert engine.model_path == "model.pt"
    
    def test_detect_objects(self):
        engine = NewDetectionEngine("model.pt")
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        result = engine.detect_objects(frame)
        assert isinstance(result, dict)
```

#### Step 4: Update Documentation
- Update API documentation in `docs/API.md`
- Add usage examples
- Update README if needed

### 5. Performance Optimization

#### Profiling Code
```python
import cProfile
import pstats

# Profile detection function
profiler = cProfile.Profile()
profiler.enable()

# Your code here
detections = detection_engine.detect_objects(frame)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

#### Memory Optimization
```python
import tracemalloc

# Track memory usage
tracemalloc.start()

# Your code here
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

tracemalloc.stop()
```

## 🔧 Common Development Tasks

### Adding New Camera Support

1. **Extend CameraManager**:
```python
class CameraManager:
    def add_camera_type(self, camera_type: str, connection_handler):
        """Add support for new camera type."""
        self.camera_handlers[camera_type] = connection_handler
```

2. **Create Camera Handler**:
```python
class NewCameraHandler:
    def connect(self, config: Dict) -> bool:
        """Connect to camera with specific protocol."""
        pass
    
    def capture_frame(self) -> np.ndarray:
        """Capture frame from camera."""
        pass
```

### Adding New Recognition Algorithm

1. **Create Recognition Class**:
```python
class NewRecognitionSystem:
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
    
    def recognize(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Recognize entities in detections."""
        pass
```

2. **Integrate with Main System**:
```python
# In main.py
if config.use_new_recognition:
    self.recognition_system = NewRecognitionSystem()
```

### Database Schema Changes

1. **Create Migration**:
```python
# database/migrations/add_new_table.py
def upgrade():
    """Add new table for feature."""
    op.create_table(
        'new_feature',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow)
    )

def downgrade():
    """Remove new table."""
    op.drop_table('new_feature')
```

2. **Update Models**:
```python
# database/models.py
class NewFeature(Base):
    __tablename__ = 'new_feature'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
```

## 🐛 Debugging

### Logging Configuration
```python
import logging

# Set up detailed logging for development
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### Common Debug Scenarios

#### Detection Issues
```python
# Debug detection pipeline
def debug_detection(frame, detection_engine):
    logger.debug(f"Frame shape: {frame.shape}")
    
    detections = detection_engine.detect_objects(frame)
    logger.debug(f"Detections found: {len(detections['humans'])} humans, {len(detections['animals'])} animals")
    
    for i, human in enumerate(detections['humans']):
        logger.debug(f"Human {i}: bbox={human['bbox']}, confidence={human['confidence']}")
```

#### Camera Connection Issues
```python
# Debug camera connectivity
def debug_camera_connection(camera_url):
    import requests
    
    try:
        response = requests.get(camera_url, timeout=5)
        logger.debug(f"Camera response: {response.status_code}")
    except Exception as e:
        logger.error(f"Camera connection failed: {e}")
```

## 📊 Performance Monitoring

### Real-time Metrics
```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'fps': 0,
            'processing_time': 0,
            'memory_usage': 0,
            'gpu_usage': 0
        }
    
    def update_metrics(self, processing_time: float):
        """Update performance metrics."""
        self.metrics['fps'] = 1.0 / processing_time
        self.metrics['processing_time'] = processing_time
        # Update other metrics
```

### Optimization Guidelines

1. **Use TensorRT** for GPU acceleration
2. **Batch processing** for multiple cameras
3. **Frame skipping** for high FPS cameras
4. **Memory pooling** for large images
5. **Async processing** for I/O operations

## 🚀 Deployment

### Production Build
```bash
# Create production build
python setup.py build

# Create installer
python setup.py bdist_wininst  # Windows
python setup.py bdist_rpm      # Linux
```

### Configuration Management
```python
# Use environment-specific configs
config_file = os.getenv('CONFIG_FILE', 'config.yaml')
settings = Settings(config_file)
```

## 📝 Contributing Guidelines

1. **Fork** the repository
2. **Create feature branch** from main
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Submit pull request** with clear description

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

---

## 🔗 Additional Resources

- **API Documentation**: `API.md`
- **Installation Guide**: `INSTALLATION.md`
- **Security Guide**: `SECURITY.md`
- **Camera Setup**: `CAMERA_SETUP.md`
