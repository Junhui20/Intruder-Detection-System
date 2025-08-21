# Intruder Detection System - Project Overview

## 🎯 Quick Start

### Essential Files (Root Directory)
- **`main.py`** - Start the application
- **`requirements.txt`** - Install dependencies
- **`config.yaml`** - Basic configuration
- **`README.md`** - Full documentation

### Quick Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Set up secure configuration
python scripts/setup_secure_config.py

# Run the system
python main.py
```

## 📂 Folder Organization

### 🧠 **core/** - Detection Engines
The brain of the system - all AI detection logic
- Face recognition (unified system)
- Object detection (YOLO11n)
- Animal/pet identification
- Camera management
- Telegram notifications

### 🖥️ **gui/** - User Interface
Modern GUI for system control and monitoring
- Real-time detection display
- Camera configuration
- Face/pet registration
- Performance monitoring
- Telegram bot management

### ⚙️ **config/** - Configuration
Secure configuration management
- Environment variable support
- Camera settings
- Detection parameters
- System preferences

### 🗄️ **database/** - Data Storage
SQLite database for all system data
- Detection logs
- Known faces/pets
- System settings
- Migration scripts

### 🛠️ **utils/** - Utilities
Helper functions and system utilities
- Image processing
- GPU optimization
- Performance tracking
- Logging system

### 📜 **scripts/** - Setup & Tools
Installation and maintenance scripts
- Automated installation
- Secure configuration setup
- Dependency checking
- Environment setup

### 🧪 **tests/** - Quality Assurance
Comprehensive testing suite
- Unit tests
- Integration tests
- Security tests
- Performance tests

### 📚 **docs/** - Documentation
Complete project documentation
- Installation guide
- Security best practices
- Development guide
- API documentation

### 🤖 **models/** - AI Models
Pre-trained models and weights
- YOLO11n detection models
- Face detection models
- Optimized model formats

### 💾 **data/** - User Data
User-specific data storage
- Known face images
- Pet photos
- Detection results
- System backups

### 📦 **dependencies/** - External Files
External dependencies and packages
- Pre-compiled wheels
- Platform-specific libraries

## 🚀 Development Workflow

### 1. Setup Development Environment
```bash
python scripts/install.py
python scripts/setup_secure_config.py
```

### 2. Run Tests
```bash
python tests/run_tests.py
```

### 3. Start Development
```bash
python main.py
```

### 4. Add New Features
- Core logic → `core/`
- GUI components → `gui/`
- Configuration → `config/`
- Tests → `tests/`
- Documentation → `docs/`

## 🔒 Security Features

### Environment Variables
- Sensitive data in environment variables
- No hardcoded secrets in code
- Secure configuration loading

### File Organization
- `.env` files excluded from git
- Proper file permissions
- Secure data storage

## 📊 System Architecture

```
User Interface (GUI) ←→ Core Detection Engines ←→ Database
        ↓                        ↓                    ↓
Configuration System    ←→    Utilities    ←→    Data Storage
        ↓                        ↓                    ↓
Scripts & Tools         ←→    Tests       ←→    Documentation
```

## 🎛️ Key Components

### Detection Pipeline
1. **Camera Input** → `core/camera_manager.py`
2. **Object Detection** → `core/detection_engine.py`
3. **Face Recognition** → `core/face_recognition.py`
4. **Pet Identification** → `core/animal_recognition.py`
5. **Notifications** → `core/notification_system.py`

### User Interface
1. **Main Dashboard** → `gui/main_window.py`
2. **Live Detection** → `gui/detection_view.py`
3. **Camera Setup** → `gui/ip_camera_manager.py`
4. **Face/Pet Management** → `gui/entity_management.py`
5. **System Monitoring** → `gui/performance_monitor.py`

## 🔧 Maintenance

### Regular Tasks
- Update AI models in `models/`
- Review security settings in `config/`
- Run tests from `tests/`
- Check logs in `data/`
- Update documentation in `docs/`

### Troubleshooting
1. Check `docs/SECURITY.md` for configuration issues
2. Run `python tests/test_security.py` for security validation
3. Use `python scripts/check_dependencies.py` for dependency issues
4. Review logs in `data/` for runtime errors

## 📈 Performance Optimization

### GPU Acceleration
- Models optimized in `models/`
- GPU utilities in `utils/gpu_optimization.py`
- Performance tracking in `utils/performance_tracker.py`

### System Monitoring
- Real-time metrics in GUI
- Performance logs in database
- Resource usage tracking

---

**This clean structure makes the project easy to navigate, maintain, and extend. Each folder has a specific purpose, and the root directory contains only essential files.**
