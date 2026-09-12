# 📦 Installation Guide - Intruder Detection System

## 🚀 Quick Start (Recommended)

The system includes smart dependency checking that only installs what you need, saving time and bandwidth.

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/intruder-detection-system.git
cd intruder-detection-system
```

### Step 2: Quick Setup
Choose your platform:

#### Windows
```cmd
setup.bat
```

#### Linux/Mac
```bash
chmod +x setup.sh
./setup.sh
```

#### Any Platform
```bash
python setup.py
```

### Step 3: Run the Application
```bash
python main.py
```

## 🔍 Smart Installation Options

### Option A: Check Dependencies First
See what's already installed vs what's needed:
```bash
python scripts/check_dependencies.py
```

**Example Output:**
```
🔍 Dependency Check Report
==================================================

✅ Satisfied Dependencies (8):
   📦 numpy 1.24.3 - Version satisfied
   📦 opencv-python 4.8.1 - Version satisfied
   📦 tkinter - Built-in module
   📦 sqlite3 - Built-in module

⚠️ Outdated Dependencies (2):
   📦 torch 1.13.0 -> >=2.0.0 (needs update)
   📦 torchvision 0.14.0 -> >=0.15.0 (needs update)

❌ Missing Dependencies (3):
   📦 ultralytics >=8.0.0 - Not installed
   📦 face-recognition >=1.3.0 - Not installed
   📦 requests >=2.31.0 - Not installed

📊 Summary:
   Total dependencies: 13
   ✅ Satisfied: 8
   ⚠️ Outdated: 2
   ❌ Missing: 3

💡 Installation Command:
   pip install ultralytics>=8.0.0 face-recognition>=1.3.0 requests>=2.31.0 torch>=2.0.0 torchvision>=0.15.0
```

### Option B: Smart Installation
Install only what's missing or outdated:
```bash
# Basic smart install
python scripts/install.py

# With GPU support (auto-detected)
python scripts/install.py --gpu

# Force reinstall everything
python scripts/install.py --force

# CPU-only mode
python scripts/install.py --no-gpu
```

**Smart Installer Features:**
- ✅ **Selective Installation**: Only installs missing/outdated packages
- ✅ **GPU Detection**: Automatically detects CUDA and installs appropriate packages
- ✅ **Version Checking**: Compares installed vs required versions
- ✅ **Virtual Environment Detection**: Warns if not in venv
- ✅ **Installation Verification**: Tests imports after installation

### Option C: Traditional Installation
If you prefer the traditional approach:
```bash
pip install -r requirements.txt
```

## 🎮 GPU Support

### Automatic GPU Detection
The smart installer automatically detects GPU support:

```bash
python scripts/install.py
```

**Detection Process:**
1. Checks for `nvidia-smi` command
2. Tests existing PyTorch CUDA availability
3. Installs appropriate packages based on detection

### Manual GPU Configuration

#### Force Enable GPU Support
```bash
python scripts/install.py --gpu
```

#### Force Disable GPU Support (CPU-only)
```bash
python scripts/install.py --no-gpu
```

#### Manual CUDA Installation
If automatic detection fails:
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 🔧 Troubleshooting Installation

### Common Issues

#### 1. Python Version Error
```
❌ Python 3.8+ required!
```
**Solution:** Install Python 3.8 or higher from [python.org](https://www.python.org/downloads/)

#### 2. Virtual Environment Warning
```
⚠️ Not in virtual environment
```
**Solution:** Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

#### 3. Face Recognition Installation Fails
**Common on Windows/Mac**
```bash
# Install dependencies first
pip install cmake dlib
pip install face-recognition
```

#### 4. OpenCV Installation Issues
```bash
# Try alternative OpenCV package
pip uninstall opencv-python
pip install opencv-python-headless
```

#### 5. CUDA/GPU Issues
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Dependency Conflicts

#### Reset Installation
```bash
# Remove all packages and reinstall
pip freeze > installed_packages.txt
pip uninstall -r installed_packages.txt -y
python scripts/install.py --force
```

#### Clean Virtual Environment
```bash
# Create fresh environment
deactivate
rm -rf venv  # Linux/Mac
rmdir /s venv  # Windows
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
python scripts/install.py
```

## 📋 Installation Verification

### Quick Test
```bash
python -c "
import cv2, numpy, torch, ultralytics, face_recognition
print('✅ All core modules imported successfully')
print(f'OpenCV: {cv2.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Full System Test
```bash
# Run comprehensive test
python scripts/check_dependencies.py

# Test core imports
python -c "
from core.detection_engine import DetectionEngine
from core.face_recognition import FaceRecognitionSystem
from core.animal_recognition import AnimalRecognitionSystem
print('✅ All core systems can be imported')
"
```

## 🌐 Platform-Specific Notes

### Windows 11 (Detailed Guide)

#### Common Windows Issues & Solutions

**Issue: "CMake not found" (Most Common)**
```powershell
# Solution 1: Install CMake (Recommended)
# 1. Download from https://cmake.org/download/
# 2. Run installer and check "Add CMake to system PATH"
# 3. Restart terminal and test: cmake --version
# 4. Install face-recognition: pip install face-recognition

# Solution 2: Skip face recognition temporarily
pip install -r requirements.txt
python main.py  # Run without face recognition
```

**Issue: "Visual Studio Build Tools required"**
```powershell
# Install build tools
pip install --upgrade setuptools wheel
# Or download Visual Studio Build Tools from Microsoft
```

**Issue: "Permission denied"**
```powershell
# Run PowerShell as Administrator
```

**Issue: "Execution policy restricted"**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Issue: "Long path names"**
- Enable long paths in Windows or move project to shorter path

#### Windows Performance Tips
1. **Use SSD**: Install on SSD for better performance
2. **Close unnecessary programs**: Free up RAM and CPU
3. **Windows Defender**: Add project folder to exclusions
4. **Power settings**: Use "High performance" mode
5. **GPU drivers**: Update NVIDIA drivers for CUDA support

#### Windows Verification
```powershell
# Test core functionality
python -c "import cv2, torch, ultralytics; print('✅ Core modules OK')"

# Test face recognition (if installed)
python -c "import face_recognition; print('✅ Face recognition OK')"

# Run the application
python main.py
```

### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt update
sudo apt install python3-dev python3-pip cmake build-essential
```

### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake
```

## 📊 Installation Size

**Typical Installation Sizes:**
- **CPU-only**: ~2.5GB
- **GPU (CUDA)**: ~4.5GB
- **Full development**: ~6GB

**Breakdown:**
- PyTorch: ~1.5GB (CPU) / ~3GB (GPU)
- OpenCV: ~200MB
- Face Recognition: ~100MB
- YOLO Models: ~50MB (downloaded on first run)
- Other dependencies: ~500MB

## 🚀 Post-Installation

### First Run
```bash
python main.py
```

### Initial Configuration
1. **Camera Setup**: Configure IP cameras in GUI
2. **Telegram Bot**: Add bot token in Notification Center
3. **Entity Registration**: Add known people and pets
4. **Threshold Tuning**: Adjust confidence levels

### Performance Optimization
```bash
# Check system performance
python -c "
from utils.gpu_optimization import GPUOptimizer
optimizer = GPUOptimizer()
print(optimizer.get_hardware_info())
"
```

## 💡 Tips for Faster Installation

1. **Use Virtual Environment**: Isolates dependencies
2. **Check First**: Run dependency check before installing
3. **Selective Install**: Use smart installer to avoid unnecessary downloads
4. **Cache Packages**: pip automatically caches downloaded packages
5. **Parallel Downloads**: pip installs multiple packages simultaneously

## 🆘 Getting Help

If you encounter issues:

1. **Check Logs**: Look for error messages in the installation output
2. **Dependency Check**: Run `python scripts/check_dependencies.py`
3. **System Requirements**: Verify your system meets minimum requirements
4. **Clean Install**: Try a fresh virtual environment
5. **GitHub Issues**: Report persistent issues with detailed error logs

## 📝 Manual Installation (Advanced)

If automated installation fails, you can install dependencies manually:

```bash
# Core AI/ML packages
pip install torch>=2.0.0 torchvision>=0.15.0
pip install ultralytics>=8.0.0
pip install opencv-python>=4.8.0
pip install face-recognition>=1.3.0
pip install numpy>=1.24.0
pip install Pillow>=10.0.0

# System and utilities
pip install requests>=2.31.0
pip install psutil>=5.9.0
pip install PyYAML>=6.0
pip install python-dotenv>=1.0.0

# Optional GPU monitoring
pip install "nvidia-ml-py>=12.535,<13"

# Test installation
python scripts/check_dependencies.py
```

---

**Ready to start detecting intruders? 🚀**

After successful installation, run `python main.py` to launch the system!
