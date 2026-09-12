#!/usr/bin/env python3
"""
Smart Installation Script for Intruder Detection System

This script provides intelligent installation that checks existing dependencies
and only installs what's needed, saving time and bandwidth.
"""

import sys
import subprocess
import os
import argparse
from check_dependencies import DependencyChecker


class SmartInstaller:
    """
    Smart installer that only installs missing or outdated dependencies.
    
    Features:
    - Pre-installation dependency checking
    - Selective package installation
    - GPU/CUDA detection and optimization
    - Virtual environment support
    - Installation verification
    """
    
    def __init__(self, force_reinstall: bool = False, gpu_support: bool = None):
        """
        Initialize smart installer.
        
        Args:
            force_reinstall: Force reinstall all packages
            gpu_support: Enable GPU support (auto-detect if None)
        """
        self.force_reinstall = force_reinstall
        self.gpu_support = gpu_support
        self.checker = DependencyChecker()
        
    def check_environment(self) -> bool:
        """
        Check if we're in a virtual environment and other prerequisites.
        
        Returns:
            True if environment is suitable for installation
        """
        print("🔍 Checking installation environment...")
        
        # Check if in virtual environment
        in_venv = (
            hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        )
        
        if in_venv:
            print("   ✅ Virtual environment detected")
        else:
            print("   ⚠️ Not in virtual environment")
            response = input("   Continue anyway? (y/N): ").lower().strip()
            if response != 'y':
                print("   💡 Recommendation: Create virtual environment first")
                print("      python -m venv venv")
                print("      source venv/bin/activate  # Linux/Mac")
                print("      venv\\Scripts\\activate     # Windows")
                return False
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("   ❌ Python 3.8+ required!")
            return False
        else:
            print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        
        # Check pip availability
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         check=True, capture_output=True)
            print("   ✅ pip available")
        except subprocess.CalledProcessError:
            print("   ❌ pip not available!")
            return False
        
        return True
    
    def detect_gpu_support(self) -> bool:
        """
        Auto-detect GPU support capability.
        
        Returns:
            True if GPU support should be enabled
        """
        if self.gpu_support is not None:
            return self.gpu_support
        
        print("🎮 Detecting GPU support...")
        
        try:
            # Try to detect NVIDIA GPU
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✅ NVIDIA GPU detected")
                return True
        except FileNotFoundError:
            pass
        
        # Check if CUDA is available through existing PyTorch installation
        try:
            import torch
            if torch.cuda.is_available():
                print("   ✅ CUDA available through existing PyTorch")
                return True
        except ImportError:
            pass
        
        print("   ⚠️ No GPU support detected, using CPU-only packages")
        return False
    
    def install_packages(self, packages: list) -> bool:
        """
        Install specified packages using pip.
        
        Args:
            packages: List of package specifications
            
        Returns:
            True if installation successful
        """
        if not packages:
            print("✅ No packages to install!")
            return True
        
        print(f"📦 Installing {len(packages)} package(s)...")
        
        # Prepare install command
        cmd = [sys.executable, "-m", "pip", "install"]
        
        # Add upgrade flag if force reinstall
        if self.force_reinstall:
            cmd.append("--upgrade")
        
        # Add packages
        cmd.extend(packages)
        
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            # Run installation
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            print("   ✅ Installation completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Installation failed!")
            print(f"   Error: {e.stderr}")
            return False
    
    def install_gpu_packages(self) -> bool:
        """
        Install GPU-specific packages if GPU support is enabled.
        
        Returns:
            True if installation successful
        """
        if not self.gpu_support:
            return True
        
        print("🎮 Installing GPU-optimized packages...")
        
        gpu_packages = [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
        
        return self.install_packages(gpu_packages)
    
    def verify_installation(self) -> bool:
        """
        Verify that installation was successful.
        
        Returns:
            True if verification passed
        """
        print("🔍 Verifying installation...")
        
        # Re-check dependencies
        self.checker = DependencyChecker()
        results = self.checker.check_dependencies()
        
        if results['missing'] or results['outdated']:
            print("   ❌ Some dependencies still missing or outdated")
            return False
        else:
            print("   ✅ All dependencies satisfied")
            return True
    
    def test_core_imports(self) -> bool:
        """
        Test importing core modules to ensure they work.
        
        Returns:
            True if core imports successful
        """
        print("🧪 Testing core module imports...")
        
        test_imports = [
            ("cv2", "OpenCV"),
            ("numpy", "NumPy"),
            ("PIL", "Pillow"),
            ("yaml", "PyYAML"),
            ("requests", "Requests")
        ]
        
        failed_imports = []
        
        for module, name in test_imports:
            try:
                __import__(module)
                print(f"   ✅ {name}")
            except ImportError as e:
                print(f"   ❌ {name}: {e}")
                failed_imports.append(name)
        
        # Test YOLO (may require download)
        try:
            from ultralytics import YOLO
            print("   ✅ Ultralytics YOLO")
        except ImportError as e:
            print(f"   ❌ Ultralytics YOLO: {e}")
            failed_imports.append("Ultralytics")
        
        try:
            import insightface
            print("   ✅ InsightFace")
        except ImportError as e:
            print(f"   ❌ InsightFace: {e}")
            failed_imports.append("InsightFace")
        
        if failed_imports:
            print(f"   ⚠️ Failed imports: {', '.join(failed_imports)}")
            return False
        else:
            print("   ✅ All core modules imported successfully")
            return True
    
    def run_installation(self) -> bool:
        """
        Run the complete installation process.
        
        Returns:
            True if installation successful
        """
        print("🚀 Smart Installation for Intruder Detection System")
        print("=" * 60)
        
        # Check environment
        if not self.check_environment():
            return False
        
        # Detect GPU support
        self.gpu_support = self.detect_gpu_support()
        
        # Check current dependencies
        print("\n🔍 Analyzing current dependencies...")
        results = self.checker.check_dependencies()
        self.checker.print_report()
        
        # Determine what needs to be installed
        packages_to_install = []
        
        if self.force_reinstall:
            # Install everything
            with open("requirements.txt", 'r') as f:
                packages_to_install = [line.strip() for line in f 
                                     if line.strip() and not line.startswith('#')]
        else:
            # Only install missing/outdated
            for pkg in results['missing']:
                if pkg['name'] not in ['tkinter', 'sqlite3']:
                    packages_to_install.append(pkg['original'])
            
            for pkg in results['outdated']:
                packages_to_install.append(pkg['original'])
        
        # Filter out GPU packages if no GPU support
        if not self.gpu_support:
            packages_to_install = [pkg for pkg in packages_to_install 
                                 if 'torch' not in pkg.lower()]
        
        # Install packages
        if packages_to_install:
            print(f"\n📦 Installing {len(packages_to_install)} package(s)...")
            if not self.install_packages(packages_to_install):
                return False
        else:
            print("\n✅ All required packages already installed!")
        
        # Install GPU packages separately if needed
        if self.gpu_support and not self.force_reinstall:
            if not self.install_gpu_packages():
                return False
        
        # Verify installation
        print("\n🔍 Verifying installation...")
        if not self.verify_installation():
            print("⚠️ Installation verification failed")
            return False
        
        # Test core imports
        if not self.test_core_imports():
            print("⚠️ Some core modules failed to import")
            return False
        
        print("\n🎉 Installation completed successfully!")
        print("\n🚀 Next steps:")
        print("   1. Configure your settings: python main.py")
        print("   2. Set up IP cameras in the GUI")
        print("   3. Add Telegram bot token")
        print("   4. Register known people and pets")
        
        return True


def main():
    """Main function for smart installation."""
    parser = argparse.ArgumentParser(description="Smart installer for Intruder Detection System")
    parser.add_argument("--force", action="store_true", 
                       help="Force reinstall all packages")
    parser.add_argument("--gpu", action="store_true", 
                       help="Force enable GPU support")
    parser.add_argument("--no-gpu", action="store_true", 
                       help="Force disable GPU support")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check dependencies, don't install")
    
    args = parser.parse_args()
    
    # Handle GPU support arguments
    gpu_support = None
    if args.gpu:
        gpu_support = True
    elif args.no_gpu:
        gpu_support = False
    
    if args.check_only:
        # Just run dependency check
        checker = DependencyChecker()
        checker.check_dependencies()
        checker.print_report()
        checker.check_system_requirements()
        return 0
    
    # Run smart installation
    installer = SmartInstaller(force_reinstall=args.force, gpu_support=gpu_support)
    
    if installer.run_installation():
        return 0
    else:
        print("\n❌ Installation failed!")
        print("💡 Try running with --force to reinstall all packages")
        return 1


if __name__ == "__main__":
    sys.exit(main())
