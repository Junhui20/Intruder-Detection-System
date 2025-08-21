#!/usr/bin/env python3
"""
Environment Setup Script for Intruder Detection System

This script sets up the environment for the intruder detection system,
including directory creation, configuration files, and initial setup.
"""

import os
import sys
import logging
import shutil
from pathlib import Path
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from database.database_manager import DatabaseManager
from config.settings import Settings

logger = logging.getLogger(__name__)

class EnvironmentSetup:
    """Environment setup utility."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.setup_results = []
    
    def setup_directories(self):
        """Create necessary directories."""
        print("📁 Setting up directories...")
        
        directories = [
            "logs",
            "data/faces",
            "data/animals", 
            "data/detections",
            "data/backups",
            "temp",
            "models"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                self.setup_results.append(f"✅ Created directory: {directory}")
            except Exception as e:
                self.setup_results.append(f"❌ Failed to create directory {directory}: {e}")
    
    def setup_configuration(self):
        """Set up configuration files."""
        print("⚙️ Setting up configuration...")
        
        try:
            # Create default config.yaml if it doesn't exist
            config_path = self.project_root / "config.yaml"
            if not config_path.exists():
                default_config = {
                    'video': {
                        'frame_width': 640,
                        'frame_height': 480,
                        'target_fps': 30,
                        'process_every_n_frames': 1
                    },
                    'detection': {
                        'yolo_confidence': 0.5,
                        'yolo_iou_threshold': 0.4,
                        'human_confidence_threshold': 0.6,
                        'animal_confidence_threshold': 0.6
                    },
                    'database': {
                        'path': 'detection_system.db'
                    },
                    'gui': {
                        'theme': 'dark',
                        'window_size': '1280x720',
                        'auto_refresh_interval': 100
                    },
                    'performance': {
                        'enable_gpu': True,
                        'enable_performance_monitoring': True,
                        'max_cpu_usage': 80.0,
                        'max_memory_usage': 4096
                    }
                }
                
                with open(config_path, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False, indent=2)
                
                self.setup_results.append("✅ Created default config.yaml")
            else:
                self.setup_results.append("ℹ️ config.yaml already exists")
                
        except Exception as e:
            self.setup_results.append(f"❌ Failed to setup configuration: {e}")
    
    def setup_database(self):
        """Initialize the SQLite database."""
        print("🗄️ Setting up database...")
        
        try:
            db_manager = DatabaseManager()
            self.setup_results.append("✅ Database initialized successfully")
            
            # Get database stats
            stats = db_manager.get_database_stats()
            for table, count in stats.items():
                self.setup_results.append(f"ℹ️ {table}: {count} records")
                
        except Exception as e:
            self.setup_results.append(f"❌ Failed to setup database: {e}")
    
    def setup_models(self):
        """Set up AI models."""
        print("🤖 Setting up AI models...")
        
        try:
            models_dir = self.project_root / "models"
            yolo_model = self.project_root / "yolo11n.pt"
            
            if yolo_model.exists():
                self.setup_results.append("✅ YOLO11n model found")
            else:
                self.setup_results.append("⚠️ YOLO11n model not found - will be downloaded on first run")
            
            # Create models directory
            models_dir.mkdir(exist_ok=True)
            self.setup_results.append("✅ Models directory ready")
            
        except Exception as e:
            self.setup_results.append(f"❌ Failed to setup models: {e}")
    
    def setup_logging(self):
        """Set up logging configuration."""
        print("📝 Setting up logging...")
        
        try:
            logs_dir = self.project_root / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            # Create log files
            log_files = [
                "detection_system.log",
                "performance.log",
                "errors.log"
            ]
            
            for log_file in log_files:
                log_path = logs_dir / log_file
                if not log_path.exists():
                    log_path.touch()
                    self.setup_results.append(f"✅ Created log file: {log_file}")
                else:
                    self.setup_results.append(f"ℹ️ Log file exists: {log_file}")
            
        except Exception as e:
            self.setup_results.append(f"❌ Failed to setup logging: {e}")
    
    def verify_dependencies(self):
        """Verify that required dependencies are installed."""
        print("🔍 Verifying dependencies...")
        
        required_packages = [
            'opencv-python',
            'ultralytics',
            'face-recognition',
            'pillow',
            'numpy',
            'psutil',
            'pyyaml'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                self.setup_results.append(f"✅ {package} installed")
            except ImportError:
                missing_packages.append(package)
                self.setup_results.append(f"❌ {package} missing")
        
        if missing_packages:
            self.setup_results.append(f"⚠️ Missing packages: {', '.join(missing_packages)}")
            self.setup_results.append("💡 Run: pip install -r requirements.txt")
    
    def run_setup(self):
        """Run the complete environment setup."""
        print("🚀 Starting Environment Setup for Intruder Detection System")
        print("=" * 60)
        
        # Run all setup steps
        self.setup_directories()
        self.setup_configuration()
        self.setup_database()
        self.setup_models()
        self.setup_logging()
        self.verify_dependencies()
        
        # Print results
        print("\n" + "=" * 60)
        print("📋 SETUP RESULTS")
        print("=" * 60)
        
        for result in self.setup_results:
            print(result)
        
        print("\n" + "=" * 60)
        print("✅ Environment setup completed!")
        print("💡 Next steps:")
        print("1. Review config.yaml and adjust settings as needed")
        print("2. Add known faces to data/faces/ directory")
        print("3. Add known animals to data/animals/ directory")
        print("4. Run: python main.py")

def main():
    """Main setup function."""
    setup = EnvironmentSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()
