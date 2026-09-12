#!/usr/bin/env python3
"""
Dependency Checker for Intruder Detection System

This script checks which dependencies are already installed and which need to be installed.
It provides detailed information about versions and compatibility.
"""

import sys
import subprocess
import importlib
from importlib.metadata import distributions
from packaging.requirements import Requirement, InvalidRequirement
from typing import Dict, List, Tuple, Optional
import re


class DependencyChecker:
    """
    Smart dependency checker that verifies installed packages and versions.
    
    Features:
    - Check if packages are already installed
    - Verify version compatibility
    - Identify missing dependencies
    - Generate optimized installation commands
    """
    
    def __init__(self, requirements_file: str = "requirements.txt"):
        """
        Initialize dependency checker.
        
        Args:
            requirements_file: Path to requirements.txt file
        """
        self.requirements_file = requirements_file
        self.required_packages = {}
        self.installed_packages = {}
        self.missing_packages = []
        self.outdated_packages = []
        self.satisfied_packages = []
        
        self._load_requirements()
        self._get_installed_packages()
    
    def _load_requirements(self):
        """Parse requirements.txt with the same grammar pip uses.

        Each entry becomes a packaging.requirements.Requirement, so bounded
        specs such as ``ultralytics>=8.4.0,<9`` and trailing comments are
        read correctly. The previous parser split on the first ``>=`` and
        kept everything after it as the version.
        """
        try:
            with open(self.requirements_file, 'r', encoding='utf-8') as f:
                for raw in f:
                    line = raw.split('#', 1)[0].strip()
                    if not line or line.startswith('-'):
                        continue
                    try:
                        req = Requirement(line)
                    except InvalidRequirement as e:
                        print(f"⚠️ Skipping unparseable requirement: {line} ({e})")
                        continue
                    self.required_packages[req.name] = {
                        'specifier': req.specifier,
                        'version': str(req.specifier) or 'any',
                        'original': line,
                    }
        except FileNotFoundError:
            print(f"❌ Requirements file not found: {self.requirements_file}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading requirements file: {e}")
            sys.exit(1)

    def _get_installed_packages(self):
        """Get list of currently installed packages."""
        try:
            for dist in distributions():
                name = dist.metadata['Name']
                if name:
                    key = name.lower().replace('_', '-')
                    self.installed_packages[key] = dist.version
        except Exception as e:
            print(f"⚠️ Warning: Could not get installed packages list: {e}")

    def check_dependencies(self) -> Dict[str, List]:
        """
        Check all dependencies and categorize them.
        
        Returns:
            Dictionary with categorized packages
        """
        self.missing_packages = []
        self.outdated_packages = []
        self.satisfied_packages = []
        
        for package_name, requirements in self.required_packages.items():
            package_lower = package_name.lower().replace('_', '-')
            
            # Handle special cases for built-in modules
            if package_name in ['tkinter', 'sqlite3']:
                try:
                    importlib.import_module(package_name)
                    self.satisfied_packages.append({
                        'name': package_name,
                        'installed_version': 'built-in',
                        'required_version': requirements['version'],
                        'status': 'Built-in module'
                    })
                    continue
                except ImportError:
                    self.missing_packages.append({
                        'name': package_name,
                        'required_version': requirements['version'],
                        'original': requirements['original'],
                        'status': 'Built-in module not available'
                    })
                    continue
            
            # Check if package is installed
            if package_lower in self.installed_packages:
                installed_version = self.installed_packages[package_lower]
                
                # Check version compatibility
                if len(requirements['specifier']) > 0:
                    spec = requirements['specifier']
                    if spec.contains(installed_version, prereleases=True):
                        self.satisfied_packages.append({
                            'name': package_name,
                            'installed_version': installed_version,
                            'required_version': requirements['version'],
                            'status': 'Version satisfied'
                        })
                    else:
                        self.outdated_packages.append({
                            'name': package_name,
                            'installed_version': installed_version,
                            'required_version': requirements['version'],
                            'original': requirements['original'],
                            'status': 'Version outdated'
                        })
                else:
                    # No version requirement, any version is fine
                    self.satisfied_packages.append({
                        'name': package_name,
                        'installed_version': installed_version,
                        'required_version': 'any',
                        'status': 'Installed'
                    })
            else:
                # Package not installed
                self.missing_packages.append({
                    'name': package_name,
                    'required_version': requirements['version'],
                    'original': requirements['original'],
                    'status': 'Not installed'
                })
        
        return {
            'missing': self.missing_packages,
            'outdated': self.outdated_packages,
            'satisfied': self.satisfied_packages
        }
    
    def print_report(self):
        """Print a detailed dependency report."""
        print("🔍 Dependency Check Report")
        print("=" * 50)
        
        # Satisfied packages
        if self.satisfied_packages:
            print(f"\n✅ Satisfied Dependencies ({len(self.satisfied_packages)}):")
            for pkg in self.satisfied_packages:
                if pkg['installed_version'] == 'built-in':
                    print(f"   📦 {pkg['name']} - {pkg['status']}")
                else:
                    print(f"   📦 {pkg['name']} {pkg['installed_version']} - {pkg['status']}")
        
        # Outdated packages
        if self.outdated_packages:
            print(f"\n⚠️ Outdated Dependencies ({len(self.outdated_packages)}):")
            for pkg in self.outdated_packages:
                print(
                    f"   📦 {pkg['name']} {pkg['installed_version']} -> "
                    f"{pkg['required_version']} (needs update)"
                )
        
        # Missing packages
        if self.missing_packages:
            print(f"\n❌ Missing Dependencies ({len(self.missing_packages)}):")
            for pkg in self.missing_packages:
                if pkg['required_version']:
                    print(f"   📦 {pkg['name']} {pkg['required_version']} - {pkg['status']}")
                else:
                    print(f"   📦 {pkg['name']} - {pkg['status']}")
        
        # Summary
        total = len(self.satisfied_packages) + len(self.outdated_packages) + len(self.missing_packages)
        print(f"\n📊 Summary:")
        print(f"   Total dependencies: {total}")
        print(f"   ✅ Satisfied: {len(self.satisfied_packages)}")
        print(f"   ⚠️ Outdated: {len(self.outdated_packages)}")
        print(f"   ❌ Missing: {len(self.missing_packages)}")
    
    def generate_install_command(self) -> str:
        """
        Generate optimized pip install command for missing/outdated packages.
        
        Returns:
            Pip install command string
        """
        packages_to_install = []
        
        # Add missing packages
        for pkg in self.missing_packages:
            if pkg['name'] not in ['tkinter', 'sqlite3']:  # Skip built-in modules
                packages_to_install.append(pkg['original'])
        
        # Add outdated packages
        for pkg in self.outdated_packages:
            packages_to_install.append(pkg['original'])
        
        if packages_to_install:
            return f"pip install {' '.join(packages_to_install)}"
        else:
            return "# All dependencies are already satisfied!"
    
    def check_system_requirements(self):
        """Check system-level requirements."""
        print("\n🖥️ System Requirements Check:")
        
        # Python version
        python_version = sys.version_info
        print(f"   🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version < (3, 8):
            print("   ❌ Python 3.8+ required!")
        else:
            print("   ✅ Python version OK")
        
        # Check for CUDA (optional)
        try:
            import torch
            if torch.cuda.is_available():
                print(f"   🎮 CUDA: Available ({torch.cuda.device_count()} GPU(s))")
            else:
                print("   ⚠️ CUDA: Not available (CPU-only mode)")
        except ImportError:
            print("   ⚠️ CUDA: Cannot check (PyTorch not installed)")
        
        # Check available disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            free_gb = free // (1024**3)
            print(f"   💾 Disk space: {free_gb}GB free")
            
            if free_gb < 5:
                print("   ⚠️ Low disk space! At least 10GB recommended")
            else:
                print("   ✅ Disk space OK")
        except Exception:
            print("   ⚠️ Could not check disk space")


def main():
    """Main function to run dependency check."""
    print("🚀 Intruder Detection System - Dependency Checker")
    print("=" * 60)
    
    # Check if requirements.txt exists
    import os
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Create checker and run analysis
    checker = DependencyChecker()
    checker.check_dependencies()
    
    # Print detailed report
    checker.print_report()
    
    # Check system requirements
    checker.check_system_requirements()
    
    # Generate install command
    install_cmd = checker.generate_install_command()
    print(f"\n💡 Installation Command:")
    print(f"   {install_cmd}")
    
    # Provide recommendations
    if checker.missing_packages or checker.outdated_packages:
        print(f"\n🔧 Next Steps:")
        print(f"   1. Run the installation command above")
        print(f"   2. Verify installation: python scripts/check_dependencies.py")
        print(f"   3. Start the application: python main.py")
    else:
        print(f"\n🎉 All dependencies satisfied! You can run:")
        print(f"   python main.py")
    
    return len(checker.missing_packages) + len(checker.outdated_packages)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
