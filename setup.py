#!/usr/bin/env python3
"""
Quick Setup Script for Intruder Detection System

This script provides a simple way to check and install dependencies.
Run this before using the main application.
"""

import sys
import os
import subprocess

def main():
    """Main setup function."""
    print("🚀 Intruder Detection System 2025 - Quick Setup")
    print("=" * 55)
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("❌ Error: requirements.txt not found!")
        print("Please run this script from the project root directory.")
        return 1
    
    # Check if scripts directory exists
    if not os.path.exists("scripts/check_dependencies.py"):
        print("❌ Error: Setup scripts not found!")
        print("Please ensure the scripts/ directory is present.")
        return 1
    
    print("🔍 Checking dependencies...")
    print("This will show you what's already installed and what needs to be installed.\n")
    
    try:
        # Run dependency checker
        result = subprocess.run([
            sys.executable, "scripts/check_dependencies.py"
        ], check=False)
        
        if result.returncode == 0:
            print("\n🎉 All dependencies are satisfied!")
            print("You can now run: python main.py")
        else:
            print(f"\n📦 Found {result.returncode} missing or outdated dependencies.")
            print("\n💡 Installation options:")
            print("   1. Smart install (recommended): python scripts/install.py")
            print("   2. Force install all: python scripts/install.py --force")
            print("   3. Manual install: pip install -r requirements.txt")
            print("   4. Check only: python scripts/check_dependencies.py")
            
            choice = input("\nWould you like to run smart installation now? (Y/n): ").lower().strip()
            if choice in ['', 'y', 'yes']:
                print("\n🚀 Running smart installation...")
                install_result = subprocess.run([
                    sys.executable, "scripts/install.py"
                ], check=False)
                
                if install_result.returncode == 0:
                    print("\n✅ Installation completed successfully!")
                    print("You can now run: python main.py")
                else:
                    print("\n❌ Installation failed. Please check the error messages above.")
                    return 1
            else:
                print("\n💡 You can install dependencies later using:")
                print("   python scripts/install.py")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error running setup: {e}")
        print("\n💡 Fallback: Try manual installation:")
        print("   pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
