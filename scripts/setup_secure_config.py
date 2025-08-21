#!/usr/bin/env python3
"""
Secure Configuration Setup Script

This script helps users set up secure configuration with environment variables.
It guides users through creating a .env file and validates the setup.
"""

import os
import sys
from pathlib import Path
import getpass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.env_config import EnvironmentConfigManager


class SecureConfigSetup:
    """Interactive setup for secure configuration."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.env_file = self.project_root / ".env"
        self.template_file = self.project_root / ".env.template"
        
    def run_setup(self):
        """Run the interactive setup process."""
        print("🔐 Secure Configuration Setup for Intruder Detection System")
        print("=" * 60)
        
        # Check if .env already exists
        if self.env_file.exists():
            print(f"⚠️ .env file already exists at {self.env_file}")
            response = input("Do you want to overwrite it? (y/N): ").lower()
            if response != 'y':
                print("Setup cancelled.")
                return False
        
        # Create .env file from template
        if not self._create_env_from_template():
            return False
        
        # Interactive configuration
        if not self._interactive_config():
            return False
        
        # Validate configuration
        if not self._validate_config():
            return False
        
        print("\n✅ Secure configuration setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Review your .env file and adjust any other settings as needed")
        print("2. Never commit the .env file to version control")
        print("3. Run the application: python main.py")
        
        return True
    
    def _create_env_from_template(self) -> bool:
        """Create .env file from template."""
        try:
            if not self.template_file.exists():
                print(f"❌ Template file not found: {self.template_file}")
                return False
            
            # Copy template to .env
            with open(self.template_file, 'r') as template:
                content = template.read()
            
            with open(self.env_file, 'w') as env_file:
                env_file.write(content)
            
            print(f"✅ Created .env file from template")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    
    def _interactive_config(self) -> bool:
        """Interactive configuration of critical settings."""
        try:
            print("\n🤖 Telegram Bot Configuration")
            print("-" * 30)
            
            # Get Telegram bot token
            bot_token = self._get_telegram_bot_token()
            if not bot_token:
                print("❌ Telegram bot token is required for notifications")
                return False
            
            # Update .env file with bot token
            self._update_env_file("TELEGRAM_BOT_TOKEN", bot_token)
            
            # Optional: Get chat ID
            print("\n💬 Optional: Telegram Chat ID")
            print("If you want to restrict notifications to a specific chat,")
            print("you can set a chat ID. Leave empty to allow all chats.")
            chat_id = input("Enter Telegram Chat ID (optional): ").strip()
            
            if chat_id:
                self._update_env_file("TELEGRAM_CHAT_ID", chat_id)
            
            return True
            
        except Exception as e:
            print(f"❌ Interactive configuration failed: {e}")
            return False
    
    def _get_telegram_bot_token(self) -> str:
        """Get Telegram bot token from user."""
        print("\nTo get a Telegram bot token:")
        print("1. Open Telegram and search for @BotFather")
        print("2. Send /newbot and follow the instructions")
        print("3. Copy the bot token provided by BotFather")
        print()
        
        while True:
            token = getpass.getpass("Enter your Telegram bot token (hidden): ").strip()
            
            if not token:
                print("❌ Bot token cannot be empty")
                continue
            
            # Basic validation
            if not token.count(':') == 1 or len(token) < 40:
                print("❌ Invalid bot token format. Should be like: 123456789:ABC-DEF1234ghIkl-zyx57W2v1u123ew11")
                retry = input("Try again? (y/N): ").lower()
                if retry != 'y':
                    return ""
                continue
            
            return token
    
    def _update_env_file(self, key: str, value: str):
        """Update a specific key in the .env file."""
        try:
            # Read current content
            with open(self.env_file, 'r') as f:
                lines = f.readlines()
            
            # Update or add the key
            key_found = False
            for i, line in enumerate(lines):
                if line.strip().startswith(f"# {key}=") or line.strip().startswith(f"{key}="):
                    lines[i] = f"{key}={value}\n"
                    key_found = True
                    break
            
            if not key_found:
                lines.append(f"\n{key}={value}\n")
            
            # Write back to file
            with open(self.env_file, 'w') as f:
                f.writelines(lines)
            
            print(f"✅ Updated {key} in .env file")
            
        except Exception as e:
            print(f"❌ Failed to update .env file: {e}")
    
    def _validate_config(self) -> bool:
        """Validate the configuration."""
        try:
            print("\n🔍 Validating configuration...")
            
            # Load environment configuration
            env_config = EnvironmentConfigManager(env_file=str(self.env_file))
            
            # Check bot token
            bot_token = env_config.get_secure("telegram.bot_token", env_var="TELEGRAM_BOT_TOKEN")
            if not bot_token:
                print("❌ Telegram bot token not found")
                return False
            
            print("✅ Telegram bot token configured")
            
            # Test loading settings
            from config.settings import Settings
            settings = Settings.load_with_env_support()
            
            if settings.bot_token:
                print("✅ Settings loaded successfully with bot token")
            else:
                print("❌ Settings loaded but bot token is missing")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False


def main():
    """Main function."""
    setup = SecureConfigSetup()
    
    try:
        success = setup.run_setup()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
