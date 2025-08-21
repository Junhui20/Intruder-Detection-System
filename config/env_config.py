"""
Environment Configuration Manager

This module provides secure configuration loading with environment variable support.
Sensitive data like API keys and tokens are loaded from environment variables first,
with fallback to configuration files.
"""

import os
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
import yaml

# Configure logging
logger = logging.getLogger(__name__)


class EnvironmentConfigManager:
    """
    Secure configuration manager with environment variable support.
    
    Features:
    - Environment variables take precedence over config files
    - Secure handling of sensitive data (API keys, tokens)
    - Automatic type conversion
    - Validation and error handling
    - Support for .env files
    """
    
    def __init__(self, config_file: str = "config.yaml", env_file: Optional[str] = ".env"):
        """
        Initialize the environment configuration manager.
        
        Args:
            config_file: Path to the main configuration file
            env_file: Path to the .env file (optional)
        """
        self.config_file = Path(config_file)
        self.env_file = Path(env_file) if env_file else None
        self.config_data = {}
        
        # Load .env file if it exists
        if self.env_file and self.env_file.exists():
            self._load_env_file()
        
        # Load main configuration
        self._load_config_file()
        
        logger.info("Environment configuration manager initialized")
    
    def _load_env_file(self):
        """Load environment variables from .env file."""
        try:
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        os.environ[key] = value
            
            logger.info(f"Loaded environment variables from {self.env_file}")
            
        except Exception as e:
            logger.warning(f"Could not load .env file {self.env_file}: {e}")
    
    def _load_config_file(self):
        """Load configuration from YAML file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_file}")
            else:
                logger.warning(f"Configuration file {self.config_file} not found")
                self.config_data = {}
                
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            self.config_data = {}
    
    def get(self, key: str, default: Any = None, env_var: Optional[str] = None, 
            config_path: Optional[str] = None, data_type: type = str) -> Any:
        """
        Get configuration value with environment variable precedence.
        
        Args:
            key: Configuration key name
            default: Default value if not found
            env_var: Environment variable name (defaults to key.upper())
            config_path: Dot-separated path in config file (defaults to key)
            data_type: Expected data type for conversion
            
        Returns:
            Configuration value with proper type conversion
        """
        # 1. Check environment variable first
        env_key = env_var or key.upper()
        env_value = os.getenv(env_key)
        
        if env_value is not None:
            try:
                return self._convert_type(env_value, data_type)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to convert env var {env_key}={env_value} to {data_type}: {e}")
        
        # 2. Check configuration file
        config_key = config_path or key
        config_value = self._get_nested_config(config_key)
        
        if config_value is not None:
            try:
                return self._convert_type(config_value, data_type)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to convert config {config_key}={config_value} to {data_type}: {e}")
        
        # 3. Return default
        return default
    
    def get_secure(self, key: str, env_var: Optional[str] = None, 
                   config_path: Optional[str] = None, required: bool = True) -> Optional[str]:
        """
        Get secure configuration value (API keys, tokens, passwords).
        
        This method prioritizes environment variables and logs warnings for
        sensitive data found in configuration files.
        
        Args:
            key: Configuration key name
            env_var: Environment variable name (defaults to key.upper())
            config_path: Dot-separated path in config file (defaults to key)
            required: Whether the value is required
            
        Returns:
            Secure configuration value or None
        """
        env_key = env_var or key.upper()
        env_value = os.getenv(env_key)
        
        if env_value:
            logger.info(f"Using secure value from environment variable {env_key}")
            return env_value
        
        # Check config file but warn about security
        config_key = config_path or key
        config_value = self._get_nested_config(config_key)
        
        if config_value:
            logger.warning(f"⚠️ SECURITY WARNING: Sensitive data '{config_key}' found in config file. "
                          f"Consider using environment variable {env_key} instead.")
            return str(config_value)
        
        if required:
            logger.error(f"Required secure configuration '{key}' not found in environment or config file")
            return None
        
        return None
    
    def _get_nested_config(self, key_path: str) -> Any:
        """Get nested configuration value using dot notation."""
        try:
            keys = key_path.split('.')
            value = self.config_data
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            
            return value
            
        except Exception:
            return None
    
    def _convert_type(self, value: Any, data_type: type) -> Any:
        """Convert value to specified type."""
        if data_type == bool:
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        elif data_type == int:
            return int(value)
        elif data_type == float:
            return float(value)
        elif data_type == str:
            return str(value)
        else:
            return value
    
    def create_env_template(self, output_file: str = ".env.template"):
        """
        Create a template .env file with all configurable environment variables.
        
        Args:
            output_file: Path to output template file
        """
        template_content = """# Intruder Detection System Environment Configuration
# Copy this file to .env and fill in your values

# =============================================================================
# SECURITY SETTINGS (REQUIRED)
# =============================================================================

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
# TELEGRAM_CHAT_ID=your_chat_id_here  # Optional: specific chat ID

# =============================================================================
# OPTIONAL SETTINGS
# =============================================================================

# Database Configuration
# DATABASE_PATH=detection_system.db

# Performance Settings
# MAX_CPU_USAGE=80.0
# MAX_MEMORY_USAGE=4096
# ENABLE_GPU=true

# Detection Thresholds
# HUMAN_CONFIDENCE_THRESHOLD=0.6
# ANIMAL_CONFIDENCE_THRESHOLD=0.6
# YOLO_CONFIDENCE=0.5

# Camera Settings
# CAMERA_CONNECTION_TIMEOUT=2
# CAMERA_BUFFER_SIZE=1

# Notification Settings
# NOTIFICATION_COOLDOWN=20
# SEND_PHOTOS=true
# PHOTO_QUALITY=85

# =============================================================================
# DEVELOPMENT SETTINGS
# =============================================================================

# Logging Level (DEBUG, INFO, WARNING, ERROR)
# LOG_LEVEL=INFO

# Enable Debug Mode
# DEBUG_MODE=false
"""
        
        try:
            with open(output_file, 'w') as f:
                f.write(template_content)
            logger.info(f"Created environment template: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to create environment template: {e}")


# Global instance for easy access
env_config = EnvironmentConfigManager()


def get_config(key: str, default: Any = None, **kwargs) -> Any:
    """Convenience function to get configuration value."""
    return env_config.get(key, default, **kwargs)


def get_secure_config(key: str, **kwargs) -> Optional[str]:
    """Convenience function to get secure configuration value."""
    return env_config.get_secure(key, **kwargs)
