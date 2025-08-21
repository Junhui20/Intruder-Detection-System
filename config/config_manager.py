#!/usr/bin/env python3
"""
Advanced Configuration Management System

This module provides runtime configuration updates, validation,
backup/restore, and migration capabilities.
"""

import os
import json
import yaml
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ConfigValidationRule:
    """Configuration validation rule."""
    path: str
    rule_type: str  # 'range', 'choices', 'type', 'required'
    params: Dict[str, Any]
    error_message: str

class ConfigManager:
    """Advanced configuration management system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to main configuration file
        """
        self.config_path = Path(config_path)
        self.backup_dir = Path("config_backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        self.config = {}
        self.validation_rules = []
        self.change_callbacks = {}
        self.config_history = []
        
        # Load configuration
        self.load_config()
        self._setup_validation_rules()
        
        logger.info("Configuration manager initialized")
    
    def load_config(self) -> bool:
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    if self.config_path.suffix.lower() == '.json':
                        self.config = json.load(f)
                    else:
                        self.config = yaml.safe_load(f) or {}
                
                logger.info(f"Configuration loaded from {self.config_path}")
                return True
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                self.config = self._get_default_config()
                self.save_config()
                return True
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = self._get_default_config()
            return False
    
    def save_config(self) -> bool:
        """Save configuration to file."""
        try:
            # Create backup before saving
            self._create_backup()
            
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix.lower() == '.json':
                    json.dump(self.config, f, indent=2)
                else:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'video.frame_width')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.error(f"Error getting config value {key_path}: {e}")
            return default

    def get_config(self) -> Dict[str, Any]:
        """
        Get the entire configuration dictionary.

        This method provides backward compatibility for code that expects
        a get_config() method returning the full config.

        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()
    
    def set(self, key_path: str, value: Any, validate: bool = True) -> bool:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'video.frame_width')
            value: Value to set
            validate: Whether to validate the value
            
        Returns:
            True if successful
        """
        try:
            # Validate if requested
            if validate and not self._validate_value(key_path, value):
                return False
            
            # Store old value for rollback
            old_value = self.get(key_path)
            
            # Set the value
            keys = key_path.split('.')
            config_ref = self.config
            
            # Navigate to parent
            for key in keys[:-1]:
                if key not in config_ref:
                    config_ref[key] = {}
                config_ref = config_ref[key]
            
            # Set final value
            config_ref[keys[-1]] = value
            
            # Log change
            self._log_config_change(key_path, old_value, value)
            
            # Trigger callbacks
            self._trigger_callbacks(key_path, value, old_value)
            
            logger.info(f"Configuration updated: {key_path} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting config value {key_path}: {e}")
            return False
    
    def update_runtime(self, updates: Dict[str, Any], validate: bool = True) -> bool:
        """
        Update multiple configuration values at runtime.
        
        Args:
            updates: Dictionary of key_path -> value updates
            validate: Whether to validate all values
            
        Returns:
            True if all updates successful
        """
        try:
            # Validate all updates first
            if validate:
                for key_path, value in updates.items():
                    if not self._validate_value(key_path, value):
                        logger.error(f"Validation failed for {key_path}")
                        return False
            
            # Apply all updates
            success_count = 0
            for key_path, value in updates.items():
                if self.set(key_path, value, validate=False):
                    success_count += 1
            
            # Save if any updates succeeded
            if success_count > 0:
                self.save_config()
                logger.info(f"Runtime configuration updated: {success_count} changes")
            
            return success_count == len(updates)
            
        except Exception as e:
            logger.error(f"Error updating runtime configuration: {e}")
            return False
    
    def validate_config(self) -> List[str]:
        """
        Validate entire configuration against rules.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        for rule in self.validation_rules:
            try:
                value = self.get(rule.path)
                if not self._validate_value_with_rule(value, rule):
                    errors.append(f"{rule.path}: {rule.error_message}")
            except Exception as e:
                errors.append(f"{rule.path}: Validation error - {e}")
        
        return errors
    
    def _validate_value(self, key_path: str, value: Any) -> bool:
        """Validate a single value against rules."""
        for rule in self.validation_rules:
            if rule.path == key_path:
                return self._validate_value_with_rule(value, rule)
        return True  # No rule found, assume valid
    
    def _validate_value_with_rule(self, value: Any, rule: ConfigValidationRule) -> bool:
        """Validate value against a specific rule."""
        try:
            if rule.rule_type == 'required':
                return value is not None
            
            elif rule.rule_type == 'type':
                expected_type = rule.params['type']
                return isinstance(value, expected_type)
            
            elif rule.rule_type == 'range':
                min_val = rule.params.get('min')
                max_val = rule.params.get('max')
                if min_val is not None and value < min_val:
                    return False
                if max_val is not None and value > max_val:
                    return False
                return True
            
            elif rule.rule_type == 'choices':
                return value in rule.params['choices']
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating rule {rule.path}: {e}")
            return False
    
    def _setup_validation_rules(self):
        """Set up configuration validation rules."""
        self.validation_rules = [
            ConfigValidationRule(
                path="video.frame_width",
                rule_type="range",
                params={"min": 320, "max": 1920},
                error_message="Frame width must be between 320 and 1920"
            ),
            ConfigValidationRule(
                path="video.frame_height",
                rule_type="range",
                params={"min": 240, "max": 1080},
                error_message="Frame height must be between 240 and 1080"
            ),
            ConfigValidationRule(
                path="video.target_fps",
                rule_type="range",
                params={"min": 1, "max": 60},
                error_message="Target FPS must be between 1 and 60"
            ),
            ConfigValidationRule(
                path="detection.yolo_confidence",
                rule_type="range",
                params={"min": 0.1, "max": 1.0},
                error_message="YOLO confidence must be between 0.1 and 1.0"
            ),
            ConfigValidationRule(
                path="gui.theme",
                rule_type="choices",
                params={"choices": ["light", "dark", "auto"]},
                error_message="Theme must be 'light', 'dark', or 'auto'"
            )
        ]
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
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
    
    def _create_backup(self):
        """Create configuration backup."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"config_backup_{timestamp}.yaml"
            
            shutil.copy2(self.config_path, backup_path)
            
            # Keep only last 10 backups
            backups = sorted(self.backup_dir.glob("config_backup_*.yaml"))
            if len(backups) > 10:
                for old_backup in backups[:-10]:
                    old_backup.unlink()
            
            logger.debug(f"Configuration backup created: {backup_path}")
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
    
    def _log_config_change(self, key_path: str, old_value: Any, new_value: Any):
        """Log configuration change."""
        change_record = {
            'timestamp': datetime.now().isoformat(),
            'key_path': key_path,
            'old_value': old_value,
            'new_value': new_value
        }
        
        self.config_history.append(change_record)
        
        # Keep only last 100 changes
        if len(self.config_history) > 100:
            self.config_history = self.config_history[-100:]
    
    def _trigger_callbacks(self, key_path: str, new_value: Any, old_value: Any):
        """Trigger registered callbacks for configuration changes."""
        for callback_key, callback_func in self.change_callbacks.items():
            try:
                if key_path.startswith(callback_key):
                    callback_func(key_path, new_value, old_value)
            except Exception as e:
                logger.error(f"Error in config callback {callback_key}: {e}")
    
    def register_change_callback(self, key_prefix: str, callback_func):
        """Register a callback for configuration changes."""
        self.change_callbacks[key_prefix] = callback_func
        logger.info(f"Registered config callback for {key_prefix}")
    
    def get_change_history(self, limit: int = 50) -> List[Dict]:
        """Get recent configuration changes."""
        return self.config_history[-limit:] if self.config_history else []
    
    def restore_backup(self, backup_filename: str) -> bool:
        """Restore configuration from backup."""
        try:
            backup_path = self.backup_dir / backup_filename
            if backup_path.exists():
                shutil.copy2(backup_path, self.config_path)
                self.load_config()
                logger.info(f"Configuration restored from {backup_filename}")
                return True
            else:
                logger.error(f"Backup file not found: {backup_filename}")
                return False
                
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False
    
    def list_backups(self) -> List[str]:
        """List available configuration backups."""
        try:
            backups = sorted(self.backup_dir.glob("config_backup_*.yaml"))
            return [backup.name for backup in backups]
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []
