#!/usr/bin/env python3
"""
Security Tests for Intruder Detection System

This module contains security-focused tests to ensure sensitive data
is properly handled and not exposed in configuration files or logs.
"""

import sys
import os
import unittest
import tempfile
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.env_config import EnvironmentConfigManager, get_secure_config
from config.settings import Settings


class TestConfigurationSecurity(unittest.TestCase):
    """Test configuration security measures."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="security_test_"))
        self.test_config_file = self.test_dir / "test_config.yaml"
        self.test_env_file = self.test_dir / ".env"
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_no_sensitive_data_in_config_file(self):
        """Test that sensitive data is not stored in config files."""
        # Create a test config file with potentially sensitive data
        config_data = {
            'telegram': {
                'bot_token': 'should_not_be_here',
                'cooldown': 20.0
            },
            'database': {
                'path': 'test.db'
            }
        }
        
        with open(self.test_config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load settings
        settings = Settings.load_with_env_support(str(self.test_config_file))
        
        # The bot token should be overridden by environment variable loading
        # Since no env var is set, it should be empty or the default
        self.assertNotEqual(settings.bot_token, 'should_not_be_here')
    
    def test_environment_variable_precedence(self):
        """Test that environment variables take precedence over config files."""
        # Set environment variable
        test_token = "test_env_token_12345"
        os.environ['TELEGRAM_BOT_TOKEN'] = test_token
        
        try:
            # Create config file with different token
            config_data = {
                'telegram': {
                    'bot_token': 'config_file_token',
                    'cooldown': 20.0
                }
            }
            
            with open(self.test_config_file, 'w') as f:
                yaml.dump(config_data, f)
            
            # Load settings
            settings = Settings.load_with_env_support(str(self.test_config_file))
            
            # Environment variable should take precedence
            self.assertEqual(settings.bot_token, test_token)
            
        finally:
            # Clean up environment variable
            if 'TELEGRAM_BOT_TOKEN' in os.environ:
                del os.environ['TELEGRAM_BOT_TOKEN']
    
    def test_secure_config_loading(self):
        """Test secure configuration loading functions."""
        # Test with no environment variable set
        secure_value = get_secure_config("telegram.bot_token", required=False)
        
        # Should return None or empty string when not set
        self.assertIn(secure_value, [None, ""])
        
        # Test with environment variable set
        test_token = "secure_test_token_67890"
        os.environ['TELEGRAM_BOT_TOKEN'] = test_token
        
        try:
            secure_value = get_secure_config("telegram.bot_token", env_var="TELEGRAM_BOT_TOKEN")
            self.assertEqual(secure_value, test_token)
            
        finally:
            # Clean up environment variable
            if 'TELEGRAM_BOT_TOKEN' in os.environ:
                del os.environ['TELEGRAM_BOT_TOKEN']
    
    def test_env_file_loading(self):
        """Test .env file loading functionality."""
        # Create test .env file
        env_content = """# Test environment file
TELEGRAM_BOT_TOKEN=env_file_token_123
TEST_SETTING=test_value
# Comment line
ANOTHER_SETTING="quoted_value"
"""
        
        with open(self.test_env_file, 'w') as f:
            f.write(env_content)
        
        # Load environment configuration
        env_config = EnvironmentConfigManager(env_file=str(self.test_env_file))
        
        # Test that environment variables were loaded
        self.assertEqual(os.environ.get('TELEGRAM_BOT_TOKEN'), 'env_file_token_123')
        self.assertEqual(os.environ.get('TEST_SETTING'), 'test_value')
        self.assertEqual(os.environ.get('ANOTHER_SETTING'), 'quoted_value')
        
        # Clean up environment variables
        for key in ['TELEGRAM_BOT_TOKEN', 'TEST_SETTING', 'ANOTHER_SETTING']:
            if key in os.environ:
                del os.environ[key]
    
    def test_configuration_validation(self):
        """Test configuration validation and error handling."""
        # Test with invalid configuration
        env_config = EnvironmentConfigManager()
        
        # Test type conversion
        os.environ['TEST_INT'] = '42'
        os.environ['TEST_FLOAT'] = '3.14'
        os.environ['TEST_BOOL_TRUE'] = 'true'
        os.environ['TEST_BOOL_FALSE'] = 'false'
        
        try:
            # Test type conversions
            int_val = env_config.get('test_int', env_var='TEST_INT', data_type=int)
            self.assertEqual(int_val, 42)
            self.assertIsInstance(int_val, int)
            
            float_val = env_config.get('test_float', env_var='TEST_FLOAT', data_type=float)
            self.assertEqual(float_val, 3.14)
            self.assertIsInstance(float_val, float)
            
            bool_val_true = env_config.get('test_bool_true', env_var='TEST_BOOL_TRUE', data_type=bool)
            self.assertTrue(bool_val_true)
            self.assertIsInstance(bool_val_true, bool)
            
            bool_val_false = env_config.get('test_bool_false', env_var='TEST_BOOL_FALSE', data_type=bool)
            self.assertFalse(bool_val_false)
            self.assertIsInstance(bool_val_false, bool)
            
        finally:
            # Clean up environment variables
            for key in ['TEST_INT', 'TEST_FLOAT', 'TEST_BOOL_TRUE', 'TEST_BOOL_FALSE']:
                if key in os.environ:
                    del os.environ[key]


class TestDataProtection(unittest.TestCase):
    """Test data protection measures."""
    
    def test_no_hardcoded_secrets(self):
        """Test that no secrets are hardcoded in the main config file."""
        config_file = project_root / "config.yaml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_content = f.read()
            
            # Check for patterns that might indicate hardcoded secrets
            suspicious_patterns = [
                r'\d{10}:[A-Za-z0-9_-]{35}',  # Telegram bot token pattern
                r'sk-[A-Za-z0-9]{48}',        # OpenAI API key pattern
                r'xoxb-[0-9]{12}-[0-9]{12}-[A-Za-z0-9]{24}',  # Slack bot token pattern
            ]
            
            import re
            for pattern in suspicious_patterns:
                matches = re.findall(pattern, config_content)
                self.assertEqual(len(matches), 0, 
                               f"Found potential hardcoded secret matching pattern: {pattern}")
    
    def test_gitignore_includes_sensitive_files(self):
        """Test that .gitignore includes sensitive files."""
        gitignore_file = project_root / ".gitignore"
        
        if gitignore_file.exists():
            with open(gitignore_file, 'r') as f:
                gitignore_content = f.read()
            
            # Check for important entries
            required_entries = ['.env', '*.db', 'logs/']
            
            for entry in required_entries:
                self.assertIn(entry, gitignore_content, 
                            f"Missing important .gitignore entry: {entry}")


def run_security_tests(verbose: bool = True) -> bool:
    """
    Run all security tests.
    
    Args:
        verbose: Whether to print detailed output
        
    Returns:
        True if all tests passed
    """
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestConfigurationSecurity))
    suite.addTest(unittest.makeSuite(TestDataProtection))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🔐 Running Security Tests for Intruder Detection System")
    print("=" * 60)
    
    success = run_security_tests()
    
    if success:
        print("\n✅ All security tests passed!")
    else:
        print("\n❌ Some security tests failed!")
        sys.exit(1)
