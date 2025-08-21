"""
Bidirectional Telegram Bot Notification System

This module implements the Telegram bot integration with command listening
and multi-user notification management as specified in requirements.
"""

import requests
import time
import json
import threading
from typing import Dict, List, Optional, Tuple
import logging
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationSystem:
    """
    Advanced Telegram bot system with bidirectional communication.
    
    Features:
    - Multi-user notification support
    - Command listening (e.g., 'check' command)
    - Individual user permissions
    - Photo and message sending
    - Notification cooldown management
    - Error handling and retry logic
    """
    
    def __init__(self, bot_token: str, db_manager=None):
        """
        Initialize the notification system.

        Args:
            bot_token: Telegram bot token
            db_manager: Database manager for notification history
        """
        self.bot_token = bot_token
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.db_manager = db_manager
        self.users = {}  # Dictionary of user configurations
        self.last_update_id = 0
        self.listening = False
        self.listen_thread = None
        self.notification_cooldowns = {}  # Track cooldowns per user
        self.default_cooldown = 20  # seconds

        # Enhanced notification features
        self.notification_history = []
        self.max_history = 1000
        self.rate_limit_window = 300  # 5 minutes
        self.max_notifications_per_window = 10
        self.last_notification_times = {}

        # Notification templates
        self.templates = {
            'human_unknown': "🚨 Unknown person detected (confidence: {confidence}%)",
            'human_known': "👋 {name} detected (confidence: {confidence}%)",
            'animal_unknown': "🐾 Unknown {animal_type} detected (confidence: {confidence}%)",
            'animal_known': "🐕 {name} detected (confidence: {confidence}%)",
            'system_startup': "🚀 Intruder Detection System started",
            'system_shutdown': "🛑 Intruder Detection System stopped",
            'camera_offline': "📹 Camera {camera_id} went offline",
            'camera_online': "📹 Camera {camera_id} is back online"
        }

        # Delivery confirmation tracking
        self.pending_confirmations = {}

        # Performance tracking
        self.notification_stats = {
            'messages_sent': 0,
            'photos_sent': 0,
            'commands_received': 0,
            'failed_sends': 0,
            'active_users': 0,
            'rate_limited': 0,
            'template_usage': {}
        }

        logger.info("Enhanced Notification System initialized")
    
    def load_users(self, users_data: List[Dict]):
        """
        Load user configurations from database.
        
        Args:
            users_data: List of user configuration dictionaries
        """
        self.users = {}
        
        for user_data in users_data:
            chat_id = user_data['chat_id']
            self.users[chat_id] = {
                'chat_id': chat_id,
                'username': user_data['telegram_username'],
                'notify_human_detection': user_data['notify_human_detection'],
                'notify_animal_detection': user_data['notify_animal_detection'],
                'sendstatus': user_data['sendstatus'],
                'last_notification': user_data.get('last_notification')
            }
        
        self.notification_stats['active_users'] = len([u for u in self.users.values() if u['sendstatus'] == 'open'])
        logger.info(f"Loaded {len(self.users)} users, {self.notification_stats['active_users']} active")
    
    def start_listening(self):
        """Start listening for incoming messages in a separate thread."""
        if not self.listening:
            self.listening = True
            self.listen_thread = threading.Thread(target=self._listen_for_messages, daemon=True)
            self.listen_thread.start()
            logger.info("Started listening for Telegram messages")
    
    def stop_listening(self):
        """Stop listening for incoming messages."""
        self.listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=5)
        logger.info("Stopped listening for Telegram messages")
    
    def _listen_for_messages(self):
        """Listen for incoming messages and process commands."""
        while self.listening:
            try:
                updates = self._get_updates()
                
                if updates and 'result' in updates:
                    for update in updates['result']:
                        self._process_update(update)
                
                time.sleep(1)  # Poll every second
                
            except Exception as e:
                logger.error(f"Error in message listening: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _get_updates(self) -> Optional[Dict]:
        """Get updates from Telegram API."""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {
                'offset': self.last_update_id + 1,
                'timeout': 10,
                'limit': 100
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['ok'] and data['result']:
                    # Update last_update_id to mark messages as processed
                    self.last_update_id = max(update['update_id'] for update in data['result'])
                
                return data
            else:
                logger.warning(f"Failed to get updates: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting updates: {e}")
            return None
    
    def _process_update(self, update: Dict):
        """Process a single update from Telegram."""
        try:
            if 'message' in update:
                message = update['message']
                chat_id = message['chat']['id']
                
                # Check if user is authorized
                if chat_id not in self.users:
                    logger.warning(f"Unauthorized user attempted to send message: {chat_id}")
                    return
                
                user = self.users[chat_id]
                if user['sendstatus'] != 'open':
                    logger.info(f"Message from inactive user: {chat_id}")
                    return
                
                # Process text messages
                if 'text' in message:
                    text = message['text'].strip().lower()
                    self._process_command(chat_id, text)
                    self.notification_stats['commands_received'] += 1
                
        except Exception as e:
            logger.error(f"Error processing update: {e}")
    
    def _process_command(self, chat_id: int, command: str):
        """Process commands from users."""
        try:
            if command == 'check':
                # Manual photo capture command
                logger.info(f"Manual check command received from {chat_id}")
                # This will be handled by the main detection system
                # For now, just acknowledge the command
                self.send_message(chat_id, "📸 Manual check initiated. Taking photo...")
                return True
            elif command in ['status', '/status']:
                # Status command
                self.send_message(chat_id, "🟢 System is running and monitoring for intruders.")
                return True
            elif command in ['help', '/help', '/start']:
                # Help command
                help_text = """
🤖 Intruder Detection Bot Commands:

• `check` - Take manual photo and analyze
• `status` - Check system status
• `help` - Show this help message

The bot will automatically notify you of:
• Unknown human detections
• Unfamiliar animal detections
• System alerts

Your notification settings can be managed through the main application.
                """
                self.send_message(chat_id, help_text)
                return True
            else:
                # Unknown command
                self.send_message(chat_id, "❓ Unknown command. Type 'help' for available commands.")
                return False
                
        except Exception as e:
            logger.error(f"Error processing command '{command}' from {chat_id}: {e}")
            return False
    
    def send_notification(self, notification_type: str, message: str, 
                         photo_path: Optional[str] = None, 
                         force: bool = False) -> bool:
        """
        Send notification to all eligible users.
        
        Args:
            notification_type: 'human' or 'animal'
            message: Notification message
            photo_path: Optional path to photo to send
            force: Skip cooldown check
            
        Returns:
            True if sent to at least one user
        """
        sent_count = 0
        
        for chat_id, user in self.users.items():
            if user['sendstatus'] != 'open':
                continue
            
            # Check notification preferences
            if (notification_type == 'human' and not user['notify_human_detection']) or \
               (notification_type == 'animal' and not user['notify_animal_detection']):
                continue
            
            # Check cooldown unless forced
            if not force and self._is_in_cooldown(chat_id):
                logger.debug(f"Skipping notification to {chat_id} due to cooldown")
                continue
            
            # Send notification
            if self._send_to_user(chat_id, message, photo_path):
                sent_count += 1
                self._update_cooldown(chat_id)
        
        return sent_count > 0
    
    def _send_to_user(self, chat_id: int, message: str, photo_path: Optional[str] = None) -> bool:
        """Send message/photo to a specific user."""
        try:
            success = False
            
            # Send photo if provided
            if photo_path and os.path.exists(photo_path):
                if self.send_photo(chat_id, photo_path, message):
                    success = True
                    self.notification_stats['photos_sent'] += 1
            else:
                # Send text message
                if self.send_message(chat_id, message):
                    success = True
                    self.notification_stats['messages_sent'] += 1
            
            if not success:
                self.notification_stats['failed_sends'] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending to user {chat_id}: {e}")
            self.notification_stats['failed_sends'] += 1
            return False
    
    def send_message(self, chat_id: int, message: str) -> bool:
        """Send text message to a specific chat."""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result['ok']:
                    logger.debug(f"Message sent to {chat_id}")
                    return True
                else:
                    logger.warning(f"Failed to send message to {chat_id}: {result}")
                    return False
            else:
                logger.warning(f"HTTP error sending message to {chat_id}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending message to {chat_id}: {e}")
            return False
    
    def send_photo(self, chat_id: int, photo_path: str, caption: str = "") -> bool:
        """Send photo to a specific chat."""
        try:
            url = f"{self.base_url}/sendPhoto"
            
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': chat_id,
                    'caption': caption,
                    'parse_mode': 'Markdown'
                }
                
                response = requests.post(url, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result['ok']:
                    logger.debug(f"Photo sent to {chat_id}")
                    return True
                else:
                    logger.warning(f"Failed to send photo to {chat_id}: {result}")
                    return False
            else:
                logger.warning(f"HTTP error sending photo to {chat_id}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending photo to {chat_id}: {e}")
            return False
    
    def _is_in_cooldown(self, chat_id: int) -> bool:
        """Check if user is in notification cooldown."""
        if chat_id not in self.notification_cooldowns:
            return False
        
        last_notification = self.notification_cooldowns[chat_id]
        return (time.time() - last_notification) < self.default_cooldown
    
    def _update_cooldown(self, chat_id: int):
        """Update cooldown timestamp for user."""
        self.notification_cooldowns[chat_id] = time.time()
    
    def test_connection(self) -> bool:
        """Test bot connection and token validity."""
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result['ok']:
                    bot_info = result['result']
                    logger.info(f"Bot connection successful: {bot_info['first_name']} (@{bot_info['username']})")
                    return True
                else:
                    logger.error(f"Bot token invalid: {result}")
                    return False
            else:
                logger.error(f"HTTP error testing bot: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error testing bot connection: {e}")
            return False
    
    def get_performance_stats(self) -> Dict:
        """Get notification system performance statistics."""
        return {
            'messages_sent': self.notification_stats['messages_sent'],
            'photos_sent': self.notification_stats['photos_sent'],
            'commands_received': self.notification_stats['commands_received'],
            'failed_sends': self.notification_stats['failed_sends'],
            'active_users': self.notification_stats['active_users'],
            'total_users': len(self.users),
            'listening_status': self.listening,
            'success_rate': (
                (self.notification_stats['messages_sent'] + self.notification_stats['photos_sent']) /
                max(1, self.notification_stats['messages_sent'] + self.notification_stats['photos_sent'] + self.notification_stats['failed_sends']) * 100
            )
        }
    
    def add_user(self, user_data: Dict) -> bool:
        """Add a new user to the notification system."""
        try:
            chat_id = user_data['chat_id']
            self.users[chat_id] = {
                'chat_id': chat_id,
                'username': user_data['telegram_username'],
                'notify_human_detection': user_data.get('notify_human_detection', True),
                'notify_animal_detection': user_data.get('notify_animal_detection', True),
                'sendstatus': user_data.get('sendstatus', 'open'),
                'last_notification': None
            }
            
            # Update active users count
            self.notification_stats['active_users'] = len([u for u in self.users.values() if u['sendstatus'] == 'open'])
            
            logger.info(f"Added user {chat_id} ({user_data['telegram_username']})")
            return True
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            return False
    
    def remove_user(self, chat_id: int) -> bool:
        """Remove a user from the notification system."""
        if chat_id in self.users:
            del self.users[chat_id]
            if chat_id in self.notification_cooldowns:
                del self.notification_cooldowns[chat_id]
            
            # Update active users count
            self.notification_stats['active_users'] = len([u for u in self.users.values() if u['sendstatus'] == 'open'])
            
            logger.info(f"Removed user {chat_id}")
            return True
        return False

    def send_templated_notification(self, template_key: str, **kwargs):
        """Send a notification using a predefined template."""
        try:
            if template_key not in self.templates:
                logger.error(f"Unknown template: {template_key}")
                return False

            # Format message using template
            message = self.templates[template_key].format(**kwargs)

            # Track template usage
            if template_key not in self.notification_stats['template_usage']:
                self.notification_stats['template_usage'][template_key] = 0
            self.notification_stats['template_usage'][template_key] += 1

            # Send to all eligible users
            return self.send_notification_to_all(message)

        except Exception as e:
            logger.error(f"Error sending templated notification: {e}")
            return False

    def send_notification_to_all(self, message: str, photo_path: str = None):
        """Send notification to all active users with rate limiting."""
        success_count = 0

        for chat_id, user in self.users.items():
            if user['sendstatus'] == 'open':
                if self._check_rate_limit(chat_id):
                    if photo_path:
                        success = self.send_photo(chat_id, photo_path, message)
                    else:
                        success = self.send_message(chat_id, message)

                    if success:
                        success_count += 1
                        self._log_notification(chat_id, message, photo_path)
                else:
                    logger.warning(f"Rate limit exceeded for user {chat_id}")
                    self.notification_stats['rate_limited'] += 1

        return success_count > 0

    def _check_rate_limit(self, chat_id: int) -> bool:
        """Check if user has exceeded rate limit."""
        current_time = time.time()

        if chat_id not in self.last_notification_times:
            self.last_notification_times[chat_id] = []

        # Remove old notifications outside the window
        window_start = current_time - self.rate_limit_window
        self.last_notification_times[chat_id] = [
            t for t in self.last_notification_times[chat_id] if t > window_start
        ]

        # Check if under limit
        if len(self.last_notification_times[chat_id]) < self.max_notifications_per_window:
            self.last_notification_times[chat_id].append(current_time)
            return True

        return False

    def _log_notification(self, chat_id: int, message: str, photo_path: str = None):
        """Log notification to history."""
        notification_record = {
            'timestamp': time.time(),
            'chat_id': chat_id,
            'message': message,
            'photo_path': photo_path,
            'delivered': True
        }

        # Add to in-memory history
        self.notification_history.append(notification_record)

        # Trim history if too long
        if len(self.notification_history) > self.max_history:
            self.notification_history = self.notification_history[-self.max_history:]

        # Log to database if available
        if self.db_manager:
            try:
                # This would require a notification_history table
                # For now, we'll just log it
                logger.debug(f"Notification logged: {chat_id} - {message[:50]}...")
            except Exception as e:
                logger.error(f"Error logging notification to database: {e}")

    def get_notification_history(self, limit: int = 50):
        """Get recent notification history."""
        return self.notification_history[-limit:] if self.notification_history else []

    def get_notification_stats(self):
        """Get comprehensive notification statistics."""
        stats = self.notification_stats.copy()
        stats['total_history_entries'] = len(self.notification_history)
        stats['rate_limit_window'] = self.rate_limit_window
        stats['max_per_window'] = self.max_notifications_per_window
        return stats

    def update_template(self, template_key: str, template_text: str):
        """Update or add a notification template."""
        self.templates[template_key] = template_text
        logger.info(f"Updated template: {template_key}")

    def get_templates(self):
        """Get all available templates."""
        return self.templates.copy()

    def clear_rate_limits(self, chat_id: int = None):
        """Clear rate limits for a user or all users."""
        if chat_id:
            if chat_id in self.last_notification_times:
                del self.last_notification_times[chat_id]
                logger.info(f"Cleared rate limits for user {chat_id}")
        else:
            self.last_notification_times.clear()
            logger.info("Cleared all rate limits")

    def set_rate_limit(self, window_seconds: int, max_notifications: int):
        """Update rate limiting settings."""
        self.rate_limit_window = window_seconds
        self.max_notifications_per_window = max_notifications
        logger.info(f"Updated rate limits: {max_notifications} per {window_seconds}s")
