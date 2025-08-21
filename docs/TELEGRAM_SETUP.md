# 📱 Telegram Bot Setup Guide - Intruder Detection System

## 🎯 Overview

This guide covers setting up the Telegram bot for receiving detection notifications and sending commands to your Intruder Detection System.

## 🤖 Creating a Telegram Bot

### Step 1: Create Bot with BotFather

1. **Open Telegram** and search for `@BotFather`
2. **Start conversation** with BotFather
3. **Send command**: `/newbot`
4. **Choose bot name**: e.g., "My Intruder Detection Bot"
5. **Choose username**: e.g., "my_intruder_bot" (must end with 'bot')
6. **Save the token**: BotFather will provide a token like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`

### Step 2: Configure Bot Settings

**Set Bot Description:**
```
/setdescription
@your_bot_username
Intruder Detection System - Sends alerts for human and animal detections with photos and status updates.
```

**Set Bot Commands:**
```
/setcommands
@your_bot_username

check - Get current system status
status - Show detection statistics  
help - Show available commands
start - Start receiving notifications
stop - Stop receiving notifications
```

**Set Bot Profile Picture:**
1. Send `/setuserpic` to BotFather
2. Select your bot
3. Upload a security camera or detection-related image

## 🔧 System Configuration

### Step 1: Add Bot Token to System

**Method 1: Environment Variable (Recommended)**
```bash
# Windows
set TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz

# Linux/Mac
export TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
```

**Method 2: Configuration File**
```yaml
# config.yaml
telegram:
  bot_token: "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
  enabled: true
  notification_cooldown: 20  # seconds
  max_users: 10
```

**Method 3: GUI Configuration**
1. Open **Notification Center** in the GUI
2. Click **"Bot Configuration"**
3. Enter your bot token
4. Click **"Test Connection"**
5. Click **"Save"** if test succeeds

### Step 2: Get Your Chat ID

**Method 1: Automatic (Recommended)**
1. Start your detection system
2. Send `/start` to your bot
3. System will automatically detect and save your chat ID

**Method 2: Manual**
1. Send any message to your bot
2. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. Find your `chat_id` in the response
4. Add it to the system via GUI or database

## 👥 User Management

### Adding Users via GUI

1. **Open Notification Center** in the main GUI
2. **Click "Add User"**
3. **Enter user details:**
   ```
   Username: @john_doe
   Chat ID: 123456789
   Notifications: ✅ Enabled
   Human Alerts: ✅ Enabled  
   Animal Alerts: ✅ Enabled
   System Alerts: ✅ Enabled
   ```
4. **Click "Save"**

### Adding Users via Bot Commands

**User Registration:**
1. User sends `/start` to the bot
2. System automatically adds user with default settings
3. Admin can modify permissions via GUI

**User Permissions:**
- **Human Detection**: Receive alerts for human detections
- **Animal Detection**: Receive alerts for animal detections  
- **System Alerts**: Receive system status and error messages
- **Photo Sharing**: Receive detection photos

### Bulk User Management

**Import Users from File:**
```json
{
  "users": [
    {
      "username": "@john_doe",
      "chat_id": 123456789,
      "human_notifications": true,
      "animal_notifications": true,
      "system_notifications": true
    },
    {
      "username": "@jane_smith", 
      "chat_id": 987654321,
      "human_notifications": true,
      "animal_notifications": false,
      "system_notifications": true
    }
  ]
}
```

## 📨 Notification Types

### Human Detection Alerts
```
🚨 HUMAN DETECTED 🚨

👤 Person: Jun Hui (92% confidence)
📍 Location: Front Door Camera
🕐 Time: 2025-08-17 14:30:25
📊 System Status: Normal

[Photo attached]
```

### Animal Detection Alerts
```
🐕 ANIMAL DETECTED 🐕

🐾 Animal: Dog - Jacky (88% confidence)
📍 Location: Back Yard Camera  
🕐 Time: 2025-08-17 14:35:10
📊 System Status: Normal

[Photo attached]
```

### Unknown Detection Alerts
```
⚠️ UNKNOWN DETECTION ⚠️

👤 Unknown Person Detected
📍 Location: Front Door Camera
🕐 Time: 2025-08-17 14:40:15
🔍 Confidence: 85%

[Photo attached]
```

### System Status Messages
```
📊 SYSTEM STATUS 📊

🟢 Status: Online
📹 Cameras: 2/2 Active
🧠 Detection: Running (58.8 FPS)
💾 Database: Connected
📱 Bot: Online
🔋 CPU: 15% | RAM: 2.1GB | GPU: 45%

Last Detection: 2 minutes ago
```

## 🎮 Bot Commands

### User Commands

**`/start`** - Register for notifications
```
Welcome to Intruder Detection System! 🛡️

You are now registered to receive detection alerts.

Available commands:
/check - System status
/status - Detection statistics
/help - Show this help
/stop - Unregister from alerts
```

**`/check`** - Get current system status
```
📊 SYSTEM STATUS

🟢 Online | 🎯 Detecting | 📹 2 Cameras Active
⚡ Performance: 58.8 FPS | 🧠 TensorRT Optimized
📊 Today: 15 humans, 8 animals detected
```

**`/status`** - Show detection statistics
```
📈 DETECTION STATISTICS (Last 24h)

👤 Humans: 15 detections
   • Jun Hui: 8 times
   • Jia Qing: 3 times  
   • Unknown: 4 times

🐾 Animals: 8 detections
   • Jacky (Dog): 5 times
   • Unknown Cat: 2 times
   • Unknown Dog: 1 time

📊 Performance: Avg 58.2 FPS
```

**`/help`** - Show available commands
```
🤖 AVAILABLE COMMANDS

/check - Current system status
/status - Detection statistics
/help - Show this help message
/start - Start receiving notifications
/stop - Stop receiving notifications

📱 You will receive automatic alerts for:
• Human detections with photos
• Animal detections with photos  
• System status updates
```

**`/stop`** - Stop receiving notifications
```
❌ Notifications Disabled

You will no longer receive detection alerts.
Send /start to re-enable notifications.
```

### Admin Commands (Future Enhancement)

**`/admin`** - Admin panel access
**`/users`** - List all registered users
**`/broadcast`** - Send message to all users
**`/settings`** - Modify system settings

## 🔧 Advanced Configuration

### Notification Customization

**Cooldown Settings:**
```yaml
telegram:
  notification_cooldown: 20  # seconds between notifications
  max_notifications_per_hour: 30
  quiet_hours:
    enabled: true
    start: "22:00"
    end: "07:00"
```

**Message Templates:**
```yaml
telegram:
  templates:
    human_detection: "🚨 Human: {name} ({confidence}%) at {location}"
    animal_detection: "🐾 Animal: {type} at {location}"
    system_alert: "⚠️ System: {message}"
```

### Photo Settings

**Image Quality:**
```yaml
telegram:
  photos:
    enabled: true
    quality: 85  # JPEG quality (1-100)
    max_size: 1024  # Max width/height in pixels
    include_overlay: true  # Include detection boxes
```

### Security Settings

**User Verification:**
```yaml
telegram:
  security:
    require_approval: true  # Admin must approve new users
    max_users: 10
    allowed_usernames: ["@admin", "@family_member"]
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Bot Not Responding

**Symptoms:**
- Bot doesn't reply to commands
- No notifications received

**Solutions:**
1. **Check bot token**: Verify token is correct
2. **Test bot manually**: Send message via Telegram
3. **Check internet connection**: Ensure system is online
4. **Restart notification system**: Restart the detection system

#### 2. "Unauthorized" Error

**Symptoms:**
- Error 401 when sending messages
- Bot token invalid

**Solutions:**
```bash
# Test bot token manually
curl "https://api.telegram.org/bot<YOUR_TOKEN>/getMe"

# Should return bot information, not error
```

#### 3. Photos Not Sending

**Symptoms:**
- Text notifications work
- Photos fail to send

**Solutions:**
1. **Check file size**: Telegram limit is 50MB
2. **Check file format**: Use JPEG/PNG
3. **Check permissions**: Ensure bot can access photo files

#### 4. High Notification Volume

**Symptoms:**
- Too many notifications
- Spam-like behavior

**Solutions:**
1. **Increase cooldown**: Set longer delay between notifications
2. **Adjust confidence thresholds**: Reduce false positives
3. **Enable quiet hours**: Disable notifications at night

### Diagnostic Commands

**Test Bot Connection:**
```python
python scripts/test_telegram_bot.py --token YOUR_TOKEN
```

**Check User Registration:**
```python
python scripts/check_telegram_users.py
```

**Send Test Notification:**
```python
python scripts/send_test_notification.py --chat-id YOUR_CHAT_ID
```

## 📊 Performance Considerations

### Message Rate Limits

**Telegram Limits:**
- **30 messages per second** to different users
- **1 message per second** to same user
- **20 MB per minute** for file uploads

**System Optimization:**
- Queue messages during high activity
- Batch notifications when possible
- Respect cooldown periods

### Resource Usage

**Memory Usage:**
- **Bot service**: ~10-20MB RAM
- **Photo processing**: ~5-10MB per photo
- **Message queue**: ~1MB per 1000 messages

**Network Usage:**
- **Text message**: ~1KB
- **Photo message**: ~100KB-2MB depending on quality

## 🔗 Integration Examples

### Custom Notification Handler

```python
class CustomTelegramHandler:
    def __init__(self, bot_token):
        self.bot_token = bot_token
        self.notification_system = NotificationSystem(bot_token)
    
    def send_custom_alert(self, message, photo_path=None):
        """Send custom alert to all users."""
        users = self.get_active_users()
        for user in users:
            self.notification_system.send_message(
                chat_id=user.chat_id,
                message=message,
                photo_path=photo_path
            )
```

### Webhook Integration

```python
# For advanced users: Use webhooks instead of polling
from flask import Flask, request

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    update = request.get_json()
    # Process Telegram update
    process_telegram_update(update)
    return 'OK'
```

---

## 📞 Support

For Telegram-specific issues:
1. **Check Telegram Bot API documentation**
2. **Test bot manually** with BotFather
3. **Verify network connectivity**
4. **Check system logs** in `logs/detection_system.log`

---

## 🔗 Related Documentation

- **Installation Guide**: `INSTALLATION.md`
- **API Documentation**: `API.md`
- **Camera Setup**: `CAMERA_SETUP.md`
- **Development Guide**: `DEVELOPMENT.md`
