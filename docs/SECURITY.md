# Security Guide

This document outlines the security measures implemented in the Intruder Detection System and best practices for secure deployment.

## 🔐 Security Features

### Environment Variable Configuration
- **Sensitive data** (API keys, tokens) are loaded from environment variables
- **Configuration files** contain no hardcoded secrets
- **Automatic precedence**: Environment variables override config file values
- **Secure loading**: Warnings when sensitive data found in config files

### Data Protection
- **Database encryption**: SQLite database with secure file permissions
- **Log sanitization**: Sensitive data excluded from log files
- **Memory protection**: Secrets cleared from memory when possible
- **File permissions**: Restricted access to configuration and data files

### Access Control
- **Telegram bot authentication**: Only authorized users can interact
- **API key validation**: Proper validation of all external API keys
- **Camera access control**: Secure IP camera authentication
- **Database access**: Protected database connections

## 🛡️ Secure Setup

### 1. Environment Variables Setup

**Option A: Interactive Setup (Recommended)**
```bash
python scripts/setup_secure_config.py
```

**Option B: Manual Setup**
```bash
# Copy template
cp .env.template .env

# Edit with your secure editor
nano .env  # or vim, code, etc.
```

### 2. Required Environment Variables

```bash
# Telegram Bot Configuration (REQUIRED)
TELEGRAM_BOT_TOKEN=your_bot_token_here

# Optional: Restrict to specific chat
TELEGRAM_CHAT_ID=your_chat_id_here
```

### 3. Optional Environment Variables

```bash
# Performance Settings
MAX_CPU_USAGE=80.0
MAX_MEMORY_USAGE=4096
ENABLE_GPU=true

# Detection Thresholds
HUMAN_CONFIDENCE_THRESHOLD=0.6
ANIMAL_CONFIDENCE_THRESHOLD=0.6
YOLO_CONFIDENCE=0.5

# Database Settings
DATABASE_PATH=detection_system.db

# Logging Settings
LOG_LEVEL=INFO
```

## 🔒 Security Best Practices

### File Security
```bash
# Set proper permissions for .env file
chmod 600 .env

# Ensure .env is in .gitignore
echo ".env" >> .gitignore

# Verify no secrets in config files
grep -r "bot_token\|api_key\|password" config.yaml
```

### Telegram Bot Security
1. **Create a dedicated bot** for the detection system
2. **Use BotFather** to generate a unique token
3. **Restrict bot permissions** to necessary commands only
4. **Monitor bot usage** through Telegram's bot analytics
5. **Revoke and regenerate tokens** if compromised

### Database Security
```bash
# Set secure database permissions
chmod 600 detection_system.db

# Regular backups with encryption
tar -czf backup.tar.gz detection_system.db
gpg -c backup.tar.gz  # Encrypt backup
```

### Network Security
- **Use HTTPS** for all external API calls
- **Validate SSL certificates** for IP cameras
- **Firewall rules** to restrict network access
- **VPN access** for remote monitoring

## 🚨 Security Monitoring

### Automated Checks
The system includes automated security checks:

```bash
# Run security tests
python tests/test_security.py

# Check for hardcoded secrets
python -c "
import re
with open('config.yaml', 'r') as f:
    content = f.read()
    if re.search(r'\d{10}:[A-Za-z0-9_-]{35}', content):
        print('⚠️ Potential bot token found in config!')
    else:
        print('✅ No hardcoded tokens detected')
"
```

### Manual Security Audit
Regular security audits should include:

1. **Configuration Review**
   - Verify no secrets in config files
   - Check file permissions
   - Review access logs

2. **Dependency Audit**
   ```bash
   pip audit  # Check for vulnerable packages
   ```

3. **Network Security**
   - Review firewall rules
   - Check for open ports
   - Validate SSL certificates

## 🔧 Troubleshooting Security Issues

### Common Issues

**Issue: "Bot token not found"**
```bash
# Check environment variable
echo $TELEGRAM_BOT_TOKEN

# Verify .env file loading
python -c "
from config.env_config import get_secure_config
token = get_secure_config('telegram.bot_token')
print('Token found' if token else 'Token missing')
"
```

**Issue: "Configuration warnings"**
- Move sensitive data from config.yaml to environment variables
- Use the secure setup script: `python scripts/setup_secure_config.py`

**Issue: "Permission denied"**
```bash
# Fix file permissions
chmod 600 .env
chmod 600 detection_system.db
chmod 755 logs/
```

### Security Incident Response

If you suspect a security breach:

1. **Immediate Actions**
   - Revoke compromised API keys/tokens
   - Change all passwords
   - Review access logs

2. **Investigation**
   - Check system logs for unusual activity
   - Review database for unauthorized changes
   - Analyze network traffic

3. **Recovery**
   - Generate new API keys/tokens
   - Update environment variables
   - Restart all services

## 📋 Security Checklist

Before deploying to production:

- [ ] All sensitive data moved to environment variables
- [ ] .env file has proper permissions (600)
- [ ] .env file is in .gitignore
- [ ] No hardcoded secrets in code or config files
- [ ] Database has secure permissions
- [ ] Firewall rules configured
- [ ] SSL certificates validated
- [ ] Regular backup strategy implemented
- [ ] Security monitoring enabled
- [ ] Incident response plan documented

## 🆘 Emergency Contacts

In case of security incidents:
- **System Administrator**: [Your contact info]
- **Security Team**: [Security team contact]
- **Telegram Support**: https://telegram.org/support

---

**Remember**: Security is an ongoing process, not a one-time setup. Regularly review and update your security measures.
