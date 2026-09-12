# 📚 Documentation Index

Welcome to the Intruder Detection System documentation! This folder contains comprehensive guides for installation, configuration, development, and usage.

## 🚀 Getting Started

### Essential Guides
- **[INSTALLATION.md](INSTALLATION.md)** - Complete setup instructions
- **[SECURITY.md](SECURITY.md)** - Security best practices and environment variable setup

### Quick Start Checklist
1. ✅ Follow [INSTALLATION.md](INSTALLATION.md) for dependency setup
2. ✅ Run `python scripts/setup_secure_config.py` for secure configuration
3. ✅ Set up cameras using [CAMERA_SETUP.md](CAMERA_SETUP.md)
4. ✅ Configure Telegram bot with [TELEGRAM_SETUP.md](TELEGRAM_SETUP.md)
5. ✅ Run `python main.py` to start the system

## 📖 Complete Documentation

### Setup & Configuration
- **[INSTALLATION.md](INSTALLATION.md)** - Installation guide with platform-specific instructions
- **[SECURITY.md](SECURITY.md)** - Security configuration and environment variables
- **[CAMERA_SETUP.md](CAMERA_SETUP.md)** - IP camera and local camera configuration
- **[TELEGRAM_SETUP.md](TELEGRAM_SETUP.md)** - Telegram bot setup and user management

### Development & Technical
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development workflow and architecture
- **[API.md](API.md)** - Code reference and API documentation

### Project Information
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes

## 🎯 Documentation by Use Case

### 🏠 **Home User Setup**
1. [INSTALLATION.md](INSTALLATION.md) - Install the system
2. [SECURITY.md](SECURITY.md) - Set up secure configuration
3. [CAMERA_SETUP.md](CAMERA_SETUP.md) - Configure your cameras
4. [TELEGRAM_SETUP.md](TELEGRAM_SETUP.md) - Set up notifications

### 👨‍💻 **Developer Setup**
1. [INSTALLATION.md](INSTALLATION.md) - Install dependencies
2. [DEVELOPMENT.md](DEVELOPMENT.md) - Development environment
3. [API.md](API.md) - Code reference
4. [SECURITY.md](SECURITY.md) - Security considerations

### 🔧 **System Administrator**
1. [SECURITY.md](SECURITY.md) - Security best practices
2. [INSTALLATION.md](INSTALLATION.md) - Deployment guide
3. [DEVELOPMENT.md](DEVELOPMENT.md) - System architecture

### 🔄 **Migration from v1.x**
1. [CHANGELOG.md](CHANGELOG.md) - What's changed in v2.0
2. [SECURITY.md](SECURITY.md) - New security features

## 🔍 Quick Reference

### Common Tasks
- **Install system**: See [INSTALLATION.md](INSTALLATION.md)
- **Set up bot token**: See [SECURITY.md](SECURITY.md#environment-variables-setup)
- **Add IP camera**: See [CAMERA_SETUP.md](CAMERA_SETUP.md#ip-camera-configuration)
- **Configure notifications**: See [TELEGRAM_SETUP.md](TELEGRAM_SETUP.md)
- **Troubleshoot issues**: Check relevant guide's troubleshooting section

### File Locations
- **Configuration**: `config.yaml` (main), `.env` (secrets)
- **Models**: `models/` folder
- **Data**: `data/` folder (faces, animals, detections)
- **Logs**: `logs/` folder
- **Database**: `detection_system.db`

## 🆘 Getting Help

### Documentation Issues
If you find issues with the documentation:
1. Check the [CHANGELOG.md](CHANGELOG.md) for recent changes
2. Verify you're using the correct version
3. Report issues via GitHub Issues

### System Issues
For system problems:
1. **Installation issues**: [INSTALLATION.md](INSTALLATION.md#troubleshooting)
2. **Security/config issues**: [SECURITY.md](SECURITY.md#troubleshooting-security-issues)
3. **Camera issues**: [CAMERA_SETUP.md](CAMERA_SETUP.md#troubleshooting)
4. **Telegram issues**: [TELEGRAM_SETUP.md](TELEGRAM_SETUP.md#troubleshooting)
5. **Development issues**: [DEVELOPMENT.md](DEVELOPMENT.md#troubleshooting)

### Support Channels
- 📖 **Documentation**: This folder
- 🐛 **Bug Reports**: GitHub Issues
- 💬 **Discussions**: GitHub Discussions
- 📧 **Email**: support@intruder-detection.com

## 📋 Documentation Standards

### File Naming
- **UPPERCASE.md** for main guides (INSTALLATION.md, SECURITY.md)
- **lowercase.md** for supplementary docs
- Clear, descriptive names

### Content Structure
- **Overview** section at the top
- **Step-by-step instructions** with code examples
- **Troubleshooting** section at the bottom
- **Cross-references** to related documentation

### Maintenance
- Documentation is updated with each release
- Version-specific information in [CHANGELOG.md](CHANGELOG.md)
- Regular review for accuracy and completeness

---

**Version**: 2.0.0  
**Total Documents**: 8 guides
