# 📹 Camera Setup Guide - Intruder Detection System

## 🎯 Overview

This guide covers setting up IP cameras, local cameras, and troubleshooting camera connections for the Intruder Detection System.

## 📱 Supported Camera Types

### 1. IP Cameras
- **HTTP/HTTPS streams**
- **RTSP streams** 
- **DroidCam** (Android phone as camera)
- **Generic IP cameras** with video endpoints

### 2. Local Cameras
- **USB webcams**
- **Built-in laptop cameras**
- **USB capture devices**

## 🔧 IP Camera Configuration

### DroidCam Setup (Recommended for Testing)

**Step 1: Install DroidCam on Android**
1. Download **DroidCam** from Google Play Store
2. Install and open the app
3. Note the **IP address** and **port** shown (e.g., `192.168.100.101:4747`)

**Step 2: Configure in Detection System**
1. Open the **IP Camera Manager** in the GUI
2. Click **"Add Camera"**
3. Enter camera details:
   ```
   Name: DroidCam Phone
   IP Address: 192.168.100.101
   Port: 4747
   URL Suffix: video
   Protocol: HTTP
   ```
4. Click **"Test Connection"** to verify
5. Click **"Save"** if test succeeds

**Step 3: Verify Connection**
- The system will automatically construct URL: `http://192.168.100.101:4747/video`
- Test in browser first: Open the URL to see if video stream loads
- If browser shows video, the detection system should work

### Generic IP Camera Setup

**Common IP Camera URLs:**
```bash
# Generic IP cameras
http://192.168.1.100:8080/video
http://192.168.1.100/mjpeg
http://192.168.1.100/stream

# Hikvision cameras
http://192.168.1.100/ISAPI/Streaming/channels/1/picture

# Dahua cameras  
http://192.168.1.100/cgi-bin/mjpg/video.cgi

# Axis cameras
http://192.168.1.100/axis-cgi/mjpg/video.cgi

# Foscam cameras
http://192.168.1.100:88/cgi-bin/CGIStream.cgi?cmd=GetMJStream
```

**Configuration Steps:**
1. Find your camera's IP address (check router admin panel)
2. Determine the video stream endpoint (check camera manual)
3. Test URL in web browser first
4. Add to system using IP Camera Manager

### RTSP Camera Setup

**RTSP URL Format:**
```bash
rtsp://username:password@192.168.1.100:554/stream1
rtsp://192.168.1.100:554/live/ch1
```

**Configuration:**
1. Enable RTSP in camera settings
2. Note username/password if required
3. Find RTSP port (usually 554)
4. Add to system with full RTSP URL

## 🖥️ Local Camera Setup

### USB Webcam Configuration

**Automatic Detection:**
- System automatically detects USB cameras
- Default camera index: `0` (first camera)
- Additional cameras: `1`, `2`, etc.

**Manual Configuration:**
1. Connect USB camera
2. Open **IP Camera Manager**
3. Click **"Test Local Camera"**
4. Select camera index if multiple cameras detected
5. Verify video feed appears

### Built-in Camera Setup

**Windows Camera Access:**
1. Ensure camera permissions are enabled:
   - Go to **Settings > Privacy > Camera**
   - Enable **"Allow apps to access your camera"**
   - Enable for **Python** applications

2. Test camera access:
   ```python
   import cv2
   cap = cv2.VideoCapture(0)
   ret, frame = cap.read()
   if ret:
       print("Camera working!")
   cap.release()
   ```

## 🔧 Advanced Configuration

### Multi-Camera Setup

**Simultaneous Cameras:**
The system supports up to **4 simultaneous cameras**:

1. **Primary Camera**: Main detection feed
2. **Secondary Cameras**: Additional monitoring points
3. **Backup Camera**: Automatic failover

**Configuration:**
```yaml
# config.yaml
cameras:
  primary:
    type: "ip"
    url: "http://192.168.1.100:8080/video"
    priority: 1
  
  secondary:
    type: "ip" 
    url: "http://192.168.1.101:8080/video"
    priority: 2
    
  backup:
    type: "local"
    index: 0
    priority: 3
```

### Camera Quality Settings

**Resolution Settings:**
```python
# High quality (slower)
camera_config = {
    'width': 1920,
    'height': 1080,
    'fps': 30
}

# Balanced (recommended)
camera_config = {
    'width': 1280,
    'height': 720,
    'fps': 30
}

# Performance (faster)
camera_config = {
    'width': 640,
    'height': 480,
    'fps': 60
}
```

### Network Optimization

**For IP Cameras:**
1. **Use wired connection** when possible
2. **Reduce video quality** if network is slow
3. **Enable camera buffering** for stable streams
4. **Set appropriate timeout** values (5-10 seconds)

**Bandwidth Requirements:**
- **720p @ 30fps**: ~2-5 Mbps
- **1080p @ 30fps**: ~5-10 Mbps
- **4K @ 30fps**: ~15-25 Mbps

## 🐛 Troubleshooting

### Common Issues

#### 1. "Camera Not Found" Error

**Symptoms:**
- Camera not detected
- Connection timeout
- Black screen in video feed

**Solutions:**
```bash
# Check camera connectivity
ping 192.168.1.100

# Test camera URL in browser
curl -I http://192.168.1.100:8080/video

# Check camera permissions (Windows)
# Settings > Privacy > Camera > Allow apps to access camera
```

#### 2. "Connection Refused" Error

**Possible Causes:**
- Wrong IP address or port
- Camera not powered on
- Firewall blocking connection
- Camera requires authentication

**Solutions:**
1. **Verify IP address**: Check router DHCP table
2. **Check camera power**: Ensure camera is on and connected
3. **Test with browser**: Open camera URL in web browser
4. **Check authentication**: Add username/password if required

#### 3. "Stream Format Not Supported"

**Symptoms:**
- Camera connects but no video
- Error about codec or format

**Solutions:**
```python
# Try different video backends
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
# or
cap = cv2.VideoCapture(url, cv2.CAP_GSTREAMER)
```

#### 4. Poor Video Quality

**Symptoms:**
- Blurry or pixelated video
- Low frame rate
- Delayed video

**Solutions:**
1. **Check network bandwidth**
2. **Reduce video resolution** in camera settings
3. **Use wired connection** instead of WiFi
4. **Adjust camera quality** settings

### Diagnostic Tools

#### Camera Connection Test
```python
# Test camera connectivity
python scripts/test_camera_connection.py --url http://192.168.1.100:8080/video
```

#### Network Diagnostics
```bash
# Test network connectivity
ping 192.168.1.100

# Test port connectivity  
telnet 192.168.1.100 8080

# Check network speed
speedtest-cli
```

#### Camera Information
```python
# Get camera capabilities
import cv2
cap = cv2.VideoCapture(0)
print(f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
cap.release()
```

## 📊 Performance Optimization

### Camera Performance Tips

1. **Use appropriate resolution**: Don't use 4K if 720p is sufficient
2. **Optimize frame rate**: 15-30 FPS is usually enough for detection
3. **Enable hardware acceleration**: Use GPU decoding when available
4. **Buffer management**: Adjust buffer size for smooth playback

### System Resource Management

**Memory Usage:**
- **720p camera**: ~50MB RAM per camera
- **1080p camera**: ~100MB RAM per camera
- **4K camera**: ~200MB RAM per camera

**CPU Usage:**
- **Software decoding**: 10-20% CPU per camera
- **Hardware decoding**: 2-5% CPU per camera

## 🔧 Configuration Examples

### DroidCam Configuration
```yaml
cameras:
  droidcam:
    name: "Phone Camera"
    type: "ip"
    ip: "192.168.100.101"
    port: 4747
    url_suffix: "video"
    protocol: "http"
    active: true
```

### Multiple IP Cameras
```yaml
cameras:
  front_door:
    name: "Front Door"
    type: "ip"
    ip: "192.168.1.100"
    port: 8080
    url_suffix: "video"
    protocol: "http"
    active: true
    
  back_yard:
    name: "Back Yard"
    type: "ip"
    ip: "192.168.1.101"
    port: 8080
    url_suffix: "stream"
    protocol: "http"
    active: true
```

### Local Camera with Backup
```yaml
cameras:
  primary:
    name: "USB Webcam"
    type: "local"
    index: 0
    active: true
    
  backup:
    name: "Built-in Camera"
    type: "local"
    index: 1
    active: false
```

## 📞 Support

For camera-specific issues:
1. **Check camera manual** for correct URL format
2. **Contact camera manufacturer** for technical support
3. **Test with VLC player** to verify stream compatibility
4. **Check system logs** in `logs/detection_system.log`

---

## 🔗 Related Documentation

- **Installation Guide**: `INSTALLATION.md`
- **API Documentation**: `docs/API.md`
- **Development Guide**: `docs/DEVELOPMENT.md`
- **Telegram Setup**: `docs/TELEGRAM_SETUP.md`
