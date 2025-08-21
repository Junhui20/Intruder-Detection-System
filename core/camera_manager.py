"""
Camera Management System

This module handles IP camera connections with HTTP/HTTPS support
and automatic fallback to local cameras as specified in requirements.
"""

import cv2
import numpy as np
import time
import requests
from typing import Dict, List, Optional, Tuple
import logging
import threading
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraManager:
    """
    Advanced camera management system with IP camera support and local fallback.
    
    Features:
    - HTTP/HTTPS protocol support
    - Flexible URL construction
    - Automatic local camera fallback
    - Connection testing and status management
    - Multiple camera support
    """
    
    def __init__(self):
        """Initialize the camera manager."""
        self.cameras = {}  # Dictionary of camera instances
        self.active_camera = None
        self.camera_configs = []  # List of camera configurations from database
        self.connection_timeout = 3  # seconds
        self.opencv_timeout = 5  # seconds for OpenCV VideoCapture
        self.fallback_enabled = True
        self.local_camera_index = 0
        self.prefer_ip_cameras = True  # Prefer IP cameras over local camera
        
        # Performance tracking
        self.camera_stats = {
            'connection_attempts': 0,
            'successful_connections': 0,
            'failed_connections': 0,
            'fallback_activations': 0,
            'frame_capture_times': []
        }

        # Health monitoring
        self.camera_health = {}
        self.health_check_interval = 30  # seconds
        self.health_monitor_thread = None
        self.health_monitoring_active = False

        # Failover settings
        self.enable_failover = True
        self.failover_attempts = 3
        self.failover_delay = 5  # seconds
        self.camera_rotation_enabled = False
        self.rotation_interval = 300  # 5 minutes
        self.rotation_thread = None

        logger.info("Advanced Camera Manager initialized")
    
    def load_camera_configs(self, camera_configs: List[Dict]):
        """
        Load camera configurations from database.

        Args:
            camera_configs: List of camera configuration dictionaries
        """
        self.camera_configs = camera_configs
        logger.info(f"Loaded {len(camera_configs)} camera configurations")

    def reload_camera_configs(self, camera_configs: List[Dict]) -> bool:
        """
        Reload camera configurations and reconnect cameras.

        Args:
            camera_configs: List of camera configuration dictionaries

        Returns:
            True if successful
        """
        try:
            logger.info("Reloading camera configurations...")

            # Release all existing cameras
            self.release_all_cameras()

            # Load new configurations
            self.load_camera_configs(camera_configs)

            # Set up cameras with new configurations
            success = self.setup_cameras()

            if success:
                logger.info(f"Successfully reloaded {len(camera_configs)} camera configurations")
            else:
                logger.warning("Camera configurations reloaded but no cameras connected")

            return success

        except Exception as e:
            logger.error(f"Failed to reload camera configurations: {e}")
            return False
    
    def setup_cameras(self) -> bool:
        """
        Set up cameras based on loaded configurations.

        Returns:
            True if at least one camera is successfully connected
        """
        success = False
        ip_camera_attempted = False

        # Try to connect to IP cameras first
        for config in self.camera_configs:
            if config['status'] == 'active':
                ip_camera_attempted = True
                camera_id = config['id']

                # Construct URL for logging
                protocol = 'https' if config.get('use_https', False) else 'http'
                video_suffix = '/video' if config.get('end_with_video', False) else ''
                camera_url = f"{protocol}://{config['ip_address']}:{config['port']}{video_suffix}"

                logger.info(f"Attempting to connect to IP camera {camera_id}: {camera_url}")

                if self._connect_ip_camera(config):
                    success = True
                    logger.info(f"[OK] Successfully connected to IP camera {camera_id}")
                else:
                    logger.warning(f"[FAILED] Failed to connect to IP camera {camera_id}: {camera_url}")

        # If no IP cameras connected and fallback is enabled, try local camera
        if not success and self.fallback_enabled:
            if ip_camera_attempted:
                logger.warning("[WARN] All IP cameras failed to connect. Attempting fallback to local camera...")

            if self._connect_local_camera():
                success = True
                self.camera_stats['fallback_activations'] += 1
                logger.info("[OK] Fallback to local camera successful")
            else:
                logger.error("[ERROR] Failed to connect to local camera")
        elif not success and ip_camera_attempted:
            logger.error("[ERROR] All IP cameras failed to connect and fallback is disabled")

        return success
    
    def _connect_ip_camera(self, config: Dict) -> bool:
        """
        Connect to an IP camera using the provided configuration.
        
        Args:
            config: Camera configuration dictionary
            
        Returns:
            True if connection successful
        """
        try:
            camera_id = config['id']
            ip_address = config['ip_address']
            port = config['port']
            use_https = config['use_https']
            end_with_video = config['end_with_video']
            
            # Construct camera URL
            protocol = 'https' if use_https else 'http'
            video_suffix = '/video' if end_with_video else ''
            camera_url = f"{protocol}://{ip_address}:{port}{video_suffix}"
            
            logger.info(f"Attempting to connect to camera: {camera_url}")

            # Check if this might be DroidCam
            is_droidcam = self._is_droidcam_url(camera_url)
            if is_droidcam:
                logger.info(f"Detected potential DroidCam URL: {camera_url}")

            # Test connection first (but don't fail immediately for DroidCam)
            connection_test_passed = self._test_camera_connection(camera_url)
            if not connection_test_passed:
                if is_droidcam:
                    logger.info(f"[INFO] DroidCam connection test failed, but attempting direct VideoCapture...")
                else:
                    logger.warning(f"[WARN] Initial connection test failed for {camera_url}, but attempting direct connection...")
                # Don't return False here - DroidCam might still work with direct VideoCapture
            
            # Create VideoCapture object with timeout handling
            logger.info(f"Creating VideoCapture for: {camera_url}")

            # Use threading to implement timeout for VideoCapture creation
            import threading
            import time

            cap = None
            exception_occurred = None

            def create_videocapture():
                nonlocal cap, exception_occurred
                try:
                    cap = cv2.VideoCapture(camera_url)
                except Exception as e:
                    exception_occurred = e

            # Start VideoCapture creation in separate thread
            thread = threading.Thread(target=create_videocapture)
            thread.daemon = True
            thread.start()

            # Wait for completion with timeout
            thread.join(timeout=self.opencv_timeout)

            if thread.is_alive():
                logger.warning(f"VideoCapture creation timed out after {self.opencv_timeout}s for: {camera_url}")
                return None

            if exception_occurred:
                logger.error(f"Exception during VideoCapture creation: {exception_occurred}")
                return None

            if cap is None:
                logger.error(f"Failed to create VideoCapture for: {camera_url}")
                return None

            # Set optimized properties for real-time processing
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
                cap.set(cv2.CAP_PROP_FPS, 30)  # Target 30 FPS

                # Additional optimizations for IP cameras (DroidCam compatible)
                if camera_url.startswith(('http', 'rtsp')):
                    # Try to set MJPEG codec (works well with DroidCam)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    # Set reasonable resolution for DroidCam
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                logger.debug(f"VideoCapture properties set for {camera_url}")
            except Exception as e:
                logger.warning(f"Failed to set some VideoCapture properties: {e}")
                # Continue anyway - basic capture might still work
            
            # Test if camera is opened and can read frames
            if cap.isOpened():
                logger.info(f"VideoCapture opened successfully for {camera_url}")

                # Try to read a frame with multiple attempts (DroidCam sometimes needs time)
                frame_read_success = False
                for attempt in range(3):
                    try:
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            logger.info(f"Successfully read frame from {camera_url} (attempt {attempt + 1})")
                            frame_read_success = True
                            break
                        else:
                            logger.warning(f"Frame read attempt {attempt + 1} failed for {camera_url}")
                            time.sleep(0.5)  # Brief pause before retry
                    except Exception as e:
                        logger.warning(f"Frame read attempt {attempt + 1} error for {camera_url}: {e}")
                        time.sleep(0.5)

                if frame_read_success:
                    self.cameras[camera_id] = {
                        'capture': cap,
                        'config': config,
                        'url': camera_url,
                        'type': 'ip',
                        'status': 'connected',
                        'last_frame_time': time.time()
                    }

                    # Set as active camera if none is set
                    if self.active_camera is None:
                        self.active_camera = camera_id

                    self.camera_stats['successful_connections'] += 1
                    logger.info(f"Camera {camera_id} successfully connected and configured")
                    return True
                else:
                    cap.release()
                    logger.warning(f"Camera {camera_url} opened but cannot read frames after multiple attempts")
            else:
                logger.warning(f"Failed to open VideoCapture for: {camera_url}")
            
            self.camera_stats['failed_connections'] += 1
            return False
            
        except Exception as e:
            logger.error(f"Error connecting to IP camera: {e}")
            self.camera_stats['failed_connections'] += 1
            return False
        finally:
            self.camera_stats['connection_attempts'] += 1
    
    def _connect_local_camera(self) -> bool:
        """
        Connect to local camera as fallback.
        
        Returns:
            True if connection successful
        """
        try:
            logger.info(f"Attempting to connect to local camera (index {self.local_camera_index})")
            
            cap = cv2.VideoCapture(self.local_camera_index)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    camera_id = 'local_0'
                    self.cameras[camera_id] = {
                        'capture': cap,
                        'config': {
                            'id': camera_id,
                            'type': 'local',
                            'index': self.local_camera_index
                        },
                        'url': f'local_camera_{self.local_camera_index}',
                        'type': 'local',
                        'status': 'connected',
                        'last_frame_time': time.time()
                    }
                    
                    # Set as active camera
                    self.active_camera = camera_id
                    self.camera_stats['successful_connections'] += 1
                    return True
                else:
                    cap.release()
                    logger.warning("Local camera opened but cannot read frames")
            else:
                logger.warning("Failed to open local camera")
            
            self.camera_stats['failed_connections'] += 1
            return False
            
        except Exception as e:
            logger.error(f"Error connecting to local camera: {e}")
            self.camera_stats['failed_connections'] += 1
            return False
        finally:
            self.camera_stats['connection_attempts'] += 1

    def _is_droidcam_url(self, camera_url: str) -> bool:
        """
        Check if the URL might be a DroidCam URL.

        Args:
            camera_url: Camera URL to check

        Returns:
            True if likely DroidCam
        """
        # Common DroidCam patterns
        droidcam_indicators = [
            ':4747',  # Default DroidCam port
            'droidcam',
            '/video',  # DroidCam video endpoint
        ]

        url_lower = camera_url.lower()
        return any(indicator in url_lower for indicator in droidcam_indicators)

    def _test_camera_connection(self, camera_url: str) -> bool:
        """
        Test if camera URL is accessible.

        Args:
            camera_url: Camera URL to test

        Returns:
            True if accessible
        """
        try:
            # For HTTP/HTTPS cameras, try a simple HEAD request first
            if camera_url.startswith(('http://', 'https://')):
                logger.debug(f"Testing HTTP connection to {camera_url}")

                try:
                    response = requests.head(camera_url, timeout=self.connection_timeout)
                    if response.status_code < 400:
                        logger.debug(f"[OK] HTTP HEAD request successful: {response.status_code}")
                        return True
                    else:
                        logger.debug(f"[FAILED] HTTP HEAD request failed: {response.status_code}")

                        # Try GET request as fallback
                        logger.debug(f"Trying GET request as fallback...")
                        response = requests.get(camera_url, timeout=self.connection_timeout, stream=True)
                        if response.status_code < 400:
                            logger.debug(f"[OK] HTTP GET request successful: {response.status_code}")
                            return True
                        else:
                            logger.debug(f"[FAILED] HTTP GET request also failed: {response.status_code}")
                            return False

                except requests.exceptions.ConnectTimeout:
                    logger.debug(f"[TIMEOUT] Connection timeout for {camera_url}")
                    return False
                except requests.exceptions.ConnectionError as e:
                    logger.debug(f"[ERROR] Connection error for {camera_url}: {e}")
                    return False
                except requests.exceptions.RequestException as e:
                    logger.debug(f"[ERROR] Request error for {camera_url}: {e}")
                    return False

            return True  # For other protocols, assume accessible

        except Exception as e:
            logger.debug(f"[ERROR] Unexpected error testing {camera_url}: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the active camera.
        
        Returns:
            Frame as numpy array or None if failed
        """
        if not self.active_camera or self.active_camera not in self.cameras:
            logger.warning("No active camera available")
            return None
        
        start_time = time.time()
        
        try:
            camera_info = self.cameras[self.active_camera]
            cap = camera_info['capture']
            
            ret, frame = cap.read()
            
            if ret and frame is not None:
                camera_info['last_frame_time'] = time.time()
                
                # Update performance stats
                capture_time = time.time() - start_time
                self.camera_stats['frame_capture_times'].append(capture_time)
                
                # Keep only last 100 capture times
                if len(self.camera_stats['frame_capture_times']) > 100:
                    self.camera_stats['frame_capture_times'] = self.camera_stats['frame_capture_times'][-100:]
                
                return frame
            else:
                logger.warning(f"Failed to capture frame from camera {self.active_camera}")
                # Try to reconnect or switch to fallback
                self._handle_camera_failure()
                return None
                
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            self._handle_camera_failure()
            return None
    
    def _handle_camera_failure(self):
        """Handle camera connection failure and attempt recovery."""
        if self.active_camera in self.cameras:
            camera_info = self.cameras[self.active_camera]
            camera_info['status'] = 'disconnected'
            
            # Try to reconnect
            if camera_info['type'] == 'ip':
                logger.info(f"Attempting to reconnect to IP camera {self.active_camera}")
                if not self._reconnect_ip_camera(camera_info):
                    # If reconnection fails, try fallback
                    if self.fallback_enabled:
                        logger.info("IP camera reconnection failed, trying local camera fallback")
                        self._connect_local_camera()
            else:
                # For local camera, try to reinitialize
                logger.info("Attempting to reinitialize local camera")
                self._connect_local_camera()
    
    def _reconnect_ip_camera(self, camera_info: Dict) -> bool:
        """Attempt to reconnect to an IP camera."""
        try:
            # Release old capture
            if 'capture' in camera_info:
                camera_info['capture'].release()
            
            # Try to reconnect
            return self._connect_ip_camera(camera_info['config'])
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False
    
    def switch_camera(self, camera_id: str) -> bool:
        """
        Switch to a different camera.
        
        Args:
            camera_id: ID of the camera to switch to
            
        Returns:
            True if switch successful
        """
        if camera_id in self.cameras and self.cameras[camera_id]['status'] == 'connected':
            self.active_camera = camera_id
            logger.info(f"Switched to camera {camera_id}")
            return True
        else:
            logger.warning(f"Cannot switch to camera {camera_id} - not available")
            return False
    
    def get_camera_status(self) -> Dict:
        """Get status of all cameras."""
        status = {
            'active_camera': self.active_camera,
            'total_cameras': len(self.cameras),
            'connected_cameras': sum(1 for cam in self.cameras.values() if cam['status'] == 'connected'),
            'cameras': {}
        }
        
        for camera_id, camera_info in self.cameras.items():
            status['cameras'][camera_id] = {
                'type': camera_info['type'],
                'status': camera_info['status'],
                'url': camera_info['url'],
                'last_frame_time': camera_info.get('last_frame_time', 0)
            }
        
        return status
    
    def get_performance_stats(self) -> Dict:
        """Get camera performance statistics."""
        capture_times = self.camera_stats['frame_capture_times']
        
        stats = {
            'connection_attempts': self.camera_stats['connection_attempts'],
            'successful_connections': self.camera_stats['successful_connections'],
            'failed_connections': self.camera_stats['failed_connections'],
            'fallback_activations': self.camera_stats['fallback_activations'],
            'connection_success_rate': (self.camera_stats['successful_connections'] / 
                                      max(1, self.camera_stats['connection_attempts']) * 100)
        }
        
        if capture_times:
            stats.update({
                'avg_capture_time': np.mean(capture_times),
                'max_capture_time': np.max(capture_times),
                'min_capture_time': np.min(capture_times),
                'estimated_fps': 1.0 / np.mean(capture_times[-10:]) if len(capture_times) >= 10 else 0
            })
        
        return stats
    
    def release_all_cameras(self):
        """Release all camera resources."""
        for camera_id, camera_info in self.cameras.items():
            try:
                if 'capture' in camera_info:
                    camera_info['capture'].release()
                logger.info(f"Released camera {camera_id}")
            except Exception as e:
                logger.error(f"Error releasing camera {camera_id}: {e}")
        
        self.cameras.clear()
        self.active_camera = None
        logger.info("All cameras released")
    
    def test_camera_connection(self, config: Dict) -> bool:
        """
        Test connection to a camera without adding it to active cameras.

        Args:
            config: Camera configuration to test

        Returns:
            True if connection successful
        """
        try:
            if config.get('type') == 'local':
                # Test local camera
                logger.info(f"Testing local camera at index {config.get('index', 0)}")
                cap = cv2.VideoCapture(config.get('index', 0))
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    result = ret and frame is not None
                    logger.info(f"Local camera test result: {result}")
                    return result
                logger.info("Local camera test failed: could not open camera")
                return False
            else:
                # Test IP camera
                ip_address = config['ip_address']
                port = config['port']
                use_https = config.get('use_https', False)
                end_with_video = config.get('end_with_video', False)

                protocol = 'https' if use_https else 'http'
                video_suffix = '/video' if end_with_video else ''
                camera_url = f"{protocol}://{ip_address}:{port}{video_suffix}"

                logger.info(f"Testing IP camera connection to: {camera_url}")
                result = self._test_camera_connection(camera_url)
                logger.info(f"IP camera test result: {result}")
                return result

        except Exception as e:
            logger.error(f"Camera connection test failed: {e}")
            return False

    def set_fallback_enabled(self, enabled: bool):
        """Enable or disable fallback to local camera."""
        self.fallback_enabled = enabled
        logger.info(f"Camera fallback {'enabled' if enabled else 'disabled'}")

    def force_ip_camera_only(self, ip_only: bool = True):
        """Force system to use only IP cameras (disable local camera fallback)."""
        self.fallback_enabled = not ip_only
        logger.info(f"IP camera only mode: {'enabled' if ip_only else 'disabled'}")

    def start_health_monitoring(self):
        """Start camera health monitoring."""
        if not self.health_monitoring_active:
            self.health_monitoring_active = True
            self.health_monitor_thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
            self.health_monitor_thread.start()
            logger.info("Camera health monitoring started")

    def stop_health_monitoring(self):
        """Stop camera health monitoring."""
        self.health_monitoring_active = False
        if self.health_monitor_thread:
            self.health_monitor_thread.join(timeout=5)
        logger.info("Camera health monitoring stopped")

    def _health_monitor_loop(self):
        """Health monitoring loop."""
        while self.health_monitoring_active:
            try:
                self._check_camera_health()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                time.sleep(10)

    def _check_camera_health(self):
        """Check health of all cameras."""
        for camera_id, camera in self.cameras.items():
            try:
                # Test camera connection
                if camera and hasattr(camera, 'isOpened') and camera.isOpened():
                    # Try to read a frame
                    ret, frame = camera.read()
                    if ret and frame is not None:
                        self.camera_health[camera_id] = {
                            'status': 'healthy',
                            'last_check': time.time(),
                            'consecutive_failures': 0
                        }
                    else:
                        self._handle_camera_failure(camera_id)
                else:
                    self._handle_camera_failure(camera_id)

            except Exception as e:
                logger.error(f"Health check failed for camera {camera_id}: {e}")
                self._handle_camera_failure(camera_id)

    def _handle_camera_failure(self, camera_id):
        """Handle camera failure."""
        if camera_id not in self.camera_health:
            self.camera_health[camera_id] = {'consecutive_failures': 0}

        self.camera_health[camera_id]['consecutive_failures'] += 1
        self.camera_health[camera_id]['status'] = 'unhealthy'
        self.camera_health[camera_id]['last_check'] = time.time()

        failures = self.camera_health[camera_id]['consecutive_failures']
        logger.warning(f"Camera {camera_id} health check failed ({failures} consecutive failures)")

        # Attempt failover if enabled
        if self.enable_failover and failures >= 3:
            self._attempt_camera_failover(camera_id)

    def _attempt_camera_failover(self, failed_camera_id):
        """Attempt to failover to another camera."""
        logger.info(f"Attempting failover from camera {failed_camera_id}")

        # Find healthy cameras
        healthy_cameras = []
        for camera_id, health in self.camera_health.items():
            if camera_id != failed_camera_id and health.get('status') == 'healthy':
                healthy_cameras.append(camera_id)

        if healthy_cameras:
            # Switch to first healthy camera
            new_camera_id = healthy_cameras[0]
            logger.info(f"Failing over to camera {new_camera_id}")
            self.switch_camera(new_camera_id)
        else:
            logger.warning("No healthy cameras available for failover")

    def switch_camera(self, camera_id):
        """Switch to a different camera."""
        try:
            if camera_id in self.cameras:
                # Disconnect current camera
                if self.active_camera:
                    self.disconnect()

                # Connect to new camera
                self.active_camera = self.cameras[camera_id]
                logger.info(f"Switched to camera {camera_id}")
                return True
            else:
                logger.error(f"Camera {camera_id} not found")
                return False

        except Exception as e:
            logger.error(f"Error switching camera: {e}")
            return False

    def get_camera_health_status(self):
        """Get health status of all cameras."""
        return self.camera_health.copy()

    def get_available_cameras(self):
        """Get list of available camera IDs."""
        return list(self.cameras.keys())

    def is_camera_healthy(self, camera_id):
        """Check if a specific camera is healthy."""
        return (camera_id in self.camera_health and
                self.camera_health[camera_id].get('status') == 'healthy')
