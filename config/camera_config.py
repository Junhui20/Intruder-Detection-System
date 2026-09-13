"""
Camera Configuration Management

A camera is a stream URL. HTTP/HTTPS MJPEG (DroidCam, IP Webcam) and RTSP
(every mainstream IP camera) are opened the same way; the URL decides.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import quote, urlsplit

logger = logging.getLogger(__name__)

DEFAULT_PORTS = {"http": 8080, "https": 8080, "rtsp": 554, "usb": 0}


def build_camera_url(
    protocol: str, host: str, port: int = 0, path: str = "", username: str = "", password: str = ""
) -> str:
    """
    Assemble a stream URL from form fields.

    Args:
        protocol: ``http``, ``https``, ``rtsp``, or ``usb`` for a webcam on this machine.
        host: IP address or hostname; the device index for ``usb``.
        port: 0 picks the protocol default (8080 for DroidCam-style HTTP, 554 for RTSP).
        path: Stream path, e.g. ``/video`` (DroidCam) or ``/stream1`` (Tapo).
        username: Optional; credentials are URL-encoded.
        password: Optional.

    Returns:
        The URL, e.g. ``rtsp://admin:pw@192.168.1.20:554/stream1``.
    """
    if protocol == "usb":  # a webcam plugged into this machine: usb:0, usb:1 ...
        return f"usb:{host or 0}"
    auth = f"{quote(username, safe='')}:{quote(password, safe='')}@" if username else ""
    port = port or DEFAULT_PORTS[protocol]
    path = "/" + path.lstrip("/") if path else ""
    return f"{protocol}://{auth}{host}:{port}{path}"


@dataclass
class CameraConfig:
    """
    One camera. ``url`` is what gets opened; the other network fields only
    exist to build it when a form supplied parts instead of a URL.
    """
    
    # Camera identification
    id: Optional[int] = None
    name: str = ""
    
    # Stream URL — the source of truth when set
    url: str = ""

    # Parts used to build the URL when ``url`` is empty
    ip_address: str = "192.168.1.100"
    port: int = 8080
    protocol: str = "http"  # http, https or rtsp
    url_suffix: str = "/video"
    username: str = ""
    password: str = ""
    
    # Connection settings
    connection_timeout: int = 3  # seconds
    read_timeout: int = 10  # seconds
    max_retries: int = 3
    
    # Status and management
    status: str = "active"  # active, inactive, error
    auto_connect: bool = True
    priority: int = 1  # Connection priority (1 = highest)
    
    # Fallback settings
    enable_fallback: bool = True
    fallback_camera_index: int = 0  # Local camera index
    
    # Video settings
    preferred_resolution: Tuple[int, int] = (640, 480)
    preferred_fps: int = 30
    buffer_size: int = 1
    
    def get_camera_url(self) -> str:
        """The URL to open: ``url`` as given, else built from the parts."""
        return self.url or build_camera_url(
            self.protocol, self.ip_address, self.port, self.url_suffix, self.username, self.password
        )

    def test_connection(self) -> Tuple[bool, str]:
        """
        Open the stream and read one frame.

        Returns:
            (success, message); the message carries the frame size or the reason.
        """
        from core.camera_manager import open_stream

        url = self.get_camera_url()
        if url.startswith("http"):  # cheap reachability check before FFmpeg's slow failure
            import requests

            try:
                requests.head(url, timeout=self.connection_timeout)
            except requests.exceptions.RequestException as e:
                return False, f"Network error: {e}"
        try:
            cap = open_stream(url)
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            cap.release()
        except Exception as e:
            return False, f"Video capture error: {e}"
        if not ok or frame is None:
            return False, "Could not read a frame"
        return True, f"Connection successful ({frame.shape[1]}x{frame.shape[0]})"
    
    def validate(self) -> Dict[str, str]:
        """
        Validate camera configuration.
        
        Returns:
            Dictionary of validation errors
        """
        errors = {}
        
        if self.url:
            parts = urlsplit(self.url)
            if parts.scheme == "usb":
                if not self.url[4:].isdigit():
                    errors['url'] = "USB cameras are usb:0, usb:1 ..."
            elif parts.scheme not in DEFAULT_PORTS or not parts.hostname:
                errors['url'] = "URL must be http(s)://host[:port]/path, rtsp://host[:port]/path or usb:0"
            return errors

        if self.protocol not in DEFAULT_PORTS:
            errors['protocol'] = "Protocol must be http, https or rtsp"

        # Validate IP address
        if not self.ip_address:
            errors['ip_address'] = "IP address is required"
        else:
            # Basic IP validation
            parts = self.ip_address.split('.')
            if len(parts) != 4:
                errors['ip_address'] = "Invalid IP address format"
            else:
                try:
                    for part in parts:
                        num = int(part)
                        if not 0 <= num <= 255:
                            errors['ip_address'] = "IP address octets must be 0-255"
                            break
                except ValueError:
                    errors['ip_address'] = "IP address must contain only numbers and dots"
        
        # Validate port
        if not 1 <= self.port <= 65535:
            errors['port'] = "Port must be between 1 and 65535"
        
        # Validate timeouts
        if self.connection_timeout <= 0:
            errors['connection_timeout'] = "Connection timeout must be positive"
        
        if self.read_timeout <= 0:
            errors['read_timeout'] = "Read timeout must be positive"
        
        # Validate retries
        if self.max_retries < 0:
            errors['max_retries'] = "Max retries cannot be negative"
        
        # Validate resolution
        if self.preferred_resolution[0] <= 0 or self.preferred_resolution[1] <= 0:
            errors['preferred_resolution'] = "Resolution dimensions must be positive"
        
        # Validate FPS
        if self.preferred_fps <= 0:
            errors['preferred_fps'] = "FPS must be positive"
        
        return errors
    
    def to_dict(self) -> Dict:
        """Fields plus the resolved ``url``, which is all the camera manager reads."""
        return {**asdict(self), "url": self.get_camera_url()}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CameraConfig':
        """Create instance from dictionary."""
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation."""
        return f"CameraConfig({self.name or self.get_camera_url()})"


class CameraConfigManager:
    """
    Manager for multiple camera configurations.
    
    Features:
    - Multiple camera management
    - Priority-based connection
    - Fallback handling
    - Configuration persistence
    """
    
    def __init__(self):
        """Initialize camera config manager."""
        self.cameras: Dict[int, CameraConfig] = {}
        self.active_camera_id: Optional[int] = None
        
    def add_camera(self, config: CameraConfig) -> int:
        """
        Add a camera configuration.
        
        Args:
            config: Camera configuration
            
        Returns:
            Camera ID
        """
        # Generate ID if not provided
        if config.id is None:
            config.id = max(self.cameras.keys(), default=0) + 1
        
        self.cameras[config.id] = config
        logger.info(f"Added camera {config.id}: {config}")
        
        return config.id
    
    def remove_camera(self, camera_id: int) -> bool:
        """
        Remove a camera configuration.
        
        Args:
            camera_id: Camera ID to remove
            
        Returns:
            True if removed
        """
        if camera_id in self.cameras:
            del self.cameras[camera_id]
            
            # Update active camera if removed
            if self.active_camera_id == camera_id:
                self.active_camera_id = None
            
            logger.info(f"Removed camera {camera_id}")
            return True
        
        return False
    
    def get_camera(self, camera_id: int) -> Optional[CameraConfig]:
        """Get camera configuration by ID."""
        return self.cameras.get(camera_id)
    
    def get_all_cameras(self) -> List[CameraConfig]:
        """Get all camera configurations."""
        return list(self.cameras.values())
    
    def get_active_cameras(self) -> List[CameraConfig]:
        """Get all active camera configurations."""
        return [cam for cam in self.cameras.values() if cam.status == "active"]
    
    def get_cameras_by_priority(self) -> List[CameraConfig]:
        """Get cameras sorted by priority (highest first)."""
        return sorted(self.cameras.values(), key=lambda x: x.priority)
    
    def test_all_cameras(self) -> Dict[int, Tuple[bool, str]]:
        """
        Test connection to all cameras.
        
        Returns:
            Dictionary of camera_id -> (success, message)
        """
        results = {}
        
        for camera_id, config in self.cameras.items():
            if config.status == "active":
                success, message = config.test_connection()
                results[camera_id] = (success, message)
                
                # Update status based on test result
                config.status = "active" if success else "error"
        
        return results
    
    def get_best_camera(self) -> Optional[CameraConfig]:
        """
        Get the best available camera based on priority and status.
        
        Returns:
            Best camera configuration or None
        """
        active_cameras = [cam for cam in self.cameras.values() 
                         if cam.status == "active" and cam.auto_connect]
        
        if not active_cameras:
            return None
        
        # Sort by priority (highest first)
        active_cameras.sort(key=lambda x: x.priority)
        return active_cameras[0]
    
    def set_active_camera(self, camera_id: int) -> bool:
        """
        Set the active camera.
        
        Args:
            camera_id: Camera ID to activate
            
        Returns:
            True if successful
        """
        if camera_id in self.cameras:
            self.active_camera_id = camera_id
            logger.info(f"Set active camera to {camera_id}")
            return True
        
        return False
    
    def get_active_camera(self) -> Optional[CameraConfig]:
        """Get the currently active camera configuration."""
        if self.active_camera_id:
            return self.cameras.get(self.active_camera_id)
        return None
    
    def load_from_database(self, db_manager) -> bool:
        """
        Load camera configurations from database.
        
        Args:
            db_manager: DatabaseManager instance
            
        Returns:
            True if successful
        """
        try:
            devices = db_manager.get_all_devices()
            
            self.cameras.clear()
            
            for device in devices:
                self.cameras[device.id] = CameraConfig(
                    id=device.id, name=f"Camera {device.id}", url=device.url, status=device.status
                )
            
            logger.info(f"Loaded {len(self.cameras)} cameras from database")
            return True
            
        except Exception as e:
            logger.error(f"Error loading cameras from database: {e}")
            return False
    
    def save_to_database(self, db_manager) -> bool:
        """
        Save camera configurations to database.
        
        Args:
            db_manager: DatabaseManager instance
            
        Returns:
            True if successful
        """
        try:
            from database.models import Device
            
            for config in self.cameras.values():
                device = Device(id=config.id, url=config.get_camera_url(), status=config.status)
                
                if config.id and db_manager.get_device(config.id):
                    db_manager.update_device(device)
                else:
                    db_manager.create_device(device)
            
            logger.info(f"Saved {len(self.cameras)} cameras to database")
            return True
            
        except Exception as e:
            logger.error(f"Error saving cameras to database: {e}")
            return False
