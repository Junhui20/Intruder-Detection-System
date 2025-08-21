"""
Camera Configuration Management

This module handles IP camera configurations and connection settings.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """
    IP Camera configuration with HTTP/HTTPS support.
    
    Features:
    - Flexible URL construction
    - Protocol support (HTTP/HTTPS)
    - Connection testing
    - Fallback configuration
    """
    
    # Camera identification
    id: Optional[int] = None
    name: str = ""
    
    # Network settings
    ip_address: str = "192.168.1.100"
    port: int = 8080
    use_https: bool = False
    
    # URL configuration
    url_suffix: str = "/video"
    end_with_video: bool = True
    custom_url: str = ""  # Override automatic URL construction
    
    # Authentication (if needed)
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
        """
        Construct the camera URL based on configuration.
        
        Returns:
            Complete camera URL
        """
        if self.custom_url:
            return self.custom_url
        
        protocol = "https" if self.use_https else "http"
        
        # Handle authentication
        auth_part = ""
        if self.username and self.password:
            auth_part = f"{self.username}:{self.password}@"
        
        # Construct base URL
        base_url = f"{protocol}://{auth_part}{self.ip_address}:{self.port}"
        
        # Add suffix
        if self.end_with_video and self.url_suffix:
            if not self.url_suffix.startswith('/'):
                self.url_suffix = '/' + self.url_suffix
            return base_url + self.url_suffix
        elif self.url_suffix:
            if not self.url_suffix.startswith('/'):
                self.url_suffix = '/' + self.url_suffix
            return base_url + self.url_suffix
        else:
            return base_url
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test camera connection.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            import requests
            import cv2
            
            url = self.get_camera_url()
            
            # Test HTTP/HTTPS connection first
            try:
                response = requests.head(url, timeout=self.connection_timeout)
                if response.status_code >= 400:
                    return False, f"HTTP error: {response.status_code}"
            except requests.exceptions.RequestException as e:
                return False, f"Network error: {e}"
            
            # Test video capture
            try:
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    return False, "Failed to open video stream"
                
                ret, frame = cap.read()
                cap.release()
                
                if not ret or frame is None:
                    return False, "Failed to capture frame"
                
                return True, f"Connection successful ({frame.shape[1]}x{frame.shape[0]})"
                
            except Exception as e:
                return False, f"Video capture error: {e}"
                
        except Exception as e:
            return False, f"Test failed: {e}"
    
    def validate(self) -> Dict[str, str]:
        """
        Validate camera configuration.
        
        Returns:
            Dictionary of validation errors
        """
        errors = {}
        
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
        """Convert to dictionary for database storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CameraConfig':
        """Create instance from dictionary."""
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation."""
        return f"CameraConfig({self.name or self.ip_address}:{self.port})"


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
                config = CameraConfig(
                    id=device.id,
                    name=f"Camera {device.id}",
                    ip_address=device.ip_address,
                    port=device.port,
                    use_https=device.use_https,
                    end_with_video=device.end_with_video,
                    status=device.status
                )
                
                self.cameras[device.id] = config
            
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
                device = Device(
                    id=config.id,
                    ip_address=config.ip_address,
                    port=config.port,
                    use_https=config.use_https,
                    end_with_video=config.end_with_video,
                    status=config.status
                )
                
                if config.id and db_manager.get_device(config.id):
                    db_manager.update_device(device)
                else:
                    db_manager.create_device(device)
            
            logger.info(f"Saved {len(self.cameras)} cameras to database")
            return True
            
        except Exception as e:
            logger.error(f"Error saving cameras to database: {e}")
            return False
