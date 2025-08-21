"""
Multi-Camera Management System

This module provides simultaneous handling of multiple IP cameras for comprehensive
surveillance coverage. It extends the single camera system to support parallel
processing of multiple camera feeds.

Features:
- Simultaneous multi-camera processing
- Resource management and load balancing
- Unified detection pipeline
- Performance monitoring per camera
- Automatic failover and recovery
- Configurable camera priorities
"""

import cv2
import numpy as np
import time
import threading
import queue
import logging
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import psutil

from core.camera_manager import CameraManager
from core.model_optimization import OptimizedDetectionEngine

# Configure logging
logger = logging.getLogger(__name__)


class CameraStatus(Enum):
    """Camera status enumeration."""
    INACTIVE = "inactive"
    CONNECTING = "connecting"
    ACTIVE = "active"
    ERROR = "error"
    RECOVERING = "recovering"


@dataclass
class CameraConfig:
    """Camera configuration data structure."""
    camera_id: str
    name: str
    ip_address: str
    port: int
    protocol: str
    video_suffix: str
    priority: int = 1  # Higher number = higher priority
    enabled: bool = True
    max_fps: int = 30
    resolution: Tuple[int, int] = (640, 480)


@dataclass
class CameraFrame:
    """Camera frame data structure."""
    camera_id: str
    frame: np.ndarray
    timestamp: float
    frame_number: int
    processing_time: float = 0.0


@dataclass
class DetectionResult:
    """Detection result data structure."""
    camera_id: str
    camera_name: str
    detections: Dict
    frame_info: Dict
    timestamp: float


class ResourceMonitor:
    """System resource monitoring for multi-camera operations."""

    def __init__(self):
        self.cpu_threshold = 80.0  # Percentage
        self.memory_threshold = 80.0  # Percentage
        self.gpu_threshold = 80.0  # Percentage

    def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # GPU monitoring (if available)
            gpu_percent = 0.0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_percent = gpus[0].load * 100
            except ImportError:
                pass

            return {
                'cpu': cpu_percent,
                'memory': memory_percent,
                'gpu': gpu_percent
            }
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {'cpu': 0.0, 'memory': 0.0, 'gpu': 0.0}

    def should_reduce_load(self) -> bool:
        """Check if system load should be reduced."""
        resources = self.get_system_resources()
        return (resources['cpu'] > self.cpu_threshold or
                resources['memory'] > self.memory_threshold or
                resources['gpu'] > self.gpu_threshold)


class MultiCameraManager:
    """
    Advanced multi-camera management system.

    Features:
    - Simultaneous camera processing
    - Resource management
    - Load balancing
    - Performance monitoring
    - Automatic recovery
    """

    def __init__(self, max_cameras: int = 4, max_workers: int = None):
        """
        Initialize the multi-camera manager.

        Args:
            max_cameras: Maximum number of simultaneous cameras
            max_workers: Maximum number of worker threads
        """
        self.max_cameras = max_cameras
        self.max_workers = max_workers or min(max_cameras, 4)

        # Camera management
        self.cameras: Dict[str, CameraManager] = {}
        self.camera_configs: Dict[str, CameraConfig] = {}
        self.camera_status: Dict[str, CameraStatus] = {}
        self.camera_threads: Dict[str, threading.Thread] = {}
        self.camera_queues: Dict[str, queue.Queue] = {}

        # Detection engine
        self.detection_engine = OptimizedDetectionEngine()

        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.running = False
        self.frame_processors = {}

        # Performance monitoring
        self.performance_stats = {
            'total_frames_processed': 0,
            'total_detections': 0,
            'average_fps': 0.0,
            'resource_usage': {},
            'camera_stats': {}
        }

        # Resource management
        self.resource_monitor = ResourceMonitor()

        # Synchronization
        self.lock = threading.RLock()
        self.stop_event = threading.Event()

        logger.info(f"Multi-camera manager initialized: max_cameras={max_cameras}, max_workers={max_workers}")

    def add_camera(self, config: CameraConfig) -> bool:
        """
        Add a camera to the multi-camera system.

        Args:
            config: Camera configuration

        Returns:
            True if camera was added successfully
        """
        with self.lock:
            if len(self.cameras) >= self.max_cameras:
                logger.warning(f"Maximum camera limit reached ({self.max_cameras})")
                return False

            if config.camera_id in self.cameras:
                logger.warning(f"Camera {config.camera_id} already exists")
                return False

            try:
                # Create camera manager
                camera_manager = CameraManager()

                # Configure camera
                camera_config = {
                    'ip_address': config.ip_address,
                    'port': config.port,
                    'protocol': config.protocol,
                    'video_suffix': config.video_suffix,
                    'name': config.name
                }

                # Test connection
                if camera_manager.test_camera_connection(camera_config):
                    self.cameras[config.camera_id] = camera_manager
                    self.camera_configs[config.camera_id] = config
                    self.camera_status[config.camera_id] = CameraStatus.INACTIVE
                    self.camera_queues[config.camera_id] = queue.Queue(maxsize=10)

                    # Initialize performance stats
                    self.performance_stats['camera_stats'][config.camera_id] = {
                        'frames_processed': 0,
                        'detections_count': 0,
                        'average_fps': 0.0,
                        'last_frame_time': 0.0,
                        'connection_uptime': 0.0
                    }

                    logger.info(f"Camera {config.camera_id} ({config.name}) added successfully")
                    return True
                else:
                    logger.error(f"Failed to connect to camera {config.camera_id}")
                    return False

            except Exception as e:
                logger.error(f"Error adding camera {config.camera_id}: {e}")
                return False

    def remove_camera(self, camera_id: str) -> bool:
        """
        Remove a camera from the multi-camera system.

        Args:
            camera_id: Camera ID to remove

        Returns:
            True if camera was removed successfully
        """
        with self.lock:
            if camera_id not in self.cameras:
                logger.warning(f"Camera {camera_id} not found")
                return False

            try:
                # Stop camera if running
                if self.camera_status[camera_id] == CameraStatus.ACTIVE:
                    self.stop_camera(camera_id)

                # Clean up resources
                del self.cameras[camera_id]
                del self.camera_configs[camera_id]
                del self.camera_status[camera_id]
                del self.camera_queues[camera_id]

                if camera_id in self.performance_stats['camera_stats']:
                    del self.performance_stats['camera_stats'][camera_id]

                logger.info(f"Camera {camera_id} removed successfully")
                return True

            except Exception as e:
                logger.error(f"Error removing camera {camera_id}: {e}")
                return False

    def start_camera(self, camera_id: str) -> bool:
        """
        Start a specific camera.

        Args:
            camera_id: Camera ID to start

        Returns:
            True if camera started successfully
        """
        with self.lock:
            if camera_id not in self.cameras:
                logger.error(f"Camera {camera_id} not found")
                return False

            if self.camera_status[camera_id] == CameraStatus.ACTIVE:
                logger.warning(f"Camera {camera_id} is already active")
                return True

            try:
                # Check resource availability
                if self.resource_monitor.should_reduce_load():
                    logger.warning(f"System resources high, delaying camera {camera_id} start")
                    return False

                # Update status
                self.camera_status[camera_id] = CameraStatus.CONNECTING

                # Start camera thread
                camera_thread = threading.Thread(
                    target=self._camera_worker,
                    args=(camera_id,),
                    daemon=True
                )
                camera_thread.start()
                self.camera_threads[camera_id] = camera_thread

                # Wait for connection
                time.sleep(1)

                if self.camera_status[camera_id] == CameraStatus.ACTIVE:
                    logger.info(f"Camera {camera_id} started successfully")
                    return True
                else:
                    logger.error(f"Failed to start camera {camera_id}")
                    return False

            except Exception as e:
                logger.error(f"Error starting camera {camera_id}: {e}")
                self.camera_status[camera_id] = CameraStatus.ERROR
                return False

    def stop_camera(self, camera_id: str) -> bool:
        """
        Stop a specific camera.

        Args:
            camera_id: Camera ID to stop

        Returns:
            True if camera stopped successfully
        """
        with self.lock:
            if camera_id not in self.cameras:
                logger.error(f"Camera {camera_id} not found")
                return False

            try:
                # Update status
                self.camera_status[camera_id] = CameraStatus.INACTIVE

                # Stop camera thread
                if camera_id in self.camera_threads:
                    thread = self.camera_threads[camera_id]
                    if thread.is_alive():
                        # Thread will stop when it checks the status
                        thread.join(timeout=5)
                    del self.camera_threads[camera_id]

                # Clear queue
                if camera_id in self.camera_queues:
                    try:
                        while not self.camera_queues[camera_id].empty():
                            self.camera_queues[camera_id].get_nowait()
                    except queue.Empty:
                        pass

                logger.info(f"Camera {camera_id} stopped successfully")
                return True

            except Exception as e:
                logger.error(f"Error stopping camera {camera_id}: {e}")
                return False

    def start_all_cameras(self) -> int:
        """
        Start all configured cameras.

        Returns:
            Number of cameras started successfully
        """
        started_count = 0

        # Sort cameras by priority (highest first)
        sorted_cameras = sorted(
            self.camera_configs.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )

        for camera_id, config in sorted_cameras:
            if config.enabled and self.start_camera(camera_id):
                started_count += 1
                # Small delay between camera starts to avoid resource spikes
                time.sleep(0.5)

        logger.info(f"Started {started_count}/{len(self.camera_configs)} cameras")
        return started_count

    def stop_all_cameras(self) -> int:
        """
        Stop all active cameras.

        Returns:
            Number of cameras stopped successfully
        """
        stopped_count = 0

        for camera_id in list(self.cameras.keys()):
            if self.stop_camera(camera_id):
                stopped_count += 1

        logger.info(f"Stopped {stopped_count} cameras")
        return stopped_count

    def _camera_worker(self, camera_id: str):
        """
        Worker thread for individual camera processing.

        Args:
            camera_id: Camera ID to process
        """
        logger.info(f"Starting camera worker for {camera_id}")

        try:
            camera_manager = self.cameras[camera_id]
            config = self.camera_configs[camera_id]

            # Configure camera
            camera_config = {
                'ip_address': config.ip_address,
                'port': config.port,
                'protocol': config.protocol,
                'video_suffix': config.video_suffix,
                'name': config.name
            }

            # Connect to camera
            if camera_manager.connect_to_camera(camera_config):
                self.camera_status[camera_id] = CameraStatus.ACTIVE
                frame_number = 0
                start_time = time.time()

                while self.camera_status[camera_id] == CameraStatus.ACTIVE:
                    try:
                        # Check system resources
                        if self.resource_monitor.should_reduce_load():
                            time.sleep(0.1)  # Reduce frame rate under high load
                            continue

                        # Capture frame
                        frame_start = time.time()
                        frame = camera_manager.get_frame()

                        if frame is not None:
                            # Create camera frame object
                            camera_frame = CameraFrame(
                                camera_id=camera_id,
                                frame=frame,
                                timestamp=time.time(),
                                frame_number=frame_number,
                                processing_time=time.time() - frame_start
                            )

                            # Add to processing queue
                            try:
                                self.camera_queues[camera_id].put_nowait(camera_frame)
                            except queue.Full:
                                # Drop oldest frame if queue is full
                                try:
                                    self.camera_queues[camera_id].get_nowait()
                                    self.camera_queues[camera_id].put_nowait(camera_frame)
                                except queue.Empty:
                                    pass

                            frame_number += 1

                            # Update performance stats
                            self.performance_stats['camera_stats'][camera_id]['frames_processed'] += 1
                            self.performance_stats['camera_stats'][camera_id]['last_frame_time'] = time.time()

                            # Calculate FPS
                            elapsed = time.time() - start_time
                            if elapsed > 0:
                                fps = frame_number / elapsed
                                self.performance_stats['camera_stats'][camera_id]['average_fps'] = fps

                        else:
                            # No frame received, small delay
                            time.sleep(0.01)

                    except Exception as e:
                        logger.error(f"Error in camera worker {camera_id}: {e}")
                        time.sleep(0.1)

                # Disconnect camera
                camera_manager.disconnect()

            else:
                logger.error(f"Failed to connect camera {camera_id}")
                self.camera_status[camera_id] = CameraStatus.ERROR

        except Exception as e:
            logger.error(f"Camera worker {camera_id} crashed: {e}")
            self.camera_status[camera_id] = CameraStatus.ERROR

        logger.info(f"Camera worker for {camera_id} stopped")

    def start_detection_processing(self):
        """Start the unified detection processing system."""
        if self.running:
            logger.warning("Detection processing is already running")
            return

        self.running = True
        self.stop_event.clear()

        # Start detection worker threads
        for i in range(self.max_workers):
            worker_thread = threading.Thread(
                target=self._detection_worker,
                args=(f"worker_{i}",),
                daemon=True
            )
            worker_thread.start()
            self.frame_processors[f"worker_{i}"] = worker_thread

        logger.info(f"Started {self.max_workers} detection workers")

    def stop_detection_processing(self):
        """Stop the unified detection processing system."""
        if not self.running:
            return

        self.running = False
        self.stop_event.set()

        # Wait for workers to finish
        for worker_id, thread in self.frame_processors.items():
            if thread.is_alive():
                thread.join(timeout=5)

        self.frame_processors.clear()
        logger.info("Detection processing stopped")

    def _detection_worker(self, worker_id: str):
        """
        Worker thread for processing frames from all cameras.

        Args:
            worker_id: Worker thread identifier
        """
        logger.info(f"Starting detection worker {worker_id}")

        while self.running and not self.stop_event.is_set():
            try:
                # Check all camera queues for frames to process
                frame_processed = False

                for camera_id, frame_queue in self.camera_queues.items():
                    if self.camera_status[camera_id] != CameraStatus.ACTIVE:
                        continue

                    try:
                        # Get frame from queue (non-blocking)
                        camera_frame = frame_queue.get_nowait()

                        # Process frame
                        detection_result = self._process_frame(camera_frame)

                        if detection_result:
                            # Handle detection result
                            self._handle_detection_result(detection_result)

                        frame_processed = True
                        break  # Process one frame per iteration

                    except queue.Empty:
                        continue

                if not frame_processed:
                    # No frames to process, small delay
                    time.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in detection worker {worker_id}: {e}")
                time.sleep(0.1)

        logger.info(f"Detection worker {worker_id} stopped")

    def _process_frame(self, camera_frame: CameraFrame) -> Optional[DetectionResult]:
        """
        Process a single frame for detections.

        Args:
            camera_frame: Frame to process

        Returns:
            Detection result or None
        """
        try:
            # Run detection
            detections = self.detection_engine.detect_objects(camera_frame.frame)

            # Create detection result
            config = self.camera_configs[camera_frame.camera_id]

            detection_result = DetectionResult(
                camera_id=camera_frame.camera_id,
                camera_name=config.name,
                detections=detections,
                frame_info={
                    'timestamp': camera_frame.timestamp,
                    'frame_number': camera_frame.frame_number,
                    'processing_time': camera_frame.processing_time,
                    'detection_time': time.time() - camera_frame.timestamp
                },
                timestamp=time.time()
            )

            # Update performance stats
            self.performance_stats['total_frames_processed'] += 1
            total_detections = len(detections.get('humans', [])) + len(detections.get('animals', []))
            self.performance_stats['total_detections'] += total_detections
            self.performance_stats['camera_stats'][camera_frame.camera_id]['detections_count'] += total_detections

            return detection_result

        except Exception as e:
            logger.error(f"Error processing frame from {camera_frame.camera_id}: {e}")
            return None

    def _handle_detection_result(self, result: DetectionResult):
        """
        Handle a detection result.

        Args:
            result: Detection result to handle
        """
        try:
            # Log significant detections
            humans = len(result.detections.get('humans', []))
            animals = len(result.detections.get('animals', []))

            if humans > 0 or animals > 0:
                logger.info(f"Camera {result.camera_name}: {humans} humans, {animals} animals detected")

            # Here you can add additional handling:
            # - Send notifications
            # - Save to database
            # - Trigger alerts
            # - Update GUI

        except Exception as e:
            logger.error(f"Error handling detection result: {e}")

    def get_camera_status(self, camera_id: str = None) -> Dict:
        """
        Get status of cameras.

        Args:
            camera_id: Specific camera ID, or None for all cameras

        Returns:
            Camera status information
        """
        with self.lock:
            if camera_id:
                if camera_id in self.cameras:
                    return {
                        'camera_id': camera_id,
                        'name': self.camera_configs[camera_id].name,
                        'status': self.camera_status[camera_id].value,
                        'stats': self.performance_stats['camera_stats'].get(camera_id, {})
                    }
                else:
                    return {}
            else:
                status = {}
                for cid in self.cameras:
                    status[cid] = {
                        'name': self.camera_configs[cid].name,
                        'status': self.camera_status[cid].value,
                        'stats': self.performance_stats['camera_stats'].get(cid, {})
                    }
                return status

    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        with self.lock:
            # Update resource usage
            self.performance_stats['resource_usage'] = self.resource_monitor.get_system_resources()

            # Calculate overall FPS
            total_fps = 0
            active_cameras = 0

            for camera_id, stats in self.performance_stats['camera_stats'].items():
                if self.camera_status[camera_id] == CameraStatus.ACTIVE:
                    total_fps += stats.get('average_fps', 0)
                    active_cameras += 1

            if active_cameras > 0:
                self.performance_stats['average_fps'] = total_fps / active_cameras

            return self.performance_stats.copy()

    def get_active_cameras(self) -> List[str]:
        """Get list of active camera IDs."""
        with self.lock:
            return [
                camera_id for camera_id, status in self.camera_status.items()
                if status == CameraStatus.ACTIVE
            ]

    def shutdown(self):
        """Shutdown the multi-camera manager."""
        logger.info("Shutting down multi-camera manager...")

        # Stop detection processing
        self.stop_detection_processing()

        # Stop all cameras
        self.stop_all_cameras()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logger.info("Multi-camera manager shutdown complete")


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    camera_configs = [
        CameraConfig(
            camera_id="front_door",
            name="Front Door Camera",
            ip_address="192.168.1.100",
            port=8080,
            protocol="http",
            video_suffix="/video",
            priority=3
        ),
        CameraConfig(
            camera_id="backyard",
            name="Backyard Camera",
            ip_address="192.168.1.101",
            port=8080,
            protocol="http",
            video_suffix="/video",
            priority=2
        ),
        CameraConfig(
            camera_id="garage",
            name="Garage Camera",
            ip_address="192.168.1.102",
            port=8080,
            protocol="http",
            video_suffix="/video",
            priority=1
        )
    ]

    # Initialize multi-camera manager
    manager = MultiCameraManager(max_cameras=4, max_workers=2)

    try:
        # Add cameras
        for config in camera_configs:
            manager.add_camera(config)

        # Start detection processing
        manager.start_detection_processing()

        # Start cameras
        manager.start_all_cameras()

        # Run for a while
        print("Multi-camera system running... Press Ctrl+C to stop")
        while True:
            time.sleep(5)

            # Print status
            stats = manager.get_performance_stats()
            print(f"Total frames: {stats['total_frames_processed']}, "
                  f"Total detections: {stats['total_detections']}, "
                  f"Average FPS: {stats['average_fps']:.1f}")

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        manager.shutdown()