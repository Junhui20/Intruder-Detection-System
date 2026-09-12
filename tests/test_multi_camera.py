#!/usr/bin/env python3
"""
Multi-Camera System Test Script

This script tests the multi-camera management system with simulated cameras
and demonstrates the simultaneous handling capabilities.
"""

import sys
import os
import time
import threading
from pathlib import Path
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.multi_camera_manager import MultiCameraManager, CameraConfig, CameraStatus
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimulatedCamera:
    """Simulated camera for testing purposes."""

    def __init__(self, camera_id: str, width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.frame_count = 0
        self.running = False

    def generate_frame(self) -> np.ndarray:
        """Generate a test frame with moving objects."""
        # Create base frame
        frame = np.random.randint(50, 100, (self.height, self.width, 3), dtype=np.uint8)

        # Add camera ID text
        cv2.putText(frame, f"Camera: {self.camera_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Add frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add timestamp
        cv2.putText(frame, f"Time: {time.time():.1f}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add moving rectangle (simulated person)
        x = int((self.frame_count * 2) % (self.width - 100))
        y = int(self.height * 0.6)
        cv2.rectangle(frame, (x, y), (x + 80, y + 120), (0, 255, 0), -1)
        cv2.putText(frame, "Person", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add moving circle (simulated animal)
        if self.frame_count % 3 == 0:  # Animal appears every 3rd frame
            x = int((self.frame_count * 3) % (self.width - 60))
            y = int(self.height * 0.8)
            cv2.circle(frame, (x + 30, y + 30), 30, (255, 0, 0), -1)
            cv2.putText(frame, "Animal", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        self.frame_count += 1
        return frame


def create_test_cameras() -> list:
    """Create test camera configurations."""
    cameras = [
        CameraConfig(
            camera_id="front_door",
            name="Front Door Camera",
            url="http://192.168.1.100:8080/video",
            priority=3,
            enabled=True
        ),
        CameraConfig(
            camera_id="backyard",
            name="Backyard Camera",
            url="http://192.168.1.101:8080/video",
            priority=2,
            enabled=True
        ),
        CameraConfig(
            camera_id="garage",
            name="Garage Camera",
            url="http://192.168.1.102:8080/video",
            priority=1,
            enabled=True
        ),
        CameraConfig(
            camera_id="side_entrance",
            name="Side Entrance Camera",
            url="http://192.168.1.103:8080/video",
            priority=1,
            enabled=False  # Disabled for testing
        )
    ]
    return cameras


def test_camera_management():
    """Test basic camera management operations."""
    print("🧪 Testing Camera Management...")
    print("=" * 50)

    # Initialize manager
    manager = MultiCameraManager(max_cameras=4, max_workers=2)

    # Create test cameras
    cameras = create_test_cameras()

    # Test adding cameras
    print("📹 Adding cameras...")
    for camera in cameras:
        success = manager.add_camera(camera)
        status = "✅" if success else "❌"
        print(f"  {status} {camera.name} ({camera.camera_id})")

    # Test camera status
    print("\n📊 Camera Status:")
    status = manager.get_camera_status()
    for camera_id, info in status.items():
        print(f"  • {info['name']}: {info['status']}")

    # Test removing a camera
    print("\n🗑️ Removing garage camera...")
    success = manager.remove_camera("garage")
    print(f"  {'✅' if success else '❌'} Garage camera removed")

    # Final status
    print("\n📊 Final Camera Status:")
    status = manager.get_camera_status()
    for camera_id, info in status.items():
        print(f"  • {info['name']}: {info['status']}")

    manager.shutdown()
    print("\n✅ Camera management test completed!")


def test_simulated_multi_camera():
    """Test multi-camera system with simulated cameras."""
    print("\n🎬 Testing Simulated Multi-Camera System...")
    print("=" * 50)

    # Create simulated cameras
    simulated_cameras = {
        "front_door": SimulatedCamera("front_door"),
        "backyard": SimulatedCamera("backyard"),
        "garage": SimulatedCamera("garage")
    }

    # Initialize manager
    manager = MultiCameraManager(max_cameras=4, max_workers=3)

    try:
        # Note: In a real scenario, you would add actual camera configs
        # For this test, we'll simulate the multi-camera processing

        print("🚀 Starting simulated multi-camera processing...")

        # Simulate processing frames from multiple cameras
        for i in range(10):  # Process 10 frames from each camera
            print(f"\n📸 Processing frame set {i + 1}/10...")

            for camera_id, sim_camera in simulated_cameras.items():
                # Generate frame
                frame = sim_camera.generate_frame()

                # Simulate detection processing
                print(f"  • {camera_id}: Generated frame {sim_camera.frame_count}")

                # In real implementation, this would be handled by the detection workers
                # Here we just simulate the processing time
                time.sleep(0.1)

            time.sleep(0.5)  # Simulate frame rate

        print("\n✅ Simulated multi-camera test completed!")

    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    finally:
        manager.shutdown()


def test_performance_monitoring():
    """Test performance monitoring capabilities."""
    print("\n📊 Testing Performance Monitoring...")
    print("=" * 50)

    # Initialize manager
    manager = MultiCameraManager(max_cameras=2, max_workers=2)

    # Simulate some performance data
    manager.performance_stats = {
        'total_frames_processed': 1500,
        'total_detections': 45,
        'average_fps': 25.3,
        'resource_usage': {
            'cpu': 45.2,
            'memory': 62.1,
            'gpu': 38.7
        },
        'camera_stats': {
            'front_door': {
                'frames_processed': 800,
                'detections_count': 25,
                'average_fps': 26.1,
                'last_frame_time': time.time(),
                'connection_uptime': 3600.0
            },
            'backyard': {
                'frames_processed': 700,
                'detections_count': 20,
                'average_fps': 24.5,
                'last_frame_time': time.time() - 1,
                'connection_uptime': 3580.0
            }
        }
    }

    # Get performance stats
    stats = manager.get_performance_stats()

    print("📈 System Performance:")
    print(f"  • Total frames processed: {stats['total_frames_processed']}")
    print(f"  • Total detections: {stats['total_detections']}")
    print(f"  • Average FPS: {stats['average_fps']:.1f}")

    print("\n💻 Resource Usage:")
    for resource, value in stats['resource_usage'].items():
        print(f"  • {resource.upper()}: {value:.1f}%")

    print("\n📹 Camera Statistics:")
    for camera_id, camera_stats in stats['camera_stats'].items():
        print(f"  • {camera_id}:")
        print(f"    - Frames: {camera_stats['frames_processed']}")
        print(f"    - Detections: {camera_stats['detections_count']}")
        print(f"    - FPS: {camera_stats['average_fps']:.1f}")
        print(f"    - Uptime: {camera_stats['connection_uptime']:.0f}s")

    manager.shutdown()
    print("\n✅ Performance monitoring test completed!")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test multi-camera management system")
    parser.add_argument("--test", choices=["management", "simulated", "performance", "all"],
                       default="all", help="Test to run")

    args = parser.parse_args()

    print("🎯 Multi-Camera System Test Suite")
    print("=" * 50)

    if args.test in ["management", "all"]:
        test_camera_management()

    if args.test in ["simulated", "all"]:
        test_simulated_multi_camera()

    if args.test in ["performance", "all"]:
        test_performance_monitoring()

    print("\n🎉 All tests completed!")
    print("\n💡 Next steps:")
    print("1. Configure real IP cameras in your network")
    print("2. Update camera configurations with actual IP addresses")
    print("3. Test with real camera feeds")
    print("4. Monitor system performance under load")


if __name__ == "__main__":
    main()