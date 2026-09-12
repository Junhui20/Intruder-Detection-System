"""Camera URL handling: RTSP and HTTP MJPEG go through the same path."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).parent.parent))

import cv2

from config.camera_config import CameraConfig, build_camera_url
from core.camera_manager import CameraManager, open_stream


class TestCameraUrls(unittest.TestCase):
    def test_builder_covers_droidcam_and_rtsp(self):
        self.assertEqual(
            build_camera_url("http", "192.168.1.5", 4747, "video"),
            "http://192.168.1.5:4747/video",
        )
        self.assertEqual(
            build_camera_url("rtsp", "192.168.1.20", path="/stream1"),
            "rtsp://192.168.1.20:554/stream1",
        )
        self.assertEqual(
            build_camera_url("rtsp", "cam", 554, "/h264", "admin", "p@ss/word"),
            "rtsp://admin:p%40ss%2Fword@cam:554/h264",
        )

    def test_explicit_url_wins_over_parts(self):
        config = CameraConfig(
            url="rtsp://cam:554/stream1", ip_address="1.2.3.4", port=80
        )
        self.assertEqual(config.get_camera_url(), "rtsp://cam:554/stream1")
        self.assertEqual(config.to_dict()["url"], "rtsp://cam:554/stream1")
        self.assertEqual(config.validate(), {})

    def test_validation_rejects_unknown_schemes(self):
        self.assertIn("url", CameraConfig(url="ftp://cam/stream").validate())
        self.assertIn("protocol", CameraConfig(protocol="onvif").validate())

    def test_rtsp_opens_through_ffmpeg_and_http_does_not(self):
        with mock.patch("core.camera_manager.cv2.VideoCapture") as capture:
            open_stream("rtsp://cam:554/stream1")
            capture.assert_called_once_with("rtsp://cam:554/stream1", cv2.CAP_FFMPEG)
            capture.reset_mock()
            open_stream("http://192.168.1.5:4747/video")
            capture.assert_called_once_with("http://192.168.1.5:4747/video")

    def test_manager_reads_the_url_it_is_given(self):
        """No rebuilding from ip/port: an RTSP config reaches the capture untouched."""
        manager = CameraManager()
        with mock.patch("core.camera_manager.open_stream") as opened:
            opened.return_value.isOpened.return_value = False
            manager._connect_ip_camera({"id": 1, "url": "rtsp://cam:554/stream1"})
            opened.assert_called_once_with("rtsp://cam:554/stream1")


if __name__ == "__main__":
    unittest.main()
