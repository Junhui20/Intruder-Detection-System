"""Web UI: fail closed without a password, then every page against a stand-in system."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).parent.parent))

from database.database_manager import DatabaseManager
from web import app as web


class FakeSystem(SimpleNamespace):
    """Just enough of IntruderDetectionSystem for the routes."""

    def __init__(self, db):
        super().__init__(
            db_manager=db,
            settings=SimpleNamespace(tier="low"),
            camera_manager=SimpleNamespace(active_camera=None),
            face_recognition=mock.Mock(),
            animal_recognition=mock.Mock(),
            latest_frame=np.zeros((48, 64, 3), dtype=np.uint8),
            detection_active=True,
            calls=[],
        )

    def start_detection(self):
        self.calls.append("start")

    def stop_detection(self):
        self.calls.append("stop")

    def apply_tier(self, tier):
        self.settings.tier = tier

    def reload_camera_configurations(self):
        self.calls.append("reload_cameras")


class TestWebUI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)  # enrolment photos and thumbnails land under data/
        self.system = FakeSystem(DatabaseManager(str(Path(self.tmp.name) / "t.db")))
        self.client = TestClient(web.create_app(self.system, "hunter2"))
        self.client.auth = ("me", "hunter2")

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def test_no_password_means_no_server(self):
        with mock.patch.dict(os.environ, {"WEB_UI_PASSWORD": ""}):
            self.assertIsNone(web.serve(self.system, "127.0.0.1", 0))

    def test_wrong_password_is_refused_everywhere(self):
        for path in ("/", "/events", "/people", "/pets", "/cameras", "/stream"):
            self.assertEqual(TestClient(web.create_app(self.system, "hunter2")).get(path).status_code, 401)
            self.assertEqual(self.client.get(path, auth=("me", "nope")).status_code, 401)

    def test_pages_render(self):
        for path in ("/", "/events", "/people", "/pets", "/cameras", "/telegram"):
            reply = self.client.get(path)
            self.assertEqual(reply.status_code, 200, path)
            self.assertIn("<nav>", reply.text)

    def test_stream_parts_are_jpegs_of_the_latest_frame(self):
        part = next(web.mjpeg(self.system))
        self.assertTrue(part.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8"))

    def test_enrol_person_saves_photos_and_reloads_faces(self):
        ok, jpeg = cv2.imencode(".jpg", np.zeros((20, 20, 3), dtype=np.uint8))
        files = [("photos", ("a.jpg", jpeg.tobytes(), "image/jpeg")), ("photos", ("b.jpg", jpeg.tobytes(), "image/jpeg"))]
        reply = self.client.post("/enrol/human", data={"name": "Ada Lovelace"}, files=files, follow_redirects=False)
        self.assertEqual(reply.status_code, 303)
        [entry] = self.system.db_manager.get_whitelist_entries(entity_type="human")
        self.assertEqual(entry.name, "Ada Lovelace")
        self.assertTrue(Path(entry.image_path).exists())
        self.assertIn('"', entry.multiple_photos)  # second photo recorded as JSON
        self.system.face_recognition.load_known_faces.assert_called()
        self.assertIn("Ada Lovelace", self.client.get("/people").text)

        self.assertEqual(self.client.delete(f"/enrol/human/{entry.id}").text, "")
        self.assertEqual(self.system.db_manager.get_whitelist_entries(entity_type="human"), [])
        self.assertFalse(Path(entry.image_path).exists())

    def test_enrol_pet_keeps_its_class(self):
        ok, jpeg = cv2.imencode(".jpg", np.zeros((20, 20, 3), dtype=np.uint8))
        self.client.post("/enrol/animal", data={"name": "Jacky", "class_id": 15},
                         files=[("photos", ("a.jpg", jpeg.tobytes(), "image/jpeg"))])
        [entry] = self.system.db_manager.get_whitelist_entries(entity_type="animal")
        self.assertEqual((entry.coco_class_id, entry.individual_id), (15, "jacky"))
        self.system.animal_recognition.load_known_pets.assert_called()

    def test_camera_add_from_parts_test_and_delete(self):
        reply = self.client.post("/cameras", data={"protocol": "rtsp", "host": "cam", "path": "/stream2"}, follow_redirects=False)
        self.assertEqual(reply.status_code, 303)
        [device] = self.system.db_manager.get_all_devices()
        self.assertEqual(device.url, "rtsp://cam:554/stream2")
        self.assertIn("reload_cameras", self.system.calls)

        with mock.patch("web.app.CameraConfig.test_connection", return_value=(False, "Could not read a frame")):
            self.assertIn('class="bad"', self.client.post(f"/cameras/{device.id}/test").text)
        self.assertEqual(self.client.delete(f"/cameras/{device.id}").status_code, 200)
        self.assertEqual(self.system.db_manager.get_all_devices(), [])

    def test_camera_with_a_bad_scheme_is_rejected(self):
        self.assertEqual(self.client.post("/cameras", data={"url": "ftp://cam/x"}).status_code, 400)

    def test_events_show_the_alert_photo(self):
        Path("data/detection_photos").mkdir(parents=True)
        Path("data/detection_photos/unknown_human_87.5%_x.jpg").write_bytes(b"jpg")
        self.system.db_manager.log_detection("human", "Unknown", 0.875, image_path="data/detection_photos/unknown_human_87.5%_x.jpg")
        text = self.client.get("/events?type=human").text
        self.assertIn("/photo/unknown_human_87.5%25_x.jpg", text)
        self.assertEqual(self.client.get("/photo/unknown_human_87.5%25_x.jpg").content, b"jpg")
        self.assertEqual(self.client.get("/photo/..%2F..%2Fetc%2Fpasswd").status_code, 404)

    def test_telegram_recipient_add_and_remove(self):
        self.system.notification_system = mock.Mock()
        self.client.post("/telegram", data={"chat_id": 7, "username": "hui", "humans": "true"}, follow_redirects=False)
        [user] = self.system.db_manager.get_all_notification_settings()
        self.assertEqual((user.chat_id, user.notify_human_detection, user.notify_animal_detection), (7, True, False))
        self.system.notification_system.load_users.assert_called()
        self.assertIn("hui", self.client.get("/telegram").text)
        self.client.delete("/telegram/7")
        self.assertEqual(self.system.db_manager.get_all_notification_settings(), [])

    def test_detection_toggle_and_tier(self):
        self.client.post("/detection", data={"on": "false"}, follow_redirects=False)
        self.assertEqual(self.system.calls[-1], "stop")
        self.client.post("/tier", data={"tier": "high"}, follow_redirects=False)
        self.assertEqual(self.client.post("/tier", data={"tier": "ultra"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()


class TestAlertPathStillWired(unittest.TestCase):
    """main.py's detection → session → screenshot path, run without cameras or models."""

    def test_an_unknown_person_opens_a_session_and_saves_a_photo(self):
        import main as entry

        tmp = tempfile.TemporaryDirectory()
        old = os.getcwd()
        os.chdir(tmp.name)
        try:
            system = entry.IntruderDetectionSystem("config.yaml")
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            person = {"bbox": (0, 0, 20, 20), "confidence": 0.9, "identity": "Unknown", "recognition_status": "unknown"}
            system._process_detections({"humans": [person], "animals": []}, frame)
            self.assertIn("human_Unknown", system.detection_sessions)
            self.assertEqual(len(list(Path("data/detection_photos").glob("*.jpg"))), 1)
        finally:
            os.chdir(old)
            tmp.cleanup()
