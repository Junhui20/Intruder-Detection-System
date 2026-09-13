"""Web UI: fail closed without a password, then every page and action against a stand-in system."""

import json
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
from database.models import WhitelistEntry
from web import app as web


class FakeSystem(SimpleNamespace):
    """Just enough of IntruderDetectionSystem for the routes."""

    def __init__(self, db):
        super().__init__(
            db_manager=db,
            settings=SimpleNamespace(tier="low", captions_enabled=True, captions_model="", notify_family=False,
                                     notification_cooldown=20, process_every_n_frames=2, ollama_host="http://x",
                                     web_host="0.0.0.0", web_port=8000),
            detection_config=SimpleNamespace(human_confidence_threshold=0.45, pet_identification_threshold=0.65,
                                             yolo_confidence=0.5),
            detection_engine=SimpleNamespace(model_format="pytorch", human_detection_enabled=True,
                                             animal_detection_enabled=True, face_recognition_enabled=True,
                                             pet_identification_enabled=True),
            camera_manager=SimpleNamespace(active_camera=None, cameras={}),
            face_recognition=mock.Mock(known_face_names=["Ada"], track_identities={}),
            animal_recognition=mock.Mock(known_pets={}),
            notification_system=None, captioner=None, performance_tracker=None, started_at=0,
            latest_frame=np.zeros((48, 64, 3), dtype=np.uint8),
            latest_raw=np.zeros((48, 64, 3), dtype=np.uint8),
            latest_detections={"humans": [{"bbox": (5, 5, 30, 40), "identity": "Unknown"}], "animals": []},
            detection_active=True, last_detection_time={}, calls=[], applied={},
        )

    def start_detection(self):
        self.calls.append("start")

    def stop_detection(self):
        self.calls.append("stop")

    def apply_tier(self, tier):
        self.settings.tier = tier

    def apply_settings(self, changes):
        self.applied.update(changes)

    def reload_camera_configurations(self):
        self.calls.append("reload_cameras")

    def reload_roster(self, kind):
        self.calls.append(f"reload_{kind}")

    def enrol(self, kind, name, photo_paths, class_id=None):
        entry_id = self.db_manager.create_whitelist_entry(WhitelistEntry(
            name=name, entity_type=kind, image_path=photo_paths[0],
            multiple_photos=json.dumps(photo_paths[1:]) if len(photo_paths) > 1 else None,
            coco_class_id=class_id if kind == "animal" else None,
        ))
        self.reload_roster(kind)
        return entry_id

    def enrol_from_frame(self, kind, name, bbox, class_id=None):
        self.calls.append(("from_frame", kind, name, tuple(bbox), class_id))
        return 7

    def forget(self, entry_id):
        entry = self.db_manager.get_whitelist_entry(entry_id)
        Path(entry.image_path).unlink(missing_ok=True)
        self.db_manager.delete_whitelist_entry(entry_id)

    def snapshot(self):
        return "data/detection_photos/snapshot.jpg"

    def test_entry(self, entry_id):
        return {"score": 0.71, "threshold": 0.45, "match": True}


JPEG = cv2.imencode(".jpg", np.zeros((20, 20, 3), dtype=np.uint8))[1].tobytes()


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
        for path in ("/", "/events", "/people", "/cameras", "/telegram", "/settings", "/status", "/stream", "/frame.json"):
            self.assertEqual(TestClient(web.create_app(self.system, "hunter2")).get(path).status_code, 401)
            self.assertEqual(self.client.get(path, auth=("me", "nope")).status_code, 401)

    def test_pages_render(self):
        for path in ("/", "/events", "/events?type=human&alerts=true&days=7", "/people", "/cameras", "/telegram",
                     "/settings", "/status"):
            reply = self.client.get(path)
            self.assertEqual(reply.status_code, 200, path)
            self.assertIn('class="strip"', reply.text)

    def test_stream_parts_are_jpegs_of_the_latest_frame(self):
        part = next(web.mjpeg(self.system))
        self.assertTrue(part.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8"))

    def test_frame_json_lists_the_boxes_for_the_picker(self):
        f = self.client.get("/frame.json").json()
        self.assertEqual((f["width"], f["height"]), (64, 48))
        self.assertEqual(f["boxes"][0]["bbox"], [5, 5, 30, 40])
        self.assertEqual(self.client.get("/frame.jpg").headers["content-type"], "image/jpeg")

    def test_enrol_from_frame_and_test_button(self):
        r = self.client.post("/enrol-from-frame", data={"kind": "human", "name": "Ada", "bbox": "[5,5,30,40]"})
        self.assertEqual(r.json(), {"id": 7})
        self.assertIn(("from_frame", "human", "Ada", (5, 5, 30, 40), None), self.system.calls)
        self.assertTrue(self.client.post("/roster/1/test").json()["match"])

    def test_enrol_person_saves_photos_and_reloads_faces(self):
        files = [("photos", ("a.jpg", JPEG, "image/jpeg")), ("photos", ("b.jpg", JPEG, "image/jpeg"))]
        reply = self.client.post("/enrol/human", data={"name": "Ada Lovelace"}, files=files, follow_redirects=False)
        self.assertEqual(reply.status_code, 303)
        [entry] = self.system.db_manager.get_whitelist_entries(entity_type="human")
        self.assertTrue(Path(entry.image_path).exists())
        self.assertEqual(len(json.loads(entry.multiple_photos)), 1)
        self.assertIn("reload_human", self.system.calls)
        self.assertIn("Ada Lovelace", self.client.get("/people").text)

        photos = self.client.get(f"/roster/{entry.id}/photos").json()["photos"]
        self.assertEqual(len(photos), 2)
        self.assertEqual(self.client.get(photos[1]).status_code, 200)
        self.client.post(f"/roster/{entry.id}/photos", files=[("photos", ("c.jpg", JPEG, "image/jpeg"))], follow_redirects=False)
        self.assertEqual(len(self.client.get(f"/roster/{entry.id}/photos").json()["photos"]), 3)

        self.assertEqual(self.client.delete(f"/roster/{entry.id}").status_code, 200)
        self.assertEqual(self.system.db_manager.get_whitelist_entries(entity_type="human"), [])
        self.assertFalse(Path(entry.image_path).exists())

    def test_enrol_pet_keeps_its_class(self):
        self.client.post("/enrol/animal", data={"name": "Jacky", "class_id": 15}, files=[("photos", ("a.jpg", JPEG, "image/jpeg"))])
        [entry] = self.system.db_manager.get_whitelist_entries(entity_type="animal")
        self.assertEqual(entry.coco_class_id, 15)

    def test_camera_add_edit_test_and_delete(self):
        reply = self.client.post("/cameras", data={"name": "Gate", "protocol": "rtsp", "host": "cam", "path": "/stream2"}, follow_redirects=False)
        self.assertEqual(reply.status_code, 303)
        [device] = self.system.db_manager.get_all_devices()
        self.assertEqual((device.name, device.url), ("Gate", "rtsp://cam:554/stream2"))
        self.assertIn("reload_cameras", self.system.calls)

        self.client.post(f"/cameras/{device.id}", data={"name": "Front gate", "url": "rtsp://cam:554/stream1", "auto": "false"}, follow_redirects=False)
        device = self.system.db_manager.get_device(device.id)
        self.assertEqual((device.name, device.url, device.status), ("Front gate", "rtsp://cam:554/stream1", "inactive"))

        with mock.patch("web.app.CameraConfig.test_connection", return_value=(False, "Could not read a frame")):
            self.assertFalse(self.client.post(f"/cameras/{device.id}/test").json()["ok"])
            self.assertIn("Could not", self.client.post("/cameras/test", data={"url": "rtsp://x:554/y"}).json()["message"])
        self.assertEqual(self.client.post("/cameras", data={"url": "ftp://cam/x"}).status_code, 400)
        self.assertEqual(self.client.delete(f"/cameras/{device.id}").status_code, 200)
        self.assertEqual(self.system.db_manager.get_all_devices(), [])

    def test_events_show_the_photo_and_caption(self):
        Path("data/detection_photos").mkdir(parents=True)
        Path("data/detection_photos/unknown_87.5%_x.jpg").write_bytes(b"jpg")
        log_id = self.system.db_manager.log_detection("human", "Unknown person", 0.875, image_path="data/detection_photos/unknown_87.5%_x.jpg", notification_sent=True)
        self.system.db_manager.set_detection_caption(log_id, "A man at the gate.")
        text = self.client.get("/events?type=human&alerts=true").text
        self.assertIn("/photo/unknown_87.5%25_x.jpg", text)
        self.assertIn("A man at the gate.", text)
        self.assertEqual(self.client.get(f"/events/{log_id}").json()["caption"], "A man at the gate.")
        self.assertEqual(self.client.get("/photo/unknown_87.5%25_x.jpg").content, b"jpg")
        self.assertEqual(self.client.get("/photo/..%2F..%2Fetc%2Fpasswd").status_code, 404)
        self.assertNotIn("Unknown person", self.client.get("/events?type=animal").text)

    def test_telegram_recipients_and_delivery(self):
        self.system.notification_system = mock.Mock(users={}, muted_until=0.0, last_alert=None)
        self.client.post("/telegram", data={"chat_id": 7, "username": "hui", "humans": "true"}, follow_redirects=False)
        [user] = self.system.db_manager.get_all_notification_settings()
        self.assertEqual((user.notify_human_detection, user.notify_animal_detection), (True, False))
        self.system.notification_system.send_message.assert_called()  # the hello
        self.client.post("/telegram/7/prefs", data={"humans": "false", "animals": "true"})
        user = self.system.db_manager.get_notification_settings(7)
        self.assertEqual((user.notify_human_detection, user.notify_animal_detection), (False, True))
        self.assertEqual(self.client.post("/telegram", data={"chat_id": 7}).status_code, 400)
        self.client.post("/telegram/delivery", data={"cooldown": 45, "notify_family": "true"})
        self.assertEqual(self.system.applied["notification_cooldown"], 45)
        self.assertTrue(self.system.applied["notify_family"])
        self.client.delete("/telegram/7")
        self.assertEqual(self.system.db_manager.get_all_notification_settings(), [])

    def test_arm_toggles_settings_tier_and_detection(self):
        self.system.notification_system = mock.Mock(users={}, muted_until=0.0, last_alert=None)
        self.assertEqual(self.client.post("/arm", data={"state": "mute", "minutes": 30}).json()["key"], "muted")
        self.assertEqual(self.client.post("/arm", data={"state": "disarm"}).json()["key"], "disarmed")
        self.assertEqual(self.client.post("/arm", data={"state": "arm"}).json()["key"], "armed")
        self.assertEqual(self.client.post("/arm", data={"state": "sideways"}).status_code, 400)
        self.client.post("/toggles", data={"human": "true", "animal": "false", "face": "true", "pet": "true", "captions": "false"})
        self.assertFalse(self.system.detection_engine.animal_detection_enabled)
        self.assertFalse(self.system.applied["captions_enabled"])
        self.client.post("/settings", data={"human_confidence_threshold": 0.5, "pet_identification_threshold": 0.6,
                                            "yolo_confidence": 0.4, "process_every_n_frames": 3, "captions_enabled": "true"})
        self.assertEqual(self.system.applied["process_every_n_frames"], 3)
        self.assertEqual(self.client.post("/tier", data={"tier": "ultra"}).status_code, 400)
        self.client.post("/detection", data={"on": "false"})
        self.assertEqual(self.system.calls[-1], "stop")
        self.assertEqual(self.client.post("/snapshot").json()["photo"], "snapshot.jpg")


class TestBotTokenComesFromTheEnvironment(unittest.TestCase):
    def test_a_token_left_in_config_yaml_is_not_used(self):
        import main as entry
        from config.settings import Settings

        system = entry.IntruderDetectionSystem("config.yaml")
        system.settings = Settings()  # TELEGRAM_BOT_TOKEN unset -> ""
        system.config_manager = mock.Mock(get=mock.Mock(return_value="123:leaked-in-yaml"))
        self.assertTrue(system._initialize_notification_system())
        self.assertIsNone(system.notification_system)


class TestAlertPathStillWired(unittest.TestCase):
    """main.py's detection → event → photo path, run without cameras or models."""

    def test_an_unknown_person_is_logged_with_a_photo_and_family_is_not(self):
        import main as entry

        tmp = tempfile.TemporaryDirectory()
        old = os.getcwd()
        os.chdir(tmp.name)
        try:
            system = entry.IntruderDetectionSystem("config.yaml")
            system.settings = SimpleNamespace(notify_family=False, captions_enabled=True)
            system.db_manager = DatabaseManager("t.db")
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            humans = [{"bbox": (0, 0, 20, 20), "confidence": 0.9, "identity": "Unknown", "recognition_status": "unknown"},
                      {"bbox": (30, 0, 50, 20), "confidence": 0.9, "identity": "Hui", "recognition_status": "known", "face_confidence": 0.8}]
            system._process_detections({"humans": humans, "animals": []}, frame)
            system._process_detections({"humans": humans, "animals": []}, frame)  # same visit: no second row
            rows = {r.entity_name: r for r in system.db_manager.get_recent_detections()}
            self.assertEqual(set(rows), {"Unknown person", "Hui"})
            self.assertTrue(Path(rows["Unknown person"].image_path).exists())
            self.assertIsNone(rows["Hui"].image_path)
            self.assertEqual(len(list(Path("data/detection_photos").glob("*.jpg"))), 1)
        finally:
            os.chdir(old)
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
