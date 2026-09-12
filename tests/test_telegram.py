"""Telegram bot: alerts out with cooldown/mute, commands in, /enroll from an alert photo."""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from core.notification_system import NotificationSystem, parse_duration

RECIPIENT = {"chat_id": 7, "telegram_username": "hui", "notify_human_detection": True,
             "notify_animal_detection": True, "sendstatus": "open"}


def telegram_ok(message_id=42):
    reply = mock.Mock()
    reply.json.return_value = {"ok": True, "result": {"message_id": message_id}}
    return reply


class TestAlerts(unittest.TestCase):
    def setUp(self):
        self.bot = NotificationSystem("token")
        self.bot.load_users([RECIPIENT])

    def test_alert_returns_ids_and_caption_edit_targets_them(self):
        with mock.patch("core.notification_system.requests.post", return_value=telegram_ok()) as post:
            self.assertEqual(self.bot.send_notification("human", "🚨 Unknown person"), [(7, 42)])
            self.assertTrue(self.bot.edit_caption(7, 42, "🚨 Unknown person\n💬 A man at the gate"))
        self.assertTrue(post.call_args.args[0].endswith("/editMessageCaption"))
        self.assertEqual(post.call_args.kwargs["data"]["message_id"], 42)

    def test_cooldown_then_disarm_then_mute_then_arm(self):
        with mock.patch("core.notification_system.requests.post", return_value=telegram_ok()):
            self.assertTrue(self.bot.send_notification("human", "1"))
            self.assertEqual(self.bot.send_notification("human", "2"), [])  # 20 s cooldown
            self.bot.notification_cooldowns.clear()
            self.bot._command(7, {"text": "/disarm"})
            self.assertEqual(self.bot.send_notification("human", "3"), [])
            self.bot._command(7, {"text": "/mute 1h30m"})
            self.assertAlmostEqual(self.bot.muted_until, __import__("time").time() + 5400, delta=2)
            self.assertEqual(self.bot.send_notification("human", "4"), [])
            self.bot._command(7, {"text": "/arm"})
            self.assertTrue(self.bot.send_notification("human", "5"))

    def test_recipient_opt_out_and_strangers_ignored(self):
        self.bot.load_users([{**RECIPIENT, "notify_animal_detection": False}])
        with mock.patch("core.notification_system.requests.post", return_value=telegram_ok()):
            self.assertEqual(self.bot.send_notification("animal", "🐾"), [])
        with mock.patch("core.notification_system.requests.get") as get, \
             mock.patch("core.notification_system.requests.post") as post:
            get.return_value.json.return_value = {"ok": True, "result": [
                {"update_id": 1, "message": {"chat": {"id": 999}, "text": "/status"}}]}
            self.bot.listening = True
            with mock.patch("core.notification_system.time.sleep", side_effect=lambda *_: setattr(self.bot, "listening", False)):
                self.bot._listen()
            post.assert_not_called()

    def test_parse_duration(self):
        self.assertEqual([parse_duration(t) for t in ("1h", "30m", "1h30m", "90", "", "soon")],
                         [3600, 1800, 5400, 5400, None, None])


class TestCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = os.getcwd()
        os.chdir(self.tmp.name)
        Path("data/detection_photos").mkdir(parents=True)
        self.photo = "data/detection_photos/alert.jpg"
        cv2.imwrite(self.photo, np.full((60, 80, 3), 90, dtype=np.uint8))
        self.system = SimpleNamespace(
            latest_frame=np.zeros((48, 64, 3), dtype=np.uint8), detection_active=True,
            camera_manager=SimpleNamespace(cameras={1: {}}), settings=SimpleNamespace(tier="low"),
            face_recognition=SimpleNamespace(known_face_names=["Ada", "Ada"]),
            animal_recognition=SimpleNamespace(known_pets={"jacky": {}}), forget=mock.Mock(),
        )
        # enrol() lands the name in the matching roster, as the real one does after reload
        self.system.enrol = mock.Mock(side_effect=lambda kind, name, paths, class_id: (
            self.system.face_recognition.known_face_names.append(name) if kind == "human"
            else self.system.animal_recognition.known_pets.__setitem__(name, {})) or 1)
        self.bot = NotificationSystem("token", system=self.system)
        self.bot.load_users([RECIPIENT])

    def tearDown(self):
        os.chdir(self.old)
        self.tmp.cleanup()

    def test_status_and_snapshot(self):
        with mock.patch("core.notification_system.requests.post", return_value=telegram_ok()) as post:
            self.bot._command(7, {"text": "/status@mybot"})
            text = post.call_args.kwargs["data"]["text"]
            self.assertIn("1 camera(s)", text)
            self.assertIn("1 people, 1 pets", text)
            self.assertIn("armed", text)
            self.bot._command(7, {"text": "/snapshot"})
            self.assertTrue(post.call_args.args[0].endswith("/sendPhoto"))
            self.assertEqual(post.call_args.kwargs["files"]["photo"][0], "frame.jpg")

    def test_enroll_needs_a_reply_to_an_alert(self):
        with mock.patch("core.notification_system.requests.post", return_value=telegram_ok()) as post:
            self.bot._command(7, {"text": "/enroll Ada"})
            self.assertIn("Reply to one of my alert photos", post.call_args.kwargs["data"]["text"])
        self.system.enrol.assert_not_called()

    def test_enroll_person_from_alert_photo(self):
        with mock.patch("core.notification_system.requests.post", return_value=telegram_ok(42)):
            self.bot.send_notification("human", "🚨", photo_path=self.photo,
                                       context={"kind": "human", "label": "person", "bbox": (0, 0, 10, 10), "class_id": None})
            self.bot._command(7, {"text": "/enroll Ada Lovelace", "reply_to_message": {"message_id": 42}})
        kind, name, paths, class_id = self.system.enrol.call_args.args
        self.assertEqual((kind, name, class_id), ("human", "Ada Lovelace", None))
        self.assertEqual(Path(paths[0]).parts[:2], ("data", "faces"))
        self.assertTrue(Path(paths[0]).name.startswith("ada_lovelace_"))
        self.assertEqual(cv2.imread(paths[0]).shape, (30, 30, 3))  # the box plus 20 px of context

    def test_enroll_that_finds_no_face_is_undone(self):
        self.system.enrol = mock.Mock(return_value=9)  # roster unchanged: no face found
        with mock.patch("core.notification_system.requests.post", return_value=telegram_ok(44)) as post:
            self.bot.send_notification("human", "🚨", photo_path=self.photo,
                                       context={"kind": "human", "label": "person", "bbox": None, "class_id": None})
            self.bot._command(7, {"text": "/enroll Ghost", "reply_to_message": {"message_id": 44}})
        self.system.forget.assert_called_once_with(9)
        self.assertIn("Could not find a face", post.call_args.kwargs["data"]["text"])

    def test_a_failing_command_replies_and_does_not_eat_the_batch(self):
        with mock.patch("core.notification_system.requests.get") as get, \
             mock.patch("core.notification_system.requests.post", return_value=telegram_ok()) as post, \
             mock.patch.object(self.bot, "status_text", side_effect=RuntimeError("boom")):
            get.return_value.json.return_value = {"ok": True, "result": [
                {"update_id": 1, "edited_message": {"chat": {"id": 7}, "text": "x"}},
                {"update_id": 2, "message": {"chat": {"id": 7}, "text": "/status"}},
                {"update_id": 3, "message": {"chat": {"id": 7}, "text": "/arm"}}]}
            self.bot.listening = True
            with mock.patch("core.notification_system.time.sleep", side_effect=lambda *_: setattr(self.bot, "listening", False)):
                self.bot._listen()
        texts = [c.kwargs["data"]["text"] for c in post.call_args_list]
        self.assertIn("⚠️ That failed: boom", texts)
        self.assertIn("🟢 Armed. Alerts are on.", texts)

    def test_enroll_pet_crops_to_its_box(self):
        with mock.patch("core.notification_system.requests.post", return_value=telegram_ok(43)):
            self.bot.send_notification("animal", "🐾", photo_path=self.photo,
                                       context={"kind": "animal", "label": "dog", "bbox": (10, 5, 50, 45), "class_id": 16})
            self.bot._command(7, {"text": "/enrol Jacky", "reply_to_message": {"message_id": 43}})
        kind, name, paths, class_id = self.system.enrol.call_args.args
        self.assertEqual((kind, class_id), ("animal", 16))
        self.assertEqual(cv2.imread(paths[0]).shape, (40, 40, 3))  # the box, not the driveway


if __name__ == "__main__":
    unittest.main()
