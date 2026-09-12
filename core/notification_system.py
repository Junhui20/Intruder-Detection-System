"""
Telegram: alerts out, commands in.

Alerts go to every recipient on the Telegram page (per-user people/animal
toggles, 20 s cooldown each). Recipients can talk back:

    /status            what is running, who is enrolled, armed or not
    /snapshot          the camera's current frame
    /arm  /disarm      alerts on / off until further notice
    /mute 1h           alerts off for a while (30m, 2h, 1h30m)
    /enroll <name>     reply to an alert photo: that person or pet is family now

The bot polls getUpdates on a thread; nobody outside the recipient list is
answered. Nothing here needs a webhook or a public address.
"""

import logging
import math
import os
import re
import shutil
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import requests

logger = logging.getLogger(__name__)

HELP = (
    "/status — what is running\n"
    "/snapshot — current camera frame\n"
    "/arm, /disarm — alerts on / off\n"
    "/mute 1h — alerts off for a while\n"
    "/enroll Name — reply to an alert photo to make them family"
)


class NotificationSystem:
    """
    The Telegram bot.

    Args:
        bot_token: From BotFather; read from the environment by the caller.
        db_manager: For /enroll.
        system: The running IntruderDetectionSystem, for /status, /snapshot and
            enrolment. None in tests.
    """

    def __init__(self, bot_token: str, db_manager=None, system=None):
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.db_manager = db_manager
        self.system = system
        self.users: Dict[int, dict] = {}
        self.last_update_id = 0
        self.listening = False
        self.listen_thread = None
        self.notification_cooldowns: Dict[int, float] = {}
        self.default_cooldown = 20  # seconds between alerts per recipient
        self.muted_until = 0.0  # epoch seconds; math.inf while disarmed
        # (chat_id, message_id) -> what that alert photo showed, for /enroll
        self.alerts: Dict[Tuple[int, int], dict] = {}
        self.last_alert: Optional[float] = None
        self.notification_stats = {
            "messages_sent": 0,
            "photos_sent": 0,
            "commands_received": 0,
            "failed_sends": 0,
            "active_users": 0,
        }

    # ── recipients ──────────────────────────────────────────────────────────

    def load_users(self, users_data: List[Dict]) -> None:
        """Replace the recipient list with notification_settings rows."""
        self.users = {
            u["chat_id"]: {
                "chat_id": u["chat_id"],
                "username": u["telegram_username"],
                "notify_human_detection": u["notify_human_detection"],
                "notify_animal_detection": u["notify_animal_detection"],
                "sendstatus": u["sendstatus"],
            }
            for u in users_data
        }
        self.notification_stats["active_users"] = sum(
            u["sendstatus"] == "open" for u in self.users.values()
        )
        logger.info(
            f"Telegram recipients: {self.notification_stats['active_users']} active"
        )

    # ── inbound ─────────────────────────────────────────────────────────────

    def start_listening(self) -> None:
        if not self.listening:
            self.listening = True
            self.listen_thread = threading.Thread(target=self._listen, daemon=True)
            self.listen_thread.start()

    def stop_listening(self) -> None:
        self.listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=5)

    def _listen(self) -> None:
        while self.listening:
            try:
                for update in self._get_updates():
                    message = update.get("message")
                    if not message or "text" not in message:  # edits, joins, stickers
                        continue
                    chat_id = message["chat"]["id"]
                    user = self.users.get(chat_id)
                    if not user or user["sendstatus"] != "open":
                        logger.warning(
                            f"Ignoring Telegram message from {chat_id}: not a recipient"
                        )
                        continue
                    self.notification_stats["commands_received"] += 1
                    try:
                        self._command(chat_id, message)
                    except Exception as e:  # one bad command must not eat the batch
                        logger.exception("Telegram command failed")
                        self.send_message(chat_id, f"⚠️ That failed: {e}")
                time.sleep(1)
            except Exception as e:
                logger.error(f"Telegram listener: {e}")
                time.sleep(5)

    def _get_updates(self) -> List[Dict]:
        params = {"offset": self.last_update_id + 1, "timeout": 10, "limit": 100}
        reply = requests.get(
            f"{self.base_url}/getUpdates", params=params, timeout=15
        ).json()
        updates = reply.get("result", []) if reply.get("ok") else []
        if updates:
            self.last_update_id = max(u["update_id"] for u in updates)
        return updates

    def _command(self, chat_id: int, message: Dict) -> None:
        word, _, arg = message["text"].strip().partition(" ")
        word = word.lower().lstrip("/").split("@")[0]
        if word in ("status", "check"):
            self.send_message(chat_id, self.status_text())
        elif word == "snapshot":
            frame = getattr(self.system, "latest_frame", None)
            if frame is None:
                self.send_message(chat_id, "No frame: detection is not running.")
            else:
                self.send_photo(
                    chat_id, cv2.imencode(".jpg", frame)[1].tobytes(), "📷 now"
                )
        elif word == "arm":
            self.muted_until = 0.0
            self.send_message(chat_id, "🟢 Armed. Alerts are on.")
        elif word == "disarm":
            self.muted_until = math.inf
            self.send_message(chat_id, "⚪ Disarmed. No alerts until /arm.")
        elif word == "mute":
            seconds = parse_duration(arg)
            if seconds is None:
                self.send_message(chat_id, "Usage: /mute 1h (also 30m, 1h30m)")
            else:
                self.muted_until = time.time() + seconds
                self.send_message(
                    chat_id, f"🔇 Muted for {arg.strip()}. /arm to end early."
                )
        elif word in ("enroll", "enrol"):
            self.send_message(chat_id, self._enroll(chat_id, message, arg.strip()))
        else:
            self.send_message(chat_id, HELP)

    def status_text(self) -> str:
        """One message: detection, cameras, roster, tier, armed state."""
        s = self.system
        if s is None:
            return "🟢 Bot is up; detection system not attached."
        cameras = len(getattr(s.camera_manager, "cameras", {}) or {})
        people = (
            len(set(s.face_recognition.known_face_names)) if s.face_recognition else 0
        )
        pets = len(s.animal_recognition.known_pets) if s.animal_recognition else 0
        if self.muted_until == math.inf:
            armed = "⚪ disarmed"
        elif time.time() < self.muted_until:
            armed = (
                f"🔇 muted for {int((self.muted_until - time.time()) // 60)} more min"
            )
        else:
            armed = "🟢 armed"
        last = (
            time.strftime("%H:%M", time.localtime(self.last_alert))
            if self.last_alert
            else "none yet"
        )
        return (
            f"{'🟢 detecting' if s.detection_active else '🔴 detection stopped'} · "
            f"{cameras} camera(s) · tier {s.settings.tier}\n"
            f"{people} people, {pets} pets enrolled\n{armed} · last alert {last}"
        )

    def _enroll(self, chat_id: int, message: Dict, name: str) -> str:
        """Make the subject of a replied-to alert photo a known person or pet."""
        reply = message.get("reply_to_message") or {}
        alert = self.alerts.get((chat_id, reply.get("message_id")))
        if not alert:
            return "Reply to one of my alert photos with /enroll Name."
        if not name:
            return "Give them a name: /enroll Name"
        if self.system is None or not Path(alert["photo_path"]).exists():
            return "That photo is gone; enrol from the web UI instead."
        kind = alert["kind"]
        folder = Path("data/faces" if kind == "human" else "data/animals")
        folder.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        path = folder / f"{slug}_{int(time.time())}.jpg"
        if alert.get("bbox"):
            x1, y1, x2, y2 = alert[
                "bbox"
            ]  # the models want the subject, not the driveway
            pad = 20 if kind == "human" else 0  # the face detector likes some context
            frame = cv2.imread(alert["photo_path"])
            cv2.imwrite(
                str(path),
                frame[max(0, y1 - pad) : y2 + pad, max(0, x1 - pad) : x2 + pad],
            )
        else:
            shutil.copy(alert["photo_path"], path)
        entry_id = self.system.enrol(kind, name, [str(path)], alert.get("class_id"))
        roster = (
            self.system.face_recognition.known_face_names
            if kind == "human"
            else self.system.animal_recognition.known_pets
        )
        if name not in roster:
            self.system.forget(entry_id)
            what = "face" if kind == "human" else "usable animal"
            return f"❌ Could not find a {what} in that photo; try a clearer alert or the web UI."
        what = "person" if kind == "human" else alert.get("label", "pet")
        return f"✅ {name} enrolled as a known {what}. Add more photos on the web UI for better matching."

    # ── outbound ────────────────────────────────────────────────────────────

    def send_notification(
        self,
        notification_type: str,
        message: str,
        photo_path: Optional[str] = None,
        force: bool = False,
        context: Optional[dict] = None,
    ) -> List[Tuple[int, int]]:
        """
        Alert every eligible recipient.

        Args:
            notification_type: 'human' or 'animal'; recipients opt in per type.
            message: Caption or text.
            photo_path: Sent as a photo when given and readable.
            force: Ignore cooldown, mute and disarm (system messages).
            context: What the photo shows (kind, bbox, class_id, label) — kept so
                a recipient can reply /enroll to it.

        Returns:
            (chat_id, message_id) for every message that went out.
        """
        if not force and time.time() < self.muted_until:
            return []
        sent = []
        for chat_id, user in self.users.items():
            wanted = (
                user[f"notify_{notification_type}_detection"]
                if notification_type in ("human", "animal")
                else True
            )
            if user["sendstatus"] != "open" or not wanted:
                continue
            if (
                not force
                and time.time() - self.notification_cooldowns.get(chat_id, 0)
                < self.default_cooldown
            ):
                continue
            message_id = self._send_to_user(chat_id, message, photo_path)
            if message_id:
                sent.append((chat_id, message_id))
                self.notification_cooldowns[chat_id] = time.time()
                if context and photo_path and os.path.exists(photo_path):
                    self.alerts[(chat_id, message_id)] = {
                        **context,
                        "photo_path": photo_path,
                    }
        if sent:
            self.last_alert = time.time()
            for key in list(self.alerts)[:-200]:  # remember the last 200 alerts
                self.alerts.pop(key, None)  # alerts run on parallel threads
        return sent

    def _send_to_user(
        self, chat_id: int, message: str, photo_path: Optional[str]
    ) -> Optional[int]:
        try:
            if photo_path and os.path.exists(photo_path):
                message_id = self.send_photo(chat_id, photo_path, message)
                self.notification_stats[
                    "photos_sent" if message_id else "failed_sends"
                ] += 1
            else:
                message_id = self.send_message(chat_id, message)
                self.notification_stats[
                    "messages_sent" if message_id else "failed_sends"
                ] += 1
            return message_id
        except Exception as e:
            logger.error(f"Error sending to {chat_id}: {e}")
            self.notification_stats["failed_sends"] += 1
            return None

    def _call(self, method: str, **kwargs) -> Optional[int]:
        """POST one Bot API method; the new message's id, or None."""
        try:
            reply = requests.post(
                f"{self.base_url}/{method}", timeout=30, **kwargs
            ).json()
            if not reply.get("ok"):
                logger.warning(f"Telegram {method} refused: {reply.get('description')}")
                return None
            return (
                reply["result"]["message_id"]
                if isinstance(reply.get("result"), dict)
                else True
            )
        except Exception as e:
            logger.error(f"Telegram {method}: {e}")
            return None

    def send_message(self, chat_id: int, message: str) -> Optional[int]:
        """Send text; returns its message_id."""
        return self._call("sendMessage", data={"chat_id": chat_id, "text": message})

    def send_photo(
        self, chat_id: int, photo: Union[str, bytes], caption: str = ""
    ) -> Optional[int]:
        """Send a photo from a path or JPEG bytes; returns its message_id."""
        data = {"chat_id": chat_id, "caption": caption}
        if isinstance(photo, bytes):
            return self._call(
                "sendPhoto",
                data=data,
                files={"photo": ("frame.jpg", photo, "image/jpeg")},
            )
        with open(photo, "rb") as f:
            return self._call("sendPhoto", data=data, files={"photo": f})

    def edit_caption(self, chat_id: int, message_id: int, caption: str) -> bool:
        """Replace the caption under an already-sent photo (plain text — VLM output has _ and *)."""
        return bool(
            self._call(
                "editMessageCaption",
                data={"chat_id": chat_id, "message_id": message_id, "caption": caption},
            )
        )

    def test_connection(self) -> bool:
        """True if the token answers getMe."""
        try:
            me = requests.get(f"{self.base_url}/getMe", timeout=10).json()
            if me.get("ok"):
                logger.info(f"Telegram bot: @{me['result'].get('username')}")
            return bool(me.get("ok"))
        except Exception as e:
            logger.error(f"Telegram getMe: {e}")
            return False

    def get_performance_stats(self) -> Dict:
        return {
            **self.notification_stats,
            "muted_until": self.muted_until,
            "remembered_alerts": len(self.alerts),
        }


def parse_duration(text: str) -> Optional[int]:
    """'1h', '30m', '1h30m', '90' (minutes) → seconds; None if unreadable."""
    text = text.strip().lower()
    if text.isdigit():
        return int(text) * 60
    parts = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?", text)
    if not text or not parts:
        return None
    hours, minutes = (int(x) if x else 0 for x in parts.groups())
    return hours * 3600 + minutes * 60 or None
