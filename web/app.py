"""
The web UI: live view, events, roster, cameras, Telegram, settings, status.

FastAPI + Jinja2 pages, a little fetch() for the actions, no build step. Every
route sits behind HTTP Basic auth against ``WEB_UI_PASSWORD``; without that
variable the server does not start at all — detection and Telegram carry on.
"""

import json
import math
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import psutil
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config.camera_config import DEFAULT_PORTS, CameraConfig, build_camera_url
from core.animal_recognition import ANIMAL_CLASSES
from database.models import Device, NotificationSettings

HERE = Path(__file__).parent
PHOTOS = Path("data/detection_photos")
ENROL = {"human": Path("data/faces"), "animal": Path("data/animals")}
templates = Jinja2Templates(directory=HERE / "templates")
templates.env.filters["basename"] = lambda p: Path(p).name


def local(t, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """detection_logs stores UTC (CURRENT_TIMESTAMP); show the machine's local time."""
    if not t:
        return ""
    stamp = datetime.strptime(str(t)[:19], "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    return stamp.astimezone().strftime(fmt)


templates.env.filters["local"] = local
templates.env.filters["clock"] = lambda t: local(t, "%H:%M")
basic = HTTPBasic()


def mjpeg(system):
    """Yield the newest annotated frame as a multipart JPEG part, ten times a second."""
    while (frame := system.latest_frame) is not None:  # ends when detection stops
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        time.sleep(0.1)


def photos_of(entry) -> List[str]:
    return [entry.image_path] + (
        json.loads(entry.multiple_photos) if entry.multiple_photos else []
    )


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def create_app(system, password: str) -> FastAPI:
    """
    Build the app around a running ``IntruderDetectionSystem``.

    Args:
        system: The orchestrator; the routes read its state and call its methods.
        password: What every request must present (any username).
    """

    def authed(credentials: HTTPBasicCredentials = Depends(basic)) -> None:
        if not secrets.compare_digest(credentials.password.encode(), password.encode()):
            raise HTTPException(401, headers={"WWW-Authenticate": "Basic"})

    app = FastAPI(dependencies=[Depends(authed)], docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    db = system.db_manager

    def bot():
        return getattr(system, "notification_system", None)

    def armed_state() -> dict:
        b = bot()
        if not b:
            return {"key": "nobot", "label": "No Telegram"}
        if b.muted_until == math.inf:
            return {"key": "disarmed", "label": "Disarmed"}
        if time.time() < b.muted_until:
            return {
                "key": "muted",
                "label": f"Muted · {int((b.muted_until - time.time()) // 60) + 1} min",
            }
        return {"key": "armed", "label": "Armed"}

    def strip() -> dict:
        cams = getattr(system.camera_manager, "cameras", {}) or {}
        tracker = getattr(system, "performance_tracker", None)
        fps = (
            tracker.fps_tracker.get_fps() if tracker and system.detection_active else 0
        )
        b = bot()
        last = (
            time.strftime("%H:%M", time.localtime(b.last_alert))
            if b and b.last_alert
            else "—"
        )
        return {
            "armed": armed_state(),
            "cams_up": len(cams),
            "cams_total": len(db.get_all_devices()),
            "fps": round(fps),
            "tier": system.settings.tier,
            "last_alert": last,
            "detecting": system.detection_active,
        }

    def page(request: Request, name: str, **ctx):
        return templates.TemplateResponse(
            request, name, {"strip": strip(), "page": name[:-5], **ctx}
        )

    # ── live ──────────────────────────────────────────────────────────────

    @app.get("/")
    def live(request: Request):
        engine = system.detection_engine
        toggles = {
            "human": getattr(engine, "human_detection_enabled", True),
            "animal": getattr(engine, "animal_detection_enabled", True),
            "face": getattr(engine, "face_recognition_enabled", True),
            "pet": getattr(engine, "pet_identification_enabled", True),
            "captions": system.settings.captions_enabled,
        }
        cameras = db.get_all_devices()
        active = getattr(system.camera_manager, "active_camera", None)
        names = {c.id: c.name or f"Camera {c.id}" for c in cameras}
        title = (
            names.get(
                active, "Local webcam" if active is not None else "No camera connected"
            )
            if cameras
            else "No cameras"
        )
        return page(
            request,
            "live.html",
            cameras=cameras,
            title=title,
            toggles=toggles,
            recent=db.get_recent_detections(limit=8),
        )

    @app.get("/stream")
    def stream():
        return StreamingResponse(
            mjpeg(system), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/frame.jpg")
    def frame_jpg():
        if system.latest_raw is None:
            raise HTTPException(404, "no frame")
        return Response(
            cv2.imencode(".jpg", system.latest_raw)[1].tobytes(),
            media_type="image/jpeg",
        )

    @app.get("/frame.json")
    def frame_json():
        frame = system.latest_frame
        if frame is None:
            raise HTTPException(404, "no frame")
        boxes = [
            {
                "kind": "human",
                "bbox": d["bbox"],
                "label": d.get("identity", "person"),
                "class_id": None,
            }
            for d in system.latest_detections.get("humans", [])
        ]
        boxes += [
            {
                "kind": "animal",
                "bbox": d["bbox"],
                "label": d.get("pet_identity", d.get("animal_type")),
                "class_id": d.get("class_id"),
            }
            for d in system.latest_detections.get("animals", [])
        ]
        return {"width": frame.shape[1], "height": frame.shape[0], "boxes": boxes}

    @app.post("/detection")
    def toggle_detection(on: bool = Form(...)):
        (system.start_detection if on else system.stop_detection)()
        return JSONResponse({"detecting": system.detection_active})

    @app.post("/toggles")
    def toggles(
        human: bool = Form(True),
        animal: bool = Form(True),
        face: bool = Form(True),
        pet: bool = Form(True),
        captions: bool = Form(True),
    ):
        e = system.detection_engine
        e.human_detection_enabled, e.animal_detection_enabled = human, animal
        e.face_recognition_enabled, e.pet_identification_enabled = face, pet
        system.apply_settings({"captions_enabled": captions})
        return JSONResponse({"ok": True})

    @app.post("/snapshot")
    def snapshot():
        path = system.snapshot()
        if not path:
            raise HTTPException(409, "no frame: detection is not running")
        return JSONResponse(
            {"photo": Path(path).name, "sent_to": len(bot().users) if bot() else 0}
        )

    @app.post("/reset-tracking")
    def reset_tracking():
        if system.face_recognition:
            system.face_recognition.track_identities.clear()
        system.last_detection_time.clear()
        return JSONResponse({"ok": True})

    @app.post("/arm")
    def arm(state: str = Form(...), minutes: int = Form(60)):
        b = bot()
        if not b:
            raise HTTPException(409, "no Telegram bot configured")
        if state not in ("arm", "disarm", "mute"):
            raise HTTPException(400, "state must be arm, disarm or mute")
        b.muted_until = {
            "arm": 0.0,
            "disarm": math.inf,
            "mute": time.time() + minutes * 60,
        }[state]
        return JSONResponse(armed_state())

    @app.post("/tier")
    def set_tier(tier: str = Form(...)):
        if tier not in ("low", "high"):
            raise HTTPException(400, "tier must be low or high")
        threading.Thread(target=system.apply_tier, args=(tier,), daemon=True).start()
        return JSONResponse({"tier": tier})

    # ── events ────────────────────────────────────────────────────────────

    @app.get("/events")
    def events(
        request: Request,
        type: str = "",
        alerts: bool = False,
        days: int = 1,
        limit: int = 100,
    ):
        rows = db.get_recent_detections(
            limit=limit,
            detection_type=type or None,
            alerts_only=alerts,
            days=days or None,
        )
        return page(
            request,
            "events.html",
            rows=rows,
            type=type,
            alerts=alerts,
            days=days,
            limit=limit,
        )

    @app.get("/events/{log_id}")
    def event(log_id: int):
        row = db.get_detection(log_id)
        if not row:
            raise HTTPException(404)
        d = row.to_dict()
        d["detected_at"] = str(d["detected_at"])
        d["photo"] = f"/photo/{Path(row.image_path).name}" if row.image_path else None
        return d

    @app.get("/photo/{name}")
    def photo(name: str):
        path = PHOTOS / Path(name).name  # no directories: the name is the whole key
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    # ── roster ────────────────────────────────────────────────────────────

    @app.get("/people")
    def people(request: Request):
        seen = db.last_seen()
        rows = db.get_whitelist_entries()
        for r in rows:
            r.photo_count = len(photos_of(r))
            r.seen = seen.get(r.name)
        return page(
            request,
            "people.html",
            people=[r for r in rows if r.entity_type == "human"],
            pets=[r for r in rows if r.entity_type == "animal"],
            classes=ANIMAL_CLASSES,
        )

    @app.get("/roster/{entry_id}/photos")
    def roster_photos(entry_id: int):
        entry = db.get_whitelist_entry(entry_id)
        if not entry:
            raise HTTPException(404)
        return {
            "name": entry.name,
            "photos": [
                f"/roster/{entry_id}/photo/{i}" for i in range(len(photos_of(entry)))
            ],
        }

    @app.get("/roster/{entry_id}/photo/{n}")
    def roster_photo(entry_id: int, n: int):
        entry = db.get_whitelist_entry(entry_id)
        paths = photos_of(entry) if entry else []
        if n >= len(paths) or not Path(paths[n]).is_file():
            raise HTTPException(404)
        return FileResponse(paths[n])

    def save_uploads(kind: str, name: str, photos: List[UploadFile]) -> List[str]:
        ENROL[kind].mkdir(parents=True, exist_ok=True)
        paths = []
        for i, photo in enumerate(photos):
            path = ENROL[kind] / f"{slug(name)}_{int(time.time())}_{i}.jpg"
            path.write_bytes(photo.file.read())
            paths.append(str(path))
        return paths

    @app.post("/enrol/{kind}")
    def enrol(
        kind: str,
        name: str = Form(...),
        photos: List[UploadFile] = File(...),
        class_id: int = Form(16),
    ):
        if kind not in ENROL or not name.strip():
            raise HTTPException(400)
        system.enrol(kind, name.strip(), save_uploads(kind, name, photos), class_id)
        return RedirectResponse("/people", 303)

    @app.post("/enrol-from-frame")
    def enrol_from_frame(
        kind: str = Form(...),
        name: str = Form(...),
        bbox: str = Form(...),
        class_id: Optional[int] = Form(None),
    ):
        if kind not in ENROL or not name.strip():
            raise HTTPException(400)
        entry_id = system.enrol_from_frame(
            kind, name.strip(), json.loads(bbox), class_id
        )
        if entry_id is None:
            raise HTTPException(409, "no frame")
        return JSONResponse({"id": entry_id})

    @app.post("/roster/{entry_id}/photos")
    def add_photos(entry_id: int, photos: List[UploadFile] = File(...)):
        entry = db.get_whitelist_entry(entry_id)
        if not entry:
            raise HTTPException(404)
        paths = photos_of(entry) + save_uploads(entry.entity_type, entry.name, photos)
        entry.image_path, entry.multiple_photos = paths[0], (
            json.dumps(paths[1:]) if len(paths) > 1 else None
        )
        db.update_whitelist_entry(entry)
        system.reload_roster(entry.entity_type)
        return RedirectResponse("/people", 303)

    @app.delete("/roster/{entry_id}")
    def forget(entry_id: int):
        system.forget(entry_id)
        return JSONResponse({"ok": True})

    @app.post("/roster/{entry_id}/test")
    def test_entry(entry_id: int):
        result = system.test_entry(entry_id)
        if result is None:
            raise HTTPException(409, "no frame, or no such entry")
        return result

    # ── cameras ───────────────────────────────────────────────────────────

    @app.get("/cameras")
    def cameras(request: Request):
        live_cams = getattr(system.camera_manager, "cameras", {}) or {}
        return page(request, "cameras.html", rows=db.get_all_devices(), live=live_cams)

    def camera_url_from(url, protocol, host, port, path, username, password) -> str:
        if protocol not in DEFAULT_PORTS:
            raise HTTPException(400, "protocol must be rtsp, http or https")
        url = url.strip() or build_camera_url(
            protocol, host.strip(), port, path.strip(), username, password
        )
        errors = CameraConfig(url=url).validate()
        if errors:
            raise HTTPException(400, errors["url"])
        return url

    @app.post("/cameras")
    def add_camera(
        name: str = Form(""),
        url: str = Form(""),
        protocol: str = Form("rtsp"),
        host: str = Form(""),
        port: int = Form(0),
        path: str = Form(""),
        username: str = Form(""),
        password: str = Form(""),
        auto: bool = Form(True),
    ):
        url = camera_url_from(url, protocol, host, port, path, username, password)
        db.create_device(
            Device(url=url, name=name.strip(), status="active" if auto else "inactive")
        )
        system.reload_camera_configurations()
        return RedirectResponse("/cameras", 303)

    @app.post("/cameras/test")
    def test_camera_url(
        url: str = Form(""),
        protocol: str = Form("rtsp"),
        host: str = Form(""),
        port: int = Form(0),
        path: str = Form(""),
        username: str = Form(""),
        password: str = Form(""),
    ):
        url = camera_url_from(url, protocol, host, port, path, username, password)
        t = time.time()
        ok, message = CameraConfig(url=url).test_connection()
        return JSONResponse(
            {"ok": ok, "message": f"{message} · {time.time() - t:.1f} s"}
        )

    @app.post("/cameras/{device_id}")
    def edit_camera(
        device_id: int,
        name: str = Form(""),
        url: str = Form(""),
        protocol: str = Form("rtsp"),
        host: str = Form(""),
        port: int = Form(0),
        path: str = Form(""),
        username: str = Form(""),
        password: str = Form(""),
        auto: bool = Form(True),
    ):
        device = db.get_device(device_id)
        if not device:
            raise HTTPException(404)
        device.url = camera_url_from(
            url, protocol, host, port, path, username, password
        )
        device.name, device.status = name.strip(), "active" if auto else "inactive"
        db.update_device(device)
        system.reload_camera_configurations()
        return RedirectResponse("/cameras", 303)

    @app.post("/cameras/{device_id}/test")
    def test_camera(device_id: int):
        device = db.get_device(device_id)
        if not device:
            raise HTTPException(404)
        t = time.time()
        ok, message = CameraConfig(url=device.url).test_connection()
        return JSONResponse(
            {"ok": ok, "message": f"{message} · {time.time() - t:.1f} s"}
        )

    @app.delete("/cameras/{device_id}")
    def delete_camera(device_id: int):
        db.delete_device(device_id)
        system.reload_camera_configurations()
        return JSONResponse({"ok": True})

    # ── telegram ──────────────────────────────────────────────────────────

    def reload_telegram_users() -> None:
        if bot():
            bot().load_users(
                [u.to_dict() for u in db.get_all_notification_settings(status="open")]
            )

    @app.get("/telegram")
    def telegram(request: Request):
        return page(
            request,
            "telegram.html",
            rows=db.get_all_notification_settings(),
            bot_ok=bool(bot()),
            cooldown=system.settings.notification_cooldown,
            captions=system.settings.captions_enabled,
            notify_family=system.settings.notify_family,
        )

    @app.post("/telegram")
    def add_recipient(
        chat_id: int = Form(...),
        username: str = Form(""),
        humans: bool = Form(False),
        animals: bool = Form(False),
    ):
        if db.get_notification_settings(chat_id):
            raise HTTPException(400, f"chat id {chat_id} is already a recipient")
        db.create_notification_settings(
            NotificationSettings(
                chat_id=chat_id,
                telegram_username=username.strip() or str(chat_id),
                notify_human_detection=humans,
                notify_animal_detection=animals,
            )
        )
        reload_telegram_users()
        if bot():
            bot().send_message(
                chat_id,
                "👋 You now get alerts from this camera. /status to check on it.",
            )
        return RedirectResponse("/telegram", 303)

    @app.post("/telegram/delivery")
    def delivery(
        cooldown: int = Form(...),
        captions: bool = Form(False),
        notify_family: bool = Form(False),
    ):
        system.apply_settings(
            {
                "notification_cooldown": cooldown,
                "captions_enabled": captions,
                "notify_family": notify_family,
            }
        )
        return JSONResponse({"ok": True})

    @app.post("/telegram/{chat_id}/prefs")
    def recipient_prefs(
        chat_id: int, humans: bool = Form(...), animals: bool = Form(...)
    ):
        user = db.get_notification_settings(chat_id)
        if not user:
            raise HTTPException(404)
        user.notify_human_detection, user.notify_animal_detection = humans, animals
        db.update_notification_settings(user)
        reload_telegram_users()
        return JSONResponse({"ok": True})

    @app.post("/telegram/{chat_id}/test")
    def test_message(chat_id: int):
        if not bot():
            raise HTTPException(409, "no Telegram bot configured")
        return JSONResponse(
            {
                "ok": bool(
                    bot().send_message(chat_id, "✅ Test message from the web UI.")
                )
            }
        )

    @app.delete("/telegram/{chat_id}")
    def delete_recipient(chat_id: int):
        db.delete_notification_settings(chat_id)
        reload_telegram_users()
        return JSONResponse({"ok": True})

    # ── settings & status ─────────────────────────────────────────────────

    @app.get("/settings")
    def settings(request: Request):
        s = system.settings
        return page(
            request,
            "settings.html",
            s=s,
            dc=system.detection_config,
            ollama=getattr(system.captioner, "enabled", False),
            bind=f"{s.web_host}:{s.web_port}",
        )

    @app.post("/settings")
    def save_settings(
        human_confidence_threshold: float = Form(...),
        pet_identification_threshold: float = Form(...),
        yolo_confidence: float = Form(...),
        process_every_n_frames: int = Form(...),
        captions_enabled: bool = Form(False),
        captions_model: str = Form(""),
    ):
        system.apply_settings(
            {
                "human_confidence_threshold": human_confidence_threshold,
                "pet_identification_threshold": pet_identification_threshold,
                "yolo_confidence": yolo_confidence,
                "process_every_n_frames": max(1, process_every_n_frames),
                "captions_enabled": captions_enabled,
                "captions_model": captions_model.strip(),
            }
        )
        return JSONResponse({"ok": True})

    @app.get("/status")
    def status(request: Request):
        today = db.counts_today()
        alerts = sum(n for k, n in today.items() if k.endswith("|1"))
        family = {
            k.split("|")[0]: n
            for k, n in today.items()
            if k.endswith("|0") and not k.startswith("Unknown")
        }
        uptime = time.time() - system.started_at
        return page(
            request,
            "status.html",
            alerts=alerts,
            family=family,
            hours=db.detections_per_hour(24),
            rss=psutil.Process().memory_info().rss // 2**20,
            uptime=f"{int(uptime // 86400)} d {int(uptime % 86400 // 3600)} h",
            caption_model=getattr(system.captioner, "model", None),
            detector=getattr(system.detection_engine, "model_format", "—"),
            people=(
                len(set(system.face_recognition.known_face_names))
                if system.face_recognition
                else 0
            ),
            pets=(
                len(system.animal_recognition.known_pets)
                if system.animal_recognition
                else 0
            ),
        )

    return app


def serve(system, host: str, port: int) -> Optional[threading.Thread]:
    """
    Start the web UI on a daemon thread, or not at all when WEB_UI_PASSWORD is unset.

    Returns:
        The server thread, or None when the UI stays off.
    """
    import logging

    password = os.environ.get("WEB_UI_PASSWORD")
    if not password:
        logging.getLogger(__name__).warning(
            "WEB_UI_PASSWORD not set: web UI off. Detection still runs."
        )
        return None
    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(
            create_app(system, password), host=host, port=port, log_level="warning"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    logging.getLogger(__name__).info(f"Web UI on http://{host}:{port}")
    return thread
