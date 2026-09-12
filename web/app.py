"""
The web UI: live view, events, enrolment, cameras, tier switch.

FastAPI + Jinja2 + htmx, no build step. Every route sits behind HTTP Basic
auth against ``WEB_UI_PASSWORD``; without that variable the server does not
start at all — detection and Telegram carry on without it.
"""

import html
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config.camera_config import DEFAULT_PORTS, CameraConfig, build_camera_url
from core.animal_recognition import ANIMAL_CLASSES
from database.models import Device, NotificationSettings, WhitelistEntry

HERE = Path(__file__).parent
PHOTOS = Path("data/detection_photos")
ENROL = {"human": Path("data/faces"), "animal": Path("data/animals")}
templates = Jinja2Templates(directory=HERE / "templates")
templates.env.filters["basename"] = lambda p: Path(p).name
basic = HTTPBasic()


def mjpeg(system):
    """Yield the newest annotated frame as a multipart JPEG part, ten times a second."""
    while (frame := system.latest_frame) is not None:  # ends when detection stops
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        time.sleep(0.1)


def create_app(system, password: str) -> FastAPI:
    """
    Build the app around a running ``IntruderDetectionSystem``.

    Args:
        system: Exposes db_manager, camera_manager, face_recognition,
            animal_recognition, latest_frame, start/stop_detection, apply_tier.
        password: What every request must present (any username).
    """

    def authed(credentials: HTTPBasicCredentials = Depends(basic)) -> None:
        if not secrets.compare_digest(credentials.password.encode(), password.encode()):
            raise HTTPException(401, headers={"WWW-Authenticate": "Basic"})

    app = FastAPI(dependencies=[Depends(authed)], docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    db = system.db_manager

    def photo_count(entry: WhitelistEntry) -> int:
        return 1 + (
            len(json.loads(entry.multiple_photos)) if entry.multiple_photos else 0
        )

    def page(request: Request, name: str, **ctx):
        return templates.TemplateResponse(
            request,
            name,
            {"tier": system.settings.tier, "photo_count": photo_count, **ctx},
        )

    @app.get("/")
    def live(request: Request):
        cameras = db.get_all_devices()
        active = getattr(system.camera_manager, "active_camera", None)
        return page(
            request,
            "live.html",
            cameras=cameras,
            active=active,
            detecting=system.detection_active,
        )

    @app.get("/stream")
    def stream():
        return StreamingResponse(
            mjpeg(system), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.post("/detection")
    def toggle_detection(on: bool = Form(...)):
        (system.start_detection if on else system.stop_detection)()
        return RedirectResponse("/", 303)

    @app.post("/tier")
    def set_tier(tier: str = Form(...)):
        if tier not in ("low", "high"):
            raise HTTPException(400, "tier must be low or high")
        threading.Thread(target=system.apply_tier, args=(tier,), daemon=True).start()
        return RedirectResponse("/", 303)

    @app.get("/events")
    def events(request: Request, type: Optional[str] = None, limit: int = 50):
        rows = db.get_recent_detections(limit=limit, detection_type=type or None)
        return page(request, "events.html", rows=rows, type=type or "", limit=limit)

    @app.get("/photo/{name}")
    def photo(name: str):
        path = PHOTOS / Path(name).name  # no directories: the name is the whole key
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/people")
    def people(request: Request):
        return page(
            request, "people.html", rows=db.get_whitelist_entries(entity_type="human")
        )

    @app.get("/pets")
    def pets(request: Request):
        return page(
            request,
            "pets.html",
            rows=db.get_whitelist_entries(entity_type="animal"),
            classes=ANIMAL_CLASSES,
        )

    @app.post("/enrol/{kind}")
    def enrol(
        kind: str,
        name: str = Form(...),
        photos: List[UploadFile] = File(...),
        class_id: int = Form(16),
    ):
        if kind not in ENROL or not name.strip():
            raise HTTPException(400)
        slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
        ENROL[kind].mkdir(parents=True, exist_ok=True)
        paths = []
        for i, photo in enumerate(photos):
            path = ENROL[kind] / f"{slug}_{int(time.time())}_{i}.jpg"
            path.write_bytes(photo.file.read())
            paths.append(str(path))
        db.create_whitelist_entry(
            WhitelistEntry(
                name=name.strip(),
                entity_type=kind,
                image_path=paths[0],
                multiple_photos=json.dumps(paths[1:]) if len(paths) > 1 else None,
                coco_class_id=class_id if kind == "animal" else None,
                individual_id=slug if kind == "animal" else None,
            )
        )
        reload_roster(kind)
        return RedirectResponse("/people" if kind == "human" else "/pets", 303)

    @app.delete("/enrol/{kind}/{entry_id}")
    def unenrol(kind: str, entry_id: int):
        entry = db.get_whitelist_entry(entry_id)
        if entry:
            for path in [entry.image_path] + (
                json.loads(entry.multiple_photos) if entry.multiple_photos else []
            ):
                Path(path).unlink(missing_ok=True)
            db.delete_whitelist_entry(entry_id)
            reload_roster(entry.entity_type)
        return HTMLResponse("")  # htmx swaps the row away

    def reload_roster(kind: str) -> None:
        rows = [e.to_dict() for e in db.get_whitelist_entries(entity_type=kind)]
        if kind == "human" and system.face_recognition:
            system.face_recognition.load_known_faces(rows)
        if kind == "animal" and system.animal_recognition:
            system.animal_recognition.load_known_pets(rows)

    @app.get("/cameras")
    def cameras(request: Request):
        return page(request, "cameras.html", rows=db.get_all_devices())

    @app.post("/cameras")
    def add_camera(
        url: str = Form(""),
        protocol: str = Form("rtsp"),
        host: str = Form(""),
        port: int = Form(0),
        path: str = Form(""),
        username: str = Form(""),
        password: str = Form(""),
    ):
        if protocol not in DEFAULT_PORTS:
            raise HTTPException(400, "protocol must be rtsp, http or https")
        url = url.strip() or build_camera_url(
            protocol, host.strip(), port, path.strip(), username, password
        )
        errors = CameraConfig(url=url).validate()
        if errors:
            raise HTTPException(400, errors["url"])
        db.create_device(Device(url=url))
        system.reload_camera_configurations()
        return RedirectResponse("/cameras", 303)

    @app.post("/cameras/{device_id}/test")
    def test_camera(device_id: int):
        device = db.get_device(device_id)
        ok, message = (
            CameraConfig(url=device.url).test_connection()
            if device
            else (False, "no such camera")
        )
        return HTMLResponse(
            f'<span class="{"ok" if ok else "bad"}">{html.escape(message)}</span>'
        )

    @app.delete("/cameras/{device_id}")
    def delete_camera(device_id: int):
        db.delete_device(device_id)
        system.reload_camera_configurations()
        return HTMLResponse("")

    @app.get("/telegram")
    def telegram(request: Request):
        return page(request, "telegram.html", rows=db.get_all_notification_settings())

    @app.post("/telegram")
    def add_telegram_user(
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
        return RedirectResponse("/telegram", 303)

    @app.delete("/telegram/{chat_id}")
    def delete_telegram_user(chat_id: int):
        db.delete_notification_settings(chat_id)
        reload_telegram_users()
        return HTMLResponse("")

    def reload_telegram_users() -> None:
        if getattr(system, "notification_system", None):
            rows = [
                u.to_dict() for u in db.get_all_notification_settings(status="open")
            ]
            system.notification_system.load_users(rows)

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
