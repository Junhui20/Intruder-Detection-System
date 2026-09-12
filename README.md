# Everything Ring, Arlo and Google charge $100–200 a year for — familiar faces, AI event captions — running on your own hardware, zero subscription. Plus one thing they don't sell: recognising *your own* pet.

A local intruder-detection system for the home. A camera (an IP camera, an
RTSP stream, or an old phone running DroidCam) feeds a YOLO detector; people and
animals are matched against the faces and pets you enrolled; every alert goes
to Telegram with a short caption written by a local vision-language model.
Nothing leaves your LAN.

> **Status: mid-rework.** The original desktop (tkinter) version is tagged
> [`v1-tkinter`](../../tree/v1-tkinter). `main` is being evolved in place — one
> problem per commit — towards the design below. Sections marked *placeholder*
> are filled in as their work lands; see [Roadmap](#roadmap).

## Two tiers, two proofs

The same features run in two model-size presets. **Low** proves that "local"
does not require good hardware. **High** proves that scaling up does not raise
the bill. Paid-tier features elsewhere — familiar faces, pet recognition, AI
captions — are available in *both* tiers; only capacity (camera count, live-view
protocol, retention) depends on the hardware.

| | **Low** | **High** |
|---|---|---|
| Reference hardware | CPU-only, 4 threads *(proxy for a Pi 5 / N100 mini PC)* | 2020 gaming laptop, RTX 3050 4 GB |
| Detector | YOLO11n *(vs YOLO26n — settled by benchmark)* | YOLO11n / YOLO26n |
| Face recognition | InsightFace `buffalo_sc` | InsightFace `buffalo_l` |
| Pet re-ID | DINOv2-small embeddings | DINOv2-base embeddings |
| Event captions | Ollama `moondream` | Ollama `qwen2.5vl` |
| Detection FPS (1 cam, 640 px) | *placeholder* | *placeholder* |
| Face-ID latency | *placeholder* | *placeholder* |
| Caption latency | *placeholder* | *placeholder* |
| Pet re-ID latency | *placeholder* | *placeholder* |
| RAM idle / under load | *placeholder* | *placeholder* |
| Pi 5 / N100 (real numbers) | *contribute yours* | — |

Every number will ship with a reproducible `python bench.py --tier low|high`.
Per-feature overrides live in `config.yaml`, so you can mix presets.

## What it costs elsewhere

| | Familiar faces | AI captions | Price (USD) | Price (MYR) |
|---|---|---|---|---|
| Google Home Premium | paid tier | yes | $10/mo or $100/yr; Advanced $200/yr | *placeholder* |
| Arlo Secure | Plus tier | yes (Secure 6) | $7.99 single / $12.99 unlimited / Plus $17.99/mo | *placeholder* |
| Ring Home Premium | yes | Video Descriptions, Premium only | $20/mo, $200/yr | *placeholder* |
| Blue Iris | via add-on | via add-on | $69.95–$99.95 one-time (Windows NVR) | *placeholder* |
| Frigate | yes (0.16+) | yes (0.15+) | free | free |
| **This project** | yes | yes (local VLM) | **free** | **free** |

USD prices verified September 2026. MYR column and local alternatives
(Tapo Care, Imou, Ezviz) are *placeholder* pending research.

## Why not Frigate?

[Frigate](https://frigate.video) is free, excellent, and better than this
project at being an NVR: 24/7 recording, timeline, Home Assistant, hardware
accelerators. If you want an NVR, use Frigate. This project does not compete
with it. It is a smaller, opinionated alert system with one feature Frigate
does not have — recognising an individual pet rather than "a dog" — and a
codebase small enough to read in an afternoon, which is the point of a
portfolio piece.

## Features

- **Detection** — YOLO11n, person tracking, 8 animal classes
  (cat, dog, horse, sheep, cow, elephant, bear, zebra), configurable thresholds,
  timer-based alerts for unknown people.
- **Familiar faces** — enrol people from photos; alerts say who it was.
  InsightFace (SCRFD + ArcFace) on ONNX Runtime, `buffalo_sc` low / `buffalo_l` high.
- **Your own pet** — enrol a pet from photos; alerts distinguish *your* cat from
  *a* cat. *(Planned: embedding-based re-ID replacing today's colour heuristic.)*
- **Event captions** — a local vision-language model writes one line per alert
  ("person in a red jacket at the side gate"). *(Planned, via Ollama; off
  gracefully if Ollama is not running.)*
- **Cameras** — IP cameras over HTTP/HTTPS, DroidCam (an old phone as a
  camera), RTSP *(planned as first-class; works today via `custom_url`)*,
  local webcam fallback, multi-camera.
- **Telegram** — alerts with photo, per-user notification settings, bot
  commands *(planned: `/status /snapshot /arm /disarm /mute 1h`, and
  `/enroll <name>` by replying to an alert photo)*.
- **Web UI** — enrolment, event history, MJPEG live view, camera management,
  tier switch. *(Planned, FastAPI + htmx, replacing the tkinter desktop app.)*
- **Storage** — SQLite, no server.

## Quick start

```bash
git clone https://github.com/Junhui20/Intruder-Detection-System.git
cd Intruder-Detection-System
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/setup_secure_config.py   # writes .env with your Telegram token
python main.py                          # add --headless to run without the desktop UI
```

Optional extras: `requirements-optional.txt` (MediaPipe, dlib for the legacy pet path),
`requirements-dev.txt` (pytest, black, flake8). GPU users: `python
scripts/install.py --gpu` installs the CUDA build of PyTorch.

Python 3.12 and 3.14 are verified. **Linux and Windows** are supported; macOS is
untested.

## Security

- Secrets (Telegram token) come from environment variables or `.env`, never
  from `config.yaml`. See [docs/SECURITY.md](docs/SECURITY.md).
- The web UI *(planned)* will refuse to start without `WEB_UI_PASSWORD` set,
  and binds to the LAN only.
- **Never port-forward this to the internet.** A camera system exposed to the
  open internet is a public webcam with a login prompt. For remote access use
  [Tailscale](https://tailscale.com) (or any WireGuard-style overlay): your
  phone joins your LAN; nothing is exposed.

## Documentation

- [Installation](docs/INSTALLATION.md)
- [Security](docs/SECURITY.md)
- [Camera setup](docs/CAMERA_SETUP.md)
- [Telegram bot setup](docs/TELEGRAM_SETUP.md)
- [Development guide](docs/DEVELOPMENT.md) · [API reference](docs/API.md)
- [Changelog](docs/CHANGELOG.md)

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q      # ~30 s; external-dependency tests are skipped when the dependency is absent
black . && flake8
```

Conventions are in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): PEP 8 at 88
columns, absolute imports, Google-style docstrings, type hints on public
methods.

## Roadmap

| | Ticket | Status |
|---|---|---|
| T1 | Cleanup + this README skeleton | done |
| T2 | CI matrix: Ubuntu × Windows × Python 3.12/3.14 | |
| T3 | RTSP first-class | |
| T4 | InsightFace face backend (replaces LBPH) | done |
| T5 | Ollama event captions + setup script | |
| T6–T7 | Pet re-ID: model choice, enrol → embed → match, PetFace eval | |
| T8 | Web UI, delete tkinter | |
| T9 | Telegram commands, `/enroll` from an alert photo | |
| T10 | `bench.py` + both tiers' numbers | |
| T11–T12 | MYR prices, final README with screenshots | |

## Current desktop interface

Until the web UI lands the tkinter app is still the interface.

![Real-time detection](Images/Real_time.jpg)

![Entity management](Images/Human_page.jpg)

## License

MIT — see [LICENSE](LICENSE).
