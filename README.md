[![tests](https://github.com/Junhui20/Intruder-Detection-System/actions/workflows/tests.yml/badge.svg)](https://github.com/Junhui20/Intruder-Detection-System/actions/workflows/tests.yml)

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
| Event captions | Ollama `qwen2.5vl:3b` | Ollama `qwen2.5vl:7b` |
| Detection FPS (1 cam, 640 px) | *placeholder* | *placeholder* |
| Face-ID latency | *placeholder* | *placeholder* |
| Caption latency | ~65 s (3b, CPU 4 threads) | 14 s (7b spills past 4 GB VRAM); 1.6 s with `captions.model: qwen2.5vl:3b` |
| Pet re-ID latency (per animal crop) | 66 ms (CPU 4 threads) | 32 ms (base) · 10 ms with small |
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
- **Your own pet** — enrol a pet from three or more photos; alerts distinguish
  *your* dog from *a* dog. DINOv2 embeddings, cosine match, `dinov2-small` low /
  `dinov2-base` high. Numbers and the caveat in [Pet re-ID](#pet-re-id).
- **Event captions** — a local vision-language model writes one line under each
  alert photo ("Blond woman in a black dress, holding cards, facing the camera").
  Ollama with `qwen2.5vl:3b` (low) / `qwen2.5vl:7b` (high); the photo goes out first,
  the caption is edited in when ready. Off, silently, when Ollama is not running.
- **Cameras** — RTSP (any Tapo / Hikvision / Dahua / Reolink / ONVIF camera,
  vendor paths in [docs/CAMERA_SETUP.md](docs/CAMERA_SETUP.md)), HTTP MJPEG,
  DroidCam (an old phone as a camera), local webcam fallback, multi-camera.
- **Telegram** — alerts with photo, per-user notification settings, bot
  commands *(planned: `/status /snapshot /arm /disarm /mute 1h`, and
  `/enroll <name>` by replying to an alert photo)*.
- **Web UI** — enrolment, event history, MJPEG live view, camera management,
  tier switch. *(Planned, FastAPI + htmx, replacing the tkinter desktop app.)*
- **Storage** — SQLite, no server.

## Pet re-ID

Enrol a pet with three or more photos; an animal box from the camera is embedded
with DINOv2 and matched by cosine similarity against them, best photo wins.
Evaluated on [DogFaceNet](https://zenodo.org/records/12578449) (CC-BY-4.0: 1,393
dogs, 8,363 aligned face crops) with the question a home actually asks — *is
this my pet, or some other dog?* — one enrolled dog against 300 strangers,
three enrolment photos, seed 0:

| | true-positive rate | at false-accept rate | threshold |
|---|---|---|---|
| `dinov2-small` (low) | 92.6 % | 1 % | cosine ≥ 0.66 |
| `dinov2-base` (high) | 94.2 % | 1 % | cosine ≥ 0.62 |
| either, looser | 98.6–98.7 % | 5 % | ≥ 0.49–0.54 |

Default threshold is 0.65 (`pet_identification_threshold`). `python bench.py
pet-eval` downloads the 72 MB set and reproduces the table.

**The gap:** those are web photos of dog faces, aligned and cropped. Your
camera sees a whole animal from above, in motion, at night. Treat the numbers
as an upper bound, and enrol photos taken by the camera itself. Cats are
untested — no freely downloadable individual-cat set was found. The PetFace
benchmark (257k individuals, 13 families) needs a research-use request form;
it was not used, and the alternative model considered (AvitoTech's CLIP) was
trained on it, which is also why it was evaluated here instead — it scored
78.8 % at 1 % FAR and was dropped.

## Quick start

```bash
git clone https://github.com/Junhui20/Intruder-Detection-System.git
cd Intruder-Detection-System
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/setup_secure_config.py   # writes .env with your Telegram token
python scripts/setup_ollama.py          # optional: installs Ollama + pulls the caption model
python main.py                          # add --headless to run without the desktop UI
```

`requirements-dev.txt` adds pytest, black and flake8. GPU users: `python
scripts/install.py --gpu` installs the CUDA build of PyTorch.

**Linux and Windows** on Python 3.12 and 3.14 are what CI tests; macOS is
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
python -m pytest tests/ -q      # ~30 s; needs internet once for the InsightFace pack
flake8                          # .flake8 ignores the v1 codebase's cosmetic noise; keep new lines clean
```

CI runs the same on Ubuntu and Windows × Python 3.12 and 3.14.

Conventions are in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): PEP 8 at 88
columns, absolute imports, Google-style docstrings, type hints on public
methods.

## Roadmap

| | Ticket | Status |
|---|---|---|
| T1 | Cleanup + this README skeleton | done |
| T2 | CI matrix: Ubuntu × Windows × Python 3.12/3.14 | done |
| T3 | RTSP first-class | done |
| T4 | InsightFace face backend (replaces LBPH) | done |
| T5 | Ollama event captions + setup script | done |
| T6–T7 | Pet re-ID: model choice, enrol → embed → match, eval | done |
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
