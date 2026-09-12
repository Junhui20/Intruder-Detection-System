# Installation

Linux and Windows on Python 3.12 or 3.14 — that is what CI tests. macOS is
untested. No compiler is needed for anything in `requirements.txt`.

## 1. Python packages

```bash
git clone https://github.com/Junhui20/Intruder-Detection-System.git
cd Intruder-Detection-System
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Linux without an NVIDIA GPU, add `--extra-index-url https://download.pytorch.org/whl/cpu`
to skip 2.5 GB of CUDA wheels. With one, `pip install torch` picks the CUDA
build on its own; for face recognition on the GPU also
`pip install onnxruntime-gpu` (its CUDA version must match torch's — check with
`python -c "import onnxruntime as o; print(o.get_available_providers())"`).

`pip install -r requirements-dev.txt` adds pytest, black, flake8 and the
FastAPI test client.

## 2. Models

Downloaded on first run, so the first start needs internet:

| What | Where | Size |
|---|---|---|
| YOLO11n | in the repo (`models/`) | 5 MB |
| InsightFace pack, low / high tier | `~/.insightface/models` | 16 MB / 280 MB |
| DINOv2 small / base | `~/.cache/huggingface` | 88 MB / 346 MB |
| Caption model (optional) | Ollama, see step 4 | 3.2 GB / 6 GB |

## 3. Secrets

```bash
python scripts/setup_secure_config.py   # writes .env
```

`.env` holds `TELEGRAM_BOT_TOKEN` (from @BotFather) and `WEB_UI_PASSWORD`.
Neither is ever read from `config.yaml`. Without the token there are no alerts;
without the password the web UI does not start. Detection runs regardless.

## 4. Captions (optional)

```bash
python scripts/setup_ollama.py          # installs Ollama, pulls the tier's model
```

Linux/macOS use Ollama's own installer (it asks for sudo); Windows users get
the download link. Skip this and alerts arrive without the one-line description.

## 5. Run

```bash
python main.py
```

Open `http://<this machine>:8000`, add a camera on the Cameras page
([CAMERA_SETUP.md](CAMERA_SETUP.md) has the RTSP paths), your chat id on the
Telegram page ([TELEGRAM_SETUP.md](TELEGRAM_SETUP.md)), then enrol people and
pets. `config.yaml` → `tier: low|high` picks the model sizes; the nav bar
switch does the same without a restart.

## Troubleshooting

- **`python main.py` exits with a config error** — the message names the
  `config.yaml` key. `tier` must be `low` or `high`.
- **"Event captions off"** in the log — Ollama is not running or the model is
  not pulled; step 4. Everything else works.
- **Face recognition slow on a GPU machine** — `onnxruntime` (CPU) is
  installed rather than `onnxruntime-gpu`, or their CUDA versions differ.
  Face-ID still works, at CPU speed (60–100 ms per frame).
- **Camera test fails** — open the URL in VLC first; if VLC plays it, this
  system will. Set the camera's I-frame interval to 1–2 s.
- **Windows: "cannot access the file because it is being used"** on the
  database — another `main.py` is still running.
- **Everything is slow on a CPU** — `tier: low`, and confirm the log says
  `Selected optimal model: pytorch`; the ONNX exports are slower on a CPU.
