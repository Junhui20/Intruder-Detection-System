#!/usr/bin/env python3
"""
Reproducible numbers for the README.

    python bench.py speed --tier low|high      # detection FPS, face / pet / caption latency, RAM
    python bench.py pet-eval [--tier low|high] [--photos 3]

`--tier low` is the CPU proxy: CUDA hidden, 4 threads (a Pi 5 / N100 class
budget on whatever CPU this is). `--tier high` uses the GPU. speed prints
markdown rows for the tier table.

pet-eval downloads DogFaceNet_224resized (Zenodo 12578449, CC-BY-4.0, 72 MB)
to ~/.cache/petreid and runs the home protocol: one enrolled dog against 300
strangers, best-of-K photos, thresholds at 1 % and 5 % false accepts.
"""

import argparse
import io
import os
import random
import sys
import time
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
sub = parser.add_subparsers(dest="cmd", required=True)
for name, help_ in (
    ("speed", "FPS, latencies and RAM for one tier"),
    ("pet-eval", "pet re-ID TPR/FAR on DogFaceNet"),
):
    p = sub.add_parser(name, help=help_)
    p.add_argument("--tier", choices=("low", "high"), default="low")
    p.add_argument(
        "--photos", type=int, default=3, help="pet-eval: enrolment photos per pet"
    )
args = parser.parse_args()
if args.tier == "low":  # must happen before torch is imported anywhere
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "4"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from core import animal_recognition, event_captions, face_recognition  # noqa: E402
from core.animal_recognition import AnimalRecognitionSystem  # noqa: E402
from core.detection_engine import DetectionEngine  # noqa: E402
from core.event_captions import EventCaptioner  # noqa: E402
from core.face_recognition import FaceRecognitionSystem  # noqa: E402

DOGFACENET = (
    "https://zenodo.org/api/records/12578449/files/DogFaceNet_224resized.zip/content"
)
CACHE = Path.home() / ".cache" / "petreid"
GPU = args.tier == "high"


def rss_mb() -> int:
    return psutil.Process().memory_info().rss // 2**20


def frame_with_people() -> np.ndarray:
    """A 640x480 frame with six faces in it — the InsightFace sample photo."""
    from insightface.data import get_image

    image = cv2.resize(get_image("t1"), (640, 443))
    return cv2.copyMakeBorder(image, 18, 19, 0, 0, cv2.BORDER_REPLICATE)


def timed(fn, n: int, warmup: int = 3) -> float:
    """Median milliseconds per call."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return 1000 * float(np.median(times))


def speed(tier: str) -> None:
    torch.set_num_threads(4 if tier == "low" else torch.get_num_threads())
    device = "cuda" if GPU and torch.cuda.is_available() else "cpu"
    label = f"{device}{', 4 threads' if device == 'cpu' else ''}"
    frame = frame_with_people()
    print(f"tier {tier} on {label}; idle RSS before models: {rss_mb()} MB\n")

    rows = []
    for weights in ("models/yolo11n.pt", "models/yolo26n.pt"):
        engine = DetectionEngine(weights, use_optimized_engine=False)
        ms = timed(lambda: engine.detect_objects(frame), 30)
        rows.append(
            f"| Detection, {Path(weights).stem}, 640 px | {1000 / ms:.1f} FPS ({ms:.0f} ms) |"
        )
    engine = DetectionEngine("models/yolo11n.pt", use_optimized_engine=True)
    ms = timed(lambda: engine.detect_objects(frame), 30)
    rows.append(
        f"| Detection as configured ({engine.model_format}) | {1000 / ms:.1f} FPS ({ms:.0f} ms) |"
    )

    faces = FaceRecognitionSystem(model=face_recognition.TIER_MODELS[tier], use_gpu=GPU)
    faces.known_face_names, faces.known_face_encodings = ["x"], [
        np.zeros(512, np.float32)
    ]
    boxes = [{"bbox": (0, 0, 640, 480), "track_id": 1}]
    ms = timed(lambda: faces.recognize_faces(frame, boxes), 20)
    rows.append(f"| Face-ID, {faces.model}, one frame with 6 faces | {ms:.0f} ms |")

    pets = AnimalRecognitionSystem(
        model=animal_recognition.TIER_MODELS[tier], use_gpu=GPU
    )
    crop = frame[100:400, 150:450]
    ms = timed(lambda: pets.embed([crop]), 20)
    rows.append(
        f"| Pet re-ID, {pets.model_name.split('/')[-1]}, one crop | {ms:.0f} ms |"
    )

    # the high tier also measures the low model on the GPU: the override for 4 GB cards
    for model in {
        tier: [event_captions.TIER_MODELS[tier]],
        "high": list(event_captions.TIER_MODELS.values())[::-1],
    }[tier]:
        try:  # unload first: num_gpu/num_thread only apply when the model loads
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=30,
            )
        except requests.RequestException:
            pass
        captioner = EventCaptioner(
            model, options={"num_gpu": 0, "num_thread": 4} if tier == "low" else None
        )
        if not captioner.enabled:
            rows.append(f"| Caption, {model} | not measured: Ollama or model missing |")
            continue
        # a different frame each call: Ollama caches the image tokens of a repeated prompt
        variants = [
            frame,
            cv2.flip(frame, 1),
            frame[20:, 30:],
            cv2.GaussianBlur(frame, (5, 5), 0),
        ]
        calls, got = iter(variants * 2), []
        ms = timed(
            lambda: got.append(captioner.describe(next(calls), "person")), 3, warmup=1
        )
        result = (
            f"{ms / 1000:.1f} s"
            if all(got)
            else f"timed out (> {captioner.timeout:.0f} s)"
        )
        rows.append(f"| Caption, {model} | {result} |")

    loaded = rss_mb()
    for _ in range(20):  # a burst of everything at once
        engine.detect_objects(frame)
        faces.recognize_faces(frame, boxes)
        pets.embed([crop])
    rows.append(
        f"| RAM (this process), models loaded / under load | {loaded} MB / {rss_mb()} MB |"
    )
    if device == "cuda":
        rows.append(
            f"| VRAM peak | {torch.cuda.max_memory_allocated() // 2**20} MB (torch) |"
        )

    print(f"| Measurement ({label}) | {tier} |\n|---|---|")
    print("\n".join(rows))


def dogfacenet() -> dict:
    """{dog_id: [image paths]} — downloaded on first call."""
    if not any(CACHE.glob("*/*/")):
        print("Downloading DogFaceNet_224resized (72 MB)...")
        CACHE.mkdir(parents=True, exist_ok=True)
        reply = requests.get(DOGFACENET, timeout=600)
        reply.raise_for_status()
        zipfile.ZipFile(io.BytesIO(reply.content)).extractall(CACHE)
    return {
        d.name: sorted(d.glob("*.jpg"))
        for d in CACHE.glob("*/*/")
        if d.is_dir() and not d.name.startswith(".")
    }


def pet_eval(tier: str, photos: int) -> None:
    system = AnimalRecognitionSystem(
        model=animal_recognition.TIER_MODELS[tier], use_gpu=GPU
    )
    dogs = {k: v for k, v in dogfacenet().items() if len(v) >= photos + 2}
    ids = sorted(dogs)
    random.Random(0).shuffle(ids)
    if len(ids) < 600:
        sys.exit(
            f"only {len(ids)} dogs have {photos + 2}+ photos; need 600 for 300 + 300"
        )
    enrolled, strangers = ids[:300], ids[300:600]
    read = lambda paths: [cv2.imread(str(p)) for p in paths]  # noqa: E731

    t = time.time()
    gallery = {d: system.embed(read(dogs[d][:photos])) for d in enrolled}
    own = {d: system.embed(read(dogs[d][photos : photos + 3])) for d in enrolled}
    stranger = system.embed(read([p for d in strangers for p in dogs[d][:3]]))
    n = sum(map(len, gallery.values())) + sum(map(len, own.values())) + len(stranger)
    per_image = 1000 * (time.time() - t) / n

    positives = np.concatenate([(own[d] @ gallery[d].T).max(1) for d in enrolled])
    negatives = np.concatenate([(stranger @ gallery[d].T).max(1) for d in enrolled])
    print(
        f"{system.model_name} on {system.device}, {photos} enrolment photo(s), "
        f"{len(enrolled)} pets × {len(strangers)} strangers, {per_image:.0f} ms/image"
    )
    for far in (0.01, 0.05):
        threshold = np.quantile(negatives, 1 - far)
        print(
            f"  FAR {far:.0%}: threshold {threshold:.2f} -> TPR {(positives >= threshold).mean():.1%}"
        )


if __name__ == "__main__":
    speed(args.tier) if args.cmd == "speed" else pet_eval(args.tier, args.photos)
