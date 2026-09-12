#!/usr/bin/env python3
"""
Reproducible numbers for the README.

    python bench.py pet-eval [--tier low|high] [--photos 3]

pet-eval downloads DogFaceNet_224resized (Zenodo 12578449, CC-BY-4.0, 72 MB)
to ~/.cache/petreid and runs the home protocol: one enrolled dog against 300
strangers, best-of-K photos, thresholds at 1 % and 5 % false accepts.
"""

import argparse
import io
import random
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.append(str(Path(__file__).resolve().parent))
from core.animal_recognition import TIER_MODELS, AnimalRecognitionSystem  # noqa: E402

DOGFACENET = (
    "https://zenodo.org/api/records/12578449/files/DogFaceNet_224resized.zip/content"
)
CACHE = Path.home() / ".cache" / "petreid"


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
    system = AnimalRecognitionSystem(model=TIER_MODELS[tier])
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
        f"{TIER_MODELS[tier]} on {system.device}, {photos} enrolment photo(s), "
        f"{len(enrolled)} pets × {len(strangers)} strangers, {per_image:.0f} ms/image"
    )
    for far in (0.01, 0.05):
        threshold = np.quantile(negatives, 1 - far)
        print(
            f"  FAR {far:.0%}: threshold {threshold:.2f} -> TPR {(positives >= threshold).mean():.1%}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pet-eval", help="pet re-ID TPR/FAR on DogFaceNet")
    p.add_argument("--tier", choices=TIER_MODELS, default="low")
    p.add_argument("--photos", type=int, default=3, help="enrolment photos per pet")
    args = parser.parse_args()
    pet_eval(args.tier, args.photos)
