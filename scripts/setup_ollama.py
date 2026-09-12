#!/usr/bin/env python3
"""
Install Ollama and pull the caption model for a tier.

    python scripts/setup_ollama.py            # tier from config.yaml
    python scripts/setup_ollama.py --tier high

Linux/macOS use Ollama's own installer (asks for sudo); Windows gets a link to
the installer. Then the model for the tier is pulled: qwen2.5vl:3b (3.2 GB) or
qwen2.5vl:7b (6 GB).
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from core.event_captions import TIER_MODELS  # noqa: E402

HOST = "http://localhost:11434"


def serving() -> bool:
    try:
        return requests.get(f"{HOST}/api/tags", timeout=2).ok
    except requests.RequestException:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--tier", choices=TIER_MODELS, help="default: tier in config.yaml"
    )
    args = parser.parse_args()
    if not args.tier:
        with open(ROOT / "config.yaml") as f:
            args.tier = (yaml.safe_load(f) or {}).get("tier", "low")
    model = TIER_MODELS[args.tier]

    if not serving():
        if not shutil.which("ollama"):
            if platform.system() == "Windows":
                print(
                    "Install Ollama from https://ollama.com/download/windows, then rerun."
                )
                return 1
            print("Installing Ollama (its installer will ask for sudo)...")
            subprocess.run(
                "curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True
            )
        subprocess.Popen(
            ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        for _ in range(20):
            if serving():
                break
            time.sleep(0.5)
        else:
            print("Ollama did not start; run `ollama serve` yourself and rerun.")
            return 1

    print(f"Pulling {model} for tier '{args.tier}'...")
    with requests.post(
        f"{HOST}/api/pull", json={"name": model}, stream=True, timeout=None
    ) as pull:
        for line in pull.iter_lines():
            status = json.loads(line)
            if "error" in status:
                print(status["error"])
                return 1
            if status.get("total"):
                print(
                    f"\r  {100 * status.get('completed', 0) // status['total']:3d}%",
                    end="",
                )
    print(
        f"\nDone. Captions will use {model}; set captions.model in config.yaml to override."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
