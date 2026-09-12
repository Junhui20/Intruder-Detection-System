"""
One-line captions for alert photos from a local vision-language model via Ollama.

``qwen2.5vl:3b`` (low tier, 3.2 GB) or ``qwen2.5vl:7b`` (high tier, 6 GB, wants
8 GB of VRAM — on a 4 GB card it spills to the CPU and takes 14 s). Not
``moondream``: in Ollama 0.34 it drops the first tokens of every answer after
the first one ("ers, they are playing a game"). Nothing leaves the machine:
Ollama runs on localhost and the frame goes to it as JPEG. The feature switches
itself off when Ollama or the model is not there.
"""

import base64
import logging
from typing import Optional

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

TIER_MODELS = {"low": "qwen2.5vl:3b", "high": "qwen2.5vl:7b"}
PROMPT = (
    "Home security camera frame. Describe the {subject} in one sentence: "
    "appearance, what they are doing, and where they are in the scene. "
    "Answer with the sentence only."
)


class EventCaptioner:
    """
    Describes what an alert frame shows, in one sentence.

    Args:
        model: Ollama model name; must already be pulled (``scripts/setup_ollama.py``).
        host: Ollama server; the default is the one ``ollama serve`` binds.
        timeout: Seconds to wait for a caption before giving up on it (the photo is
            already delivered; a late caption only delays the edit).
    """

    def __init__(
        self,
        model: str = "qwen2.5vl:3b",
        host: str = "http://localhost:11434",
        timeout: float = 120,  # the 3b needs ~65 s on a 4-thread CPU; the photo is already out
        options: Optional[dict] = None,
    ):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        # Ollama request options; bench.py adds num_gpu=0/num_thread=4 for the CPU tier
        self.options = {"num_predict": 60, "temperature": 0.2, **(options or {})}
        self.enabled = self._model_is_served() and self._warm()

    def _model_is_served(self) -> bool:
        try:
            names = [
                m["name"]
                for m in requests.get(f"{self.host}/api/tags", timeout=2).json()[
                    "models"
                ]
            ]
        except (requests.RequestException, ValueError, KeyError) as e:
            logger.warning(
                f"Event captions off: Ollama not reachable at {self.host} ({e})"
            )
            return False
        if not any(n == self.model or n.startswith(self.model + ":") for n in names):
            logger.warning(
                f"Event captions off: model {self.model} not pulled (have {names or 'none'})"
            )
            return False
        logger.info(f"Event captions: {self.model} at {self.host}")
        return True

    def _warm(self) -> bool:
        """Load the model now and keep it resident; a cold load takes 20 s+."""
        try:
            requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "keep_alive": -1, "options": self.options},
                timeout=120,
            ).raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning(f"Event captions off: {self.model} failed to load ({e})")
            return False

    def describe(self, frame: np.ndarray, subject: str = "person") -> Optional[str]:
        """
        Caption one frame.

        Args:
            frame: BGR image, typically the saved alert screenshot.
            subject: What the detector found — ``person``, ``dog`` — so the model
                describes that rather than the wallpaper.

        Returns:
            One sentence, or None when disabled, timed out, or the model returned nothing.
        """
        if not self.enabled:
            return None
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        try:
            reply = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": PROMPT.format(subject=subject),
                    "images": [base64.b64encode(jpeg.tobytes()).decode()],
                    "stream": False,
                    "keep_alive": -1,
                    "options": self.options,
                },
                timeout=self.timeout,
            )
            body = reply.json()
            if "error" in body:
                raise requests.RequestException(body["error"])
            text = body.get("response", "").strip()
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"Caption failed: {e}")
            return None
        return text.split("\n")[0].strip() or None
