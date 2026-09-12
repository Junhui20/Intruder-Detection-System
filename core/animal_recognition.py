"""
Individual pet re-identification on DINOv2 embeddings.

YOLO says "a dog"; this module says "Jacky". Each enrolled photo becomes a
DINOv2 embedding, an animal crop from the camera is matched by cosine
similarity, and the best of a pet's photos decides. ``dinov2-small`` (low
tier) or ``dinov2-base`` (high tier), both Apache-2.0, ~90 / ~350 MB, fetched
from Hugging Face on first use.

Measured on DogFaceNet (1393 dogs, aligned face crops), one enrolled pet
against 300 strangers, three enrolment photos: small 92.6 % of the pet's
photos recognised at 1 % false accepts (cosine ≥ 0.66); base 94.2 % (≥ 0.62).
Those are web face crops, not doorway body crops — treat them as an upper
bound. ``python bench.py pet-eval`` reproduces them.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    import torch
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    AutoModel = None

logger = logging.getLogger(__name__)

TIER_MODELS = {"low": "facebook/dinov2-small", "high": "facebook/dinov2-base"}
ANIMAL_CLASSES = {
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
}


class AnimalRecognitionSystem:
    """
    Tells an enrolled pet apart from any other animal of its kind.

    Args:
        confidence_threshold: Minimum YOLO confidence for an animal box to be considered.
        pet_identification_threshold: Minimum cosine similarity to name a pet.
        model: Hugging Face id of the DINOv2 checkpoint.
        use_gpu: Run on CUDA when available.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        pet_identification_threshold: float = 0.65,
        model: str = "facebook/dinov2-small",
        use_gpu: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.pet_identification_threshold = pet_identification_threshold
        self.model_name = model
        self.known_pets: Dict[str, dict] = {}  # name -> {class_id, embeddings}
        self.processing_times: List[float] = []
        self.animals_processed = 0
        self.pets_recognized = 0
        self.model = None
        if AutoModel is None:
            logger.warning("transformers not installed; pet identification disabled")
            return
        try:
            self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            self.processor = AutoImageProcessor.from_pretrained(model)
            self.model = AutoModel.from_pretrained(model).eval().to(self.device)
            logger.info(f"Pet re-ID: {model} on {self.device}")
        except Exception as e:
            logger.error(f"Pet re-ID model {model} failed to load: {e}")

    def embed(self, images: List[np.ndarray]) -> np.ndarray:
        """
        L2-normalised DINOv2 CLS embeddings of BGR crops, one row per image.

        Args:
            images: Animal crops of any size.

        Returns:
            (n, dim) float32 array; rows have unit norm.
        """
        rows = []
        for i in range(0, len(images), 32):  # batches of 32 fit a 4 GB card
            rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images[i : i + 32]]
            pixels = self.processor(images=rgb, return_tensors="pt")["pixel_values"]
            with torch.no_grad():
                cls = self.model(pixel_values=pixels.to(self.device)).last_hidden_state[
                    :, 0
                ]
            rows.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        return np.concatenate(rows)

    def add_known_pet(self, name: str, class_id: int, image_paths: List[str]) -> bool:
        """
        Enrol a pet from its photos. Three or more photos is what the eval assumed.

        Args:
            name: The pet's name as shown in alerts.
            class_id: COCO class (15 cat, 16 dog, ...); only boxes of that class are compared.
            image_paths: Photos of the pet, ideally cropped to the animal.

        Returns:
            True if at least one photo was readable.
        """
        images = [
            im
            for p in image_paths
            if os.path.exists(p) and (im := cv2.imread(p)) is not None
        ]
        if class_id not in ANIMAL_CLASSES:
            logger.warning(
                f"{name} not enrolled: class {class_id!r} is not an animal YOLO reports"
            )
            return False
        if not images or self.model is None:
            logger.warning(f"No photos enrolled for {name}")
            return False
        self.known_pets[name] = {"class_id": class_id, "embeddings": self.embed(images)}
        return True

    def load_known_pets(self, pets_data: List[Dict]) -> None:
        """
        Replace the roster with whitelist rows (``name``/``individual_id``,
        ``coco_class_id``, ``image_path``, optional JSON ``multiple_photos``).
        """
        roster = {}
        for row in pets_data:
            try:
                extra = json.loads(row.get("multiple_photos") or "[]")
            except ValueError:
                extra = []  # a bad row loses its extra photos, not the whole roster
            name = row.get("individual_id") or row["name"]
            if self.add_known_pet(
                name, row["coco_class_id"], [row["image_path"]] + extra
            ):
                roster[name] = self.known_pets.pop(name)
        self.known_pets = (
            roster  # one assignment: the detection thread reads it mid-frame
        )
        logger.info(f"Enrolled {len(self.known_pets)} pet(s)")

    def identify_animals(
        self, frame: np.ndarray, animal_detections: List[Dict]
    ) -> List[Dict]:
        """
        Annotate animal detections with ``pet_identity``, ``identification_confidence``
        and ``recognition_status`` (``known_pet`` / ``unknown_animal``).

        Args:
            frame: BGR frame the detections came from.
            animal_detections: Dicts with ``bbox`` and ``class_id`` from the detector.

        Returns:
            The same list, annotated in place.
        """
        start = time.time()
        for det in animal_detections:
            x1, y1, x2, y2 = det["bbox"]
            crop = frame[max(0, y1) : y2, max(0, x1) : x2]
            name, score = None, 0.0
            candidates = {
                n: p
                for n, p in self.known_pets.items()
                if p["class_id"] == det["class_id"]
            }
            if candidates and crop.size:
                try:
                    query = self.embed([crop])[0]
                except Exception as e:  # one bad crop must not cost the frame
                    logger.warning(f"Pet embedding failed: {e}")
                    candidates = {}
                for n, pet in candidates.items():
                    sim = float((pet["embeddings"] @ query).max())
                    if sim >= self.pet_identification_threshold and sim > score:
                        name, score = n, sim
            self.animals_processed += 1
            self.pets_recognized += bool(name)
            det.update(
                pet_identity=name
                or f"Unknown {ANIMAL_CLASSES.get(det['class_id'], 'animal')}",
                identification_confidence=score,
                recognition_status="known_pet" if name else "unknown_animal",
            )
        self.processing_times = (self.processing_times + [time.time() - start])[-100:]
        return animal_detections

    def update_confidence_thresholds(
        self,
        general_threshold: Optional[float] = None,
        pet_threshold: Optional[float] = None,
    ) -> None:
        """Set either threshold (0–1); None leaves it alone."""
        if general_threshold is not None and 0 <= general_threshold <= 1:
            self.confidence_threshold = general_threshold
        if pet_threshold is not None and 0 <= pet_threshold <= 1:
            self.pet_identification_threshold = pet_threshold

    def get_performance_stats(self) -> Dict:
        """Counters and timing for the monitoring UI."""
        return {
            "model": self.model_name if self.model is not None else "none",
            "total_animals_processed": self.animals_processed,
            "known_pet_identifications": self.pets_recognized,
            "avg_processing_time": (
                float(np.mean(self.processing_times)) if self.processing_times else 0.0
            ),
            "pet_identification_threshold": self.pet_identification_threshold,
            "known_pets_count": len(self.known_pets),
        }
