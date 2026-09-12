"""
Face recognition on InsightFace: SCRFD detection + ArcFace embeddings over ONNX Runtime.

One model pack per tier — ``buffalo_sc`` (low, MobileFaceNet) or ``buffalo_l``
(high, ResNet-50). Packs download to ``~/.insightface/models`` on first use.
A match is the cosine similarity between a face's embedding and the enrolled
ones; on a unit-normalised embedding that is a dot product.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

logger = logging.getLogger(__name__)

TIER_MODELS = {"low": "buffalo_sc", "high": "buffalo_l"}


class FaceRecognitionSystem:
    """
    Names the people YOLO found by matching their faces against enrolled photos.

    Args:
        confidence_threshold: Minimum cosine similarity to accept a match.
        max_faces_per_frame: Cap on person boxes processed per frame.
        model: InsightFace pack name, ``buffalo_sc`` or ``buffalo_l``.
        use_gpu: Run on CUDA when ``onnxruntime-gpu`` is installed.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.45,
        max_faces_per_frame: int = 10,
        model: str = "buffalo_sc",
        use_gpu: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_faces_per_frame = max_faces_per_frame
        self.model = model
        self.known_face_names: List[str] = []
        self.known_face_encodings: List[np.ndarray] = []
        # track_id -> {name, score, misses, frame}; carries an identity through
        # frames where the face is turned away
        self.track_identities: Dict[int, dict] = {}
        self.max_misses = 15
        self.frame_count = 0
        self.processing_times: List[float] = []
        self.faces_processed = 0
        self.faces_recognized = 0
        self.app = None
        self.backend_type = "none"
        if FaceAnalysis is None:
            logger.warning("insightface not installed; face recognition disabled")
            return
        # TensorRT is listed as available whenever onnxruntime-gpu is installed,
        # then fails to load without the TensorRT libraries; CUDA and CPU are enough.
        providers = [
            p
            for p in onnxruntime.get_available_providers()
            if p != "TensorrtExecutionProvider"
        ]
        if not use_gpu:
            providers = ["CPUExecutionProvider"]
        try:
            self.app = FaceAnalysis(
                name=model,
                providers=providers,
                allowed_modules=["detection", "recognition"],
            )
            self.app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1)
            self.backend_type = "insightface"
            logger.info(f"Face recognition: {model} on {providers[0]}")
        except Exception as e:
            logger.error(f"InsightFace pack {model} failed to load: {e}")

    def _embed(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Embedding of the most confident face in ``image``, or None."""
        faces = self.app.get(image)
        if not faces:  # tight crops give SCRFD no context; pad and retry
            h, w = image.shape[:2]
            padded = cv2.copyMakeBorder(
                image, h // 2, h // 2, w // 2, w // 2, cv2.BORDER_REPLICATE
            )
            faces = self.app.get(padded)
        return max(faces, key=lambda f: f.det_score).normed_embedding if faces else None

    def add_known_face(self, name: str, image_path: str) -> bool:
        """
        Enrol one photo of ``name``.

        Args:
            name: Person's name as stored in the whitelist.
            image_path: Photo containing their face.

        Returns:
            True if a face was found and enrolled.
        """
        image = (
            cv2.imread(image_path) if self.app and os.path.exists(image_path) else None
        )
        embedding = self._embed(image) if image is not None else None
        if embedding is None:
            logger.warning(f"No face enrolled for {name} from {image_path}")
            return False
        self.known_face_names.append(name)
        self.known_face_encodings.append(embedding)
        return True

    def remove_known_face(self, name: str) -> None:
        """Drop every enrolled embedding for ``name``."""
        keep = [i for i, n in enumerate(self.known_face_names) if n != name]
        self.known_face_names = [self.known_face_names[i] for i in keep]
        self.known_face_encodings = [self.known_face_encodings[i] for i in keep]

    def load_known_faces(self, faces_data: List[Dict]) -> None:
        """
        Replace the roster with whitelist rows (``name``, ``image_path`` and the
        optional JSON list ``multiple_photos``). Every photo of a person counts.

        Args:
            faces_data: Rows from the whitelist table; a photo with no detectable
                face is skipped, not fatal.
        """
        names, encodings = [], []
        for row in faces_data:
            extra = json.loads(row.get("multiple_photos") or "[]")
            for path in [row["image_path"], *extra]:
                image = cv2.imread(path) if self.app and os.path.exists(path) else None
                embedding = self._embed(image) if image is not None else None
                if embedding is None:
                    logger.warning(f"No face found for {row['name']} in {path}")
                    continue
                names.append(row["name"])
                encodings.append(embedding)
        # one assignment each: the detection thread reads these mid-frame
        self.known_face_names, self.known_face_encodings = names, encodings
        logger.info(f"Enrolled {len(set(names))} person(s), {len(names)} photo(s)")

    def recognize_faces(
        self, frame: np.ndarray, human_detections: List[Dict]
    ) -> List[Dict]:
        """
        Annotate person detections with ``identity``, ``face_confidence`` and
        ``recognition_status``.

        Args:
            frame: BGR frame the detections came from.
            human_detections: Dicts with ``bbox`` (x1, y1, x2, y2) and optional
                ``track_id``.

        Returns:
            The same list, annotated in place. Untouched when nobody is enrolled.
        """
        self.frame_count += 1
        self.track_identities = {
            t: v
            for t, v in self.track_identities.items()
            if self.frame_count - v["frame"] <= 30
        }
        names, encodings = (
            self.known_face_names,
            self.known_face_encodings,
        )  # one snapshot
        if not human_detections or not encodings:
            return human_detections
        start = time.time()
        known = np.stack(encodings)
        try:
            faces = self.app.get(frame)
        except Exception as e:  # a bad frame must not cost the caller its detections
            logger.error(f"Face detection failed: {e}")
            return human_detections
        taken = set()
        for det in human_detections[: self.max_faces_per_frame]:
            x1, y1, x2, y2 = det["bbox"]
            inside = [
                f
                for f in faces
                if x1 <= (f.bbox[0] + f.bbox[2]) / 2 <= x2
                and y1 <= (f.bbox[1] + f.bbox[3]) / 2 <= y2
            ]
            name, score = None, 0.0
            if inside:
                sims = known @ max(inside, key=lambda f: f.det_score).normed_embedding
                i = int(sims.argmax())
                if sims[i] >= self.confidence_threshold and names[i] not in taken:
                    name, score = names[i], float(sims[i])
            name, score = self._smooth(det.get("track_id"), name, score)
            if name:
                taken.add(name)
                self.faces_recognized += 1
            det.update(
                identity=name or "Unknown",
                face_confidence=score,
                recognition_status="known" if name else "unknown",
            )
            self.faces_processed += 1
        self.processing_times = (self.processing_times + [time.time() - start])[-100:]
        return human_detections

    def _smooth(
        self, track_id: Optional[int], name: Optional[str], score: float
    ) -> Tuple[Optional[str], float]:
        """Hold a track's identity across misses; only a stronger match replaces it."""
        if track_id is None:
            return name, score
        t = self.track_identities.setdefault(
            track_id, {"name": None, "score": 0.0, "misses": 0}
        )
        t["frame"] = self.frame_count
        if name and (t["name"] in (None, name) or score > t["score"]):
            t.update(
                name=name,
                score=score if t["name"] != name else 0.8 * t["score"] + 0.2 * score,
                misses=0,
            )
        else:
            t["misses"] += 1
            if t["misses"] > self.max_misses:
                t.update(name=None, score=0.0)
        return t["name"], t["score"]

    def update_confidence_threshold(self, new_threshold: float) -> None:
        """Set the minimum similarity for a match (0–1)."""
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold

    def get_performance_stats(self) -> Dict:
        """Counters and timing for the monitoring UI."""
        return {
            "backend": f"{self.backend_type}:{self.model}",
            "total_faces_processed": self.faces_processed,
            "successful_recognitions": self.faces_recognized,
            "recognition_accuracy": (
                100 * self.faces_recognized / self.faces_processed
                if self.faces_processed
                else 0
            ),
            "avg_processing_time": (
                float(np.mean(self.processing_times)) if self.processing_times else 0.0
            ),
            "confidence_threshold": self.confidence_threshold,
            "known_faces_count": len(set(self.known_face_names)),
            "active_tracks": len(self.track_identities),
        }
