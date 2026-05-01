"""MegaDetector v6 wrapper (class-agnostic animal localizer)."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Detection:
    box_xyxy: tuple[int, int, int, int]
    score: float
    label: str  # "animal" | "person" | "vehicle"


class Detector:
    """Thin wrapper around PytorchWildlife.MegaDetectorV6.

    Loads weights once at construction. Returns Detection records with
    absolute pixel xyxy coordinates.
    """

    def __init__(self, device: str = "auto", det_threshold: float = 0.2,
                 keep_labels: tuple[str, ...] = ("animal",)):
        from PytorchWildlife.models import detection as pw_detection

        self.device = self._resolve_device(device)
        self.det_threshold = det_threshold
        self.keep_labels = set(keep_labels)
        self._model = pw_detection.MegaDetectorV6(device=self.device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection on a single RGB image (H, W, 3) numpy array."""
        result = self._model.single_image_detection(
            image, det_conf_thres=self.det_threshold
        )
        return self._to_detections(result)

    def detect_batch(self, images: Iterable[np.ndarray],
                     batch_size: int = 8) -> list[list[Detection]]:
        """Run detection on a batch of RGB image arrays."""
        images = list(images)
        results = self._model.batch_image_detection(
            images, batch_size=batch_size, det_conf_thres=self.det_threshold
        )
        return [self._to_detections(r) for r in results]

    def _to_detections(self, result: dict) -> list[Detection]:
        det = result["detections"]
        # supervision.Detections object: .xyxy (N,4) abs px, .confidence (N,), .class_id (N,)
        out: list[Detection] = []
        for box, conf, cid in zip(det.xyxy, det.confidence, det.class_id):
            label = self._model.CLASS_NAMES.get(int(cid), str(cid))
            if label not in self.keep_labels:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            out.append(Detection(
                box_xyxy=(x1, y1, x2, y2),
                score=float(conf),
                label=label,
            ))
        return out
