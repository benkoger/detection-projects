"""MegaDetector v6 wrapper (class-agnostic animal localizer)."""

from contextlib import contextmanager
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

    # MegaDetector v6 variants exposed by PytorchWildlife. yolov9-c is the
    # default: compact yolov9 backbone, fast on a single GPU and accurate
    # enough for general camera-trap use.
    DEFAULT_VERSION = "MDV6-yolov9-c"

    def __init__(self, device: str = "auto", det_threshold: float = 0.2,
                 keep_labels: tuple[str, ...] = ("animal",),
                 version: str = DEFAULT_VERSION):
        self._allowlist_ultralytics_globals()
        from PytorchWildlife.models import detection as pw_detection

        self.device = self._resolve_device(device)
        self.det_threshold = det_threshold
        self.keep_labels = set(keep_labels)
        with self._weights_only_false():
            self._model = pw_detection.MegaDetectorV6(
                device=self.device, version=version,
            )

    @staticmethod
    def _allowlist_ultralytics_globals() -> None:
        # PyTorch 2.6 made torch.load default to weights_only=True, which
        # rejects the pickled ultralytics classes inside MegaDetector .pt
        # checkpoints. Ultralytics' own torch_load patch tries to set
        # weights_only=False but doesn't always take effect (e.g. on torch
        # 2.11). Whitelisting the relevant classes makes the safe path work.
        import torch

        try:
            from ultralytics.nn.tasks import DetectionModel
            from ultralytics.nn.modules import (
                Conv, C2f, SPPF, Detect, DFL, Bottleneck, C3, C2,
            )
        except Exception:
            return

        safe = [DetectionModel, Conv, C2f, SPPF, Detect, DFL, Bottleneck, C3, C2]
        # Pull anything else ultralytics expects to round-trip; ignore
        # missing names (different ultralytics versions ship slightly
        # different module sets).
        for name in ("Concat", "Upsample", "C2fAttn", "RepC3", "ELAN1",
                     "AConv", "ADown", "RepNCSPELAN4", "SPPELAN", "Silence"):
            try:
                mod = __import__("ultralytics.nn.modules", fromlist=[name])
                safe.append(getattr(mod, name))
            except (ImportError, AttributeError):
                pass

        try:
            torch.serialization.add_safe_globals(safe)
        except Exception:
            pass

    @staticmethod
    @contextmanager
    def _weights_only_false():
        # Force torch.load default to weights_only=False while inside this
        # block. Ultralytics' own monkeypatch tries to do this but isn't
        # taking effect on torch 2.11. We trust the cached MegaDetector
        # checkpoint, so this is safe in our context.
        import torch

        original = torch.load

        def patched(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original(*args, **kwargs)

        torch.load = patched
        try:
            yield
        finally:
            torch.load = original

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
