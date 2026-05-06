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

    # MegaDetector v6 variants exposed by PytorchWildlife. yolov9-e is the
    # extended yolov9 backbone (~60M params): ~1.5x slower than yolov9-c
    # but materially better small-object recall, which matters for
    # zoomed-out camera-trap deployments.
    DEFAULT_VERSION = "MDV6-yolov9-e"

    def __init__(self, device: str = "auto", det_threshold: float = 0.10,
                 keep_labels: tuple[str, ...] = ("animal",),
                 version: str = DEFAULT_VERSION):
        self._allowlist_ultralytics_globals()
        self._silence_ultralytics()
        from PytorchWildlife.models import detection as pw_detection

        self.device = self._resolve_device(device)
        self.det_threshold = det_threshold
        self.keep_labels = set(keep_labels)
        with self._weights_only_false():
            self._model = pw_detection.MegaDetectorV6(
                device=self.device, version=version,
            )

    @staticmethod
    def _silence_ultralytics() -> None:
        # Tiled inference makes ~20 detector calls per image. Ultralytics'
        # default per-call "0: 1280x1280 1 animal, 24.9ms" log line then
        # drowns out our own progress lines. Bump its logger to WARNING and
        # set the global "verbose=False" env var for good measure.
        import os
        os.environ["YOLO_VERBOSE"] = "False"
        try:
            from ultralytics.utils import LOGGER
            import logging
            LOGGER.setLevel(logging.WARNING)
        except Exception:
            pass

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

    def detect(self, image: np.ndarray,
               tile: bool = False,
               tile_size: int = 480,
               overlap: float = 0.2,
               nms_iou: float = 0.3) -> list[Detection]:
        """Run detection on a single RGB image (H, W, 3) numpy array.

        With ``tile=True`` (default), uses SAHI-style sliced inference: the
        full image plus a grid of overlapping ``tile_size`` tiles are each
        run through the detector, results are merged in full-image
        coordinates, and NMS deduplicates overlapping boxes (IoU > nms_iou).

        - **Zoomed-out shots**: tiny animals get effectively magnified by
          tiling, so MegaDetector sees them at training resolution.
        - **Zoomed-in shots**: the full-image pass catches large animals
          even if they straddle tile boundaries; NMS dominates because
          the full-image detection is higher confidence.
        - **Small images** (both sides ≤ tile_size): tiling is skipped and
          this collapses to a plain full-image detection.
        """
        H, W = image.shape[:2]

        if not tile or (W <= tile_size and H <= tile_size):
            return self._to_detections(
                self._model.single_image_detection(
                    image, det_conf_thres=self.det_threshold
                )
            )

        all_dets: list[Detection] = []

        # Full-image pass — catches large animals that span multiple tiles.
        all_dets.extend(self._to_detections(
            self._model.single_image_detection(
                image, det_conf_thres=self.det_threshold
            )
        ))

        # Sliced passes.
        stride = max(1, int(tile_size * (1.0 - overlap)))
        ys = list(range(0, max(H - tile_size, 0) + 1, stride))
        xs = list(range(0, max(W - tile_size, 0) + 1, stride))
        # Ensure the last row/column reaches the image edge.
        if ys and ys[-1] + tile_size < H:
            ys.append(H - tile_size)
        if xs and xs[-1] + tile_size < W:
            xs.append(W - tile_size)
        if not ys: ys = [0]
        if not xs: xs = [0]

        for y in ys:
            for x in xs:
                tile_img = image[y:y + tile_size, x:x + tile_size]
                tile_dets = self._to_detections(
                    self._model.single_image_detection(
                        tile_img, det_conf_thres=self.det_threshold
                    )
                )
                # Translate tile-local boxes back to full-image coords.
                for d in tile_dets:
                    tx1, ty1, tx2, ty2 = d.box_xyxy
                    all_dets.append(Detection(
                        box_xyxy=(tx1 + x, ty1 + y, tx2 + x, ty2 + y),
                        score=d.score,
                        label=d.label,
                    ))

        return self._nms(all_dets, iou_thresh=nms_iou)

    @staticmethod
    def _nms(dets: list[Detection], iou_thresh: float = 0.3) -> list[Detection]:
        """Per-class greedy NMS. Keeps the highest-confidence box and
        suppresses boxes of the same label whose IoU exceeds the threshold."""
        if not dets:
            return []

        def iou(a, b) -> float:
            ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            return inter / ua if ua > 0 else 0.0

        kept: list[Detection] = []
        # Process highest-confidence first.
        remaining = sorted(dets, key=lambda d: d.score, reverse=True)
        while remaining:
            head = remaining.pop(0)
            kept.append(head)
            remaining = [
                d for d in remaining
                if d.label != head.label or iou(head.box_xyxy, d.box_xyxy) < iou_thresh
            ]
        return kept

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
