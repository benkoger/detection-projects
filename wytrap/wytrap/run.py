"""End-to-end orchestration: image(s) -> detections + species labels."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import sys
import time

import numpy as np
from PIL import Image

from wytrap.classifier import Classifier
from wytrap.detector import Detector
from wytrap.io import (
    DetectionRecord,
    ImageRecord,
    append_jsonl,
    output_path_for,
    save_record,
)
from wytrap.species_lists import load_species

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _iter_images(folder: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for p in sorted(folder.glob(pattern)):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = box
    W, H = image.size
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(x1 + 1, min(x2, W))
    y2 = max(y1 + 1, min(y2, H))
    return image.crop((x1, y1, x2, y2))


def process_image(image_path: str | Path,
                  detector: Detector,
                  classifier: Classifier,
                  merges: dict[str, str] | None = None) -> ImageRecord:
    """Run detector + classifier on one image. Returns an ImageRecord."""
    image_path = Path(image_path)
    merges = merges or {}

    pil = Image.open(image_path).convert("RGB")
    arr = np.asarray(pil)

    detections = detector.detect(arr)
    record = ImageRecord(
        image_path=str(image_path),
        image_size=[pil.size[0], pil.size[1]],
        detections=[],
    )
    if not detections:
        return record

    crops = [_crop(pil, d.box_xyxy) for d in detections]
    classifications = classifier.classify_batch(crops)

    for det, cls in zip(detections, classifications):
        canonical = merges.get(cls.fine_label, cls.fine_label)
        record.detections.append(DetectionRecord(
            box_xyxy=list(det.box_xyxy),
            det_score=det.score,
            det_label=det.label,
            label=canonical,
            fine_label=cls.fine_label,
            cls_score=cls.score,
            topk=[[name, score] for name, score in cls.topk],
        ))
    return record


def process_folder(input_dir: str | Path,
                   output_dir: str | Path,
                   species: str | Sequence[str] = "wyoming_all",
                   det_threshold: float = 0.2,
                   cls_topk: int = 5,
                   batch_size: int = 8,
                   device: str = "auto",
                   recursive: bool = False,
                   resume: bool = False,
                   jsonl_path: str | Path | None = None,
                   merges: dict[str, str] | None = None,
                   log: callable = print) -> dict:
    """Run the full pipeline over a folder of images. Returns summary dict."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    species_list = load_species(species) if isinstance(species, str) else list(species)
    log(f"[wytrap] {len(species_list)} species in classifier list")
    log(f"[wytrap] loading detector (device={device}, threshold={det_threshold})")
    detector = Detector(device=device, det_threshold=det_threshold)
    log(f"[wytrap] loading classifier (BioCLIP-2, topk={cls_topk})")
    classifier = Classifier(species=species_list, topk=cls_topk)

    images = list(_iter_images(input_dir, recursive))
    log(f"[wytrap] {len(images)} images under {input_dir}")

    n_done = n_skipped = n_failed = 0
    t_start = time.time()
    for i, image_path in enumerate(images, 1):
        out_path = output_path_for(image_path, output_dir, input_root=input_dir)
        if resume and out_path.exists():
            n_skipped += 1
            continue
        t0 = time.time()
        try:
            record = process_image(image_path, detector, classifier, merges=merges)
            save_record(record, out_path)
            if jsonl_path:
                append_jsonl(record, jsonl_path)
            n_done += 1
            dt = time.time() - t0
            log(f"[wytrap] {i}/{len(images)} {image_path.name} "
                f"({len(record.detections)} det, {dt:.2f}s)")
        except Exception as e:
            n_failed += 1
            err_record = ImageRecord(
                image_path=str(image_path),
                image_size=[0, 0],
                error=f"{type(e).__name__}: {e}",
            )
            save_record(err_record, out_path)
            print(f"[wytrap] FAILED {image_path}: {e}", file=sys.stderr)

    elapsed = time.time() - t_start
    log(f"[wytrap] done: {n_done} processed, {n_skipped} skipped, "
        f"{n_failed} failed in {elapsed:.1f}s")
    return {
        "processed": n_done,
        "skipped": n_skipped,
        "failed": n_failed,
        "elapsed_seconds": elapsed,
    }
