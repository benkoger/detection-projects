"""End-to-end orchestration: image(s) -> detections + species labels."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence
import sys
import time

import numpy as np
from PIL import Image

from wytrap.classifier import Classifier, bioclip_cache_status
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


def _make_logger(log: callable, t_start: float):
    def _log(msg: str = "", *, banner: bool = False) -> None:
        if banner:
            bar = "=" * 60
            log(bar)
            log(f"  {msg}")
            log(bar)
            return
        elapsed = time.time() - t_start
        stamp = datetime.now().strftime("%H:%M:%S")
        log(f"[wytrap {stamp} +{elapsed:6.1f}s] {msg}")
    return _log


def _fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:4.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


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

    t_start = time.time()
    plog = _make_logger(log, t_start)

    species_list = load_species(species) if isinstance(species, str) else list(species)

    plog("Initializing wytrap pipeline", banner=True)
    plog(f"input dir         : {input_dir}")
    plog(f"output dir        : {output_dir}")
    plog(f"species list      : {len(species_list)} names "
         f"({species if isinstance(species, str) else 'custom sequence'})")
    plog(f"det threshold     : {det_threshold}")
    plog(f"cls topk          : {cls_topk}")
    plog(f"device requested  : {device}")
    plog(f"recursive / resume: {recursive} / {resume}")
    if jsonl_path:
        plog(f"jsonl aggregate   : {jsonl_path}")

    plog("Loading MegaDetector v6", banner=True)
    detector = Detector(device=device, det_threshold=det_threshold)
    plog(f"detector ready (device resolved to: {detector.device}, "
         f"keep_labels={sorted(detector.keep_labels)})")

    plog("Loading BioCLIP-2 classifier", banner=True)
    cache_info = bioclip_cache_status()
    if cache_info["status"] == "cached":
        plog(f"weights source    : LOCAL CACHE ({cache_info['cache_dir']})")
        for fname, path in cache_info["cached_paths"].items():
            plog(f"  hit             : {fname} -> {path}")
    elif cache_info["status"] == "partial":
        plog(f"weights source    : PARTIAL CACHE ({cache_info['cache_dir']}) "
             f"- will download {cache_info['missing']}")
    elif cache_info["status"] == "missing":
        plog(f"weights source    : REMOTE (HuggingFace) - downloading to "
             f"{cache_info['cache_dir']} on first use")
    else:
        plog("weights source    : unknown (huggingface_hub probe failed)")
    classifier = Classifier(species=species_list, topk=cls_topk)
    plog(f"classifier ready ({len(classifier.species)} text embeddings cached)")

    plog("Scanning input folder", banner=True)
    images = list(_iter_images(input_dir, recursive))
    plog(f"found {len(images)} image(s) (extensions: {sorted(IMAGE_EXTS)})")
    if not images:
        plog("No images to process. Exiting.")
        return {"processed": 0, "skipped": 0, "failed": 0, "elapsed_seconds": 0.0,
                "label_counts": {}}

    plog("Processing images", banner=True)
    n_done = n_skipped = n_failed = 0
    n_detections_total = 0
    label_counts: Counter[str] = Counter()
    t_loop = time.time()

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
            n_detections_total += len(record.detections)
            dt = time.time() - t0

            # Build a one-line preview of what we found.
            if record.detections:
                # Most-confident detection's label + top species.
                top = max(record.detections, key=lambda d: d.det_score)
                preview = (f"{top.label} ({top.cls_score:.2f}); "
                           f"{len(record.detections)} box(es)")
                for d in record.detections:
                    label_counts[d.label] += 1
            else:
                preview = "no detections"

            avg = (time.time() - t_loop) / max(n_done, 1)
            remaining = (len(images) - i) * avg
            plog(f"{i:>4}/{len(images)} {image_path.name:<40} "
                 f"| {dt:5.2f}s | {preview} | ETA {_fmt_eta(remaining)}")
        except Exception as e:
            n_failed += 1
            err_record = ImageRecord(
                image_path=str(image_path),
                image_size=[0, 0],
                error=f"{type(e).__name__}: {e}",
            )
            save_record(err_record, out_path)
            print(f"[wytrap] FAILED {image_path}: {e}", file=sys.stderr)
            plog(f"{i:>4}/{len(images)} {image_path.name:<40} | FAILED: {e}")

    elapsed = time.time() - t_start
    proc_elapsed = time.time() - t_loop
    throughput = (n_done / proc_elapsed) if proc_elapsed > 0 and n_done else 0.0

    plog("Run complete", banner=True)
    plog(f"processed         : {n_done}")
    plog(f"skipped (resume)  : {n_skipped}")
    plog(f"failed            : {n_failed}")
    plog(f"total detections  : {n_detections_total}")
    plog(f"throughput        : {throughput:.2f} img/s "
         f"({(1.0/throughput):.2f}s/img)" if throughput else "throughput        : n/a")
    plog(f"wall time         : {_fmt_eta(elapsed)} "
         f"(processing {_fmt_eta(proc_elapsed)})")

    if label_counts:
        plog("top labels found  :")
        for name, count in label_counts.most_common(10):
            plog(f"  {count:>5}  {name}")

    return {
        "processed": n_done,
        "skipped": n_skipped,
        "failed": n_failed,
        "elapsed_seconds": elapsed,
        "label_counts": dict(label_counts),
    }
