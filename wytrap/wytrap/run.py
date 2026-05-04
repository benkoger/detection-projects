"""End-to-end orchestration: image(s) -> detections + species labels."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence
import os
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


def _make_logger(log: callable, t_start: float,
                 log_file_handle=None):
    """Return a `plog(msg, banner=False)` that writes to `log` and (if given)
    also appends to a file. Each line is timestamped + has elapsed seconds."""
    def _emit(line: str) -> None:
        log(line)
        if log_file_handle is not None:
            log_file_handle.write(line + "\n")
            log_file_handle.flush()

    def _log(msg: str = "", *, banner: bool = False) -> None:
        if banner:
            bar = "=" * 60
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _emit(bar)
            _emit(f"  [{stamp}] {msg}")
            _emit(bar)
            return
        elapsed = time.time() - t_start
        stamp = datetime.now().strftime("%H:%M:%S")
        _emit(f"[wytrap {stamp} +{elapsed:7.1f}s] {msg}")
    return _log


def _log_environment(plog) -> None:
    """Print hostname / SLURM / GPU context up front so .out files are useful
    when a job fails 10 hours in and you need to know which node it ran on."""
    plog(f"hostname          : {os.uname().nodename}")
    plog(f"pid               : {os.getpid()}")
    plog(f"SLURM_JOB_ID      : {os.environ.get('SLURM_JOB_ID', '<not slurm>')}")
    plog(f"SLURM_NODELIST    : {os.environ.get('SLURM_NODELIST', '<n/a>')}")
    plog(f"HF_HOME           : {os.environ.get('HF_HOME', '<unset>')}")
    plog(f"TORCH_HOME        : {os.environ.get('TORCH_HOME', '<unset>')}")
    try:
        import torch
        plog(f"torch version     : {torch.__version__}")
        plog(f"CUDA available    : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            plog(f"GPU               : {torch.cuda.get_device_name(0)}")
            plog(f"CUDA version      : {torch.version.cuda}")
    except ImportError:
        plog("torch             : not importable")


def _fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:4.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def _iter_images(folder: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for p in sorted(folder.glob(pattern)):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        # Skip macOS AppleDouble metadata stubs (e.g. ._IMG_0001.JPG) and
        # other hidden dotfiles that PIL cannot decode.
        if p.name.startswith("."):
            continue
        yield p


def _crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = box
    W, H = image.size
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(x1 + 1, min(x2, W))
    y2 = max(y1 + 1, min(y2, H))
    return image.crop((x1, y1, x2, y2))


def assess_box_quality(box: tuple[int, int, int, int],
                       image_size: tuple[int, int],
                       edge_margin_frac: float = 0.01,
                       min_box_area_frac: float = 0.005,
                       max_aspect_ratio: float = 5.0) -> tuple[str, str]:
    """Heuristic check for whether a detection box is likely to contain a
    mostly-whole animal vs. just an edge sliver.

    Returns (quality, reason). quality is one of:
      - "ok":    box looks classifiable
      - "edge":  box touches the image border (likely partial animal)
      - "small": box is a small fraction of the image (few pixels for BioCLIP)
      - "thin":  extreme aspect ratio (long-thin box, often a leg/tail)

    First failing check wins; thresholds are configurable. Edge takes priority
    because edge-clipped boxes drive the most BioCLIP errors in practice.
    """
    x1, y1, x2, y2 = box
    W, H = image_size
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    margin_w = edge_margin_frac * W
    margin_h = edge_margin_frac * H
    if (x1 <= margin_w or y1 <= margin_h
            or x2 >= W - margin_w or y2 >= H - margin_h):
        sides = []
        if x1 <= margin_w: sides.append("L")
        if y1 <= margin_h: sides.append("T")
        if x2 >= W - margin_w: sides.append("R")
        if y2 >= H - margin_h: sides.append("B")
        return "edge", f"touches {''.join(sides)} border"

    if (bw * bh) / float(W * H) < min_box_area_frac:
        return "small", f"area {(bw*bh)/(W*H):.4f} < {min_box_area_frac}"

    ar = max(bw / bh, bh / bw)
    if ar > max_aspect_ratio:
        return "thin", f"aspect ratio {ar:.1f} > {max_aspect_ratio}"

    return "ok", ""


def process_image(image_path: str | Path,
                  detector: Detector,
                  classifier: Classifier,
                  merges: dict[str, str] | None = None,
                  edge_margin_frac: float = 0.01,
                  min_box_area_frac: float = 0.005,
                  max_aspect_ratio: float = 5.0,
                  skip_classification_when_bad: bool = False) -> ImageRecord:
    """Run detector + classifier on one image. Returns an ImageRecord.

    Each detection is tagged with a `quality` field based on box geometry;
    when `skip_classification_when_bad` is True, bad-quality boxes get
    `label="skipped"` and BioCLIP isn't invoked on them (saves compute).
    """
    image_path = Path(image_path)
    merges = merges or {}

    pil = Image.open(image_path).convert("RGB")
    arr = np.asarray(pil)
    W, H = pil.size

    detections = detector.detect(arr)
    record = ImageRecord(
        image_path=str(image_path),
        image_size=[W, H],
        detections=[],
    )
    if not detections:
        return record

    qualities = [
        assess_box_quality(
            d.box_xyxy, (W, H),
            edge_margin_frac=edge_margin_frac,
            min_box_area_frac=min_box_area_frac,
            max_aspect_ratio=max_aspect_ratio,
        )
        for d in detections
    ]

    # Decide which boxes to actually feed to BioCLIP.
    to_classify_idx: list[int] = []
    crops: list[Image.Image] = []
    for i, (det, (q, _)) in enumerate(zip(detections, qualities)):
        if skip_classification_when_bad and q != "ok":
            continue
        to_classify_idx.append(i)
        crops.append(_crop(pil, det.box_xyxy))

    classifications = classifier.classify_batch(crops) if crops else []
    cls_by_idx = dict(zip(to_classify_idx, classifications))

    for i, (det, (q, reason)) in enumerate(zip(detections, qualities)):
        cls = cls_by_idx.get(i)
        if cls is None:
            # Box was skipped — preserve the detection but note it.
            record.detections.append(DetectionRecord(
                box_xyxy=list(det.box_xyxy),
                det_score=det.score,
                det_label=det.label,
                label="skipped",
                fine_label="",
                scientific_label="",
                cls_score=0.0,
                topk=[],
                quality=q,
                quality_reason=reason,
            ))
            continue

        canonical = merges.get(cls.fine_label, cls.fine_label)
        record.detections.append(DetectionRecord(
            box_xyxy=list(det.box_xyxy),
            det_score=det.score,
            det_label=det.label,
            label=canonical,
            fine_label=cls.fine_label,
            scientific_label=cls.scientific_label,
            cls_score=cls.score,
            topk=[t.to_dict() for t in cls.topk],
            quality=q,
            quality_reason=reason,
        ))
    return record


def process_folder(input_dir: str | Path,
                   output_dir: str | Path,
                   species: str | Sequence = "wyoming_all",
                   det_threshold: float = 0.2,
                   cls_topk: int = 5,
                   batch_size: int = 8,
                   device: str = "auto",
                   recursive: bool = False,
                   resume: bool = False,
                   jsonl_path: str | Path | None = None,
                   merges: dict[str, str] | None = None,
                   edge_margin_frac: float = 0.01,
                   min_box_area_frac: float = 0.005,
                   max_aspect_ratio: float = 5.0,
                   skip_classification_when_bad: bool = False,
                   log: callable = print,
                   log_file: str | Path | None = None) -> dict:
    """Run the full pipeline over a folder of images. Returns summary dict."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default the persistent log file to live next to the JSON outputs so
    # the run survives even after the SLURM .out file is gone.
    if log_file is None:
        log_file = output_dir / "wytrap.log"
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_file, "a")

    t_start = time.time()
    plog = _make_logger(log, t_start, log_file_handle=log_fh)

    species_list = load_species(species) if isinstance(species, str) else list(species)

    plog("Initializing wytrap pipeline", banner=True)
    _log_environment(plog)
    plog(f"log file          : {log_file}")
    plog(f"input dir         : {input_dir}")
    plog(f"output dir        : {output_dir}")
    plog(f"species list      : {len(species_list)} names "
         f"({species if isinstance(species, str) else 'custom sequence'})")
    plog(f"det threshold     : {det_threshold}")
    plog(f"cls topk          : {cls_topk}")
    plog(f"box quality       : edge<{edge_margin_frac}, "
         f"area<{min_box_area_frac}, ar>{max_aspect_ratio}, "
         f"skip_bad={skip_classification_when_bad}")
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
    classifier = Classifier(species=species_list, topk=cls_topk, device=device)
    plog(f"classifier ready on {classifier.device} "
         f"({len(classifier.species)} text embeddings cached)")

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
    n_quality_ok = 0
    quality_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    t_loop = time.time()

    for i, image_path in enumerate(images, 1):
        out_path = output_path_for(image_path, output_dir, input_root=input_dir)
        if resume and out_path.exists():
            n_skipped += 1
            continue
        t0 = time.time()
        try:
            record = process_image(
                image_path, detector, classifier, merges=merges,
                edge_margin_frac=edge_margin_frac,
                min_box_area_frac=min_box_area_frac,
                max_aspect_ratio=max_aspect_ratio,
                skip_classification_when_bad=skip_classification_when_bad,
            )
            save_record(record, out_path)
            if jsonl_path:
                append_jsonl(record, jsonl_path)
            n_done += 1
            n_detections_total += len(record.detections)
            dt = time.time() - t0

            # Build a one-line preview of what we found. The headline label
            # is the most-confident *ok-quality* detection if any exist;
            # otherwise the most-confident detection regardless of quality.
            for d in record.detections:
                quality_counts[d.quality] += 1
                if d.quality == "ok":
                    n_quality_ok += 1
                    label_counts[d.label] += 1
            if record.detections:
                ok_dets = [d for d in record.detections if d.quality == "ok"]
                pool = ok_dets or record.detections
                top = max(pool, key=lambda d: d.det_score)
                tag = "" if top.quality == "ok" else f" [{top.quality}]"
                preview = (f"{top.label} ({top.cls_score:.2f}){tag}; "
                           f"{len(record.detections)} box(es), "
                           f"{len(ok_dets)} ok")
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
    plog(f"total detections  : {n_detections_total} "
         f"(ok={n_quality_ok}, "
         f"{', '.join(f'{q}={n}' for q, n in sorted(quality_counts.items()) if q != 'ok')})")
    plog(f"throughput        : {throughput:.2f} img/s "
         f"({(1.0/throughput):.2f}s/img)" if throughput else "throughput        : n/a")
    plog(f"wall time         : {_fmt_eta(elapsed)} "
         f"(processing {_fmt_eta(proc_elapsed)})")

    if label_counts:
        plog("top labels (ok quality only):")
        for name, count in label_counts.most_common(10):
            plog(f"  {count:>5}  {name}")

    log_fh.close()
    return {
        "processed": n_done,
        "skipped": n_skipped,
        "failed": n_failed,
        "elapsed_seconds": elapsed,
        "label_counts": dict(label_counts),
        "quality_counts": dict(quality_counts),
        "log_file": str(log_file),
    }
