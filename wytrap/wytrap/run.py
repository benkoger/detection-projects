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

from wytrap.classifier import Classification, Classifier, bioclip_cache_status
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


def _pad_box(box: tuple[int, int, int, int],
             image_size: tuple[int, int],
             factor: float) -> tuple[int, int, int, int]:
    """Center-expand a box by `factor`, clamped to image bounds.

    factor=2.0 doubles each side around the box center. Clamping at the
    image edge means a box that's already nearly full-frame stays roughly
    the same size — the "padded" pass becomes a no-op in that case, which
    is correct.
    """
    x1, y1, x2, y2 = box
    W, H = image_size
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * factor, (y2 - y1) * factor
    nx1 = max(0, int(round(cx - bw / 2)))
    ny1 = max(0, int(round(cy - bh / 2)))
    nx2 = min(W, int(round(cx + bw / 2)))
    ny2 = min(H, int(round(cy + bh / 2)))
    return (nx1, ny1, max(nx1 + 1, nx2), max(ny1 + 1, ny2))


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
                  skip_classification_when_bad: bool = False,
                  tile: bool = True,
                  tile_size: int = 480,
                  tile_overlap: float = 0.2,
                  multiscale: bool = True,
                  multiscale_pad: float = 2.0) -> ImageRecord:
    """Run detector + classifier on one image. Returns an ImageRecord.

    With ``multiscale=True`` (default), each detection is classified at
    three crop scales (tight / padded / full image) and the highest-scoring
    scale wins. The whole-image classification is computed once per image
    and shared across all detections, so cost is ~2N + 1 BioCLIP forward
    passes per image (N detections), not 3N.

    Each detection is also tagged with a ``quality`` field based on box
    geometry; when ``skip_classification_when_bad`` is True, bad-quality
    boxes get ``label="skipped"`` and BioCLIP isn't invoked on them.
    """
    image_path = Path(image_path)
    merges = merges or {}

    pil = Image.open(image_path).convert("RGB")
    arr = np.asarray(pil)
    W, H = pil.size

    detections = detector.detect(arr, tile=tile, tile_size=tile_size,
                                 overlap=tile_overlap)
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
    tight_crops: list[Image.Image] = []
    padded_crops: list[Image.Image] = []
    for i, (det, (q, _)) in enumerate(zip(detections, qualities)):
        if skip_classification_when_bad and q != "ok":
            continue
        to_classify_idx.append(i)
        tight_crops.append(_crop(pil, det.box_xyxy))
        if multiscale:
            padded_crops.append(_crop(pil, _pad_box(det.box_xyxy, (W, H),
                                                   multiscale_pad)))

    cls_tight = classifier.classify_batch(tight_crops) if tight_crops else []
    cls_padded = (classifier.classify_batch(padded_crops)
                  if multiscale and padded_crops else [None] * len(tight_crops))
    # Whole-image classification: computed once, shared across all detections.
    cls_full = classifier.classify(pil) if multiscale and tight_crops else None

    by_idx: dict[int, tuple[Classification, str, dict, bool]] = {}
    for k, det_idx in enumerate(to_classify_idx):
        ct = cls_tight[k]
        scale_scores = {"tight": ct.score}
        winner_name, winner_cls = "tight", ct
        agree = True
        if multiscale:
            cp = cls_padded[k]
            scale_scores["padded"] = cp.score
            scale_scores["full"]   = cls_full.score
            for name, c in (("padded", cp), ("full", cls_full)):
                if c.score > winner_cls.score:
                    winner_name, winner_cls = name, c
            # Cross-scale agreement: do all three scales pick the same top-1
            # species? Disagreement is a strong signal that the high-score
            # winner is overconfident on an uninformative crop (the
            # 0.99-bison-called-moose pattern).
            top1_tight  = ct.fine_label
            top1_padded = cp.fine_label
            top1_full   = cls_full.fine_label
            agree = (top1_tight == top1_padded == top1_full)
        by_idx[det_idx] = (winner_cls, winner_name, scale_scores, agree)

    for i, (det, (q, reason)) in enumerate(zip(detections, qualities)):
        if i not in by_idx:
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

        cls, scale_name, scale_scores, agree = by_idx[i]
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
            scale=scale_name,
            scale_scores={k: round(v, 4) for k, v in scale_scores.items()},
            cross_scale_agree=agree,
        ))
    return record


def process_folder(input_dir: str | Path,
                   output_dir: str | Path,
                   species: str | Sequence = "wyoming_all",
                   det_threshold: float = 0.10,
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
                   tile: bool = True,
                   tile_size: int = 480,
                   tile_overlap: float = 0.2,
                   multiscale: bool = True,
                   multiscale_pad: float = 2.0,
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
    plog(f"sliced detection  : tile={tile}, tile_size={tile_size}, "
         f"overlap={tile_overlap}")
    plog(f"multi-scale cls   : multiscale={multiscale}, "
         f"pad_factor={multiscale_pad}")
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
                tile=tile, tile_size=tile_size, tile_overlap=tile_overlap,
                multiscale=multiscale, multiscale_pad=multiscale_pad,
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
                scale_tag = f" @{top.scale}" if top.scale != "tight" else ""
                preview = (f"{top.label} ({top.cls_score:.2f}){scale_tag}{tag}; "
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
