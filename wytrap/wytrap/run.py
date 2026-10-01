"""The per-image loop: detect, tag box quality, classify, write records.

Two entry points share it:

    process_folder   `wytrap detect`   images -> MegaDetector -> classifier -> records
    reclassify       `wytrap classify` records of an earlier run -> another classifier,
                     same boxes and quality tags, only the labels change

Both write one JSON per image (mirroring the input tree) plus
`all_records.jsonl` and `manifest.json`; BioCLIP runs also write
`prompts.json` for `wytrap calibrate`.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence
import json
import os
import sys
import time

import numpy as np
from PIL import Image

from wytrap.classifiers import BoxClassifier, BoxInput
from wytrap.detector import Detector
from wytrap.io import (
    DetectionRecord,
    ImageRecord,
    append_jsonl,
    common_root,
    output_path_for,
    save_record,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------

def _make_logger(log: callable, t_start: float, log_file_handle=None):
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
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        # Skip macOS AppleDouble metadata stubs (e.g. ._IMG_0001.JPG) and
        # other hidden dotfiles that PIL cannot decode.
        if p.name.startswith("."):
            continue
        yield p


# --------------------------------------------------------------------------
# box quality
# --------------------------------------------------------------------------

def assess_box_quality(box: tuple[int, int, int, int],
                       image_size: tuple[int, int],
                       min_pixel_side: int = 60,
                       border_overlap_truncated: float = 0.20,
                       max_aspect_ratio: float = 8.0,
                       border_tolerance_px: int = 2) -> tuple[str, str]:
    """Classifier-feasibility check for a detection box.

    Returns (quality, reason). quality is one of:
      - "ok":          box has enough pixels and the animal looks fully framed
      - "low_pixels":  short side < min_pixel_side — the classifier won't have signal
      - "truncated":   substantial fraction of box perimeter sits on the image
                       border — animal almost certainly extends out of frame.
                       Treated as PASCAL VOC "difficult": neither TP nor FP at
                       eval time; kept for analysis but not committed.
      - "thin":        extreme aspect ratio (sliver, leg/tail) — rare; dropped.

    Why these criteria:
      - **Absolute pixel size**, not fraction-of-image. The classifier doesn't
        care about resolution; it cares about how many pixels of animal it sees.
      - **Perimeter-overlap fraction** captures *degree* of truncation rather
        than just "box touches edge". A close-up bison filling 95% of the
        frame has high perimeter overlap → truncated (we can't see the whole
        animal). A small pronghorn standing at the right edge has ~25%
        overlap → truncated. A distant pronghorn centered with no edge
        contact has 0% overlap → ok regardless of size.
      - **Larger aspect-ratio cap (8:1)** lets through snakes, distant
        elongated views, animals at odd angles.

    First failing check wins; tight thresholds first.
    """
    x1, y1, x2, y2 = box
    W, H = image_size
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # 1) Pixel-size floor. Below this, the classifier can't recover signal
    #    even with multi-scale upsampling.
    if min(bw, bh) < min_pixel_side:
        return "low_pixels", f"min side {min(bw, bh)}px < {min_pixel_side}px"

    # 2) Aspect-ratio cap. Filters tiny slivers; rare in practice.
    ar = max(bw / bh, bh / bw)
    if ar > max_aspect_ratio:
        return "thin", f"aspect ratio {ar:.1f} > {max_aspect_ratio}"

    # 3) Truncation — fraction of box perimeter coincident with image edge.
    #    Each box side that lies within `border_tolerance_px` of an image
    #    border contributes its length to the overlap.
    overlap_len = 0
    if x1 <= border_tolerance_px:                  overlap_len += bh   # left
    if y1 <= border_tolerance_px:                  overlap_len += bw   # top
    if (W - x2) <= border_tolerance_px:            overlap_len += bh   # right
    if (H - y2) <= border_tolerance_px:            overlap_len += bw   # bottom
    perimeter = 2 * (bw + bh)
    overlap_frac = overlap_len / perimeter if perimeter > 0 else 0.0
    if overlap_frac >= border_overlap_truncated:
        return "truncated", (f"{overlap_frac:.0%} of perimeter on image edge "
                             f">= {border_overlap_truncated:.0%}")

    return "ok", ""


# --------------------------------------------------------------------------
# classification of one image's boxes
# --------------------------------------------------------------------------

def classify_boxes(classifier: BoxClassifier | None, image: Image.Image, image_path: str,
                   dets: list[DetectionRecord], merges: dict[str, str] | None = None) -> None:
    """Fill the classification fields of `dets` in place. Boxes the classifier
    returns None for keep the detector's label (person, vehicle) or are marked
    "skipped" (animal boxes it chose not to classify)."""
    merges = merges or {}
    boxes = [BoxInput(tuple(d.box_xyxy), d.det_score, d.det_label, d.quality) for d in dets]
    results = classifier.classify_image(image, image_path, boxes) if classifier and boxes \
        else [None] * len(boxes)
    for d, r in zip(dets, results):
        if r is None:
            passthrough = d.det_label if d.det_label != "animal" else "skipped"
            d.label = passthrough
            d.fine_label = passthrough if passthrough != "skipped" else ""
            d.scientific_label, d.cls_score, d.topk = "", 0.0, []
            d.lineage, d.scale, d.scale_scores = {}, "tight", {}
            d.cross_scale_agree, d.prompt_logp, d.source = True, {}, ""
            continue
        d.label = merges.get(r.label, r.label)
        d.fine_label = r.label
        d.scientific_label = r.scientific
        d.cls_score = r.score
        d.topk = r.topk
        d.lineage = r.lineage
        d.scale = r.scale
        d.scale_scores = r.scale_scores
        d.cross_scale_agree = r.cross_scale_agree
        d.prompt_logp = r.prompt_logp
        d.source = r.source


def process_image(image_path: str | Path,
                  detector: Detector,
                  classifier: BoxClassifier | None,
                  merges: dict[str, str] | None = None,
                  min_pixel_side: int = 60,
                  border_overlap_truncated: float = 0.20,
                  max_aspect_ratio: float = 8.0,
                  tile: bool = False,
                  tile_size: int = 480,
                  tile_overlap: float = 0.2) -> ImageRecord:
    """Run detector + classifier on one image. Returns an ImageRecord."""
    image_path = Path(image_path)
    pil = Image.open(image_path).convert("RGB")
    W, H = pil.size
    detections = detector.detect(np.asarray(pil), tile=tile, tile_size=tile_size,
                                 overlap=tile_overlap)
    record = ImageRecord(image_path=str(image_path), image_size=[W, H])
    for det in detections:
        q, reason = assess_box_quality(det.box_xyxy, (W, H), min_pixel_side=min_pixel_side,
                                       border_overlap_truncated=border_overlap_truncated,
                                       max_aspect_ratio=max_aspect_ratio)
        record.detections.append(DetectionRecord(
            box_xyxy=list(det.box_xyxy), det_score=det.score, det_label=det.label,
            label="", fine_label="", scientific_label="", cls_score=0.0, topk=[],
            quality=q, quality_reason=reason))
    classify_boxes(classifier, pil, str(image_path), record.detections, merges)
    return record


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

def _preview(record: ImageRecord) -> str:
    """One-line summary: the most-confident ok-quality box, else any box."""
    if not record.detections:
        return "no detections"
    ok = [d for d in record.detections if d.quality == "ok"]
    top = max(ok or record.detections, key=lambda d: d.det_score)
    tag = "" if top.quality == "ok" else f" [{top.quality}]"
    scale_tag = f" @{top.scale}" if top.scale not in ("tight", "") else ""
    return (f"{top.label} ({top.cls_score:.2f}){scale_tag}{tag}; "
            f"{len(record.detections)} box(es), {len(ok)} ok")


def _write_manifest(output_dir: Path, classifier: BoxClassifier | None, extra: dict) -> None:
    m = {"wytrap_version": _version(), "classifier": classifier.describe() if classifier else None}
    m.update(extra)
    (output_dir / "manifest.json").write_text(json.dumps(m, indent=2, default=str))
    prompts = getattr(classifier, "prompts", None)
    if prompts:
        # `wytrap calibrate` needs the prompt order and the bias in force
        (output_dir / "prompts.json").write_text(json.dumps(
            {"prompts": prompts, "common": classifier.common_names(),
             "prompt_bias": getattr(classifier, "prompt_bias_path", None)}, indent=1))


def _version() -> str:
    from wytrap import __version__
    return __version__


def _run(items: Sequence, make_record, output_dir: Path, input_root: Path | None,
         jsonl_path: Path | None, resume: bool, plog, t_start: float) -> dict:
    """Shared loop: progress, ETA, per-image JSON, aggregate JSONL, summary."""
    n_done = n_skipped = n_failed = 0
    n_detections = 0
    quality_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    t_loop = time.time()
    for i, item in enumerate(items, 1):
        image_path = Path(item if not isinstance(item, dict) else item["image_path"])
        out_path = output_path_for(image_path, output_dir, input_root=input_root)
        if resume and out_path.exists():
            n_skipped += 1
            continue
        t0 = time.time()
        try:
            record = make_record(item)
            save_record(record, out_path)
            if jsonl_path:
                append_jsonl(record, jsonl_path)
            n_done += 1
            n_detections += len(record.detections)
            for d in record.detections:
                quality_counts[d.quality] += 1
                if d.quality == "ok":
                    label_counts[d.label] += 1
            avg = (time.time() - t_loop) / max(n_done, 1)
            plog(f"{i:>4}/{len(items)} {image_path.name:<40} | {time.time() - t0:5.2f}s | "
                 f"{_preview(record)} | ETA {_fmt_eta((len(items) - i) * avg)}")
        except Exception as e:
            n_failed += 1
            save_record(ImageRecord(image_path=str(image_path), image_size=[0, 0],
                                    error=f"{type(e).__name__}: {e}"), out_path)
            print(f"[wytrap] FAILED {image_path}: {e}", file=sys.stderr)
            plog(f"{i:>4}/{len(items)} {image_path.name:<40} | FAILED: {e}")

    elapsed = time.time() - t_start
    proc = time.time() - t_loop
    plog("Run complete", banner=True)
    plog(f"processed         : {n_done}")
    plog(f"skipped (resume)  : {n_skipped}")
    plog(f"failed            : {n_failed}")
    plog(f"total detections  : {n_detections} "
         f"({', '.join(f'{k}={v}' for k, v in sorted(quality_counts.items()))})")
    if n_done and proc > 0:
        plog(f"throughput        : {n_done / proc:.2f} img/s ({proc / n_done:.2f}s/img)")
    plog(f"wall time         : {_fmt_eta(elapsed)} (processing {_fmt_eta(proc)})")
    if label_counts:
        plog("top labels (ok quality only):")
        for lab, n in label_counts.most_common(10):
            plog(f"    {n:>5}  {lab}")
    return {"processed": n_done, "skipped": n_skipped, "failed": n_failed,
            "elapsed_seconds": elapsed, "label_counts": dict(label_counts),
            "quality_counts": dict(quality_counts)}


def _open_log(output_dir: Path, log_file, log):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(log_file) if log_file else output_dir / "wytrap.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_file, "a")
    t_start = time.time()
    return _make_logger(log, t_start, log_file_handle=fh), t_start, log_file


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def process_folder(input_dir: str | Path,
                   output_dir: str | Path,
                   classifier: BoxClassifier | None,
                   detector_version: str = Detector.DEFAULT_VERSION,
                   det_threshold: float = 0.20,
                   det_imgsz: int | None = None,
                   keep_labels: Sequence[str] = ("animal",),
                   device: str = "auto",
                   recursive: bool = True,
                   resume: bool = True,
                   jsonl_path: str | Path | None = None,
                   merges: dict[str, str] | None = None,
                   min_pixel_side: int = 60,
                   border_overlap_truncated: float = 0.20,
                   max_aspect_ratio: float = 8.0,
                   tile: bool = False,
                   tile_size: int = 480,
                   tile_overlap: float = 0.2,
                   log: callable = print,
                   log_file: str | Path | None = None) -> dict:
    """`wytrap detect`: MegaDetector, then `classifier` (None = detection only)."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    plog, t_start, log_file = _open_log(output_dir, log_file, log)

    plog("Initializing wytrap pipeline", banner=True)
    _log_environment(plog)
    plog(f"log file          : {log_file}")
    plog(f"input dir         : {input_dir}")
    plog(f"output dir        : {output_dir}")
    plog(f"classifier        : {classifier.name if classifier else 'none (detection only)'}")
    if classifier:
        for k, v in classifier.describe().items():
            if k != "classifier" and v is not None and not isinstance(v, (dict, list)):
                plog(f"  {k:<16}: {v}")
    plog(f"det threshold     : {det_threshold}")
    plog(f"box quality       : min_pixel_side={min_pixel_side}px, "
         f"border_overlap>={border_overlap_truncated:.0%}, ar>{max_aspect_ratio}")
    plog(f"sliced detection  : tile={tile}, tile_size={tile_size}, overlap={tile_overlap}")
    plog(f"recursive / resume: {recursive} / {resume}")
    if jsonl_path:
        plog(f"jsonl aggregate   : {jsonl_path}")

    plog(f"Loading MegaDetector ({detector_version})", banner=True)
    detector = Detector(device=device, det_threshold=det_threshold, version=detector_version,
                        imgsz=det_imgsz, keep_labels=tuple(keep_labels))
    plog(f"detector ready (device {detector.device}, imgsz={detector.imgsz}, "
         f"keep_labels={sorted(detector.keep_labels)})")

    plog("Scanning input folder", banner=True)
    images = list(_iter_images(input_dir, recursive))
    plog(f"found {len(images)} image(s) (extensions: {sorted(IMAGE_EXTS)})")
    _write_manifest(output_dir, classifier, {
        "command": "detect", "input": str(input_dir), "images": len(images),
        "detector": {"version": detector_version, "threshold": det_threshold,
                     "imgsz": detector.imgsz, "keep_labels": sorted(detector.keep_labels),
                     "tile": tile},
        "box_quality": {"min_pixel_side": min_pixel_side,
                        "border_overlap_truncated": border_overlap_truncated,
                        "max_aspect_ratio": max_aspect_ratio}})
    if not images:
        plog("No images to process. Exiting.")
        return {"processed": 0, "skipped": 0, "failed": 0, "elapsed_seconds": 0.0,
                "label_counts": {}}

    plog("Processing images", banner=True)
    return _run(images, lambda p: process_image(
        p, detector, classifier, merges=merges, min_pixel_side=min_pixel_side,
        border_overlap_truncated=border_overlap_truncated, max_aspect_ratio=max_aspect_ratio,
        tile=tile, tile_size=tile_size, tile_overlap=tile_overlap),
        output_dir, input_dir, Path(jsonl_path) if jsonl_path else None, resume, plog, t_start)


def reclassify(records_path: str | Path,
               output_dir: str | Path,
               classifier: BoxClassifier,
               jsonl_path: str | Path | None = None,
               merges: dict[str, str] | None = None,
               log: callable = print,
               log_file: str | Path | None = None) -> dict:
    """`wytrap classify`: re-label the boxes of an earlier run with another
    classifier. Boxes, scores and quality tags are copied; the per-image JSONs
    mirror the images' tree below their common root."""
    records_path, output_dir = Path(records_path), Path(output_dir)
    plog, t_start, log_file = _open_log(output_dir, log_file, log)
    plog("Initializing wytrap classify", banner=True)
    _log_environment(plog)
    plog(f"log file          : {log_file}")
    plog(f"records           : {records_path}")
    plog(f"output dir        : {output_dir}")
    plog(f"classifier        : {classifier.name}")
    for k, v in classifier.describe().items():
        if k != "classifier" and v is not None and not isinstance(v, (dict, list)):
            plog(f"  {k:<16}: {v}")

    records = [json.loads(l) for l in open(records_path) if l.strip()]
    root = common_root([r["image_path"] for r in records])
    plog(f"{len(records)} records, images under {root}")
    if jsonl_path is None:
        jsonl_path = output_dir / "all_records.jsonl"
    jsonl_path = Path(jsonl_path)
    if jsonl_path.exists():        # a re-run must not append to the old aggregate
        jsonl_path.unlink()
    _write_manifest(output_dir, classifier, {
        "command": "classify", "records": str(records_path), "images": len(records)})

    def make_record(r: dict) -> ImageRecord:
        dets = [DetectionRecord(**{k: v for k, v in d.items() if k in DetectionRecord.__dataclass_fields__})
                for d in (r.get("detections") or [])]
        rec = ImageRecord(image_path=r["image_path"], image_size=r["image_size"],
                          detections=dets, error=r.get("error"))
        if dets:
            with Image.open(r["image_path"]) as im:
                pil = im.convert("RGB")
            classify_boxes(classifier, pil, r["image_path"], dets, merges)
        return rec

    plog("Classifying", banner=True)
    return _run(records, make_record, output_dir, root, jsonl_path, False, plog, t_start)
