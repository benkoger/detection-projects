"""Arrange a raw camera-trap dump into the folder layout the rest of wytrap
expects (the same one the Idaho subsets use):

    <out>/images/<camera>/<file>.jpg   one flat folder per camera
    <out>/labels.json                  {"images": [{file_name, image_id, seq_id,
                                                    location, datetime, labels}, ...],
                                        "categories": []}
                                       labels are empty; a review fills them in,
                                       after which `wytrap eval` can score runs
    <out>/manifest.json                where the images came from, per-camera
                                       counts and date ranges, sequence settings

Cameras are the first-level folders of --source. Files below a camera are
flattened into one folder, with the sub-path folded into the name
(100RECNX/IMG_0001.JPG -> 100RECNX_IMG_0001.JPG), because Reconyx cameras
restart file numbering in every 100RECNX/101RECNX/... folder.

Timestamps come from EXIF (DateTimeOriginal). Sequences are frames from
one camera less than --seq-gap seconds apart; there is no camera-side
sequence id in a raw dump, so this is the best available grouping.

By default files are MOVED (a rename within the filesystem, so instant and
no duplicate of a 400 GB tree). --copy and --link keep the source.

Usage:
    wytrap ingest --source /data/CameraTrap_raw --out /data/CameraTrap_test
    wytrap ingest --source /data/CameraTrap_test --out /data/CameraTrap_test  # in place
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
RESERVED = {"images", "labels.json", "manifest.json"}      # never treated as cameras
EXIF_DATETIME_ORIGINAL, EXIF_DATETIME = 36867, 306


def exif_datetime(path: Path) -> str | None:
    """EXIF capture time as 'YYYY-MM-DD HH:MM:SS', or None."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif()
            raw = ex.get(EXIF_DATETIME_ORIGINAL) or ex.get_ifd(0x8769).get(EXIF_DATETIME_ORIGINAL) \
                or ex.get(EXIF_DATETIME)
        if not raw:
            return None
        return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def flat_name(rel: Path) -> str:
    return "_".join(rel.parts)


def place(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "move":
        os.replace(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "link":
        os.symlink(src.resolve(), dst)


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--source", required=True, help="raw dump: one folder per camera")
    ap.add_argument("--out", required=True, help="data folder to create (may equal --source)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--copy", action="store_true", help="copy files instead of moving them")
    g.add_argument("--link", action="store_true", help="symlink files instead of moving them")
    ap.add_argument("--seq-gap", type=float, default=60.0,
                    help="seconds between frames that starts a new sequence (default 60)")
    ap.add_argument("--workers", type=int, default=16, help="threads for reading EXIF")
    ap.add_argument("--dry-run", action="store_true", help="report what would happen, touch nothing")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    return run(ap.parse_args(argv))


def run(args: argparse.Namespace) -> int:
    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    mode = "copy" if args.copy else "link" if args.link else "move"
    if not source.is_dir():
        raise SystemExit(f"--source {source} is not a folder")
    cameras = sorted(p for p in source.iterdir()
                     if p.is_dir() and not p.name.startswith(".") and p.name not in RESERVED
                     and not p.name.startswith("output"))
    if not cameras:
        raise SystemExit(f"no camera folders under {source}")
    print(f"[ingest] {len(cameras)} camera folders under {source} -> {out}/images ({mode})")

    # ---- plan every file
    plan: list[tuple[str, Path, Path]] = []      # (camera, src, dst)
    for cam in cameras:
        files = sorted(p for p in cam.rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))
        for f in files:
            rel = f.relative_to(cam)
            plan.append((cam.name, f, out / "images" / cam.name / flat_name(rel)))
        print(f"[ingest]   {cam.name}: {len(files)} images")
    print(f"[ingest] {len(plan)} images total")
    if args.dry_run:
        for cam, src, dst in plan[:5]:
            print(f"[ingest]   {src} -> {dst}")
        return 0

    # ---- EXIF before moving (paths change)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        stamps = list(ex.map(lambda t: exif_datetime(t[1]), plan))
    n_missing = sum(1 for s in stamps if s is None)
    print(f"[ingest] EXIF read in {time.time() - t0:.0f}s; {n_missing} images without a timestamp")

    # ---- place files
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for cam, src, dst in plan:
        place(src, dst, mode)
    print(f"[ingest] files placed in {time.time() - t0:.0f}s")
    if mode == "move":
        # drop the emptied camera trees so the folder is not scanned twice
        for cam in cameras:
            for d in sorted((p for p in cam.rglob("*") if p.is_dir()), reverse=True):
                try:
                    d.rmdir()
                except OSError:
                    pass
            try:
                cam.rmdir()
            except OSError:
                pass

    # ---- sequences: per camera, by time, split at gaps
    images: list[dict] = []
    per_cam: dict[str, dict] = {}
    by_cam: dict[str, list[tuple[str | None, Path]]] = {}
    for (cam, _, dst), stamp in zip(plan, stamps):
        by_cam.setdefault(cam, []).append((stamp, dst))
    for cam, items in by_cam.items():
        items.sort(key=lambda t: (t[0] is None, t[0] or "", t[1].name))
        seq_n, prev = 0, None
        n_seq = 0
        for stamp, dst in items:
            t = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S") if stamp else None
            if prev is None or t is None or (t - prev).total_seconds() > args.seq_gap:
                seq_n += 1
                n_seq += 1
            prev = t if t else prev
            rel = dst.relative_to(out / "images")
            images.append({
                "image_id": str(rel),
                "file_name": str(rel),
                "seq_id": f"{cam}_seq{seq_n:05d}",
                "location": cam,
                "datetime": stamp,
                "labels": [],
            })
        dated = [s for s, _ in items if s]
        per_cam[cam] = {"images": len(items), "sequences": n_seq,
                        "first": min(dated) if dated else None,
                        "last": max(dated) if dated else None,
                        "no_timestamp": sum(1 for s, _ in items if not s)}

    (out / "labels.json").write_text(json.dumps({
        "images": images, "categories": [],
        "note": "unlabelled deployment data; fill 'labels' from review before `wytrap eval`",
    }, indent=1))
    (out / "manifest.json").write_text(json.dumps({
        "source": str(source), "mode": mode, "created": datetime.now().isoformat(timespec="seconds"),
        "cameras": per_cam, "images": len(images), "no_timestamp": n_missing,
        "seq_gap_seconds": args.seq_gap, "name_flattening": "sub-path joined with '_'",
    }, indent=2))
    print(f"[ingest] wrote {out / 'labels.json'} ({len(images)} images, "
          f"{sum(c['sequences'] for c in per_cam.values())} sequences) and {out / 'manifest.json'}")
    return 0
