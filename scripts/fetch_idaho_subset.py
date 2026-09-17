"""Download a stratified subset of the LILA Idaho Camera Traps dataset.

The full dataset is ~1.5 M images / 1.45 TB with sequence-level labels and
no bounding boxes. This pulls a manageable, class-balanced slice straight
over HTTPS (no cloud CLI or credentials needed) so the wytrap pipeline can
be exercised end to end on a GPU node.

What it writes under --out:

    images/<loc>/<file>.jpg      the sampled images, mirroring LILA's layout
    labels.json                  image-level ground truth:
                                   {"images": [{"file_name", "image_id", "seq_id",
                                                "location", "datetime",
                                                "labels": [...]}, ...],
                                    "categories": [...]}
    manifest.json                sampling parameters + counts, for repro

Resumable: already-downloaded files are skipped, so re-running with a
larger --per-class only fetches the new ones.

Options that matter for evaluation (see scripts/eval_image_level.py):
    --whole-sequences      download every frame of each sampled sequence
                           (per-class counts then refer to sequences), so
                           detection can be scored per sequence and the
                           label-on-empty-frame noise is softened.
    --negatives-per-class  also sample N images per camera-problem label
                           (snow on lens, foggy lens, ...) as hard negatives.

Usage (from the repo root, on a login node with outbound network):

    python scripts/fetch_idaho_subset.py --out /project/uwyo-0007/data/idaho-subset \\
        --per-class 100 [--classes deer,elk,moose,...] [--workers 16] [--seed 42]

Dataset: https://lila.science/datasets/idaho-camera-traps/
Credit Idaho Department of Fish and Game; images may not be sold.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

METADATA_URL = ("https://lilawildlife.blob.core.windows.net/lila-wildlife/"
                "idaho-camera-traps/idaho-camera-traps.json.zip")
IMAGE_BASE = ("https://lilawildlife.blob.core.windows.net/lila-wildlife/"
              "idaho-camera-traps/public/")

# Classes that overlap with the Wyoming species lists in wytrap, plus
# "empty" so the detector's false-positive rate is exercised too.
# LILA removed every image labelled human, vehicle or domestic dog from the
# public bucket (the metadata still lists them; downloads 404), so those are
# not sampled. "horse" images all carry a human co-label and are gone too.
DEFAULT_CLASSES = [
    "deer", "elk", "moose", "pronghorn", "bighorn sheep",
    "wolf", "coyote", "fox", "bear", "mountain lion", "bobcat",
    "lagomorph", "rabbit", "skunk", "squirrel", "turkey", "grouse",
    "cattle",
    "empty",
]

# Labels that describe camera problems rather than content. Usable as hard
# negatives (--negatives-per-class); never sampled as positives.
CAMERA_PROBLEM_LABELS = {
    "snow on lens", "foggy lens", "vegetation obstruction", "malfunction",
    "misdirected", "foggy weather", "lens obscured", "sun", "tilted",
}
# Unidentified animals. Excluded from both positives and negatives.
AMBIGUOUS_LABELS = {"unknown", "unknown canid", "unknown cervid"}
JUNK_LABELS = CAMERA_PROBLEM_LABELS | AMBIGUOUS_LABELS
# Idaho attached "other" as a second tag to nearly every rare-species image
# (all 147 bighorn sheep, 307 of 313 rabbits, ...). It carries no content
# information, so it is stripped before the single-label check.
IGNORED_COLABELS = {"other"}


def load_metadata(cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "idaho-camera-traps.json.zip"
    if not zip_path.exists():
        print(f"[fetch] downloading metadata (~27 MB) -> {zip_path}")
        with requests.get(METADATA_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".json"))
        print(f"[fetch] parsing {name} (~575 MB JSON, takes a minute)")
        return json.load(io.TextIOWrapper(zf.open(name), encoding="utf-8"))


def sample(meta: dict, classes: list[str], per_class: int, seed: int,
           one_per_sequence: bool, whole_sequences: bool = False,
           negatives_per_class: int = 0,
           max_frames_per_seq: int = 10) -> tuple[list[dict], dict[str, int]]:
    cats = {c["id"]: c["name"] for c in meta["categories"]}
    img_labels: dict[str, set[str]] = defaultdict(set)
    for a in meta["annotations"]:
        img_labels[a["image_id"]].add(cats[a["category_id"]])
    images = {im["id"]: im for im in meta["images"]}
    seq_frames: dict[str, list[str]] = defaultdict(list)
    for im in meta["images"]:
        seq_frames[im["seq_id"]].append(im["id"])

    # Single-label images per target class, plus junk-only images per junk
    # label when hard negatives were requested.
    by_class: dict[str, list[str]] = defaultdict(list)
    for img_id, labels in img_labels.items():
        labels = labels - IGNORED_COLABELS
        if len(labels) != 1:          # keep single-label images only
            continue
        (label,) = labels
        if label in classes and label not in JUNK_LABELS:
            by_class[label].append(img_id)
        elif negatives_per_class and label in CAMERA_PROBLEM_LABELS:
            by_class[label].append(img_id)

    targets = [(c, per_class) for c in classes if c not in JUNK_LABELS]
    if negatives_per_class:
        targets += [(j, negatives_per_class) for j in sorted(CAMERA_PROBLEM_LABELS)
                    if by_class.get(j)]

    rng = random.Random(seed)
    chosen: list[dict] = []
    counts: dict[str, int] = {}
    for cls, n_want in targets:
        ids = by_class.get(cls, [])
        rng.shuffle(ids)
        picked, seen_seq = [], set()
        for img_id in ids:
            seq = images[img_id]["seq_id"]
            if (one_per_sequence or whole_sequences) and seq in seen_seq:
                continue
            seen_seq.add(seq)
            if whole_sequences:
                # every frame of the sequence carries the sequence label;
                # bursts run to 40 frames, so cap and keep frame order
                frames = sorted(seq_frames[seq],
                                key=lambda i: images[i].get("frame_num", 0))
                picked.extend(frames[:max_frames_per_seq])
            else:
                picked.append(img_id)
            if len(seen_seq) >= n_want:
                break
        counts[cls] = len(picked)
        for img_id in picked:
            im = images[img_id]
            chosen.append({
                "image_id": img_id,
                "file_name": im["file_name"],
                "seq_id": im["seq_id"],
                "location": im["location"],
                "datetime": im.get("datetime"),
                "labels": [cls],
            })
    return chosen, counts


def download_one(rec: dict, out_dir: Path, retries: int = 3) -> tuple[str, bool, str]:
    dest = out_dir / "images" / rec["file_name"]
    if dest.exists() and dest.stat().st_size > 0:
        return rec["file_name"], True, "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = IMAGE_BASE + rec["file_name"]
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.rename(dest)
            return rec["file_name"], True, "downloaded"
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2 * attempt)
    return rec["file_name"], False, err


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                    help="comma-separated LILA category names")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--allow-multi-frame", action="store_true",
                    help="allow several frames from the same sequence "
                         "(default: at most one image per sequence)")
    ap.add_argument("--whole-sequences", action="store_true",
                    help="download all frames of each sampled sequence; "
                         "--per-class then counts sequences")
    ap.add_argument("--max-frames-per-seq", type=int, default=10,
                    help="with --whole-sequences, keep at most this many "
                         "frames per sequence (first N by frame_num)")
    ap.add_argument("--negatives-per-class", type=int, default=0,
                    help="also sample this many images per camera-problem "
                         "label (snow on lens, foggy lens, ...) as hard negatives")
    ap.add_argument("--metadata-cache", default=None,
                    help="where to keep the metadata zip (default: <out>/_meta)")
    ap.add_argument("--dry-run", action="store_true",
                    help="sample and write labels.json but download nothing")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    meta = load_metadata(Path(args.metadata_cache or out / "_meta"))
    known = {c["name"] for c in meta["categories"]}
    unknown = [c for c in classes if c not in known]
    if unknown:
        print(f"[fetch] WARNING unknown classes ignored: {unknown}")
        classes = [c for c in classes if c in known]

    chosen, counts = sample(meta, classes, args.per_class, args.seed,
                            one_per_sequence=not args.allow_multi_frame,
                            whole_sequences=args.whole_sequences,
                            negatives_per_class=args.negatives_per_class,
                            max_frames_per_seq=args.max_frames_per_seq)
    print("[fetch] sampled per class:", json.dumps(counts))
    print(f"[fetch] total images: {len(chosen)}")

    all_classes = list(dict.fromkeys(classes + list(counts)))
    labels = {
        "images": chosen,
        "categories": [{"id": i, "name": c} for i, c in enumerate(all_classes)],
        "source": "https://lila.science/datasets/idaho-camera-traps/",
        "label_level": "image (from LILA sequence-level labels; no boxes)",
    }
    (out / "labels.json").write_text(json.dumps(labels, indent=2))

    ok = failed = 0
    failures: list[dict] = []
    if not args.dry_run:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(download_one, rec, out) for rec in chosen]
            for i, fut in enumerate(as_completed(futs), 1):
                name, good, msg = fut.result()
                if good:
                    ok += 1
                else:
                    failed += 1
                    failures.append({"file_name": name, "error": msg})
                if i % 100 == 0 or i == len(futs):
                    rate = i / max(time.time() - t0, 1e-6)
                    print(f"[fetch] {i}/{len(futs)} ok={ok} failed={failed} "
                          f"({rate:.1f} img/s)", flush=True)

    if not args.dry_run:
        failed_names = {f["file_name"] for f in failures}
        if failed_names:
            kept = [rec for rec in chosen if rec["file_name"] not in failed_names]
            print(f"[fetch] dropping {len(failed_names)} failed downloads from labels.json "
                  f"({len(kept)} images remain)")
            labels["images"] = kept
            (out / "labels.json").write_text(json.dumps(labels, indent=2))

    manifest = {
        "per_class": args.per_class, "seed": args.seed, "classes": classes,
        "one_per_sequence": not args.allow_multi_frame,
        "whole_sequences": args.whole_sequences,
        "negatives_per_class": args.negatives_per_class,
        "max_frames_per_seq": args.max_frames_per_seq,
        "counts": counts, "downloaded_ok": ok, "download_failed": failed,
        "failures": failures, "image_base": IMAGE_BASE,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[fetch] wrote {out / 'labels.json'} and {out / 'manifest.json'}")
    if not args.dry_run and ok == 0:
        print("[fetch] ERROR: nothing downloaded")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
