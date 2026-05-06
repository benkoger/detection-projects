"""Build a tiny stratified evaluation set for fast iteration.

Picks ~N images per merged GT class so the wytrap pipeline can be exercised
end-to-end in ~10 minutes instead of ~10 hours. Symlinks images (no copies)
and writes a subset COCO json with just the chosen images and their
annotations. The eval script and the sbatch wrappers work against this
folder unchanged.

Forced-include list: hand-picked images that exercise known failure modes
(extreme close-up, extreme far-shot, BioCLIP-confused-with-moose). These
get added on top of the per-class sample so a tiny run always touches the
edge cases we've been tracking.

Usage:
    python scripts/build_tiny_eval.py \\
        --gt     /path/to/combined.json \\
        --images /path/to/images \\
        --out    /path/to/tiny \\
        [--per-class 20] [--seed 42] \\
        [--include file1.JPG,file2.JPG,...]

Output (under --out):
    images/             symlinks pointing back to originals
    combined.json       subset COCO with kept images + their annotations
    manifest.json       record of how the subset was built (for repro)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in (REPO_ROOT, REPO_ROOT / "wytrap"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

from helpers.helpers import YNP_EVAL_MERGES  # noqa: E402

# Hand-picked files that exercise specific failure modes we've been tracking.
# Edit this list as new pathological cases come up.
DEFAULT_FORCED_FILES = [
    "YNP_12C_5.8.24to6.3.24-05110575.JPG",      # bison filling frame, fur-only crop
    "8A_5.6.24to6.4.24_P2-RCNX1393.JPG",         # far-shot bison treeline
    "YNP_5C_4.29.24to6.13.24-05070471.JPG",      # bison called moose at 0.99
    "YNP_8A_5.6.24to6.4.24_P1-RCNX0009.JPG",     # 8A pronghorn p10 case
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--gt", required=True,
                   help="Full COCO json with ground-truth annotations.")
    p.add_argument("--images", required=True,
                   help="Folder of source images (will be symlinked).")
    p.add_argument("--out", required=True,
                   help="Output dir. Creates <out>/images/ (symlinks), "
                        "<out>/combined.json, <out>/manifest.json.")
    p.add_argument("--per-class", type=int, default=20,
                   help="Target images per merged GT class (default 20).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for stratified sampling (default 42). "
                        "Same seed → same subset, so re-running is "
                        "reproducible for A/B comparisons.")
    p.add_argument("--include",
                   help="Comma-separated list of file_names that MUST be "
                        "included in the subset. Defaults to a hand-picked "
                        "list of known-tricky images.")
    p.add_argument("--no-default-includes", action="store_true",
                   help="Don't include the default forced files. Use only "
                        "files passed via --include.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    gt_path = Path(args.gt)
    images_dir = Path(args.images)
    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    print(f"loading {gt_path}")
    with open(gt_path) as f:
        coco = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    im_id_to_im = {im["id"]: im for im in coco["images"]}

    # Per-image set of merged classes (image counts toward each).
    image_classes: dict[int, set[str]] = defaultdict(set)
    n_orphans = 0
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid not in cat_id_to_name:
            n_orphans += 1
            continue
        raw = cat_id_to_name[cid]
        merged = YNP_EVAL_MERGES.get(raw, raw)
        image_classes[ann["image_id"]].add(merged)
    if n_orphans:
        print(f"warning: dropped {n_orphans} orphan annotations")

    # Group image ids by each class they contain.
    by_class: dict[str, list[int]] = defaultdict(list)
    for im_id, classes in image_classes.items():
        for c in classes:
            by_class[c].append(im_id)

    print(f"GT class coverage: {len(by_class)} classes")
    for c, ims in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        print(f"  {c:<20} available={len(ims)}")

    # Forced includes (by basename).
    forced = []
    if not args.no_default_includes:
        forced += DEFAULT_FORCED_FILES
    if args.include:
        forced += [s.strip() for s in args.include.split(",") if s.strip()]
    forced_set = set(forced)
    name_to_im = {im["file_name"]: im for im in coco["images"]}
    forced_ids = []
    missing_forced = []
    for fname in forced_set:
        if fname in name_to_im:
            forced_ids.append(name_to_im[fname]["id"])
        else:
            missing_forced.append(fname)
    if missing_forced:
        print(f"warning: {len(missing_forced)} forced-include filenames not "
              f"in GT: {missing_forced}")

    # Stratified sample.
    rng = random.Random(args.seed)
    chosen: set[int] = set(forced_ids)
    for c in sorted(by_class.keys()):
        candidates = [i for i in by_class[c] if i not in chosen]
        rng.shuffle(candidates)
        # How many more we need from this class.
        already_in = sum(1 for i in chosen if c in image_classes[i])
        need = max(0, args.per_class - already_in)
        chosen.update(candidates[:need])

    print(f"\nselected {len(chosen)} images "
          f"(forced={len(forced_ids)}, "
          f"per-class target={args.per_class})")

    # Per-class actual counts after selection.
    actual_counts: dict[str, int] = defaultdict(int)
    for im_id in chosen:
        for c in image_classes.get(im_id, set()):
            actual_counts[c] += 1
    for c in sorted(actual_counts):
        print(f"  {c:<20} kept={actual_counts[c]}")

    # Build subset COCO.
    sub_images = [im_id_to_im[i] for i in chosen]
    sub_image_ids = set(chosen)
    sub_anns = [ann for ann in coco["annotations"]
                if ann["image_id"] in sub_image_ids
                and ann["category_id"] in cat_id_to_name]
    sub_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": sub_images,
        "annotations": sub_anns,
    }
    sub_path = out_dir / "combined.json"
    with open(sub_path, "w") as f:
        json.dump(sub_coco, f, indent=2)
    print(f"wrote {sub_path}  ({len(sub_images)} images, "
          f"{len(sub_anns)} annotations)")

    # Symlink images. Skip & overwrite-safe.
    n_linked = n_skipped = n_missing = 0
    for im in sub_images:
        src = images_dir / im["file_name"]
        if not src.exists():
            n_missing += 1
            continue
        # Source might be in a subdir; symlink at flat path under out/images/.
        dest = out_dir / "images" / im["file_name"]
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            dest.symlink_to(src.resolve())
            n_linked += 1
        except OSError as e:
            print(f"warning: could not symlink {im['file_name']}: {e}")
            n_skipped += 1
    print(f"linked {n_linked} images into {out_dir / 'images'}  "
          f"(missing={n_missing}, failed={n_skipped})")

    manifest = {
        "source_gt": str(gt_path),
        "source_images": str(images_dir),
        "per_class_target": args.per_class,
        "seed": args.seed,
        "forced_files": sorted(forced_set),
        "missing_forced": missing_forced,
        "n_images": len(sub_images),
        "n_annotations": len(sub_anns),
        "actual_per_class": dict(actual_counts),
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
