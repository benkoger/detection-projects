"""Draw a run's boxes and labels onto its images, for review by eye.

Reads one or more wytrap run folders (all_records.jsonl) that share the same
detector boxes, and writes a JPEG per image with every box outlined and
labelled: the classifier's label and score on each box, the MegaDetector
score, and the quality tag when the box is not `ok`. With several runs the
label block lists one line per classifier, and boxes they disagree on are
drawn in orange instead of green. People are red, vehicles blue, bad-quality
boxes thin and grey.

Usage:
    wytrap draw --run /runs/cam01-speciesnet --out /runs/cam01-drawn
    wytrap draw --merged /data/output/shirley/merged --out /data/output/shirley/drawn \\
        --per-label 20                     # the 20 most confident images per headline label
    wytrap draw --merged ... --out ... --label "ferret,weasel" --min-det 0.3
    wytrap draw --merged ... --out ... --images review.txt --remap /project/uwyo-0007/data=/Users/me/data

`--merged` takes the folder `wytrap merge` wrote and draws every arm listed
in its summary.json; `--run NAME=DIR` names arms by hand (the first supplies
the boxes). Images without a box above `--min-det` are skipped unless
`--empty` is given. Output JPEGs mirror the image tree below its common root
and are downscaled so the long side is `--max-size` px; `index.csv` beside
them lists what was drawn.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from wytrap.io import common_root
from wytrap.merge import box_key, headline, load_arm

COLOUR = {
    "animal": (46, 204, 64),     # green: an animal every arm agrees on
    "disagree": (255, 153, 0),   # orange: arms name it differently
    "person": (255, 65, 54),
    "vehicle": (0, 116, 217),
    "bad": (160, 160, 160),      # low_pixels / truncated / thin
}
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arial.ttf",
]


# --------------------------------------------------------------------------
# drawing one image (runs in a worker process)
# --------------------------------------------------------------------------

def _font(size: int):
    from PIL import ImageFont
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)      # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def draw_one(task: dict) -> tuple[str, str | None, str]:
    """Open, downscale, draw every box with its label block, save.
    Returns (image_path, out_path or None, error message or "")."""
    from PIL import Image, ImageDraw, ImageOps
    src = Path(task["src"])
    out = Path(task["out"])
    try:
        img = Image.open(src)
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:  # noqa: BLE001
        return task["image"], None, f"cannot open {src}: {e}"
    W, H = img.size
    long = max(W, H)
    scale = min(1.0, task["max_size"] / long) if task["max_size"] else 1.0
    if scale < 1.0:
        img = img.resize((round(W * scale), round(H * scale)), Image.Resampling.LANCZOS)
    W, H = img.size
    long = max(W, H)
    width = task["line_width"] or max(2, round(long / 500))
    font = _font(task["font_size"] or max(12, round(long / 70)))
    draw = ImageDraw.Draw(img)
    pad = max(2, width)

    for b in task["boxes"]:
        x1, y1, x2, y2 = [v * scale for v in b["xyxy"]]
        colour = tuple(b["colour"])
        w = max(1, width // 2) if b["thin"] else width
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=w)
        # label block: one line per entry, above the box when there is room
        lines = b["lines"]
        sizes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
        lh = max(s[3] - s[1] for s in sizes) + pad
        bw = max(s[2] - s[0] for s in sizes) + 2 * pad
        bh = lh * len(lines) + pad
        bx = min(max(0, x1), max(0, W - bw))
        by = y1 - bh if y1 - bh >= 0 else y1
        draw.rectangle([bx, by, bx + bw, by + bh], fill=colour)
        text = (0, 0, 0) if sum(colour) > 380 else (255, 255, 255)
        for i, ln in enumerate(lines):
            draw.text((bx + pad, by + pad / 2 + i * lh), ln, fill=text, font=font)

    if task.get("title"):
        tb = draw.textbbox((0, 0), task["title"], font=font)
        draw.rectangle([0, H - (tb[3] - tb[1]) - 2 * pad, tb[2] - tb[0] + 2 * pad, H],
                       fill=(0, 0, 0))
        draw.text((pad, H - (tb[3] - tb[1]) - pad - tb[1]), task["title"],
                  fill=(255, 255, 255), font=font)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, "JPEG", quality=task["jpeg_quality"])
    except Exception as e:  # noqa: BLE001
        return task["image"], None, f"cannot write {out}: {e}"
    return task["image"], str(out), ""


# --------------------------------------------------------------------------
# choosing images and composing their label blocks
# --------------------------------------------------------------------------

def _arms_from_merged(merged: Path) -> list[tuple[str, Path]]:
    """The arms a `wytrap merge` folder was built from: its summary.json when
    the merge recorded them, otherwise every run folder beside it (the layout
    deploy_infer.sbatch writes), BioCLIP first since it supplied the boxes."""
    summary_path = merged / "summary.json"
    if summary_path.exists():
        arms = json.loads(summary_path.read_text()).get("arms")
        if arms:
            return [(name, Path(d)) for name, d in arms.items()]
    runs = sorted(d for d in merged.parent.iterdir()
                  if d.is_dir() and (d / "all_records.jsonl").exists())
    if not runs:
        raise SystemExit(f"{summary_path} is missing (merge not finished, or an older merge "
                         f"that did not record its arms) and no run folder with "
                         f"all_records.jsonl sits beside {merged}; name them with --run")
    runs.sort(key=lambda d: (not d.name.startswith("bioclip"), d.name))
    arms, seen = [], set()
    for d in runs:
        name = d.name.split("-")[0]
        n, base = 1, name
        while name in seen:
            n += 1
            name = f"{base}{n}"
        seen.add(name)
        arms.append((name, d))
    print(f"[draw] {summary_path.name} not usable; arms from run folders beside it: "
          + ", ".join(f"{n}={d.name}" for n, d in arms))
    return arms


def _remap(path: str, remaps: list[tuple[str, str]]) -> str:
    for old, new in remaps:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def _fmt_label(d: dict | None) -> str:
    if not d:
        return "-"
    lab = d.get("label") or d.get("det_label") or "?"
    if d.get("det_label", "animal") != "animal":
        return lab
    sc = d.get("cls_score")
    return f"{lab} {sc:.2f}" if sc else lab


def box_block(d: dict, per_arm: list[tuple[str, dict | None]]) -> tuple[list[str], tuple, bool]:
    """Label lines, outline colour and whether to draw the box thin, for one
    detector box `d` and its record in each arm (the first arm is `d`)."""
    det_label = d.get("det_label", "animal")
    quality = d.get("quality", "ok")
    thin = quality not in ("ok", "")
    det_line = f"MD {d['det_score']:.2f}" + (f" · {quality}" if thin else "")
    if det_label != "animal":
        return [det_label, det_line], COLOUR[det_label], thin
    if len(per_arm) == 1:
        lines = [_fmt_label(per_arm[0][1]), det_line]
        colour = COLOUR["animal"]
    else:
        lines = [det_line] + [f"{name}: {_fmt_label(dd)}" for name, dd in per_arm]
        labels = {(dd or {}).get("label") for _, dd in per_arm} - {None, "", "skipped"}
        colour = COLOUR["animal"] if len(labels) <= 1 else COLOUR["disagree"]
    if thin:
        colour = COLOUR["bad"]
    return lines, colour, thin


def add_arguments(ap: argparse.ArgumentParser) -> None:
    src = ap.add_argument_group("runs")
    src.add_argument("--run", action="append", default=[], metavar="[NAME=]DIR",
                     help="a run folder (all_records.jsonl); repeat for several classifiers "
                          "on the same boxes, the first supplies the boxes")
    src.add_argument("--merged", metavar="DIR",
                     help="a `wytrap merge` output folder: draw every arm in its summary.json")
    ap.add_argument("--out", required=True, help="folder for the drawn JPEGs and index.csv")
    ap.add_argument("--remap", action="append", default=[], metavar="OLD=NEW",
                    help="rewrite image paths that start with OLD (records written on the "
                         "cluster, images synced elsewhere); repeatable")
    sel = ap.add_argument_group("which images")
    sel.add_argument("--min-det", type=float, default=0.5,
                     help="draw boxes at or above this det_score (default 0.5)")
    sel.add_argument("--empty", action="store_true",
                     help="also write images with no box above the floor")
    sel.add_argument("--label", default=None,
                     help="comma-separated substrings: keep images where any arm's label on "
                          "any drawn box contains one (case-insensitive); 'person' and "
                          "'vehicle' work too")
    sel.add_argument("--camera", default=None,
                     help="substring of the first folder below the images' common root")
    sel.add_argument("--images", metavar="FILE",
                     help="text file of image paths (absolute, relative to the common root, "
                          "or bare file names), one per line; the `image` column of "
                          "images.csv works")
    sel.add_argument("--per-label", type=int, default=None, metavar="N",
                     help="at most N images per headline label of the first arm, the most "
                          "confident first (or random with --random)")
    sel.add_argument("--limit", type=int, default=None, help="at most this many images in all")
    sel.add_argument("--random", action="store_true",
                     help="sample --per-label and --limit at random instead of by confidence")
    sel.add_argument("--seed", type=int, default=0)
    look = ap.add_argument_group("appearance")
    look.add_argument("--max-size", type=int, default=1600,
                      help="long side of the output in px (default 1600; 0 keeps full size)")
    look.add_argument("--font-size", type=int, default=0, help="px; default scales with the image")
    look.add_argument("--line-width", type=int, default=0, help="px; default scales with the image")
    look.add_argument("--jpeg-quality", type=int, default=88)
    look.add_argument("--no-title", action="store_true",
                      help="omit the file name written in the bottom-left corner")
    ap.add_argument("--jobs", type=int, default=4, help="worker processes (default 4)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    return run(ap, ap.parse_args(argv))


def run(ap: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    # ---- the arms --------------------------------------------------------
    arms: list[tuple[str, Path]] = []
    if args.merged:
        arms += _arms_from_merged(Path(args.merged))
    for spec in args.run:
        name, _, d = spec.rpartition("=")
        d = Path(d)
        arms.append((name or d.name, d))
    if not arms:
        ap.error("give --run DIR (repeatable) or --merged DIR")
    for name, d in arms:
        if not (d / "all_records.jsonl").exists():
            ap.error(f"{d}/all_records.jsonl not found (arm {name})")
    remaps = [tuple(r.split("=", 1)) for r in args.remap]
    for r in remaps:
        if len(r) != 2:
            ap.error("--remap takes OLD=NEW")

    names = [n for n, _ in arms]
    base = load_arm(arms[0][1])
    lookup = {n: {ip: {box_key(d): d for d in (r.get("detections") or [])}
                  for ip, r in load_arm(d).items()} for n, d in arms[1:]}
    images = sorted(base)
    root = common_root(images)
    print(f"[draw] {len(images)} images in {arms[0][1]}; arms: {', '.join(names)}; "
          f"image root {root}")

    def rel_of(ip: str) -> str:
        return str(Path(ip).relative_to(root)) if ip.startswith(str(root)) else Path(ip).name

    # ---- which images ---------------------------------------------------
    wanted: set[str] | None = None
    if args.images:
        wanted = set()
        for line in Path(args.images).read_text().splitlines():
            line = line.strip().strip('"')
            if line and not line.startswith("#"):
                wanted.add(line)
    substrings = [s.strip().lower() for s in (args.label or "").split(",") if s.strip()]

    chosen: list[tuple[str, list[dict], list[list[tuple[str, dict | None]]]]] = []
    for ip in images:
        rel = rel_of(ip)
        if wanted is not None and not ({ip, rel, Path(ip).name} & wanted):
            continue
        camera = rel.split("/")[0] if "/" in rel else ""
        if args.camera and args.camera.lower() not in camera.lower():
            continue
        r = base[ip]
        dets = [d for d in (r.get("detections") or []) if d["det_score"] >= args.min_det]
        if not dets and not args.empty:
            continue
        per_box = []
        for d in dets:
            k = box_key(d)
            per_box.append([(names[0], d)] + [(n, lookup[n].get(ip, {}).get(k)) for n in names[1:]])
        if substrings:
            texts = [((dd or {}).get("label") or (dd or {}).get("det_label") or "").lower()
                     for pb in per_box for _, dd in pb]
            if not any(s in t for s in substrings for t in texts):
                continue
        chosen.append((ip, dets, per_box))

    rng = random.Random(args.seed)
    if args.per_label is not None:
        groups: dict[str, list] = defaultdict(list)
        for item in chosen:
            h = headline(item[1])
            groups[h["label"] if h else ""].append(item)
        chosen = []
        for lab, items in sorted(groups.items()):
            if args.random:
                rng.shuffle(items)
            else:
                items.sort(key=lambda it: -((headline(it[1]) or {}).get("cls_score") or 0.0))
            chosen += items[:args.per_label]
    if args.limit is not None and len(chosen) > args.limit:
        if args.random:
            rng.shuffle(chosen)
        chosen = chosen[:args.limit]
    chosen.sort(key=lambda it: it[0])
    if not chosen:
        print("[draw] nothing to draw after filtering")
        return 1

    # ---- tasks -----------------------------------------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks, meta = [], []
    for ip, dets, per_box in chosen:
        rel = rel_of(ip)
        boxes = []
        for d, pb in zip(dets, per_box):
            lines, colour, thin = box_block(d, pb)
            boxes.append({"xyxy": d["box_xyxy"], "lines": lines, "colour": colour, "thin": thin})
        tasks.append({
            "image": ip, "src": _remap(ip, remaps), "out": str(out / rel),
            "boxes": boxes, "max_size": args.max_size, "font_size": args.font_size,
            "line_width": args.line_width, "jpeg_quality": args.jpeg_quality,
            "title": None if args.no_title else rel,
        })
        row = {"image": rel, "out": str(Path(rel)), "n_boxes": len(dets)}
        for n in names:
            h = headline([dd for pb in per_box for nn, dd in pb if nn == n and dd])
            row[f"{n}_label"] = h["label"] if h else ""
            row[f"{n}_score"] = round(float(h["cls_score"]), 3) if h else 0.0
        row["n_person"] = sum(d.get("det_label") == "person" for d in dets)
        row["n_vehicle"] = sum(d.get("det_label") == "vehicle" for d in dets)
        meta.append(row)

    print(f"[draw] drawing {len(tasks)} images with {args.jobs} workers -> {out}")
    results = (Pool(args.jobs).imap(draw_one, tasks, chunksize=8)
               if args.jobs > 1 else map(draw_one, tasks))
    n_ok, errors = 0, []
    for i, (ip, op, err) in enumerate(results, 1):
        if err:
            errors.append((ip, err))
        else:
            n_ok += 1
        if i % 500 == 0 or i == len(tasks):
            print(f"[draw] {i}/{len(tasks)} ({len(errors)} failed)")
    ok_images = {t["image"] for t in tasks} - {ip for ip, _ in errors}
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0]))
        w.writeheader()
        w.writerows(m for m, t in zip(meta, tasks) if t["image"] in ok_images)
    for ip, err in errors[:10]:
        print(f"[draw] FAILED {err}")
    if len(errors) > 10:
        print(f"[draw] ... {len(errors) - 10} more failures")
    if errors and n_ok == 0 and not remaps:
        print("[draw] no image could be opened: if the records were written on another "
              "machine, map their paths with --remap OLD=NEW")
    print(f"[draw] wrote {n_ok} images and {out}/index.csv")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
