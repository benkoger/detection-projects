"""Merge several classifier arms run on the SAME detector boxes into one
side-by-side table, for unlabelled deployment data (no ground truth).

Each arm is a wytrap-format run folder (all_records.jsonl). Boxes are joined
on (image_path, box_xyxy), which is exact because every arm re-labels the
boxes of one source wytrap run rather than detecting again.

Writes:
    <out>/boxes.csv    one row per detector box: geometry, quality, and each
                       arm's label + score, plus how many arms agree
    <out>/images.csv   one row per image: headline label per arm (highest
                       det_score quality-ok animal box, else any box), person
                       and vehicle counts, consensus label
    <out>/summary.json label counts per arm, agreement rates, empties

Usage:
    wytrap merge --out /path/merged \\
        --arm bioclip=/path/output-bioclip-...-calib \\
        --arm speciesnet=/path/output-speciesnet-ens-... \\
        --arm wusa=/path/output-addax-wusa-...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from wytrap.io import common_root


def load_arm(run_dir: Path) -> dict[str, dict]:
    recs = {}
    with open(run_dir / "all_records.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["image_path"]] = r
    return recs


def box_key(d: dict) -> tuple:
    return tuple(int(v) for v in d["box_xyxy"])


def headline(dets: list[dict]) -> dict | None:
    """The box that names the image: best quality-ok animal, else best animal,
    else nothing (person/vehicle-only images are counted, not named)."""
    animals = [d for d in dets if d.get("det_label", "animal") == "animal"
               and d.get("label") not in ("", "skipped")]
    if not animals:
        return None
    ok = [d for d in animals if d.get("quality", "ok") == "ok"]
    return max(ok or animals, key=lambda d: d["det_score"])


def combine(out_root: Path) -> int:
    """Concatenate per-camera merged tables; the camera column already names
    the folder each row came from."""
    for fname in ("images.csv", "boxes.csv"):
        parts = sorted(out_root.glob(f"*/merged/{fname}"))
        if not parts:
            print(f"[merge] no */merged/{fname} under {out_root}")
            continue
        rows, fields = [], None
        for p in parts:
            with open(p) as f:
                rd = csv.DictReader(f)
                fields = fields or rd.fieldnames
                rows.extend(rd)
        target = out_root / f"all_{fname}"
        with open(target, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"[merge] {len(parts)} cameras, {len(rows)} rows -> {target}")
        if fname == "images.csv":
            cons = Counter(r["consensus"] or "(no consensus / no animal)" for r in rows)
            for lab, n in cons.most_common(15):
                print(f"         {n:7d}  {lab}")
    return 0


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=DIR",
                    help="a classifier run folder; the first one supplies the boxes")
    ap.add_argument("--out", help="output folder (required with --arm)")
    ap.add_argument("--min-det", type=float, default=0.5,
                    help="drop boxes below this det_score from the tables (default 0.5). "
                         "Runs record boxes down to 0.20 so the floor can be chosen here "
                         "without re-running; 0.30 kept the most true animals on Idaho, "
                         "0.50 the fewest false ones")
    ap.add_argument("--combine", metavar="OUT_ROOT",
                    help="instead of merging arms: concatenate every <OUT_ROOT>/*/merged/"
                         "images.csv and boxes.csv (one array task per camera) into "
                         "<OUT_ROOT>/all_images.csv and all_boxes.csv")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    return run(ap, ap.parse_args(argv))


def run(ap: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.combine:
        return combine(Path(args.combine))
    if not args.arm or not args.out:
        ap.error("--arm and --out are required (or use --combine)")

    arms: list[tuple[str, dict[str, dict]]] = []
    for spec in args.arm:
        name, d = spec.split("=", 1)
        arms.append((name, load_arm(Path(d))))
    names = [n for n, _ in arms]
    base_name, base = arms[0]
    images = sorted(base)
    root = common_root(images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # per-box lookup for the other arms
    lookup: dict[str, dict[str, dict[tuple, dict]]] = {}
    for name, recs in arms[1:]:
        lookup[name] = {ip: {box_key(d): d for d in (r.get("detections") or [])}
                        for ip, r in recs.items()}

    label_counts = {n: Counter() for n in names}
    agree_hist = Counter()
    n_images = n_empty = n_person_only = 0
    box_rows, img_rows = [], []
    for ip in images:
        r = base[ip]
        rel = str(Path(ip).relative_to(root)) if str(ip).startswith(str(root)) else ip
        camera = rel.split("/")[0] if "/" in rel else ""
        dets = [d for d in (r.get("detections") or []) if d["det_score"] >= args.min_det]
        n_images += 1
        n_person = sum(1 for d in dets if d.get("det_label") == "person")
        n_vehicle = sum(1 for d in dets if d.get("det_label") == "vehicle")
        n_animal = sum(1 for d in dets if d.get("det_label", "animal") == "animal")
        if not dets:
            n_empty += 1
        elif not n_animal:
            n_person_only += 1

        per_arm_dets = {base_name: {box_key(d): d for d in dets}}
        for name in names[1:]:
            per_arm_dets[name] = lookup[name].get(ip, {})

        for d in dets:
            k = box_key(d)
            row = {"image": rel, "camera": camera,
                   "x1": k[0], "y1": k[1], "x2": k[2], "y2": k[3],
                   "det_label": d.get("det_label", "animal"),
                   "det_score": round(d["det_score"], 3),
                   "quality": d.get("quality", "ok")}
            labels = []
            for name in names:
                dd = per_arm_dets[name].get(k)
                lab = (dd or {}).get("label", "")
                sc = (dd or {}).get("cls_score", 0.0)
                row[f"{name}_label"] = lab
                row[f"{name}_score"] = round(float(sc), 3)
                if d.get("det_label", "animal") == "animal" and lab not in ("", "skipped"):
                    labels.append(lab)
            row["n_agree"] = Counter(labels).most_common(1)[0][1] if labels else 0
            box_rows.append(row)

        img = {"image": rel, "camera": camera, "n_boxes": len(dets),
               "n_animal": n_animal, "n_person": n_person, "n_vehicle": n_vehicle}
        heads = []
        for name in names:
            h = headline(list(per_arm_dets[name].values()))
            img[f"{name}_label"] = h["label"] if h else ""
            img[f"{name}_score"] = round(float(h["cls_score"]), 3) if h else 0.0
            if h:
                heads.append(h["label"])
                label_counts[name][h["label"]] += 1
        if heads:
            lab, n = Counter(heads).most_common(1)[0]
            img["consensus"] = lab if n > len(names) / 2 else ""
            img["n_agree"] = n
            agree_hist[n] += 1
        else:
            img["consensus"] = ""
            img["n_agree"] = 0
        img_rows.append(img)

    for fname, rows in (("boxes.csv", box_rows), ("images.csv", img_rows)):
        if rows:
            with open(out / fname, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    n_named = sum(agree_hist.values())
    summary = {
        "arms": {n: str(Path(spec.split('=', 1)[1])) for n, spec in zip(names, args.arm)},
        "images": n_images, "empty": n_empty, "person_or_vehicle_only": n_person_only,
        "animal_images": n_named,
        "all_arms_agree": agree_hist.get(len(names), 0) / n_named if n_named else None,
        "agreement_histogram": {str(k): v for k, v in sorted(agree_hist.items())},
        "headline_label_counts": {n: dict(c.most_common()) for n, c in label_counts.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[merge] {n_images} images: {n_empty} empty, {n_person_only} person/vehicle only, "
          f"{n_named} with an animal")
    if n_named:
        print(f"[merge] all {len(names)} arms agree on {summary['all_arms_agree']:.1%} "
              f"of animal images; histogram {dict(agree_hist)}")
    for n in names:
        top = ", ".join(f"{l} {c}" for l, c in label_counts[n].most_common(8))
        print(f"[merge] {n:>12}: {top}")
    print(f"[merge] wrote {out}/boxes.csv, images.csv, summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
