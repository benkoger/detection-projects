"""Evaluate wytrap outputs against IMAGE-LEVEL labels (no boxes).

Companion to scripts/eval_pipeline.py for datasets like LILA Idaho Camera
Traps, whose labels are per sequence and carry no bounding boxes. Two
questions are answered:

  1. Detection as presence/absence. For every image, "did wytrap produce
     at least one animal box at det_score >= t?" (any quality tag) is scored
     against "does the label name an animal?". Swept over t. Empty images, camera-problem
     labels (snow on lens, ...), humans and vehicles all count as negatives
     (the detector only keeps 'animal' boxes), and false-positive rate is
     also reported per negative type and per location.

  2. Classification given the image. Among animal-labelled images with at
     least one detection, an image-level species is derived from the boxes
     (--agg max_score: the highest-det_score box wins; --agg vote: sum of
     det_score*cls_score per label) and compared with the label after both
     sides pass through helpers.IDAHO_EVAL_MERGES. Top-1 / top-k accuracy,
     confusion matrix, per-class precision/recall, accuracy-vs-coverage over
     a cls_score floor, and splits by hour-of-day and wytrap quality tag.

  With --sequence-level, frames are grouped by seq_id first: a sequence is
  "detected" if any frame is, and its label vote pools every frame's boxes.
  Use it with subsets fetched via --whole-sequences to soften the
  sequence-label-on-empty-frame noise.

Usage:
    python scripts/eval_image_level.py \\
        --labels /path/to/labels.json \\
        --pred   /path/to/output-wytrap \\
        [--out   /path/to/output-wytrap/eval] \\
        [--det-threshold 0.50] [--agg max_score|vote] [--quality ok|all] \\
        [--sequence-level] [--no-merge] [--top-ks 1,3,5]

Writes <out>/metrics.json, per_image.csv, confusion_matrix.csv/.png, eval.log.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in (REPO_ROOT, REPO_ROOT / "wytrap"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

from helpers.helpers import IDAHO_EVAL_MERGES  # noqa: E402
from helpers.vocab import Vocab  # noqa: E402
from wytrap.io import load_record  # noqa: E402

log = logging.getLogger("wytrap.eval_image")

# Labels that mean "no animal should be detected". Everything else in the
# labels file is treated as an animal class.
NEGATIVE_LABELS = {
    "empty", "human", "vehicle",
    "snow on lens", "foggy lens", "vegetation obstruction", "malfunction",
    "misdirected", "foggy weather", "lens obscured", "sun", "tilted",
}
DEFAULT_DET_SWEEP = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
DEFAULT_CLS_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_labels(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    return d["images"]


def load_predictions(pred_dir: Path) -> dict[str, dict]:
    """Map the last two path components ('loc_x/file.jpg') -> record dict."""
    jsonl = pred_dir / "all_records.jsonl"
    recs: dict[str, dict] = {}
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                recs[_key(d["image_path"])] = d
        log.info("loaded %d records from %s", len(recs), jsonl)
        return recs
    n = 0
    for p in pred_dir.rglob("*.json"):
        if "eval" in p.parts or p.name in ("manifest.json", "labels.json"):
            continue
        try:
            r = load_record(p)
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", p, e)
            continue
        d = {"image_path": r.image_path, "image_size": r.image_size,
             "error": r.error,
             "detections": [vars(x) for x in r.detections]}
        recs[_key(d["image_path"])] = d
        n += 1
    log.info("loaded %d per-image JSONs from %s", n, pred_dir)
    return recs


def _key(image_path: str) -> str:
    parts = Path(image_path).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


# --------------------------------------------------------------------------
# per-image reduction
# --------------------------------------------------------------------------

def merged(name: str, merges: dict[str, str]) -> str:
    return merges.get(name, name)


VOCAB: Vocab | None = None   # set from --vocab; replaces string merges
RELATION_COUNTS: Counter = Counter()


def resolve_box_label(b: dict, merges: dict[str, str]) -> str:
    """Eval label for one box: taxon-node resolution when a vocab is loaded,
    otherwise the legacy string merge. Unresolvable -> 'outside vocabulary'."""
    if VOCAB is None:
        return merged(b["fine_label"], merges)
    lab, rel = VOCAB.resolve(lineage=b.get("lineage"), scientific=b.get("scientific_label"),
                             common=b.get("fine_label"))
    RELATION_COUNTS[rel] += 1
    if lab is None:
        return "outside vocabulary"
    if rel == "coarser":
        # Ancestor of one or more nodes: not credited at top-1, but scored
        # by the hierarchical metric if it contains the ground-truth node.
        rn = VOCAB.pred_rank_name(b.get("lineage"), b.get("scientific_label"))
        return f"{rn[0]}:{rn[1]} (coarser)" if rn else "outside vocabulary"
    return lab


def resolve_topk_entry(e: dict, merges: dict[str, str]) -> str:
    if VOCAB is None:
        return merged(e["common"], merges)
    lab, rel = VOCAB.resolve(lineage=e.get("lineage"), scientific=e.get("scientific"),
                             common=e.get("common"))
    return lab if lab and rel != "coarser" else "outside vocabulary"


def rollup_consistent(pred: str | None, gt: str) -> bool:
    """A coarser prediction whose taxon contains the ground-truth node."""
    if VOCAB is None or not pred or not pred.endswith(" (coarser)"):
        return False
    rank, _, name = pred[:-len(" (coarser)")].partition(":")
    return VOCAB.is_ancestor(rank, name, gt)


def usable_boxes(rec: dict, quality: str) -> list[dict]:
    dets = rec.get("detections") or []
    if quality == "all":
        return dets
    return [d for d in dets if d.get("quality", "ok") == "ok"]


def image_prediction(boxes: list[dict], agg: str, merges: dict[str, str],
                     top_k: int) -> tuple[str | None, float, list[str]]:
    """Return (top1_label, cls_score, topk_labels) for one image or sequence."""
    if not boxes:
        return None, 0.0, []
    if agg == "max_score":
        b = max(boxes, key=lambda d: d["det_score"])
        top1 = resolve_box_label(b, merges)
        # The ranked list starts with the record's own prediction. For an
        # ensemble (e.g. SpeciesNet roll-up) fine_label can differ from the
        # classifier's raw topk[0]; top-1 must score the final answer.
        topk: list[str] = [top1]
        for e in b.get("topk", []):
            m = resolve_topk_entry(e, merges)
            if m not in topk:
                topk.append(m)
        return top1, float(b["cls_score"]), topk[:top_k]
    if agg == "vote":
        score: dict[str, float] = defaultdict(float)
        for b in boxes:
            for e in b.get("topk", []):
                score[resolve_topk_entry(e, merges)] += b["det_score"] * e["score"]
        ranked = sorted(score.items(), key=lambda kv: -kv[1])
        top1, s = ranked[0]
        total = sum(score.values()) or 1.0
        return top1, s / total, [k for k, _ in ranked[:top_k]]
    raise ValueError(agg)


def hour_bucket(dt: str | None) -> str:
    if not dt:
        return "unknown"
    try:
        h = int(dt[11:13])
    except (ValueError, IndexError):
        return "unknown"
    return "day" if 7 <= h < 19 else "night"


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def evaluate(items: list[dict], args, merges: dict[str, str]) -> dict:
    """items: [{key, gt, is_animal, boxes(all quality-filtered), location,
    hour, seq_id}] with boxes already filtered by quality."""
    out: dict = {}

    # ---- detection sweep
    det_rows = []
    for t in args.det_sweep:
        tp = fn = fp = tn = 0
        fp_by_type: Counter = Counter()
        n_by_type: Counter = Counter()
        fp_by_loc: Counter = Counter()
        n_by_loc: Counter = Counter()
        for it in items:
            # Presence/absence: any box counts, whatever its quality tag. A
            # truncated or low-pixel animal is still an animal in the frame.
            has = any(b["det_score"] >= t for b in it["all_boxes"])
            if it["is_animal"]:
                tp += has
                fn += not has
            else:
                n_by_type[it["gt"]] += 1
                n_by_loc[it["location"]] += 1
                if has:
                    fp += 1
                    fp_by_type[it["gt"]] += 1
                    fp_by_loc[it["location"]] += 1
                else:
                    tn += 1
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        det_rows.append({
            "det_threshold": t, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "fp_rate_by_negative_type": {
                k: round(fp_by_type[k] / n_by_type[k], 4) for k in sorted(n_by_type)},
            "fp_rate_by_location_top10": dict(sorted(
                ((k, round(fp_by_loc[k] / n_by_loc[k], 4)) for k in n_by_loc
                 if n_by_loc[k] >= 5), key=lambda kv: -kv[1])[:10]),
        })
    out["detection"] = det_rows
    best = max(det_rows, key=lambda r: r["f1"])
    log.info("detection: best F1 %.3f at det>=%.2f (P %.3f R %.3f)",
             best["f1"], best["det_threshold"], best["precision"], best["recall"])

    # ---- classification at fixed det threshold
    t = args.det_threshold
    cls_items = []
    for it in items:
        if not it["is_animal"]:
            continue
        boxes = [b for b in it["boxes"] if b["det_score"] >= t]
        if not boxes:
            continue
        top1, score, topk = image_prediction(boxes, args.agg, merges, max(args.top_ks))
        cls_items.append({**it, "pred": top1, "cls_score": score, "topk": topk,
                          "n_boxes": len(boxes),
                          "worst_quality": _worst_quality(it["all_boxes"], t)})
    n_animal = sum(it["is_animal"] for it in items)
    out["classification"] = {
        "det_threshold": t, "agg": args.agg,
        "animal_images": n_animal, "animal_images_with_detection": len(cls_items),
    }
    if not cls_items:
        log.warning("no animal images with detections at det>=%.2f", t)
        return out, []

    topk_acc = {}
    for k in args.top_ks:
        hits = sum(it["gt"] in it["topk"][:k] for it in cls_items)
        topk_acc[f"top{k}"] = round(hits / len(cls_items), 4)
    out["classification"]["accuracy"] = topk_acc
    n = len(cls_items)
    rolled = [it for it in cls_items if it["pred"] and it["pred"].endswith(" (coarser)")]
    consistent = sum(rollup_consistent(it["pred"], it["gt"]) for it in rolled)
    out["classification"]["hierarchical"] = {
        # correct at the node's rank, or a roll-up to an ancestor of the true node
        "top1_or_consistent_rollup": round((sum(it["gt"] == it["pred"] for it in cls_items)
                                            + consistent) / n, 4),
        "rollup_rate": round(len(rolled) / n, 4),
        "rollup_consistent_rate": round(consistent / len(rolled), 4) if rolled else None,
        "outside_vocabulary_rate": round(sum(it["pred"] == "outside vocabulary"
                                             for it in cls_items) / n, 4),
    }
    log.info("classification (%d imgs): %s", len(cls_items),
             ", ".join(f"{k} {v:.3f}" for k, v in topk_acc.items()))

    # accuracy vs coverage
    cov_rows = []
    for c in args.cls_sweep:
        kept = [it for it in cls_items if it["cls_score"] >= c]
        acc = (sum(it["gt"] == it["pred"] for it in kept) / len(kept)) if kept else 0.0
        cov_rows.append({"cls_min_conf": c, "coverage": round(len(kept) / len(cls_items), 4),
                         "top1_accuracy": round(acc, 4), "n": len(kept)})
    out["classification"]["accuracy_vs_coverage"] = cov_rows

    # per-class P/R and confusion
    classes = sorted({it["gt"] for it in cls_items} | {it["pred"] for it in cls_items})
    conf: dict[str, Counter] = {c: Counter() for c in classes}
    for it in cls_items:
        conf[it["gt"]][it["pred"]] += 1
    per_class = {}
    for c in classes:
        tp = conf[c][c]
        support = sum(conf[c].values())
        predicted = sum(conf[g][c] for g in classes)
        per_class[c] = {
            "support": support,
            "recall": round(tp / support, 4) if support else None,
            "precision": round(tp / predicted, 4) if predicted else None,
            "top_confusions": [k for k, _ in conf[c].most_common(4) if k != c][:3],
        }
    out["classification"]["per_class"] = per_class
    out["classification"]["confusion_classes"] = classes

    # splits
    def _split(keyfn):
        groups: dict[str, list] = defaultdict(list)
        for it in cls_items:
            groups[keyfn(it)].append(it["gt"] == it["pred"])
        return {k: {"n": len(v), "top1_accuracy": round(sum(v) / len(v), 4)}
                for k, v in sorted(groups.items())}
    out["classification"]["by_time_of_day"] = _split(lambda it: it["hour"])
    out["classification"]["by_worst_quality"] = _split(lambda it: it["worst_quality"])
    out["classification"]["by_n_boxes"] = _split(
        lambda it: "1" if it["n_boxes"] == 1 else ("2-3" if it["n_boxes"] <= 3 else "4+"))
    return out, cls_items, conf, classes


def _worst_quality(all_boxes: list[dict], t: float) -> str:
    order = ["ok", "truncated", "thin", "low_pixels", "skipped", "small", "edge"]
    qs = [b.get("quality", "ok") for b in all_boxes if b["det_score"] >= t]
    if not qs:
        return "none"
    return max(qs, key=lambda q: order.index(q) if q in order else len(order))


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def write_confusion(conf, classes, out_dir: Path) -> None:
    with open(out_dir / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt \\ pred"] + classes)
        for g in classes:
            w.writerow([g] + [conf[g][p] for p in classes])
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        m = np.array([[conf[g][p] for p in classes] for g in classes], dtype=float)
        row = m.sum(axis=1, keepdims=True)
        norm = np.divide(m, row, out=np.zeros_like(m), where=row > 0)
        fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(classes)),) * 2)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
        ax.set_yticks(range(len(classes)), classes, fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
        for i in range(len(classes)):
            for j in range(len(classes)):
                if m[i, j]:
                    ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=6,
                            color="white" if norm[i, j] > 0.5 else "black")
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log.warning("confusion_matrix.png not written: %s", e)


def write_per_image(items: list[dict], cls_by_key: dict[str, dict], t: float,
                    out_dir: Path) -> None:
    with open(out_dir / "per_image.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "seq_id", "location", "hour", "gt", "is_animal",
                    "n_boxes_at_t", "max_det_score", "n_boxes_any_quality", "pred", "cls_score",
                    "topk", "correct"])
        for it in items:
            boxes = [b for b in it["boxes"] if b["det_score"] >= t]
            c = cls_by_key.get(it["key"])
            w.writerow([
                it["key"], it["seq_id"], it["location"], it["hour"], it["gt"],
                int(it["is_animal"]), len(boxes),
                round(max((b["det_score"] for b in it["all_boxes"]), default=0.0), 4),
                sum(b["det_score"] >= t for b in it["all_boxes"]),
                c["pred"] if c else "", round(c["cls_score"], 4) if c else "",
                "|".join(c["topk"]) if c else "",
                int(c["gt"] == c["pred"]) if c else "",
            ])


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build_items(labels: list[dict], preds: dict[str, dict], merges: dict[str, str],
                quality: str, sequence_level: bool) -> tuple[list[dict], int]:
    items, missing = [], 0
    for im in labels:
        key = im["file_name"]
        rec = preds.get(key)
        if rec is None:
            missing += 1
            continue
        gt_raw = im["labels"][0] if im.get("labels") else "empty"
        gt = merged(gt_raw, merges)
        items.append({
            "key": key, "seq_id": im.get("seq_id", key), "location": im.get("location", "?"),
            "hour": hour_bucket(im.get("datetime")),
            "gt": gt, "is_animal": gt_raw not in NEGATIVE_LABELS,
            "boxes": usable_boxes(rec, quality),
            "all_boxes": rec.get("detections") or [],
        })
    if not sequence_level:
        return items, missing
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        groups[it["seq_id"]].append(it)
    seq_items = []
    for seq, frames in groups.items():
        f0 = frames[0]
        seq_items.append({
            **f0, "key": seq,
            "boxes": [b for f in frames for b in f["boxes"]],
            "all_boxes": [b for f in frames for b in f["all_boxes"]],
        })
    return seq_items, missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default=None, help="default: <pred>/eval")
    ap.add_argument("--det-threshold", type=float, default=0.50,
                    help="det_score floor for the classification section")
    ap.add_argument("--det-sweep", default=",".join(map(str, DEFAULT_DET_SWEEP)))
    ap.add_argument("--cls-sweep", default=",".join(map(str, DEFAULT_CLS_SWEEP)))
    ap.add_argument("--agg", choices=["max_score", "vote"], default="max_score")
    ap.add_argument("--quality", choices=["ok", "all"], default="ok")
    ap.add_argument("--top-ks", default="1,3,5")
    ap.add_argument("--sequence-level", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--vocab", default=None,
                    help="taxon-node vocabulary CSV (taxonomy/idaho_vocab.csv); predictions "
                         "resolve by lineage / scientific name instead of string merges")
    args = ap.parse_args(argv)
    args.det_sweep = [float(x) for x in args.det_sweep.split(",")]
    args.cls_sweep = [float(x) for x in args.cls_sweep.split(",")]
    args.top_ks = [int(x) for x in args.top_ks.split(",")]

    pred_dir = Path(args.pred)
    out_dir = Path(args.out) if args.out else pred_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out_dir / "eval.log", mode="w")])
    merges = {} if args.no_merge else IDAHO_EVAL_MERGES
    global VOCAB
    if args.vocab:
        VOCAB = Vocab.load(args.vocab)
        log.info("vocabulary: %s (%d labels, %d member taxa)", args.vocab,
                 len(VOCAB.labels), len(VOCAB.members))

    labels = load_labels(Path(args.labels))
    preds = load_predictions(pred_dir)
    items, missing = build_items(labels, preds, merges, args.quality, args.sequence_level)
    unit = "sequences" if args.sequence_level else "images"
    log.info("%d labelled images, %d without predictions, %d %s evaluated",
             len(labels), missing, len(items), unit)
    log.info("gt distribution: %s", dict(Counter(it["gt"] for it in items).most_common()))

    result = evaluate(items, args, merges)
    metrics, cls_items = result[0], result[1]
    metrics["config"] = {k: v for k, v in vars(args).items()}
    if VOCAB is not None:
        metrics["vocab_resolution"] = dict(RELATION_COUNTS)
        log.info("prediction relation to vocabulary nodes: %s", dict(RELATION_COUNTS))
    metrics["n_labelled"] = len(labels)
    metrics["n_missing_predictions"] = missing
    metrics["unit"] = unit
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_per_image(items, {c["key"]: c for c in cls_items}, args.det_threshold, out_dir)
    if len(result) == 4:
        write_confusion(result[2], result[3], out_dir)
    log.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
