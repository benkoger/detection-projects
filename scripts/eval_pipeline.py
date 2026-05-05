"""Evaluate wytrap (MegaDetector + BioCLIP) outputs against COCO ground truth.

Standalone counterpart to ``pipeline_vs_frcnn.ipynb``: takes a COCO json with
GT boxes/labels and a folder of wytrap per-image JSONs, IoU-matches the
predictions to GT, and reports detection precision/recall, classification
top-1 / top-3 / top-5 accuracy, and the top-k "recovery rate" (cases where
the GT label was in top-5 but not top-1).

Usage:
    python scripts/eval_pipeline.py \\
        --gt   /path/to/combined.json \\
        --pred /path/to/output-pipeline \\
        [--out /path/to/output-pipeline/eval] \\
        [--iou 0.5] [--quality ok|all] \\
        [--no-merge]                    # disable GT label merging

Writes <out>/metrics.json, <out>/confusion_matrix.csv, <out>/confusion_matrix.png,
and <out>/eval.log. Logs go to stdout too so SLURM .out captures them.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make `helpers/` and `wytrap/` importable when running from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in (REPO_ROOT, REPO_ROOT / "wytrap"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

from helpers.helpers import DEFAULT_CATEGORY_MERGES
from wytrap.io import load_record


log = logging.getLogger("wytrap.eval")


def setup_logging(log_file: Path | None, level: int = logging.INFO) -> None:
    fmt = "%(asctime)s [%(levelname)-7s] %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--gt", required=True,
                   help="COCO json with ground-truth boxes/labels.")
    p.add_argument("--pred", required=True,
                   help="Folder of wytrap per-image JSON outputs.")
    p.add_argument("--out",
                   help="Eval output dir. Default: <pred>/eval/.")
    p.add_argument("--iou", type=float, default=0.5,
                   help="IoU threshold for matching predictions to GT (default 0.5).")
    p.add_argument("--quality", default="ok", choices=["ok", "all"],
                   help="Restrict to quality=='ok' detections (default) or "
                        "evaluate every detection regardless of quality.")
    p.add_argument("--no-merge", action="store_true",
                   help="Skip the predator/deer category merge. By default, GT "
                        "labels (coyote/wolf/fox/black bear/grizzly bear/mule "
                        "deer/white-tailed deer) are collapsed to Canid/Bear/Deer "
                        "to align with the ynp_testbed species list.")
    p.add_argument("--top-ks", default="1,3,5",
                   help="Comma-separated k values for top-k accuracy (default '1,3,5').")
    return p.parse_args()


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(gt_path: Path,
            merges: dict[str, str]) -> tuple[dict[str, list], dict[str, str]]:
    """Returns (gt_by_file, file_to_date).

    gt_by_file: {file_basename: [(xyxy_box, merged_label), ...]}
    file_to_date: {file_basename: 'YYYY-MM-DD' or '' if not parseable}
    """
    with open(gt_path) as f:
        coco = json.load(f)
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    im_id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}

    file_to_date: dict[str, str] = {}
    for im in coco["images"]:
        # date_captured is ISO-8601 like "2024-08-28T17:57:49.000+00:00".
        # We want just the calendar day. Tolerate missing/malformed values.
        raw = (im.get("date_captured") or "").strip()
        date = raw.split("T", 1)[0] if "T" in raw else raw[:10]
        file_to_date[im["file_name"]] = date

    gt_by_file: dict[str, list] = defaultdict(list)
    for ann in coco["annotations"]:
        fname = im_id_to_file[ann["image_id"]]
        if fname.startswith("."):
            continue
        x, y, w, h = ann["bbox"]
        raw_label = id_to_name[ann["category_id"]]
        label = merges.get(raw_label, raw_label)
        gt_by_file[fname].append((
            [int(x), int(y), int(x + w), int(y + h)], label
        ))
    return gt_by_file, file_to_date


def load_pred(pred_dir: Path,
              gt_files: set[str],
              merges: dict[str, str],
              quality_filter: set[str] | None) -> tuple[dict[str, list], Counter]:
    """Returns ({file_basename: [(xyxy, merged_top1, [merged_topk_labels])]}, quality_counts)."""
    pred_by_file: dict[str, list] = defaultdict(list)
    quality_counts: Counter[str] = Counter()
    n_records_seen = 0
    n_records_matched = 0
    for jf in pred_dir.rglob("*.json"):
        # skip eval/metrics.json from prior runs
        if jf.parent.name == "eval":
            continue
        try:
            rec = load_record(jf)
        except Exception as e:
            log.warning("could not read %s: %s", jf, e)
            continue
        n_records_seen += 1
        fname = Path(rec.image_path).name
        if fname.startswith("."):
            continue
        if fname not in gt_files:
            continue
        n_records_matched += 1
        for det in rec.detections:
            quality_counts[det.quality] += 1
            if quality_filter is not None and det.quality not in quality_filter:
                continue
            top1 = merges.get(det.label, det.label)
            topk = [merges.get(t.get("common", ""), t.get("common", ""))
                    for t in (det.topk or [])]
            pred_by_file[fname].append((det.box_xyxy, top1, topk))
    log.info("scanned %d pipeline JSONs, matched %d to GT filenames",
             n_records_seen, n_records_matched)
    return pred_by_file, quality_counts


def evaluate(gt_by_file: dict[str, list],
             pred_by_file: dict[str, list],
             iou_thresh: float,
             top_ks: list[int]) -> dict:
    tp = fp = fn = 0
    matched = 0
    correct_at = {k: 0 for k in top_ks}
    confusion: dict[str, Counter] = defaultdict(Counter)
    in_topk_only: dict[str, Counter] = defaultdict(Counter)
    max_k = max(top_ks)

    for fname, gts in gt_by_file.items():
        preds = pred_by_file.get(fname, [])
        gt_used = [False] * len(gts)
        for pbox, plabel, ptopk in preds:
            best_iou, best_j = 0.0, -1
            for j, (gbox, _) in enumerate(gts):
                if gt_used[j]:
                    continue
                v = iou(pbox, gbox)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= iou_thresh:
                tp += 1
                gt_used[best_j] = True
                glabel = gts[best_j][1]
                matched += 1
                confusion[glabel][plabel] += 1
                for k in top_ks:
                    if glabel in ptopk[:k]:
                        correct_at[k] += 1
                if plabel != glabel and glabel in ptopk[:max_k]:
                    in_topk_only[glabel][plabel] += 1
            else:
                fp += 1
        fn += sum(1 for u in gt_used if not u)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    metrics = {
        "tp": tp, "fp": fp, "fn": fn, "matched": matched,
        "precision": precision, "recall": recall,
        "iou_thresh": iou_thresh,
    }
    for k in top_ks:
        metrics[f"top{k}_acc"] = correct_at[k] / matched if matched else 0.0
    if 1 in top_ks and max_k != 1:
        metrics[f"top{max_k}_recovery"] = (
            (correct_at[max_k] - correct_at[1]) / matched if matched else 0.0
        )
    metrics["confusion"] = {gt: dict(row) for gt, row in confusion.items()}
    metrics["in_topk_only"] = {gt: dict(row) for gt, row in in_topk_only.items()}
    return metrics


def write_confusion(metrics: dict, out_dir: Path) -> None:
    confusion = metrics["confusion"]
    labels = sorted(set(confusion.keys()) |
                    {p for row in confusion.values() for p in row})
    if not labels:
        log.warning("confusion matrix is empty; skipping CSV/PNG")
        return

    csv_path = out_dir / "confusion_matrix.csv"
    with open(csv_path, "w") as f:
        f.write("gt\\pred," + ",".join(labels) + "\n")
        for gt in labels:
            row = confusion.get(gt, {})
            f.write(gt + "," + ",".join(str(row.get(p, 0)) for p in labels) + "\n")
    log.info("wrote %s", csv_path)

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib/numpy not available, skipping confusion PNG")
        return

    n = len(labels)
    mat = np.zeros((n, n), dtype=int)
    for i, gt in enumerate(labels):
        for j, pred in enumerate(labels):
            mat[i, j] = confusion.get(gt, {}).get(pred, 0)

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * n + 2), max(7, 0.6 * n + 1)))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
    ax.set_title("Wytrap (MegaDetector + BioCLIP) confusion matrix")
    for i in range(n):
        for j in range(n):
            if mat[i, j]:
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() / 2 else "black",
                        fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    png_path = out_dir / "confusion_matrix.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log.info("wrote %s", png_path)


def build_species_timeseries(file_to_date: dict[str, str],
                             gt_by_file: dict[str, list],
                             pred_by_file: dict[str, list]
                             ) -> tuple[list[str], list[str],
                                        dict[tuple[str, str], int],
                                        dict[tuple[str, str], int]]:
    """Group species presence by calendar day for both GT and predictions.

    Returns (sorted_dates, sorted_species, gt_presence, pred_presence).
    *_presence values are int counts of detections that day; 0 means absent.
    """
    gt_presence: dict[tuple[str, str], int] = defaultdict(int)
    for fname, gts in gt_by_file.items():
        date = file_to_date.get(fname, "")
        if not date:
            continue
        for _box, label in gts:
            gt_presence[(date, label)] += 1

    pred_presence: dict[tuple[str, str], int] = defaultdict(int)
    for fname, preds in pred_by_file.items():
        date = file_to_date.get(fname, "")
        if not date:
            continue
        for _box, label, _topk in preds:
            pred_presence[(date, label)] += 1

    dates = sorted({d for (d, _) in gt_presence} | {d for (d, _) in pred_presence})
    species = sorted({s for (_, s) in gt_presence} | {s for (_, s) in pred_presence})
    return dates, species, dict(gt_presence), dict(pred_presence)


def write_species_timeseries(file_to_date: dict[str, str],
                             gt_by_file: dict[str, list],
                             pred_by_file: dict[str, list],
                             out_dir: Path) -> None:
    """Render a per-day species presence/absence plot and matching CSV.

    Two stacked panels:
      (top)    GT — what the annotators actually saw on each day.
      (bottom) Predictions — what wytrap thinks was present.

    Cell color encodes count of detections on that day (white=absent).
    """
    dates, species, gt_pres, pred_pres = build_species_timeseries(
        file_to_date, gt_by_file, pred_by_file
    )
    if not dates or not species:
        log.warning("not enough data for species timeseries plot "
                    "(dates=%d, species=%d)", len(dates), len(species))
        return

    # CSV with one row per (date, species) and columns gt_count, pred_count.
    csv_path = out_dir / "species_timeseries.csv"
    with open(csv_path, "w") as f:
        f.write("date,species,gt_count,pred_count\n")
        for d in dates:
            for s in species:
                g = gt_pres.get((d, s), 0)
                p = pred_pres.get((d, s), 0)
                if g or p:
                    f.write(f"{d},{s},{g},{p}\n")
    log.info("wrote %s", csv_path)

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib/numpy not available, skipping timeseries PNG")
        return

    # Build dense calendar index (every day from min..max, even gaps) so the
    # x-axis reflects real time rather than "days we happened to see anything".
    from datetime import date as _date, timedelta
    try:
        d_min = min(_date.fromisoformat(d) for d in dates)
        d_max = max(_date.fromisoformat(d) for d in dates)
    except ValueError:
        log.warning("non-ISO date_captured values in GT; skipping timeseries PNG")
        return

    n_days = (d_max - d_min).days + 1
    full_dates = [(d_min + timedelta(days=i)).isoformat() for i in range(n_days)]
    date_idx = {d: i for i, d in enumerate(full_dates)}

    n_sp = len(species)
    gt_mat = np.zeros((n_sp, n_days), dtype=int)
    pred_mat = np.zeros((n_sp, n_days), dtype=int)
    for (d, s), c in gt_pres.items():
        if d in date_idx:
            gt_mat[species.index(s), date_idx[d]] = c
    for (d, s), c in pred_pres.items():
        if d in date_idx:
            pred_mat[species.index(s), date_idx[d]] = c

    fig_h = max(4.0, 0.35 * n_sp + 1.2)
    fig_w = max(10.0, 0.12 * n_days + 3.0)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h * 2),
                             sharex=True, sharey=True)
    vmax = max(gt_mat.max(), pred_mat.max(), 1)

    for ax, mat, title in [(axes[0], gt_mat,   "Ground truth (annotated)"),
                           (axes[1], pred_mat, "Wytrap predictions")]:
        im = ax.imshow(mat, aspect="auto", cmap="viridis",
                       vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_yticks(range(n_sp))
        ax.set_yticklabels(species)
        ax.set_title(f"{title}  ({mat.sum()} total detections)")
        ax.set_ylabel("species")
        fig.colorbar(im, ax=ax, label="detections / day")

    # X tick density: roughly one label per ~7 days, but always at least 6.
    step = max(1, n_days // max(6, n_days // 7))
    tick_idx = list(range(0, n_days, step))
    axes[1].set_xticks(tick_idx)
    axes[1].set_xticklabels([full_dates[i] for i in tick_idx],
                            rotation=45, ha="right", fontsize=8)
    axes[1].set_xlabel("date")

    fig.suptitle("Species presence per day  (white = absent)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    png_path = out_dir / "species_timeseries.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log.info("wrote %s", png_path)


def main() -> int:
    args = parse_args()

    pred_dir = Path(args.pred)
    out_dir = Path(args.out) if args.out else (pred_dir / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "eval.log")

    top_ks = sorted({int(k) for k in args.top_ks.split(",") if k.strip()})
    quality_filter = {"ok"} if args.quality == "ok" else None
    merges = {} if args.no_merge else DEFAULT_CATEGORY_MERGES

    log.info("gt              : %s", args.gt)
    log.info("pred dir        : %s", pred_dir)
    log.info("out dir         : %s", out_dir)
    log.info("iou threshold   : %s", args.iou)
    log.info("quality filter  : %s", args.quality)
    log.info("merge GT labels : %s", "no" if args.no_merge else "yes (DEFAULT_CATEGORY_MERGES)")
    log.info("top-k           : %s", top_ks)

    gt_by_file, file_to_date = load_gt(Path(args.gt), merges=merges)
    log.info("loaded %d GT images, %d boxes",
             len(gt_by_file), sum(len(v) for v in gt_by_file.values()))

    gt_files = set(gt_by_file.keys())
    pred_by_file, quality_counts = load_pred(
        pred_dir, gt_files=gt_files, merges=merges,
        quality_filter=quality_filter,
    )
    log.info("quality breakdown across all loaded preds: %s",
             dict(quality_counts))
    log.info("eligible (quality-filtered) prediction images: %d, boxes: %d",
             len(pred_by_file), sum(len(v) for v in pred_by_file.values()))

    metrics = evaluate(gt_by_file, pred_by_file,
                       iou_thresh=args.iou, top_ks=top_ks)
    metrics["quality_counts"] = dict(quality_counts)
    metrics["quality_filter"] = args.quality
    metrics["merge_applied"] = not args.no_merge
    metrics["n_gt_images"] = len(gt_by_file)
    metrics["n_gt_boxes"] = sum(len(v) for v in gt_by_file.values())

    log.info("===== detection =====")
    log.info("  precision : %.3f", metrics["precision"])
    log.info("  recall    : %.3f", metrics["recall"])
    log.info("  tp/fp/fn  : %d / %d / %d",
             metrics["tp"], metrics["fp"], metrics["fn"])

    log.info("===== classification (matched detections only, n=%d) =====",
             metrics["matched"])
    for k in top_ks:
        log.info("  top-%d acc : %.3f", k, metrics[f"top{k}_acc"])
    max_k = max(top_ks)
    if 1 in top_ks and max_k != 1:
        log.info("  top-%d recovery (in top-%d but not top-1): %.3f",
                 max_k, max_k, metrics[f"top{max_k}_recovery"])

    log.info("===== where top-%d saves us (gt label in top-%d but top-1 wrong) =====",
             max_k, max_k)
    for gt_label, confused in sorted(metrics["in_topk_only"].items(),
                                     key=lambda kv: -sum(kv[1].values())):
        total = sum(confused.values())
        most = ", ".join(f"{name}({n})"
                         for name, n in
                         Counter(confused).most_common(3))
        log.info("  %-20s %5d  most-confused-with: %s", gt_label, total, most)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("wrote %s", metrics_path)

    write_confusion(metrics, out_dir)
    write_species_timeseries(file_to_date, gt_by_file, pred_by_file, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
