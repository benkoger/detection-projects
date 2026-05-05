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

from helpers.helpers import DEFAULT_CATEGORY_MERGES, YNP_EVAL_MERGES
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
                   help="Skip the eval-time category merge. By default, both GT "
                        "labels and predicted species are collapsed via "
                        "helpers.YNP_EVAL_MERGES so e.g. predicted "
                        "'yellow-bellied marmot' counts as a hit on GT 'rodent'.")
    p.add_argument("--merge-map", default="ynp_eval",
                   choices=["ynp_eval", "default"],
                   help="Which merge map to apply. 'ynp_eval' (default) "
                        "collapses canids/bears/deer/rodents/birds; 'default' "
                        "is the predator/deer-only map used by FRCNN training.")
    p.add_argument("--top-ks", default="1,3,5",
                   help="Comma-separated k values for top-k accuracy (default '1,3,5').")
    p.add_argument("--cls-min-confidence", type=float, default=0.30,
                   help="Below this BioCLIP top-1 score, a matched detection "
                        "counts as 'abstained' rather than committed. "
                        "Classification metrics report only on committed; "
                        "abstention is its own line item. Default 0.30.")
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
    orphan_ids: set[int] = set()
    n_orphans = 0
    for ann in coco["annotations"]:
        fname = im_id_to_file.get(ann["image_id"])
        if fname is None or fname.startswith("."):
            continue
        cat_id = ann["category_id"]
        if cat_id not in id_to_name:
            orphan_ids.add(cat_id)
            n_orphans += 1
            continue
        x, y, w, h = ann["bbox"]
        raw_label = id_to_name[cat_id]
        label = merges.get(raw_label, raw_label)
        gt_by_file[fname].append((
            [int(x), int(y), int(x + w), int(y + h)], label
        ))
    if n_orphans:
        log.warning("dropped %d annotations with unknown category_id(s) %s "
                    "(not in coco['categories']).",
                    n_orphans, sorted(orphan_ids))
    return gt_by_file, file_to_date


def load_pred(pred_dir: Path,
              gt_files: set[str],
              merges: dict[str, str],
              quality_filter: set[str] | None
              ) -> tuple[dict[str, list], Counter, Counter]:
    """Load wytrap per-image JSONs.

    Returns:
        pred_by_file: {file_basename: [(xyxy, merged_top1, [merged_topk_labels],
                                         cls_score, scale)]}
        quality_counts: Counter of quality field values across all preds.
        scale_counts:   Counter of scale field values across all preds.
    """
    pred_by_file: dict[str, list] = defaultdict(list)
    quality_counts: Counter[str] = Counter()
    scale_counts:   Counter[str] = Counter()
    n_records_seen = 0
    n_records_matched = 0
    for jf in pred_dir.rglob("*.json"):
        # skip eval/metrics.json from prior runs
        if jf.parent.name == "eval":
            continue
        # skip macOS AppleDouble stubs (._FILENAME) — they look like JSON by
        # extension but are 4KB binary blobs.
        if jf.name.startswith("."):
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
            scale_counts[det.scale] += 1
            if quality_filter is not None and det.quality not in quality_filter:
                continue
            top1 = merges.get(det.label, det.label)
            topk = [merges.get(t.get("common", ""), t.get("common", ""))
                    for t in (det.topk or [])]
            pred_by_file[fname].append((det.box_xyxy, top1, topk,
                                        float(det.cls_score), det.scale))
    log.info("scanned %d pipeline JSONs, matched %d to GT filenames",
             n_records_seen, n_records_matched)
    return pred_by_file, quality_counts, scale_counts


def evaluate(gt_by_file: dict[str, list],
             pred_by_file: dict[str, list],
             iou_thresh: float,
             top_ks: list[int],
             cls_min_confidence: float = 0.30) -> dict:
    """Evaluate predictions vs GT.

    Detection P/R is computed over all matched/unmatched boxes regardless of
    classification confidence. Classification metrics (top-1/3/5 accuracy,
    confusion matrix) are computed only on **committed** matches:
    cls_score >= cls_min_confidence. Matched-but-abstained predictions are
    counted in `abstained` for separate reporting.

    Each `pred_by_file` entry is a 5-tuple
    (xyxy_box, top1_label, topk_labels, cls_score, scale).
    """
    tp = fp = fn = 0
    matched = 0
    committed = 0
    correct_at = {k: 0 for k in top_ks}
    confusion: dict[str, Counter] = defaultdict(Counter)
    in_topk_only: dict[str, Counter] = defaultdict(Counter)
    # Per-class abstention tracking (matched detections only).
    matched_per_class: Counter[str] = Counter()
    abstained_per_class: Counter[str] = Counter()
    # Per-scale tracking (committed only).
    scale_n: Counter[str] = Counter()
    scale_correct: Counter[str] = Counter()
    # Detections matched-but-abstained where the GT label was nonetheless
    # in the top-k — these are recoverable if the threshold is lowered.
    abstained_with_gt_in_topk = 0
    # Raw matched-detection records for the threshold sweep.
    matched_records: list[tuple[float, str, str, list, str]] = []
    max_k = max(top_ks)

    for fname, gts in gt_by_file.items():
        preds = pred_by_file.get(fname, [])
        gt_used = [False] * len(gts)
        for pbox, plabel, ptopk, cls_score, scale in preds:
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
                matched_per_class[glabel] += 1
                matched_records.append((cls_score, plabel, glabel, ptopk, scale))
                if cls_score >= cls_min_confidence:
                    committed += 1
                    confusion[glabel][plabel] += 1
                    scale_n[scale] += 1
                    if plabel == glabel:
                        scale_correct[scale] += 1
                    for k in top_ks:
                        if glabel in ptopk[:k]:
                            correct_at[k] += 1
                    if plabel != glabel and glabel in ptopk[:max_k]:
                        in_topk_only[glabel][plabel] += 1
                else:
                    abstained_per_class[glabel] += 1
                    if glabel in ptopk[:max_k]:
                        abstained_with_gt_in_topk += 1
            else:
                fp += 1
        fn += sum(1 for u in gt_used if not u)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    abstained = matched - committed

    metrics = {
        "tp": tp, "fp": fp, "fn": fn,
        "matched": matched,
        "committed": committed,
        "abstained": abstained,
        "abstention_rate": (abstained / matched) if matched else 0.0,
        "abstained_but_gt_in_topk": abstained_with_gt_in_topk,
        "precision": precision, "recall": recall,
        "iou_thresh": iou_thresh,
        "cls_min_confidence": cls_min_confidence,
    }
    for k in top_ks:
        metrics[f"top{k}_acc"] = (correct_at[k] / committed) if committed else 0.0
    if 1 in top_ks and max_k != 1:
        metrics[f"top{max_k}_recovery"] = (
            (correct_at[max_k] - correct_at[1]) / committed if committed else 0.0
        )
    metrics["confusion"] = {gt: dict(row) for gt, row in confusion.items()}
    metrics["in_topk_only"] = {gt: dict(row) for gt, row in in_topk_only.items()}

    # Per-class abstention rate (matched detections only).
    abstention_by_class = {}
    for cls in sorted(matched_per_class):
        m = matched_per_class[cls]
        a = abstained_per_class[cls]
        abstention_by_class[cls] = {
            "matched": m, "abstained": a,
            "rate": (a / m) if m else 0.0,
        }
    metrics["abstention_by_class"] = abstention_by_class

    # Scale breakdown (committed only).
    scale_breakdown = {}
    for s in sorted(scale_n):
        n = scale_n[s]
        scale_breakdown[s] = {
            "n":         n,
            "correct":   scale_correct[s],
            "top1_acc":  (scale_correct[s] / n) if n else 0.0,
        }
    metrics["scale_breakdown"] = scale_breakdown

    # Threshold sweep — re-bin matched_records over a fixed grid.
    sweep = []
    for thresh in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60):
        cm, cc = 0, 0
        for cls_score, plabel, glabel, ptopk, scale in matched_records:
            if cls_score >= thresh:
                cm += 1
                if plabel == glabel:
                    cc += 1
        sweep.append({
            "threshold":     thresh,
            "committed":     cm,
            "committed_pct": (cm / matched) if matched else 0.0,
            "abstention":    1.0 - ((cm / matched) if matched else 0.0),
            "top1_acc":      (cc / cm) if cm else 0.0,
        })
    metrics["threshold_sweep"] = sweep
    return metrics


def write_confusion(metrics: dict, out_dir: Path,
                    suffix: str = "",
                    title_qualifier: str = "") -> None:
    """Write confusion_matrix{suffix}.csv and .png based on metrics["confusion"].

    `suffix` is appended to the output filenames (e.g. "_committed", "_all")
    so multiple variants can coexist in one run. `title_qualifier` is an
    extra string to splice into the figure title (e.g. "committed only,
    cls_score >= 0.30" or "all detections, no confidence threshold").
    """
    confusion = metrics["confusion"]
    labels = sorted(set(confusion.keys()) |
                    {p for row in confusion.values() for p in row})
    if not labels:
        log.warning("confusion matrix is empty; skipping CSV/PNG (suffix=%r)",
                    suffix)
        return

    csv_path = out_dir / f"confusion_matrix{suffix}.csv"
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

    # Row-normalize so the color encodes per-class behavior, not the absolute
    # frequency of the class. Otherwise the most common class (bison) saturates
    # the colorbar and the rest of the matrix looks empty. Raw counts are
    # preserved in confusion_matrix*.csv and shown as cell annotations.
    row_sum = mat.sum(axis=1, keepdims=True).astype(float)
    row_sum[row_sum == 0] = 1.0  # avoid /0 for any all-zero rows
    norm = mat / row_sum

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * n + 2), max(7, 0.6 * n + 1)))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    abstention_by_class = metrics.get("abstention_by_class", {})
    yticklabels = []
    for i, l in enumerate(labels):
        info = abstention_by_class.get(l, {})
        committed = int(mat[i].sum())
        abstained = info.get("abstained", 0)
        if abstained:
            yticklabels.append(f"{l} (n={committed}, abstained={abstained})")
        else:
            yticklabels.append(f"{l} (n={committed})")
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    title = "Wytrap (MegaDetector + BioCLIP) confusion matrix"
    if title_qualifier:
        title += f" — {title_qualifier}"
    ax.set_title(title + "\ncolor = row-normalized fraction; "
                         "cell text = pct (raw count)")
    for i in range(n):
        for j in range(n):
            if mat[i, j]:
                pct = norm[i, j] * 100
                ax.text(j, i, f"{pct:.0f}%\n({mat[i, j]})",
                        ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black",
                        fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("fraction of GT class predicted as column")
    fig.tight_layout()
    png_path = out_dir / f"confusion_matrix{suffix}.png"
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
        for entry in preds:
            # entries are 5-tuples; older callers may pass 3-tuples.
            label = entry[1]
            pred_presence[(date, label)] += 1

    dates = sorted({d for (d, _) in gt_presence} | {d for (d, _) in pred_presence})
    species = sorted({s for (_, s) in gt_presence} | {s for (_, s) in pred_presence})
    return dates, species, dict(gt_presence), dict(pred_presence)


def write_species_timeseries(file_to_date: dict[str, str],
                             gt_by_file: dict[str, list],
                             pred_by_file: dict[str, list],
                             out_dir: Path,
                             suffix: str = "",
                             title_qualifier: str = "") -> None:
    """Render a per-day species presence/absence plot and matching CSV.

    Two stacked panels:
      (top)    GT — what the annotators actually saw on each day.
      (bottom) Predictions — what wytrap thinks was present.

    `suffix` is appended to the output filenames (e.g. "_committed", "_all").
    `title_qualifier` adds a description to the figure title.
    """
    dates, species, gt_pres, pred_pres = build_species_timeseries(
        file_to_date, gt_by_file, pred_by_file
    )
    if not dates or not species:
        log.warning("not enough data for species timeseries plot "
                    "(dates=%d, species=%d, suffix=%r)",
                    len(dates), len(species), suffix)
        return

    # CSV with one row per (date, species) and columns gt_count, pred_count.
    csv_path = out_dir / f"species_timeseries{suffix}.csv"
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

    # Binary presence/absence: white = no detections that day, solid color =
    # one or more. We don't care about density on this plot — it's a "did we
    # see species X on date Y" answer, not a count.
    from matplotlib.colors import ListedColormap
    gt_cmap   = ListedColormap(["white", "#1f77b4"])   # blue for GT
    pred_cmap = ListedColormap(["white", "#d62728"])   # red for predictions

    for ax, mat, title, cmap in [
        (axes[0], gt_mat,   "Ground truth (annotated)",  gt_cmap),
        (axes[1], pred_mat, "Wytrap predictions",        pred_cmap),
    ]:
        ax.imshow((mat > 0).astype(int), aspect="auto", cmap=cmap,
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(n_sp))
        ax.set_yticklabels(species)
        n_present = int((mat > 0).sum())
        ax.set_title(f"{title}  "
                     f"({n_present} species-days, {mat.sum()} detections)")
        ax.set_ylabel("species")
        # faint grid lines so dense days near each other are still readable
        ax.set_xticks([i - 0.5 for i in range(1, n_days)], minor=True)
        ax.set_yticks([i - 0.5 for i in range(1, n_sp)], minor=True)
        ax.grid(which="minor", color="lightgray", linewidth=0.3)
        ax.tick_params(which="minor", length=0)

    # X tick density: roughly one label per ~7 days, but always at least 6.
    step = max(1, n_days // max(6, n_days // 7))
    tick_idx = list(range(0, n_days, step))
    axes[1].set_xticks(tick_idx)
    axes[1].set_xticklabels([full_dates[i] for i in tick_idx],
                            rotation=45, ha="right", fontsize=8)
    axes[1].set_xlabel("date")

    title = "Species presence per day  (white = absent, color = present)"
    if title_qualifier:
        title += f"\n{title_qualifier}"
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()
    png_path = out_dir / f"species_timeseries{suffix}.png"
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
    merge_map = {
        "ynp_eval": YNP_EVAL_MERGES,
        "default":  DEFAULT_CATEGORY_MERGES,
    }[args.merge_map]
    merges = {} if args.no_merge else merge_map

    log.info("gt              : %s", args.gt)
    log.info("pred dir        : %s", pred_dir)
    log.info("out dir         : %s", out_dir)
    log.info("iou threshold   : %s", args.iou)
    log.info("quality filter  : %s", args.quality)
    log.info("merge labels    : %s",
             "no" if args.no_merge else f"yes ({args.merge_map}, {len(merges)} entries)")
    log.info("top-k           : %s", top_ks)
    log.info("commit threshold: cls_score >= %.2f", args.cls_min_confidence)

    gt_by_file, file_to_date = load_gt(Path(args.gt), merges=merges)
    log.info("loaded %d GT images, %d boxes",
             len(gt_by_file), sum(len(v) for v in gt_by_file.values()))

    gt_files = set(gt_by_file.keys())
    pred_by_file, quality_counts, scale_counts = load_pred(
        pred_dir, gt_files=gt_files, merges=merges,
        quality_filter=quality_filter,
    )
    log.info("quality breakdown across all loaded preds: %s",
             dict(quality_counts))
    log.info("scale breakdown across all loaded preds:   %s",
             dict(scale_counts))
    log.info("eligible (quality-filtered) prediction images: %d, boxes: %d",
             len(pred_by_file), sum(len(v) for v in pred_by_file.values()))

    metrics = evaluate(gt_by_file, pred_by_file,
                       iou_thresh=args.iou, top_ks=top_ks,
                       cls_min_confidence=args.cls_min_confidence)
    metrics["quality_counts"] = dict(quality_counts)
    metrics["scale_counts"]   = dict(scale_counts)
    metrics["quality_filter"] = args.quality
    metrics["merge_applied"] = not args.no_merge
    metrics["n_gt_images"] = len(gt_by_file)
    metrics["n_gt_boxes"] = sum(len(v) for v in gt_by_file.values())

    log.info("===== detection =====")
    log.info("  precision : %.3f", metrics["precision"])
    log.info("  recall    : %.3f", metrics["recall"])
    log.info("  tp/fp/fn  : %d / %d / %d",
             metrics["tp"], metrics["fp"], metrics["fn"])

    log.info("===== classification (committed only, n=%d / matched=%d) =====",
             metrics["committed"], metrics["matched"])
    for k in top_ks:
        log.info("  top-%d acc : %.3f", k, metrics[f"top{k}_acc"])
    max_k = max(top_ks)
    if 1 in top_ks and max_k != 1:
        log.info("  top-%d recovery (in top-%d but not top-1): %.3f",
                 max_k, max_k, metrics[f"top{max_k}_recovery"])

    log.info("===== abstention (matched detections, threshold=%.2f) =====",
             metrics["cls_min_confidence"])
    log.info("  matched              : %d", metrics["matched"])
    log.info("  committed            : %d  (%.0f%%)",
             metrics["committed"],
             100 * (1.0 - metrics["abstention_rate"]))
    log.info("  abstained            : %d  (%.0f%%)",
             metrics["abstained"], 100 * metrics["abstention_rate"])
    log.info("  abstained but GT in top-%d: %d  (recoverable if threshold lowered)",
             max_k, metrics["abstained_but_gt_in_topk"])

    log.info("===== abstention by class =====")
    for cls, info in sorted(metrics["abstention_by_class"].items(),
                            key=lambda kv: -kv[1]["rate"]):
        log.info("  %-20s %5.0f%%  abstained  (%d/%d)",
                 cls, 100 * info["rate"], info["abstained"], info["matched"])

    log.info("===== scale breakdown (committed only) =====")
    for scale_name, info in sorted(metrics["scale_breakdown"].items()):
        log.info("  scale=%-8s n=%5d  top1=%.3f",
                 scale_name, info["n"], info["top1_acc"])

    log.info("===== threshold sweep (matched detections) =====")
    log.info("  threshold   committed   abstention   top-1 | committed")
    for row in metrics["threshold_sweep"]:
        marker = "  ←" if abs(row["threshold"] - metrics["cls_min_confidence"]) < 1e-9 else ""
        log.info("  %.2f          %5d        %5.0f%%       %.3f%s",
                 row["threshold"], row["committed"],
                 100 * row["abstention"], row["top1_acc"], marker)

    log.info("===== where top-%d saves us (gt label in top-%d but top-1 wrong, "
             "committed only) =====", max_k, max_k)
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

    # ----------- side-by-side outputs: committed vs all -----------
    # For comparison, also compute a second metrics pass with NO confidence
    # filter and emit a parallel set of confusion matrices + timeseries.
    # The committed view is the "trustworthy answer"; the all view shows
    # what we'd get if we just used every prediction regardless of score.
    log.info("Computing 'all detections' metrics pass for side-by-side comparison")
    metrics_all = evaluate(gt_by_file, pred_by_file,
                           iou_thresh=args.iou, top_ks=top_ks,
                           cls_min_confidence=0.0)
    metrics_all["quality_filter"] = args.quality
    metrics_all["merge_applied"] = not args.no_merge
    metrics_all_path = out_dir / "metrics_all.json"
    with open(metrics_all_path, "w") as f:
        json.dump(metrics_all, f, indent=2)
    log.info("wrote %s", metrics_all_path)

    # Pred-by-file filtered to committed-only for the timeseries.
    pred_committed: dict[str, list] = defaultdict(list)
    for fname, preds in pred_by_file.items():
        for entry in preds:
            cls_score = entry[3]
            if cls_score >= args.cls_min_confidence:
                pred_committed[fname].append(entry)

    threshold_str = f"cls_score >= {args.cls_min_confidence:.2f}"
    write_confusion(metrics, out_dir,
                    suffix="_committed",
                    title_qualifier=f"committed only ({threshold_str})")
    write_confusion(metrics_all, out_dir,
                    suffix="_all",
                    title_qualifier="all detections (no confidence threshold)")

    write_species_timeseries(file_to_date, gt_by_file, pred_committed,
                             out_dir, suffix="_committed",
                             title_qualifier=(
                                 f"Predictions filtered to committed "
                                 f"({threshold_str}) — "
                                 f"{sum(len(v) for v in pred_committed.values())} "
                                 f"of {sum(len(v) for v in pred_by_file.values())} preds"))
    write_species_timeseries(file_to_date, gt_by_file, pred_by_file,
                             out_dir, suffix="_all",
                             title_qualifier=(
                                 "Predictions: all detections (no confidence threshold)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
