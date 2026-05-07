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
        # mode='w' so each eval run starts a fresh log instead of
        # accumulating across re-runs.
        handlers.append(logging.FileHandler(log_file, mode="w"))
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
    p.add_argument("--no-cross-scale-agree", action="store_true",
                   help="Don't require cross-scale agreement for committed "
                        "predictions. By default a detection only commits "
                        "when its three scales all picked the same top-1 "
                        "species — without this filter ~90%% of FPs would "
                        "slip through. Disable for diagnostic comparisons.")
    p.add_argument("--date-source", default="auto",
                   choices=["auto", "ocr", "coco"],
                   help="Where dates come from for the timeseries. "
                        "'ocr' (preferred) reads file_to_date.json (built "
                        "by scripts/ocr_burnin_dates.py) — accurate when "
                        "COCO date_captured is wrong / EXIF was stripped. "
                        "'coco' uses the date_captured field. 'auto' "
                        "(default) prefers the OCR JSON if found next to "
                        "--pred, else falls back to COCO.")
    p.add_argument("--ocr-dates",
                   help="Explicit path to file_to_date.json. Overrides the "
                        "auto-discovery next to --pred.")
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
              ) -> tuple[dict[str, list], Counter, Counter, dict[str, str]]:
    """Load wytrap per-image JSONs.

    Returns:
        pred_by_file: {file_basename: [(xyxy, merged_top1, [merged_topk_labels],
                                         cls_score, scale, fine_label)]}
        quality_counts: Counter of quality field values across all preds.
        scale_counts:   Counter of scale field values across all preds.
        fname_to_image_path: {file_basename: absolute image path on disk}
    """
    pred_by_file: dict[str, list] = defaultdict(list)
    quality_counts: Counter[str] = Counter()
    scale_counts:   Counter[str] = Counter()
    fname_to_image_path: dict[str, str] = {}
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
        fname_to_image_path[fname] = rec.image_path
        for det in rec.detections:
            quality_counts[det.quality] += 1
            scale_counts[det.scale] += 1
            if quality_filter is not None and det.quality not in quality_filter:
                continue
            top1 = merges.get(det.label, det.label)
            topk = [merges.get(t.get("common", ""), t.get("common", ""))
                    for t in (det.topk or [])]
            pred_by_file[fname].append((det.box_xyxy, top1, topk,
                                        float(det.cls_score), det.scale,
                                        det.fine_label,
                                        bool(det.cross_scale_agree)))
    log.info("scanned %d pipeline JSONs, matched %d to GT filenames",
             n_records_seen, n_records_matched)
    return pred_by_file, quality_counts, scale_counts, fname_to_image_path


def evaluate(gt_by_file: dict[str, list],
             pred_by_file: dict[str, list],
             iou_thresh: float,
             top_ks: list[int],
             cls_min_confidence: float = 0.30,
             require_cross_scale_agree: bool = True) -> dict:
    """Evaluate predictions vs GT.

    Detection P/R is computed over all matched/unmatched boxes regardless of
    classification confidence. Classification metrics (top-1/3/5 accuracy,
    confusion matrix) are computed only on **committed** matches:

        cls_score >= cls_min_confidence
        AND (cross_scale_agree OR not require_cross_scale_agree)

    Cross-scale agreement is a strong noise filter — on the tiny eval, FPs
    agree across scales only ~10% of the time vs ~78% for true positives,
    so requiring agreement materially improves precision at moderate cost
    to committed recall.

    Each `pred_by_file` entry is a 7-tuple
    (xyxy_box, top1_label, topk_labels, cls_score, scale,
     fine_label, cross_scale_agree).
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
    matched_records: list[tuple[float, str, str, list, str, bool]] = []
    max_k = max(top_ks)

    for fname, gts in gt_by_file.items():
        preds = pred_by_file.get(fname, [])
        gt_used = [False] * len(gts)
        for pbox, plabel, ptopk, cls_score, scale, _fine, agree in preds:
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
                matched_records.append(
                    (cls_score, plabel, glabel, ptopk, scale, agree))
                committed_here = (
                    cls_score >= cls_min_confidence
                    and (agree or not require_cross_scale_agree)
                )
                if committed_here:
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

    # Threshold sweep — re-bin matched_records over a fixed grid. Honors
    # the same require_cross_scale_agree gate so the sweep reflects the
    # current commit policy, not a hypothetical agree-disabled one.
    sweep = []
    for thresh in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60):
        cm, cc = 0, 0
        for cls_score, plabel, glabel, ptopk, scale, agree in matched_records:
            if cls_score < thresh:
                continue
            if require_cross_scale_agree and not agree:
                continue
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
    metrics["require_cross_scale_agree"] = require_cross_scale_agree
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

    row_sum = mat.sum(axis=1, keepdims=True).astype(float)
    row_sum[row_sum == 0] = 1.0  # avoid /0 for any all-zero rows
    norm = mat / row_sum

    # Wider figure + small left margin keeps the title clear of the colorbar.
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * n + 3), max(7, 0.6 * n + 1)))
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
    title = "Confusion matrix"
    if title_qualifier:
        title += f" — {title_qualifier}"
    ax.set_title(title)
    fig.text(0.5, 0.01,
             "row-normalized; cell text = pct (raw count)",
             ha="center", fontsize=9, color="gray")
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


def select_representatives(pred_by_file: dict[str, list],
                           gt_by_file: dict[str, list],
                           file_to_date: dict[str, str]
                           ) -> dict[tuple[str, str], dict]:
    """For each (date, species) cell, pick something to thumbnail.

    Output schema per cell:
        {
            "pred_fname":     <fname of best pred image, or None>
            "pred_box":       [x1,y1,x2,y2] or None
            "pred_score":     float or None
            "pred_scale":     str or None
            "pred_fine":      str or None
            "gt_fname":       <fname of a GT image for this cell, or None>
            "gt_box":         [x1,y1,x2,y2] or None
            "thumb_fname":    fname of the JPEG to draw on (pref pred, else gt)
            "boxes_to_draw":  [(box, color_rgb), ...]  used by render_thumbnail
        }

    Pred selection: tight-scale wins preferred, then highest cls_score.
    GT selection: just the first GT annotation (no real ranking criterion).
    Boxes drawn:
        - red on the pred image if a pred exists
        - blue on the gt image if no pred exists (GT-only cell)
        - if pred image happens to be the same as a GT image, draw both
          (red pred + blue GT) so the user can eyeball the IoU
    """
    PRED_COLOR = (220, 30, 30)   # red
    GT_COLOR   = (50, 100, 220)  # blue

    # ---- pred reps ----
    by_cell_pred: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for fname, preds in pred_by_file.items():
        date = file_to_date.get(fname, "")
        if not date:
            continue
        for entry in preds:
            box, merged_label, _topk, cls_score, scale, fine_label = entry[:6]
            by_cell_pred[(date, merged_label)].append({
                "fname":      fname,
                "box":        box,
                "cls_score":  cls_score,
                "scale":      scale,
                "fine_label": fine_label,
            })
    pred_reps: dict[tuple[str, str], dict] = {}
    for cell, dets in by_cell_pred.items():
        tight = [d for d in dets if d["scale"] == "tight"]
        pool = tight if tight else dets
        pred_reps[cell] = max(pool, key=lambda d: d["cls_score"])

    # ---- GT reps (one per cell, by date+merged label) ----
    by_cell_gt: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for fname, gts in gt_by_file.items():
        date = file_to_date.get(fname, "")
        if not date:
            continue
        for box, label in gts:
            by_cell_gt[(date, label)].append({"fname": fname, "box": box})
    # Stable rep: just pick the first one (cells are sorted ish).
    gt_reps = {cell: items[0] for cell, items in by_cell_gt.items()}

    # ---- merge ----
    out: dict[tuple[str, str], dict] = {}
    for cell in set(pred_reps) | set(gt_reps):
        p = pred_reps.get(cell)
        g = gt_reps.get(cell)

        if p is not None:
            # Pred-cell: draw the pred box on the pred image. If the GT for
            # this cell happens to be on the same image, draw both.
            thumb_fname = p["fname"]
            boxes = [(p["box"], PRED_COLOR)]
            same_image_gt = next((it["box"] for it in by_cell_gt.get(cell, [])
                                  if it["fname"] == p["fname"]), None)
            if same_image_gt is not None:
                boxes.append((same_image_gt, GT_COLOR))
            out[cell] = {
                "pred_fname":    p["fname"],
                "pred_box":      p["box"],
                "pred_score":    p["cls_score"],
                "pred_scale":    p["scale"],
                "pred_fine":     p["fine_label"],
                "gt_fname":      g["fname"] if g else None,
                "gt_box":        same_image_gt,
                "thumb_fname":   thumb_fname,
                "boxes_to_draw": boxes,
            }
        else:
            # GT-only cell: draw a GT box on a GT image, in blue.
            thumb_fname = g["fname"]
            out[cell] = {
                "pred_fname":    None,
                "pred_box":      None,
                "pred_score":    None,
                "pred_scale":    None,
                "pred_fine":     None,
                "gt_fname":      g["fname"],
                "gt_box":        g["box"],
                "thumb_fname":   thumb_fname,
                "boxes_to_draw": [(g["box"], GT_COLOR)],
            }
    return out


def _safe(name: str) -> str:
    """Filename-safe slug for species/date strings."""
    import re as _re
    return _re.sub(r"[^A-Za-z0-9._-]", "_", name)


def render_thumbnail(image_path: str | Path,
                     boxes: list[tuple[list[int], tuple[int, int, int]]],
                     out_path: Path,
                     max_size: int = 1024,
                     box_width: int = 4) -> bool:
    """Open the image, draw all `boxes` (each as (xyxy, color_rgb)), downscale
    so the long edge is `max_size`, save as JPEG. Returns True on success,
    False on failure (missing/unreadable image, etc.).

    Used to render multiple-color overlays — e.g. red predicted + blue GT
    on the same thumbnail when a cell has both.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        log.warning("Pillow not available; cannot render thumbnails")
        return False

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        log.warning("could not open %s for thumbnail: %s", image_path, e)
        return False

    # Draw boxes at full image resolution so the line is crisp post-scale.
    draw = ImageDraw.Draw(img)
    for box, color in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

    W, H = img.size
    long_edge = max(W, H)
    if long_edge > max_size:
        scale = max_size / long_edge
        img = img.resize((int(W * scale), int(H * scale)),
                         Image.Resampling.LANCZOS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=85)
    return True


def write_interactive_timeseries(file_to_date: dict[str, str],
                                 gt_by_file: dict[str, list],
                                 pred_by_file: dict[str, list],
                                 fname_to_image_path: dict[str, str],
                                 out_dir: Path,
                                 cls_min_confidence: float,
                                 suffix: str = "_committed") -> None:
    """Generate an interactive HTML timeseries with hover-preview thumbnails.

    Each cell shows whether the predictions found that species on that day
    (within the supplied pred_by_file). On hover, a representative thumbnail
    (highest cls_score in that cell, tight-scale preferred) is loaded into a
    fixed preview pane. Thumbnails live in a sibling folder so the HTML is
    self-contained.
    """
    representative = select_representatives(pred_by_file, gt_by_file,
                                            file_to_date)

    # Compute the same date and species axes the static plot uses.
    dates, species, gt_pres, pred_pres = build_species_timeseries(
        file_to_date, gt_by_file, pred_by_file
    )
    if not dates or not species:
        log.warning("no data for interactive timeseries (suffix=%r)", suffix)
        return

    # Dense calendar (every day in the range, including gaps).
    from datetime import date as _date, timedelta
    try:
        d_min = min(_date.fromisoformat(d) for d in dates)
        d_max = max(_date.fromisoformat(d) for d in dates)
    except ValueError:
        log.warning("non-ISO date_captured values; skipping interactive HTML")
        return
    n_days = (d_max - d_min).days + 1
    full_dates = [(d_min + timedelta(days=i)).isoformat() for i in range(n_days)]

    # Wipe and recreate so we never serve stale thumbnails: the
    # representative for a (date, species) cell may have changed since the
    # last run (different filter, different best-scoring detection), but
    # the filename is keyed only on (date, species), so without wiping
    # we'd reuse an old box-on-image that no longer matches.
    thumb_dir = out_dir / f"timeseries_thumbs{suffix}"
    if thumb_dir.exists():
        import shutil
        shutil.rmtree(thumb_dir)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    # Render representative thumbnails. One per (date, species) with data.
    n_thumbs = n_pred = n_gt_only = n_both_boxes = 0
    cell_meta: dict[tuple[str, str], dict] = {}
    for (d, s), info in representative.items():
        thumb_fname = info["thumb_fname"]
        # Look up the absolute image path. fname_to_image_path comes from
        # the per-image JSONs (only files with at least one detection),
        # so for GT-only cells the JSON may not exist — fall back to
        # constructing the path from the input dir.
        image_path = fname_to_image_path.get(thumb_fname)
        if not image_path:
            # Fall back to looking next to a sibling pred image if any.
            # As a last-ditch, try common image-folder env. Worst case,
            # the thumbnail just doesn't render.
            example = next(iter(fname_to_image_path.values()), None)
            if example:
                from os.path import dirname, join
                image_path = join(dirname(example), thumb_fname)

        if not image_path:
            continue

        thumb_name = f"{_safe(d)}__{_safe(s)}.jpg"
        thumb_path = thumb_dir / thumb_name
        if not thumb_path.exists():
            ok = render_thumbnail(image_path, info["boxes_to_draw"], thumb_path)
            if not ok:
                continue
        n_thumbs += 1
        if info["pred_box"] is not None:
            n_pred += 1
            if info["gt_box"] is not None:
                n_both_boxes += 1
        else:
            n_gt_only += 1

        cell_meta[(d, s)] = {
            "thumb":      f"timeseries_thumbs{suffix}/{thumb_name}",
            "fname":      thumb_fname,
            "score":      info["pred_score"],
            "scale":      info["pred_scale"],
            "fine_label": info["pred_fine"],
            "kind":       "pred" if info["pred_box"] is not None else "gt",
            "has_gt_box": info["gt_box"] is not None,
        }
    log.info("rendered %d thumbnails into %s "
             "(%d pred + %d GT-only, %d showing both boxes)",
             n_thumbs, thumb_dir, n_pred, n_gt_only, n_both_boxes)

    # Build the SVG heatmap. Columns = days, rows = species. Each filled
    # cell carries its metadata as data-attributes for JS hover.
    cell_w = 8        # px per day
    cell_h = 22       # px per species row
    left_pad = 180    # space for species labels
    top_pad = 60      # space for date labels and title
    bottom_pad = 80
    n_sp = len(species)
    svg_w = left_pad + n_days * cell_w + 20
    svg_h = top_pad + n_sp * cell_h + bottom_pad

    # Heuristic step for date axis labels (~1 label per ~7 days, min 6).
    step = max(1, n_days // max(6, n_days // 7))

    svg_parts = [
        f'<svg viewBox="0 0 {svg_w} {svg_h}" '
        f'xmlns="http://www.w3.org/2000/svg" id="heatmap" '
        f'preserveAspectRatio="xMinYMin meet">',
        '<style>'
        '.cell{stroke:#eee;stroke-width:0.5;cursor:pointer}'
        '.cell.gt{fill:#1f77b4}'
        '.cell.pred{fill:#d62728}'
        '.cell.both{fill:#7e3a8a}'
        '.cell.empty{fill:white}'
        '.cell:hover{stroke:#000;stroke-width:1.5}'
        '.label{font-family:sans-serif;font-size:11px;fill:#333}'
        '.date{font-family:sans-serif;font-size:9px;fill:#666}'
        '</style>',
    ]

    # Species labels (y axis)
    for i, sp in enumerate(species):
        y = top_pad + i * cell_h + cell_h / 2 + 4
        svg_parts.append(
            f'<text class="label" x="{left_pad - 6}" y="{y}" '
            f'text-anchor="end">{sp}</text>'
        )

    # Date labels (x axis) — place rotated 45° at the bottom.
    for j in range(0, n_days, step):
        x = left_pad + j * cell_w + cell_w / 2
        y = top_pad + n_sp * cell_h + 14
        svg_parts.append(
            f'<text class="date" x="{x}" y="{y}" '
            f'transform="rotate(45 {x} {y})">{full_dates[j]}</text>'
        )

    # Cells
    for i, sp in enumerate(species):
        for j, d in enumerate(full_dates):
            x = left_pad + j * cell_w
            y = top_pad + i * cell_h
            has_gt   = bool(gt_pres.get((d, sp), 0))
            has_pred = bool(pred_pres.get((d, sp), 0))
            if has_gt and has_pred:
                klass = "cell both"
            elif has_pred:
                klass = "cell pred"
            elif has_gt:
                klass = "cell gt"
            else:
                klass = "cell empty"

            attrs = [f'class="{klass}"',
                     f'x="{x}"', f'y="{y}"',
                     f'width="{cell_w}"', f'height="{cell_h}"',
                     f'data-date="{d}"', f'data-species="{sp}"',
                     f'data-gt="{int(gt_pres.get((d, sp), 0))}"',
                     f'data-pred="{int(pred_pres.get((d, sp), 0))}"']
            meta = cell_meta.get((d, sp))
            if meta:
                attrs += [
                    f'data-thumb="{meta["thumb"]}"',
                    f'data-fname="{meta["fname"]}"',
                    f'data-kind="{meta["kind"]}"',
                    f'data-has-gt-box="{int(meta["has_gt_box"])}"',
                ]
                # Pred-side attrs only meaningful when kind == 'pred'.
                if meta["kind"] == "pred":
                    attrs += [
                        f'data-score="{meta["score"]:.2f}"',
                        f'data-scale="{meta["scale"]}"',
                        f'data-fine="{meta["fine_label"]}"',
                    ]
            svg_parts.append('<rect ' + ' '.join(attrs) + ' />')

    # Legend
    legend_y = svg_h - 25
    svg_parts.append(
        f'<rect class="cell gt" x="{left_pad}" y="{legend_y}" '
        f'width="{cell_w}" height="{cell_h - 6}" />'
        f'<text class="label" x="{left_pad + cell_w + 6}" '
        f'y="{legend_y + cell_h - 11}">GT only</text>'
        f'<rect class="cell pred" x="{left_pad + 100}" y="{legend_y}" '
        f'width="{cell_w}" height="{cell_h - 6}" />'
        f'<text class="label" x="{left_pad + 100 + cell_w + 6}" '
        f'y="{legend_y + cell_h - 11}">Predicted only</text>'
        f'<rect class="cell both" x="{left_pad + 220}" y="{legend_y}" '
        f'width="{cell_w}" height="{cell_h - 6}" />'
        f'<text class="label" x="{left_pad + 220 + cell_w + 6}" '
        f'y="{legend_y + cell_h - 11}">GT and predicted</text>'
    )

    svg_parts.append('</svg>')
    svg_str = "\n".join(svg_parts)

    if suffix == "_strict":
        view_label = (f"strict (cls_score >= {cls_min_confidence:.2f}, "
                      f"quality=ok, scale=tight, cross-scale agree)")
    else:
        view_label = (f"committed (cls_score >= {cls_min_confidence:.2f}, "
                      f"quality=ok)")
    title = f"Wytrap species presence per day — {view_label}"
    html = f'''<!doctype html>
<html><head><meta charset="utf-8" />
<title>{title}</title>
<style>
body {{font-family:sans-serif;margin:20px;color:#222}}
h1 {{font-size:16px;margin:0 0 10px}}
.note {{font-size:11px;color:#666;margin-bottom:14px}}
.layout {{display:flex;gap:24px;align-items:flex-start}}
#heatmap-wrap {{flex:1 1 auto;overflow-x:auto;border:1px solid #ddd;
                background:#fff;padding:8px}}
#heatmap {{display:block;width:100%;height:auto;min-width:1000px}}
#preview {{flex:0 0 auto;width:560px;position:sticky;top:20px}}
#preview img {{width:100%;height:auto;border:1px solid #ccc;background:#f8f8f8}}
#caption {{font-size:12px;margin-top:8px;line-height:1.4}}
#caption .meta {{color:#666;font-size:11px;margin-top:2px}}
.placeholder {{color:#999;font-style:italic;font-size:12px;text-align:center;
               padding:60px 0;border:1px dashed #ccc}}
</style></head>
<body>
<h1>{title}</h1>
<div class="note">
Hover any cell to see a representative image with bounding box.
Cell color: <b style="color:#1f77b4">blue</b>=GT only ·
<b style="color:#d62728">red</b>=predicted only ·
<b style="color:#7e3a8a">purple</b>=both.
Empty cells have neither.
</div>

<div class="layout">
  <div id="heatmap-wrap">
    {svg_str}
  </div>
  <div id="preview">
    <div id="preview-img-wrap"><div class="placeholder">
      hover a cell with a prediction to see its image
    </div></div>
    <div id="caption"></div>
  </div>
</div>

<script>
const wrap = document.getElementById('preview-img-wrap');
const cap = document.getElementById('caption');
document.querySelectorAll('rect.cell').forEach(el => {{
  el.addEventListener('mouseenter', () => {{
    const date = el.getAttribute('data-date');
    const sp = el.getAttribute('data-species');
    const gt = el.getAttribute('data-gt');
    const pred = el.getAttribute('data-pred');
    const thumb = el.getAttribute('data-thumb');
    const fname = el.getAttribute('data-fname');
    const kind = el.getAttribute('data-kind');
    const hasGtBox = el.getAttribute('data-has-gt-box');
    if (thumb && kind === 'pred') {{
      const fine = el.getAttribute('data-fine');
      const score = el.getAttribute('data-score');
      const scale = el.getAttribute('data-scale');
      const colorNote = (hasGtBox === '1')
        ? '<span style="color:#d62728">red</span>=pred &middot; ' +
          '<span style="color:#1f77b4">blue</span>=GT'
        : '<span style="color:#d62728">red</span>=pred';
      wrap.innerHTML = `<img src="${{thumb}}" alt="${{fname}}" />`;
      cap.innerHTML =
        `<b>${{sp}}</b> on ${{date}}<br/>` +
        `<span class="meta">${{fname}} · pred=${{fine}} · ` +
        `score=${{score}} · scale=${{scale}}<br/>` +
        `${{colorNote}} · gt_count=${{gt}} pred_count=${{pred}}</span>`;
    }} else if (thumb && kind === 'gt') {{
      // GT-only cell: blue box on the GT-annotated image. The model didn't
      // commit a prediction here.
      wrap.innerHTML = `<img src="${{thumb}}" alt="${{fname}}" />`;
      cap.innerHTML =
        `<b>${{sp}}</b> on ${{date}}<br/>` +
        `<span class="meta">${{fname}} · ` +
        `<span style="color:#1f77b4">blue</span>=GT (no committed prediction)<br/>` +
        `gt_count=${{gt}} pred_count=${{pred}}</span>`;
    }} else if (pred !== '0') {{
      wrap.innerHTML = '<div class="placeholder">no thumbnail available ' +
                       '(check eval.log for failures)</div>';
      cap.innerHTML =
        `<b>${{sp}}</b> on ${{date}} · gt=${{gt}} pred=${{pred}}`;
    }} else {{
      wrap.innerHTML = '<div class="placeholder">no detections this day</div>';
      cap.innerHTML = `<b>${{sp}}</b> on ${{date}} · empty`;
    }}
  }});
}});
</script>
</body></html>
'''

    html_path = out_dir / f"species_timeseries{suffix}.html"
    with open(html_path, "w") as f:
        f.write(html)
    log.info("wrote %s", html_path)


def image_level_eval(gt_by_file: dict[str, list],
                     pred_by_file: dict[str, list],
                     cls_min_confidence: float,
                     require_cross_scale_agree: bool) -> dict:
    """Image-level "has-animal" eval, the metric that matters for triage.

    At the box level, one image with 5 background FPs counts as 5 FPs. For
    the use case of "filter out images that contain nothing," what we care
    about is per-image: does wytrap correctly say this image has/doesn't
    have an animal?

    Bucket each image by GT presence + committed-prediction presence:
        TP: GT > 0  AND  committed_pred > 0   (correctly flagged as non-empty)
        FN: GT > 0  AND  committed_pred == 0  (missed animal — bad for triage)
        FP: GT == 0 AND  committed_pred > 0   (false alarm on empty image)
        TN: GT == 0 AND  committed_pred == 0  (correctly empty)

    Returns a metrics dict with TP/FP/FN/TN, precision, recall, F1,
    accuracy, and the empty-image precision/recall as separate stats.
    """
    tp = fp = fn = tn = 0
    # Walk every image we know about (union of GT and pred filenames).
    all_files = set(gt_by_file) | set(pred_by_file)
    for fname in all_files:
        n_gt = len(gt_by_file.get(fname, []))
        # Count committed-quality predictions for this image.
        n_committed = 0
        for entry in pred_by_file.get(fname, []):
            cls_score = entry[3]
            agree = entry[6] if len(entry) > 6 else True
            if cls_score < cls_min_confidence:
                continue
            if require_cross_scale_agree and not agree:
                continue
            n_committed += 1
        if n_gt > 0 and n_committed > 0:
            tp += 1
        elif n_gt > 0 and n_committed == 0:
            fn += 1
        elif n_gt == 0 and n_committed > 0:
            fp += 1
        else:
            tn += 1

    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    # "Empty-image" pair: among images we labeled empty, how many really were?
    empty_precision = tn / (tn + fn) if (tn + fn) else 0.0
    empty_recall    = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_images": n,
        "has_animal_precision": precision,
        "has_animal_recall":    recall,
        "has_animal_f1":        f1,
        "accuracy":             (tp + tn) / n if n else 0.0,
        "empty_image_precision": empty_precision,
        "empty_image_recall":    empty_recall,
    }


def calibration_analysis(gt_by_file: dict[str, list],
                         pred_by_file: dict[str, list],
                         iou_thresh: float) -> dict:
    """Treat matched detections as a labeled mini-dataset and learn what
    inference-time parameter changes would improve precision/recall.

    Computes:
      - det_score distribution: matched-correct, matched-wrong, unmatched
      - cls_score distribution: matched-correct vs matched-wrong
      - scale distribution:     matched vs unmatched
      - quality distribution:   matched vs unmatched (proxy via prefilter)
      - cross_scale_agree:      effect on accuracy

    Then suggests per-knob recommendations (det_threshold, cls_min_confidence,
    whether to drop a scale, whether to require cross_scale_agree).
    """
    # Bucket every prediction.
    matched_correct: list[dict] = []
    matched_wrong:   list[dict] = []
    unmatched:       list[dict] = []

    for fname, gts in gt_by_file.items():
        preds = pred_by_file.get(fname, [])
        gt_used = [False] * len(gts)
        # We don't have det_score in the pred tuple — but cls_score is enough
        # for the BioCLIP side; det_score we can recover from the original
        # JSON via a second pass if needed. For now, focus on cls_score.
        for entry in preds:
            box, plabel, _topk, cls_score, scale, _fine = entry[:6]
            agree = entry[6] if len(entry) > 6 else True
            best_iou, best_j = 0.0, -1
            for j, (gbox, _) in enumerate(gts):
                if gt_used[j]:
                    continue
                v = iou(box, gbox)
                if v > best_iou:
                    best_iou, best_j = v, j
            row = {"cls_score": cls_score, "scale": scale,
                   "agree": agree, "plabel": plabel}
            if best_iou >= iou_thresh:
                gt_used[best_j] = True
                glabel = gts[best_j][1]
                if plabel == glabel:
                    matched_correct.append(row)
                else:
                    matched_wrong.append(row)
            else:
                unmatched.append(row)

    n_mc, n_mw, n_um = len(matched_correct), len(matched_wrong), len(unmatched)
    log.info("===== calibration: matched-correct / matched-wrong / unmatched =====")
    log.info("  bucket sizes: %d / %d / %d", n_mc, n_mw, n_um)

    def pct(rows, pred):
        if not rows: return 0.0
        return sum(1 for r in rows if pred(r)) / len(rows)

    # cls_score thresholds: at each cutoff, what's the fraction of each
    # bucket that passes? Recall over matched-correct vs FP rate over wrong+unmatched.
    log.info("  cls_score gates (fraction passing at each cutoff):")
    log.info(f"    {'cut':>4} {'mc':>6} {'mw':>6} {'um':>6}")
    for cut in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        f_mc = pct(matched_correct, lambda r: r["cls_score"] >= cut)
        f_mw = pct(matched_wrong,   lambda r: r["cls_score"] >= cut)
        f_um = pct(unmatched,       lambda r: r["cls_score"] >= cut)
        log.info(f"    {cut:>4.2f} {f_mc:>6.1%} {f_mw:>6.1%} {f_um:>6.1%}")

    # Scale distribution
    log.info("  scale distribution by bucket (fraction of bucket):")
    log.info(f"    {'scale':<8} {'mc':>6} {'mw':>6} {'um':>6}")
    for s in ("tight", "padded", "full"):
        f_mc = pct(matched_correct, lambda r, s=s: r["scale"] == s)
        f_mw = pct(matched_wrong,   lambda r, s=s: r["scale"] == s)
        f_um = pct(unmatched,       lambda r, s=s: r["scale"] == s)
        log.info(f"    {s:<8} {f_mc:>6.1%} {f_mw:>6.1%} {f_um:>6.1%}")

    # cross-scale agreement
    f_mc_agree = pct(matched_correct, lambda r: r["agree"])
    f_mw_agree = pct(matched_wrong,   lambda r: r["agree"])
    f_um_agree = pct(unmatched,       lambda r: r["agree"])
    log.info("  cross_scale_agree: mc=%.1f%% mw=%.1f%% um=%.1f%%",
             100 * f_mc_agree, 100 * f_mw_agree, 100 * f_um_agree)

    # Recommendations: pick the cls_score where matched-correct retention is
    # >= 0.90 but unmatched retention is minimised. Heuristic.
    best_cut = 0.30
    best_score = -1.0
    for cut in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        f_mc = pct(matched_correct, lambda r, c=cut: r["cls_score"] >= c)
        f_um = pct(unmatched,       lambda r, c=cut: r["cls_score"] >= c)
        # Maximise matched recall - unmatched FP rate, with a floor of 90%
        # matched recall (don't sacrifice too much).
        if f_mc < 0.90:
            continue
        s = f_mc - f_um
        if s > best_score:
            best_score, best_cut = s, cut

    if f_mc_agree > f_mw_agree + 0.10 and f_mc_agree > f_um_agree + 0.10:
        agree_recommendation = "yes — matched-correct agrees noticeably more often than wrong/unmatched"
    else:
        agree_recommendation = "no clear benefit — leave optional"

    # Find the worst-performing scale (fraction of bucket that's unmatched).
    scale_um = {
        s: sum(1 for r in unmatched if r["scale"] == s) / max(n_um, 1)
        for s in ("tight", "padded", "full")
    }
    worst_scale = max(scale_um, key=lambda k: scale_um[k])

    log.info("===== calibration recommendations =====")
    log.info("  cls_min_confidence: try %.2f  (keeps >=90%% mc, drops most um)", best_cut)
    log.info("  cross_scale_agree filter: %s", agree_recommendation)
    log.info("  worst scale by FP share: %s (%.1f%% of unmatched)",
             worst_scale, 100 * scale_um[worst_scale])

    return {
        "bucket_sizes": {"matched_correct": n_mc,
                         "matched_wrong":   n_mw,
                         "unmatched":       n_um},
        "recommendation": {
            "cls_min_confidence":      best_cut,
            "cross_scale_agree_helps": agree_recommendation,
            "worst_scale_by_fp":       worst_scale,
        },
        "scale_distribution": {
            s: {
                "matched_correct": sum(1 for r in matched_correct if r["scale"] == s),
                "matched_wrong":   sum(1 for r in matched_wrong if r["scale"] == s),
                "unmatched":       sum(1 for r in unmatched if r["scale"] == s),
            } for s in ("tight", "padded", "full")
        },
        "cross_scale_agree_rate": {
            "matched_correct": f_mc_agree,
            "matched_wrong":   f_mw_agree,
            "unmatched":       f_um_agree,
        },
    }


def resolve_dates(coco_dates: dict[str, str],
                  pred_dir: Path,
                  source: str = "auto",
                  ocr_path: str | None = None) -> dict[str, str]:
    """Decide whether to use OCR-derived dates or COCO date_captured.

    `coco_dates` is the {fname: date} map built from COCO. OCR dates live in
    `file_to_date.json` produced by scripts/ocr_burnin_dates.py (parses the
    burned-in timestamp from the JPEG's top/bottom strips).

    Modes:
        'ocr':  require the OCR JSON, error otherwise.
        'coco': always use COCO dates.
        'auto': prefer the OCR JSON if found, else COCO.
    """
    if source == "coco":
        log.info("date source: COCO date_captured (per --date-source coco)")
        return coco_dates

    candidate_paths = []
    if ocr_path:
        candidate_paths.append(Path(ocr_path))
    candidate_paths += [
        pred_dir / "file_to_date.json",
        pred_dir.parent / "file_to_date.json",
    ]
    found = next((p for p in candidate_paths if p.exists()), None)

    if source == "ocr" and not found:
        raise FileNotFoundError(
            f"--date-source ocr but no file_to_date.json found at "
            f"{[str(p) for p in candidate_paths]}. Build one with "
            f"scripts/ocr_burnin_dates.py.")
    if not found:
        log.info("date source: COCO date_captured (no OCR JSON found)")
        return coco_dates

    with open(found) as f:
        ocr_dates = json.load(f)
    log.info("date source: OCR (%s, %d entries)", found, len(ocr_dates))

    # Combine: OCR dates take precedence; COCO fills any gaps.
    combined = dict(coco_dates)  # start with COCO as fallback
    n_overridden = n_added = 0
    for fname, date in ocr_dates.items():
        if not date:  # empty string means OCR didn't find a date
            continue
        if fname in combined:
            if combined[fname] != date:
                n_overridden += 1
        else:
            n_added += 1
        combined[fname] = date
    log.info("OCR dates: overrode %d COCO entries, added %d new entries",
             n_overridden, n_added)
    return combined


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
    file_to_date = resolve_dates(file_to_date, pred_dir,
                                 source=args.date_source,
                                 ocr_path=args.ocr_dates)
    log.info("loaded %d GT images, %d boxes",
             len(gt_by_file), sum(len(v) for v in gt_by_file.values()))

    gt_files = set(gt_by_file.keys())
    pred_by_file, quality_counts, scale_counts, fname_to_image_path = load_pred(
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
                       cls_min_confidence=args.cls_min_confidence,
                       require_cross_scale_agree=not args.no_cross_scale_agree)
    metrics["quality_counts"] = dict(quality_counts)
    metrics["scale_counts"]   = dict(scale_counts)
    metrics["quality_filter"] = args.quality
    metrics["merge_applied"] = not args.no_merge
    metrics["n_gt_images"] = len(gt_by_file)
    metrics["n_gt_boxes"] = sum(len(v) for v in gt_by_file.values())

    log.info("===== detection (per box, IoU=%.2f) =====", args.iou)
    log.info("  precision : %.3f", metrics["precision"])
    log.info("  recall    : %.3f", metrics["recall"])
    log.info("  tp/fp/fn  : %d / %d / %d",
             metrics["tp"], metrics["fp"], metrics["fn"])

    img_level = image_level_eval(
        gt_by_file, pred_by_file,
        cls_min_confidence=args.cls_min_confidence,
        require_cross_scale_agree=not args.no_cross_scale_agree,
    )
    metrics["image_level"] = img_level
    log.info("===== image-level 'has animal' (committed: cls>=%.2f%s) =====",
             args.cls_min_confidence,
             "" if args.no_cross_scale_agree else " + cross_scale_agree")
    log.info("  images           : %d", img_level["n_images"])
    log.info("  TP (correct hit) : %d", img_level["tp"])
    log.info("  FN (missed)      : %d", img_level["fn"])
    log.info("  FP (false alarm) : %d", img_level["fp"])
    log.info("  TN (empty,  ok)  : %d", img_level["tn"])
    log.info("  has-animal P/R/F : %.3f / %.3f / %.3f",
             img_level["has_animal_precision"],
             img_level["has_animal_recall"],
             img_level["has_animal_f1"])
    log.info("  accuracy         : %.3f", img_level["accuracy"])
    log.info("  empty-image P/R  : %.3f / %.3f   (when we say empty, "
             "is the image really empty? / how many true-empties did we catch?)",
             img_level["empty_image_precision"],
             img_level["empty_image_recall"])

    log.info("===== classification (committed only, n=%d / matched=%d) =====",
             metrics["committed"], metrics["matched"])
    for k in top_ks:
        log.info("  top-%d acc : %.3f", k, metrics[f"top{k}_acc"])
    max_k = max(top_ks)
    if 1 in top_ks and max_k != 1:
        log.info("  top-%d recovery (in top-%d but not top-1): %.3f",
                 max_k, max_k, metrics[f"top{max_k}_recovery"])

    agree_note = ("+ cross_scale_agree"
                  if metrics.get("require_cross_scale_agree", True)
                  else "no agree filter")
    log.info("===== abstention (matched detections, threshold=%.2f, %s) =====",
             metrics["cls_min_confidence"], agree_note)
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
                           cls_min_confidence=0.0,
                           require_cross_scale_agree=False)
    metrics_all["quality_filter"] = args.quality
    metrics_all["merge_applied"] = not args.no_merge
    metrics_all_path = out_dir / "metrics_all.json"
    with open(metrics_all_path, "w") as f:
        json.dump(metrics_all, f, indent=2)
    log.info("wrote %s", metrics_all_path)

    # Calibration: what do the matched detections tell us about the right
    # inference-time knobs? Writes calibration_analysis.json plus a log block.
    calibration = calibration_analysis(gt_by_file, pred_by_file,
                                       iou_thresh=args.iou)
    cal_path = out_dir / "calibration_analysis.json"
    with open(cal_path, "w") as f:
        json.dump(calibration, f, indent=2)
    log.info("wrote %s", cal_path)

    # Pred-by-file filtered to committed-only for the timeseries.
    # Committed = cls_score >= threshold AND cross_scale_agree (matches the
    # evaluate() definition). The agreement gate is the bigger filter — on
    # tiny eval, FPs agree only ~10% of the time vs ~78% for TPs.
    pred_committed: dict[str, list] = defaultdict(list)
    # "Strict" view for the interactive HTML: additionally requires
    # scale=='tight'. Sharpest filter we have — for visual confirmation only.
    pred_strict: dict[str, list] = defaultdict(list)
    require_agree = not args.no_cross_scale_agree
    n_dropped_score = n_dropped_disagree = n_dropped_nontight = 0
    for fname, preds in pred_by_file.items():
        for entry in preds:
            cls_score = entry[3]
            scale = entry[4]
            agree = entry[6] if len(entry) > 6 else True
            if cls_score < args.cls_min_confidence:
                n_dropped_score += 1
                continue
            if require_agree and not agree:
                n_dropped_disagree += 1
                continue
            pred_committed[fname].append(entry)
            if scale != "tight":
                n_dropped_nontight += 1
                continue
            pred_strict[fname].append(entry)
    log.info("committed timeseries: kept %d preds  "
             "(dropped %d below cls threshold, %d cross-scale-disagree)",
             sum(len(v) for v in pred_committed.values()),
             n_dropped_score, n_dropped_disagree)
    log.info("strict timeseries:    kept %d preds  "
             "(of those that survived committed: %d non-tight)",
             sum(len(v) for v in pred_strict.values()), n_dropped_nontight)

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

    # Interactive HTML version. Two flavors:
    #   _committed: ok-quality + cls_score >= threshold.
    #   _strict:    additionally requires scale=='tight' + cross_scale_agree.
    # Strict is the cleanest visual confirmation; committed is more permissive
    # so you can see what's lurking just below the strict bar.
    write_interactive_timeseries(file_to_date, gt_by_file, pred_committed,
                                 fname_to_image_path, out_dir,
                                 cls_min_confidence=args.cls_min_confidence,
                                 suffix="_committed")
    write_interactive_timeseries(file_to_date, gt_by_file, pred_strict,
                                 fname_to_image_path, out_dir,
                                 cls_min_confidence=args.cls_min_confidence,
                                 suffix="_strict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
