"""Diagnose why the wytrap pipeline produces too many or too few detections.

Walks the per-image JSON outputs and reports distributions that surface the
three usual suspects:

  1) SAHI tile duplication       -> high inter-detection IoU within an image
  2) det_threshold too permissive -> FPs cluster at low det_score
  3) Model over-firing on texture -> uniformly distributed det_scores;
                                      median detections-per-image is high
                                      even on images that shouldn't have any

Optional: pass --gt /path/to/coco.json to also report matched vs unmatched
distributions. Useful for comparing two runs (old NMS 0.5 vs new NMS 0.3,
single-scale vs multi-scale, etc.).

Usage:
    python scripts/diagnose_detections.py \\
        --pred /path/to/output-pipeline \\
        [--gt   /path/to/combined.json] \\
        [--out  /path/to/output-pipeline/eval/diagnose.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in (REPO_ROOT, REPO_ROOT / "wytrap"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

from wytrap.io import load_record  # noqa: E402

log = logging.getLogger("wytrap.diagnose")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--pred", required=True,
                   help="Folder of wytrap per-image JSONs.")
    p.add_argument("--gt",
                   help="Optional COCO json with ground-truth boxes — enables "
                        "matched/unmatched diagnostics.")
    p.add_argument("--out", help="Path for diagnose.json. Default: "
                                  "<pred>/eval/diagnose.json.")
    p.add_argument("--iou-thresh", type=float, default=0.5,
                   help="IoU threshold for matching to GT (default 0.5).")
    return p.parse_args()


def setup_logging(out_path: Path) -> None:
    fmt = "%(asctime)s [%(levelname)-7s] %(name)s | %(message)s"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(out_path)]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers,
                        force=True)


def quantiles(values: list[float], qs=(0.5, 0.9, 0.99, 1.0)) -> dict:
    if not values:
        return {q: 0 for q in qs}
    s = sorted(values)
    n = len(s)
    return {q: s[min(n - 1, int(round(q * (n - 1))))] for q in qs}


def histogram(values: list[float], bins: list[float]) -> list[int]:
    """Returns a count for each bin where bin[i] <= v < bin[i+1].
    Final bin captures values >= bins[-1]."""
    counts = [0] * (len(bins))
    for v in values:
        placed = False
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                placed = True
                break
        if not placed and v >= bins[-1]:
            counts[-1] += 1
    return counts


def fmt_hist(label: str, values: list[float], bins: list[float]) -> str:
    counts = histogram(values, bins)
    n = sum(counts) or 1
    width = 40
    out = [f"  {label}  (n={sum(counts)})"]
    for i, c in enumerate(counts):
        if i < len(bins) - 1:
            edge = f"[{bins[i]:.2f},{bins[i + 1]:.2f})"
        else:
            edge = f">={bins[-1]:.2f}    "
        bar = "#" * int(width * c / n)
        out.append(f"    {edge:<14}  {c:>7}  {bar}")
    return "\n".join(out)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    args = parse_args()

    pred_dir = Path(args.pred)
    out_path = Path(args.out) if args.out else (pred_dir / "eval" / "diagnose.json")
    setup_logging(out_path.with_suffix(".log"))

    log.info("scanning %s for per-image JSONs", pred_dir)
    json_files = [p for p in pred_dir.rglob("*.json")
                  if p.parent.name != "eval" and not p.name.startswith(".")]
    log.info("found %d JSON files", len(json_files))

    # Per-image aggregates
    per_image_total: list[int] = []
    per_image_ok:    list[int] = []
    per_image_other: list[int] = []
    quality_counts: Counter[str] = Counter()
    scale_counts:   Counter[str] = Counter()
    label_counts:   Counter[str] = Counter()
    det_scores: list[float] = []
    cls_scores: list[float] = []
    det_scores_by_quality: dict[str, list[float]] = defaultdict(list)
    det_scores_by_scale:   dict[str, list[float]] = defaultdict(list)
    cls_scores_by_scale:   dict[str, list[float]] = defaultdict(list)
    pair_ious: list[float] = []
    n_high_iou_pairs = 0
    n_total_pairs = 0
    cross_scale_agree_counts = Counter()
    n_records_with_error = 0

    fname_to_image_path: dict[str, str] = {}
    for jf in json_files:
        try:
            rec = load_record(jf)
        except Exception as e:
            log.warning("could not read %s: %s", jf, e)
            continue
        if rec.error:
            n_records_with_error += 1
            continue
        fname = Path(rec.image_path).name
        fname_to_image_path[fname] = rec.image_path
        n_total = len(rec.detections)
        n_ok = sum(1 for d in rec.detections if d.quality == "ok")
        per_image_total.append(n_total)
        per_image_ok.append(n_ok)
        per_image_other.append(n_total - n_ok)

        for d in rec.detections:
            quality_counts[d.quality] += 1
            scale_counts[d.scale] += 1
            label_counts[d.label] += 1
            det_scores.append(d.det_score)
            cls_scores.append(d.cls_score)
            det_scores_by_quality[d.quality].append(d.det_score)
            det_scores_by_scale[d.scale].append(d.det_score)
            cls_scores_by_scale[d.scale].append(d.cls_score)
            cross_scale_agree_counts[bool(d.cross_scale_agree)] += 1

        # Inter-detection IoU within this image (ok-quality only).
        ok_dets = [d for d in rec.detections if d.quality == "ok"]
        for i in range(len(ok_dets)):
            for j in range(i + 1, len(ok_dets)):
                v = iou(ok_dets[i].box_xyxy, ok_dets[j].box_xyxy)
                pair_ious.append(v)
                n_total_pairs += 1
                if v > 0.3:
                    n_high_iou_pairs += 1

    # ----- report -----
    log.info("=" * 70)
    log.info("DETECTIONS-PER-IMAGE")
    log.info("=" * 70)
    qs = quantiles(per_image_total, (0.5, 0.9, 0.99, 1.0))
    log.info("  total      : median=%d  p90=%d  p99=%d  max=%d",
             qs[0.5], qs[0.9], qs[0.99], qs[1.0])
    qs = quantiles(per_image_ok, (0.5, 0.9, 0.99, 1.0))
    log.info("  ok-quality : median=%d  p90=%d  p99=%d  max=%d",
             qs[0.5], qs[0.9], qs[0.99], qs[1.0])
    qs = quantiles(per_image_other, (0.5, 0.9, 0.99, 1.0))
    log.info("  filtered   : median=%d  p90=%d  p99=%d  max=%d",
             qs[0.5], qs[0.9], qs[0.99], qs[1.0])

    log.info("=" * 70)
    log.info("QUALITY / SCALE DISTRIBUTIONS")
    log.info("=" * 70)
    n_total_dets = sum(quality_counts.values())
    log.info("  quality counts (of %d total detections):", n_total_dets)
    for q, c in quality_counts.most_common():
        log.info("    %-10s %7d  (%.1f%%)", q, c, 100 * c / max(n_total_dets, 1))
    log.info("  scale counts:")
    for s, c in scale_counts.most_common():
        log.info("    %-10s %7d  (%.1f%%)", s, c, 100 * c / max(n_total_dets, 1))
    log.info("  cross_scale_agree counts:")
    for k in (True, False):
        c = cross_scale_agree_counts[k]
        log.info("    %-10s %7d  (%.1f%%)",
                 str(k), c, 100 * c / max(n_total_dets, 1))

    log.info("=" * 70)
    log.info("DET_SCORE DISTRIBUTION")
    log.info("=" * 70)
    bins = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    log.info("\n%s", fmt_hist("det_score (all)", det_scores, bins))
    for q in ("ok", "small", "edge", "thin"):
        if q in det_scores_by_quality:
            log.info("\n%s", fmt_hist(f"det_score (quality={q})",
                                       det_scores_by_quality[q], bins))

    log.info("=" * 70)
    log.info("CLS_SCORE DISTRIBUTION (by winning scale)")
    log.info("=" * 70)
    bins = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    log.info("\n%s", fmt_hist("cls_score (all)", cls_scores, bins))
    for s in ("tight", "padded", "full"):
        if s in cls_scores_by_scale:
            log.info("\n%s", fmt_hist(f"cls_score (scale={s})",
                                       cls_scores_by_scale[s], bins))

    log.info("=" * 70)
    log.info("INTER-DETECTION IoU (within image, ok-quality only)")
    log.info("=" * 70)
    if n_total_pairs:
        log.info("  total pairs               : %d", n_total_pairs)
        log.info("  pairs with IoU > 0.3      : %d  (%.1f%%)",
                 n_high_iou_pairs,
                 100 * n_high_iou_pairs / n_total_pairs)
        log.info("  pairs with IoU > 0.5      : %d  (%.1f%%)",
                 sum(1 for v in pair_ious if v > 0.5),
                 100 * sum(1 for v in pair_ious if v > 0.5) / n_total_pairs)
        log.info("  pairs with IoU > 0.7      : %d  (%.1f%%)",
                 sum(1 for v in pair_ious if v > 0.7),
                 100 * sum(1 for v in pair_ious if v > 0.7) / n_total_pairs)
        log.info("\n%s", fmt_hist("pairwise IoU", pair_ious,
                                   [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]))
    else:
        log.info("  (no images had >1 ok-quality detection — no pairs)")

    log.info("=" * 70)
    log.info("LABEL DISTRIBUTION (top 20)")
    log.info("=" * 70)
    for lbl, c in label_counts.most_common(20):
        log.info("  %-30s %7d  (%.1f%%)",
                 lbl, c, 100 * c / max(n_total_dets, 1))

    summary = {
        "n_jsons": len(json_files),
        "n_with_error": n_records_with_error,
        "per_image_total": quantiles(per_image_total),
        "per_image_ok":    quantiles(per_image_ok),
        "quality_counts": dict(quality_counts),
        "scale_counts":   dict(scale_counts),
        "cross_scale_agree": {str(k): v for k, v in cross_scale_agree_counts.items()},
        "det_score_quantiles": quantiles(det_scores, (0.1, 0.5, 0.9)),
        "cls_score_quantiles": quantiles(cls_scores, (0.1, 0.5, 0.9)),
        "pair_iou_summary": {
            "total_pairs": n_total_pairs,
            "high_iou_pairs_gt_0.3": n_high_iou_pairs,
            "high_iou_pairs_gt_0.5": sum(1 for v in pair_ious if v > 0.5),
            "high_iou_pairs_gt_0.7": sum(1 for v in pair_ious if v > 0.7),
        },
        "label_top20": dict(label_counts.most_common(20)),
    }

    # Optional GT comparison.
    if args.gt:
        from collections import defaultdict as _dd
        log.info("=" * 70)
        log.info("MATCHED VS UNMATCHED (against GT)")
        log.info("=" * 70)
        with open(args.gt) as f:
            coco = json.load(f)
        cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        im_id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}
        gt_by_file = _dd(list)
        for ann in coco["annotations"]:
            cid = ann["category_id"]
            if cid not in cat_id_to_name:
                continue
            fname = im_id_to_file[ann["image_id"]]
            x, y, w, h = ann["bbox"]
            gt_by_file[fname].append([int(x), int(y), int(x + w), int(y + h)])

        # Bucket every prediction.
        matched_det_scores: list[float] = []
        unmatched_det_scores: list[float] = []
        matched_cls_scores: list[float] = []
        unmatched_cls_scores: list[float] = []
        matched_scale = Counter()
        unmatched_scale = Counter()
        n_imgs_no_gt = n_imgs_no_pred = 0
        for jf in json_files:
            try:
                rec = load_record(jf)
            except Exception:
                continue
            fname = Path(rec.image_path).name
            gts = gt_by_file.get(fname, [])
            ok_dets = [d for d in rec.detections if d.quality == "ok"]
            if not gts:
                n_imgs_no_gt += 1
                # All ok dets are unmatched
                for d in ok_dets:
                    unmatched_det_scores.append(d.det_score)
                    unmatched_cls_scores.append(d.cls_score)
                    unmatched_scale[d.scale] += 1
                continue
            if not ok_dets:
                n_imgs_no_pred += 1
                continue
            gt_used = [False] * len(gts)
            for d in ok_dets:
                best_iou, best_j = 0.0, -1
                for j, gbox in enumerate(gts):
                    if gt_used[j]:
                        continue
                    v = iou(d.box_xyxy, gbox)
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= args.iou_thresh:
                    gt_used[best_j] = True
                    matched_det_scores.append(d.det_score)
                    matched_cls_scores.append(d.cls_score)
                    matched_scale[d.scale] += 1
                else:
                    unmatched_det_scores.append(d.det_score)
                    unmatched_cls_scores.append(d.cls_score)
                    unmatched_scale[d.scale] += 1

        log.info("  images: %d total, %d with no GT, %d with no ok-pred",
                 len(json_files), n_imgs_no_gt, n_imgs_no_pred)
        log.info("  matched   ok dets: %d", len(matched_det_scores))
        log.info("  unmatched ok dets: %d", len(unmatched_det_scores))
        log.info("\n%s",
                 fmt_hist("det_score (matched)", matched_det_scores,
                          [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]))
        log.info("\n%s",
                 fmt_hist("det_score (unmatched)", unmatched_det_scores,
                          [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]))
        log.info("  scale (matched):   %s", dict(matched_scale))
        log.info("  scale (unmatched): %s", dict(unmatched_scale))

        summary["gt_diagnostic"] = {
            "matched": len(matched_det_scores),
            "unmatched": len(unmatched_det_scores),
            "matched_det_score_quantiles": quantiles(matched_det_scores,
                                                     (0.1, 0.5, 0.9)),
            "unmatched_det_score_quantiles": quantiles(unmatched_det_scores,
                                                       (0.1, 0.5, 0.9)),
            "matched_scale": dict(matched_scale),
            "unmatched_scale": dict(unmatched_scale),
        }

    # ----- recommendations -----
    log.info("=" * 70)
    log.info("HEURISTIC RECOMMENDATIONS")
    log.info("=" * 70)

    # If duplicates dominate, NMS is the lever.
    if n_total_pairs and n_high_iou_pairs / n_total_pairs > 0.05:
        log.info("  - >5%% of intra-image pairs overlap at IoU>0.3: tighten "
                 "Detector NMS (currently 0.3, try 0.2) or increase tile "
                 "overlap so the same animal lands centred in fewer tiles.")
    # If FPs cluster at low det_score, raise det_threshold.
    if "gt_diagnostic" in summary:
        m_q = summary["gt_diagnostic"]["matched_det_score_quantiles"]
        u_q = summary["gt_diagnostic"]["unmatched_det_score_quantiles"]
        if u_q.get(0.5, 0) < m_q.get(0.5, 0) - 0.10:
            log.info("  - unmatched det_score median (%.2f) is much lower "
                     "than matched (%.2f): raise det_threshold to ~%.2f to "
                     "kill FPs without losing TPs.",
                     u_q[0.5], m_q[0.5], (u_q[0.5] + m_q[0.5]) / 2)
    # If full-scale dominates, suggests BioCLIP fallback noise.
    full_share = scale_counts.get("full", 0) / max(n_total_dets, 1)
    if full_share > 0.40:
        log.info("  - %.0f%% of detections won at scale=full: suggests many "
                 "tight crops are uninformative noise. Tighter NMS / higher "
                 "det_threshold should reduce this.", 100 * full_share)
    # If cross_scale_agree is mostly False, BioCLIP isn't stable.
    agree_share = cross_scale_agree_counts[True] / max(n_total_dets, 1)
    if agree_share < 0.5:
        log.info("  - only %.0f%% of detections have cross_scale_agree=True: "
                 "BioCLIP disagrees with itself across scales. Consider "
                 "filtering downstream to scale=tight + agree=True.",
                 100 * agree_share)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("wrote %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
