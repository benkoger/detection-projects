"""Evaluate a trained FRCNN run on its validation split.

Loads cfg.json + final_model.pth from a run folder, runs inference on the
validation COCO dataset, and produces:

  - <run>/eval/metrics.json        : per-class precision/recall/F1, mAP@0.5,
                                      mAP@0.5:0.95, num_gt, num_pred
  - <run>/eval/confusion_matrix.csv : rows = GT class, cols = predicted class
                                      (last column = "missed", last row =
                                      "background -> predicted as X")
  - <run>/eval/confusion_matrix.png : heatmap rendering
  - <run>/eval/eval.log             : same logs as stdout

Usage:
    python scripts/eval_model.py --run-folder $MODEL_PATH/runs/<id>
                                 [--score-threshold 0.5]
                                 [--iou-threshold 0.5]
                                 [--weights final_model.pth]

In sbatch, $RUN_ID is exported by the train job's chained submission and
the script resolves --run-folder against $MODEL_PATH/runs/$RUN_ID.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv

from koger_detection.obj_det.engine import (
    collate_fn,
    get_detection_model,
    worker_init_fn,
)
from koger_detection.obj_det.mydatasets import CocoDetection


log = logging.getLogger("wytrap.eval")


def setup_logging(log_file: Path | None, level: int = logging.INFO) -> None:
    fmt = "%(asctime)s [%(levelname)-7s] %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-folder",
                   help="Path to the training run folder. Defaults to "
                        "$MODEL_PATH/runs/$RUN_ID if --run-folder is omitted "
                        "and $RUN_ID is set.")
    p.add_argument("--weights", default="final_model.pth",
                   help="Filename within run-folder to evaluate. Default: "
                        "final_model.pth. Use 'model-epoch-N.pth' to evaluate "
                        "an intermediate checkpoint.")
    p.add_argument("--score-threshold", type=float, default=0.5,
                   help="Confidence threshold for keeping predicted boxes.")
    p.add_argument("--iou-threshold", type=float, default=0.5,
                   help="IoU threshold for matching predictions to GT boxes.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def resolve_run_folder(args: argparse.Namespace) -> Path:
    if args.run_folder:
        return Path(args.run_folder)
    run_id = os.environ.get("RUN_ID")
    model_path = os.environ.get("MODEL_PATH")
    if run_id and model_path:
        return Path(model_path) / "runs" / run_id
    raise SystemExit("Must pass --run-folder or set RUN_ID + MODEL_PATH env vars")


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def evaluate_run(run_folder: Path, weights: str,
                 score_thr: float, iou_thr: float,
                 batch_size: int, num_workers: int) -> dict:
    cfg_path = run_folder / "cfg.json"
    weights_path = run_folder / weights
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    with open(cfg_path) as f:
        cfg = json.load(f)

    log.info("run folder    : %s", run_folder)
    log.info("weights       : %s", weights_path.name)
    log.info("score thresh  : %g", score_thr)
    log.info("iou thresh    : %g", iou_thr)

    # Build model + load weights.
    model = get_detection_model(**cfg["model"])
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device        : %s", device)
    model.to(device).eval()

    # Load val COCO -- we want both the GT and the category names.
    val_json = cfg["training"]["val_json_path"]
    image_folder = cfg["training"]["image_folder"]
    with open(val_json) as f:
        val_coco = json.load(f)
    id_to_name = {c["id"]: c["name"] for c in val_coco["categories"]}
    name_order = [id_to_name[i] for i in sorted(id_to_name)]
    log.info("num classes   : %d (%s)", len(name_order), ", ".join(name_order))

    val_aug = A.Compose([
        A.ToFloat(max_value=255),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc",
                                 label_fields=["class_labels", "area"]))
    dataset = CocoDetection(image_folder, val_json, transform=val_aug)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )
    log.info("val images    : %d", len(dataset))

    # Walk the dataset, collecting per-image preds + GT.
    BG = "background"
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # For mAP we accumulate (score, is_tp) per class across the whole set,
    # plus the total number of GT boxes per class. mAP@0.5 with the IoU
    # threshold provided.
    per_class_dets: dict[str, list[tuple[float, int]]] = defaultdict(list)
    per_class_gt: dict[str, int] = defaultdict(int)

    n_done = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for tgt, out in zip(targets, outputs):
                gt_boxes = tgt["boxes"].cpu().numpy()
                gt_labels = [id_to_name[int(l)] for l in tgt["labels"].cpu().numpy()]
                for lbl in gt_labels:
                    per_class_gt[lbl] += 1

                pred_boxes = out["boxes"].cpu().numpy()
                pred_scores = out["scores"].cpu().numpy()
                pred_labels = [id_to_name[int(l)] for l in out["labels"].cpu().numpy()]

                # Keep predictions sorted by score, descending.
                order = np.argsort(-pred_scores)
                pred_boxes = pred_boxes[order]
                pred_scores = pred_scores[order]
                pred_labels = [pred_labels[i] for i in order]

                gt_used = [False] * len(gt_boxes)

                for pbox, pscore, plabel in zip(pred_boxes, pred_scores, pred_labels):
                    # Confusion matrix uses score threshold.
                    counted_for_confusion = pscore >= score_thr

                    best_iou, best_j = 0.0, -1
                    for j, gbox in enumerate(gt_boxes):
                        if gt_used[j]:
                            continue
                        v = iou_xyxy(pbox, gbox)
                        if v > best_iou:
                            best_iou, best_j = v, j

                    is_tp = 0
                    if best_iou >= iou_thr and best_j >= 0:
                        # Match (correct or wrong class).
                        if counted_for_confusion:
                            confusion[gt_labels[best_j]][plabel] += 1
                        if plabel == gt_labels[best_j]:
                            gt_used[best_j] = True
                            is_tp = 1
                    else:
                        # No matching GT -> false positive against background.
                        if counted_for_confusion:
                            confusion[BG][plabel] += 1
                    per_class_dets[plabel].append((float(pscore), is_tp))

                # Any GT not matched at all -> missed.
                for j, used in enumerate(gt_used):
                    if not used:
                        confusion[gt_labels[j]][BG] += 1

            n_done += len(images)
            if n_done % 50 == 0:
                log.info("eval progress: %d/%d", n_done, len(dataset))

    # ---- Per-class precision / recall / F1 (at score_thr, iou_thr) ----
    per_class_metrics: dict[str, dict] = {}
    for cls in name_order:
        tp = sum(1 for s, t in per_class_dets.get(cls, []) if s >= score_thr and t == 1)
        fp = sum(1 for s, t in per_class_dets.get(cls, []) if s >= score_thr and t == 0)
        fn = per_class_gt.get(cls, 0) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class_metrics[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "num_gt": per_class_gt.get(cls, 0),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # ---- mAP@iou_thr (Pascal-VOC-style: AP = area under PR curve) ----
    aps: dict[str, float] = {}
    for cls in name_order:
        dets = sorted(per_class_dets.get(cls, []), key=lambda kv: -kv[0])
        n_gt = per_class_gt.get(cls, 0)
        if n_gt == 0 or not dets:
            aps[cls] = 0.0
            continue
        tps = np.array([t for _, t in dets], dtype=np.float64)
        fps = 1 - tps
        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(fps)
        recalls = cum_tp / n_gt
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        # 11-point interpolation
        ap = 0.0
        for r in np.linspace(0, 1, 11):
            mask = recalls >= r
            ap += (precisions[mask].max() if mask.any() else 0.0) / 11
        aps[cls] = float(ap)

    macro_p = np.mean([m["precision"] for m in per_class_metrics.values()])
    macro_r = np.mean([m["recall"] for m in per_class_metrics.values()])
    macro_f1 = np.mean([m["f1"] for m in per_class_metrics.values()])
    mAP = float(np.mean(list(aps.values())))

    log.info("=" * 60)
    log.info("EVAL @ score>=%g, IoU>=%g", score_thr, iou_thr)
    log.info("=" * 60)
    log.info("%-20s %5s %5s %5s %7s %7s %7s %6s",
             "class", "GT", "TP", "FP", "P", "R", "F1", "AP")
    for cls in name_order:
        m = per_class_metrics[cls]
        log.info("%-20s %5d %5d %5d %7.3f %7.3f %7.3f %6.3f",
                 cls, m["num_gt"], m["tp"], m["fp"],
                 m["precision"], m["recall"], m["f1"], aps[cls])
    log.info("-" * 60)
    log.info("%-20s %5s %5s %5s %7.3f %7.3f %7.3f %6.3f",
             "MACRO AVG", "", "", "", macro_p, macro_r, macro_f1, mAP)

    # ---- Persist ----
    out_dir = run_folder / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "score_threshold": score_thr,
        "iou_threshold": iou_thr,
        "weights": weights,
        "per_class": per_class_metrics,
        "ap_per_class": aps,
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
            "mAP": mAP,
        },
        "name_order": name_order,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)
    log.info("wrote %s", out_dir / "metrics.json")

    # Confusion matrix CSV: rows = GT (incl. background = "extra prediction"),
    # cols = predicted (incl. background = "missed"). Background row is FPs;
    # background col is FNs.
    rows = name_order + [BG]   # GT side
    cols = name_order + [BG]   # predicted side
    with open(out_dir / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt\\pred"] + cols)
        for r in rows:
            row_counts = [confusion.get(r, {}).get(c, 0) for c in cols]
            w.writerow([r] + row_counts)
    log.info("wrote %s", out_dir / "confusion_matrix.csv")

    # Optional heatmap PNG.
    try:
        import matplotlib.pyplot as plt
        mat = np.array([[confusion.get(r, {}).get(c, 0) for c in cols] for r in rows])
        fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.5),
                                         max(6, len(rows) * 0.5)))
        im = ax.imshow(mat, aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
        ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
        ax.set_title(f"Confusion (score>={score_thr}, IoU>={iou_thr})")
        for i in range(len(rows)):
            for j in range(len(cols)):
                if mat[i, j]:
                    ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                            fontsize=8, color="white" if mat[i, j] > mat.max() / 2 else "black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
        log.info("wrote %s", out_dir / "confusion_matrix.png")
    except Exception as e:
        log.warning("could not render confusion_matrix.png: %s", e)

    return metrics_payload


def main() -> int:
    args = parse_args()
    load_dotenv()
    run_folder = resolve_run_folder(args)
    setup_logging(log_file=run_folder / "eval" / "eval.log",
                  level=getattr(logging, args.log_level))
    log.info("hostname           : %s", os.uname().nodename)
    log.info("SLURM_JOB_ID       : %s",
             os.environ.get("SLURM_JOB_ID", "<not slurm>"))
    evaluate_run(run_folder=run_folder,
                 weights=args.weights,
                 score_thr=args.score_threshold,
                 iou_thr=args.iou_threshold,
                 batch_size=args.batch_size,
                 num_workers=args.num_workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
