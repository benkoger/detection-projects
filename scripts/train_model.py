"""SLURM-friendly equivalent of example_notebooks/train_model-clean.ipynb.

Runs the same FRCNN training pipeline as the notebook (helpers cleanup ->
CocoDetection -> get_detection_model -> koger train() loop) but as a plain
Python script so it can be submitted via sbatch with a >8 h time limit.

Usage:
    python scripts/train_model.py [--num-epochs 30] [--readme "..."]
                                  [--research-project YNP-BisonGraze]
                                  [--run-id <name>]

All other settings (paths, augmentations, model hyperparameters) live in this
file in the same form as the notebook so it's easy to keep in sync. Edit
here, not via more flags.

Logs are written both to stdout (for SLURM .out files) and to
<run_folder>/train.log so they survive after the job ends.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Make `helpers/` importable when running from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv

from helpers.helpers import (
    assign_supercategories,
    merge_categories,
    rebalance_train_val,
    remove_images_by_pattern,
)
from koger_detection.obj_det.engine import (
    get_detection_model,
    train,
)
from koger_detection.utils.lr_scheduler import get_lr_scheduler


log = logging.getLogger("wytrap.train")


def setup_logging(log_file: Path | None = None,
                  level: int = logging.INFO) -> None:
    """Configure root logger to write to stdout (and optionally a file)."""
    fmt = "%(asctime)s [%(levelname)-7s] %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-project", default="YNP-BisonGraze",
                   help="Subfolder under $ROOT/annotations and $MODEL_PATH/runs.")
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--readme",
                   default="Increasing rpn proposals to 512. "
                           "Using full current augmentation regime.")
    p.add_argument("--run-id",
                   help="Optional run folder name (default: current timestamp). "
                        "In sbatch, set this to $SLURM_JOB_ID so eval can find it.")
    p.add_argument("--skip-clean", action="store_true",
                   help="Skip the helpers cleanup step. Use when 'clean/' "
                        "already exists.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def coco_stats(json_path: str) -> tuple[int, int, int]:
    with open(json_path) as f:
        coco = json.load(f)
    per_image_counts: dict[int, int] = {}
    per_image_labels: dict[int, set[int]] = {}
    for ann in coco["annotations"]:
        per_image_counts[ann["image_id"]] = per_image_counts.get(ann["image_id"], 0) + 1
        per_image_labels.setdefault(ann["image_id"], set()).add(ann["category_id"])
    n_images = len(coco["images"])
    max_anns = max(per_image_counts.values()) if per_image_counts else 0
    max_label = max((max(s) for s in per_image_labels.values()), default=0)
    return n_images, max_anns, max_label


def main() -> int:
    args = parse_args()

    load_dotenv()
    root = os.environ.get("ROOT")
    model_path = os.environ.get("MODEL_PATH")
    if not root or not model_path:
        # Logging not yet set up at this point — write directly to stderr.
        print("ERROR: ROOT and MODEL_PATH must be set in .env", file=sys.stderr)
        return 2

    # Decide run folder up front so we can put the log file inside it.
    run_name = args.run_id or datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    run_folder = Path(model_path) / "runs" / run_name
    run_folder.mkdir(parents=True, exist_ok=False)

    setup_logging(log_file=run_folder / "train.log",
                  level=getattr(logging, args.log_level))

    log.info("hostname           : %s", os.uname().nodename)
    log.info("SLURM_JOB_ID       : %s",
             os.environ.get("SLURM_JOB_ID", "<not slurm>"))
    log.info("CUDA available     : %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("GPU                : %s", torch.cuda.get_device_name(0))
    log.info("run folder         : %s", run_folder)

    research_project = args.research_project
    image_folder = os.path.join(root, "annotations", research_project, "images")
    train_json_path = os.path.join(root, "annotations", research_project, "train.json")
    val_json_path   = os.path.join(root, "annotations", research_project, "val.json")

    # ---------------- 1. Annotation cleanup -----------------
    clean_dir = os.path.join(root, "annotations", research_project, "clean")
    if args.skip_clean:
        train_json_path = os.path.join(clean_dir, "train.json")
        val_json_path   = os.path.join(clean_dir, "val.json")
        log.info("reusing cleaned splits at %s", clean_dir)
    else:
        log.info("annotation cleanup: merging categories, dropping YNP_XA, "
                 "rebalancing to 0.75/0.25")
        train_coco = assign_supercategories(merge_categories(train_json_path))
        val_coco   = assign_supercategories(merge_categories(val_json_path))
        train_coco = remove_images_by_pattern(train_coco)
        val_coco   = remove_images_by_pattern(val_coco)

        train_json_path = os.path.join(clean_dir, "train.json")
        val_json_path   = os.path.join(clean_dir, "val.json")
        rebalance_train_val(
            train_coco, val_coco,
            fraction_val=0.25,
            train_out=train_json_path,
            val_out=val_json_path,
        )

    # ---------------- 2. Stats / class count -----------------
    n_train, max_train_anns, train_max_label = coco_stats(train_json_path)
    n_val,   max_val_anns,   val_max_label   = coco_stats(val_json_path)
    log.info("train: %d images, max %d anns/image", n_train, max_train_anns)
    log.info("val  : %d images, max %d anns/image", n_val, max_val_anns)
    num_classes = max(train_max_label, val_max_label) + 1
    log.info("num_classes (incl. background): %d", num_classes)

    # ---------------- 3. Augmentations -----------------
    bbox_params = A.BboxParams(format="pascal_voc",
                               label_fields=["class_labels", "area"])
    train_aug = A.Compose([
        A.ToFloat(max_value=255),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.geometric.resize.RandomScale(0.2, interpolation=cv2.INTER_LINEAR, p=0.75),
        A.geometric.transforms.PadIfNeeded(min_height=1024, min_width=1024,
                                            border_mode=cv2.BORDER_CONSTANT,
                                            value=0, p=1.0),
        A.crops.transforms.RandomCrop(1024, 1024, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.1,
                                   brightness_by_max=True, p=0.75),
        A.Blur(p=0.1),
        ToTensorV2(),
    ], bbox_params=bbox_params)
    val_aug = A.Compose([
        A.ToFloat(max_value=255),
        ToTensorV2(),
    ], bbox_params=bbox_params)

    # ---------------- 4. Cfg -----------------
    cfg = {
        "model": {
            "model_type": "bbox_v2",
            "num_classes": num_classes,
            "trainable_backbone_layers": 5,
            "rpn_batch_size_per_image": 512,
            "rpn_pre_nms_top_n_train": 4000,
            "rpn_post_nms_top_n_train": 2000,
            "rpn_pre_nms_top_n_test": 4000,
            "rpn_post_nms_top_n_test": 2000,
            "box_detections_per_img": 700,
            "box_nms_thresh": 0.7,
            "box_batch_size_per_image": 512,
            "box_positive_fraction": 0.5,
            "fixed_size": [1024, 1024],
        },
        "training": {
            "image_folder": image_folder,
            "train_json_path": train_json_path,
            "val_json_path": val_json_path,
            "batch_size": 4,
            "num_workers": 4,
            "num_epochs": args.num_epochs,
            "run_folder": str(run_folder),
            "epochs_per_val": 1,
            "optimizer": {
                "name": "SGD",
                "lr": 0.005,
                "momentum": 0.9,
                "weight_decay": 0.0005,
            },
            "lr_scheduler": {
                "name": "ReduceOnPlateau",
                "patience": 4,
                "factor": 0.3,
            },
        },
        "train_aug": train_aug.to_dict(),
        "val_aug":   val_aug.to_dict(),
        "readme":    args.readme,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }

    with open(run_folder / "cfg.json", "w") as f:
        json.dump(cfg, f,
                  default=lambda o: f"<<non-serializable: {str(o)}>>",
                  indent=4)
    log.info("wrote cfg.json")

    # ---------------- 5. Model + optimizer -----------------
    log.info("building model")
    model = get_detection_model(**cfg["model"])
    params = [p for p in model.parameters() if p.requires_grad]

    cfg_t = cfg["training"]
    cfg_t["optimizer"].pop("name")
    optimizer = torch.optim.SGD(params, **cfg_t["optimizer"])
    lr_scheduler = get_lr_scheduler(optimizer, **cfg_t["lr_scheduler"])

    # ---------------- 6. Train -----------------
    log.info("starting training: %d epochs, batch_size=%d, lr=%g",
             cfg_t["num_epochs"], cfg_t["batch_size"],
             cfg_t["optimizer"]["lr"])
    train(cfg, model, optimizer, lr_scheduler, train_aug, val_aug)
    log.info("training done. checkpoints + final_model.pth in %s", run_folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
