"""SpeciesNet comparison arms that write wytrap-format records.

Two arms, both scored by scripts/eval_image_level.py exactly like a wytrap run:

  classifier   Same images, same MegaDetector boxes as an existing wytrap run
               (its all_records.jsonl), only the classifier swapped: each box
               is cropped the way SpeciesNet's always-crop model expects and
               classified by SpeciesNet. Isolates BioCLIP 2 vs SpeciesNet.

  ensemble     SpeciesNet as shipped: its own MegaDetector v5a, classifier,
               taxonomy roll-up and geofence (country/admin1). With
               --detections-from <wytrap all_records.jsonl> the detector step
               is replaced by the wytrap run's boxes (e.g. MegaDetector v6),
               so everything after detection is SpeciesNet's and everything
               up to it is shared with wytrap.

SpeciesNet labels are 'uuid;class;order;family;genus;species;common'. They are
mapped to the Idaho vocabulary by taxonomy (genus odocoileus -> deer, family
leporidae -> lagomorph, ...) so the eval's merges line up; anything unmapped
keeps SpeciesNet's common name. Roll-ups such as 'odocoileus species' or
'canine family' land on the right Idaho class when one exists.

Usage:
    python scripts/run_speciesnet_arms.py classifier \\
        --wytrap-records /path/output-wytrap/all_records.jsonl \\
        --output /path/output-speciesnet-clf [--model hf:...] [--batch-size 32]

    python scripts/run_speciesnet_arms.py ensemble \\
        --images /path/images --output /path/output-speciesnet-ens \\
        [--detections-from /path/output-wytrap/all_records.jsonl] \\
        [--country USA --admin1 ID] [--no-geofence]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in (REPO_ROOT, REPO_ROOT / "wytrap"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

from wytrap.io import DetectionRecord, ImageRecord, append_jsonl, save_record  # noqa: E402
from wytrap.run import IMAGE_EXTS, assess_box_quality  # noqa: E402

# Mirror used by AddaxAI Connect; avoids Kaggle credentials.
DEFAULT_MODEL = "hf:Addax-Data-Science/SPECIESNET-v4-0-2-A"

from helpers.helpers import taxonomy_to_idaho  # noqa: E402


def speciesnet_label_to_names(label: str) -> tuple[str, str]:
    """Return (idaho_or_common_name, scientific) for a SpeciesNet label string."""
    parts = label.split(";")
    if len(parts) != 7:
        return label, label
    _, cls, order, family, genus, species, common = parts
    sci = " ".join(p for p in (genus, species) if p) or common
    return taxonomy_to_idaho(cls, order, family, genus, species, common), sci


def sn_lineage(label: str) -> dict[str, str]:
    parts = label.split(";")
    if len(parts) != 7:
        return {}
    _, cls, order, family, genus, species, _ = parts
    return {"class": cls, "order": order, "family": family, "genus": genus, "species": species}


def _topk(classes: list[str], scores: list[float]) -> list[dict]:
    out = []
    for c, s in zip(classes, scores):
        name, sci = speciesnet_label_to_names(c)
        out.append({"common": name, "scientific": sci, "score": float(s),
                    "lineage": sn_lineage(c)})
    return out


def masked_classifications(cls: dict, k: int = 5) -> dict:
    """Softmax over SpeciesNet's target_logits (the shared candidate set) and
    return a classifications dict in SpeciesNet's own format, top-k only."""
    import numpy as np
    labels = cls.get("target_classes") or []
    logits = np.asarray(cls.get("target_logits") or [], dtype=np.float64)
    if not len(labels) or not len(logits):
        return cls
    p = np.exp(logits - logits.max())
    p /= p.sum()
    order = np.argsort(-p)[:k]
    return {"classes": [labels[i] for i in order], "scores": [float(p[i]) for i in order]}


def write_target_file(model: str, vocab_path: str, out_dir: Path) -> Path:
    """SpeciesNet target-species file = vocab.candidate_classes over its labels."""
    from speciesnet.utils import ModelInfo
    from helpers.vocab import Vocab
    labels = [l.strip() for l in open(ModelInfo(model).classifier_labels, encoding="utf-8") if l.strip()]
    cand = Vocab.load(vocab_path).candidate_classes({l: sn_lineage(l) for l in labels})
    path = out_dir / "speciesnet_targets.txt"
    path.write_text("\n".join(l for l in labels if l in cand) + "\n")
    _log(f"candidate set: {len(cand)} of {len(labels)} SpeciesNet labels -> {path}")
    return path


def _log(msg: str) -> None:
    print(f"[speciesnet {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# arm 1: SpeciesNet classifier on wytrap boxes
# ----------------------------------------------------------------------------

def run_classifier_arm(args) -> int:
    from speciesnet import BBox, SpeciesNetClassifier
    from speciesnet.utils import load_rgb_image

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "all_records.jsonl"
    if jsonl.exists():
        jsonl.unlink()

    records = [json.loads(l) for l in open(args.wytrap_records) if l.strip()]
    _log(f"{len(records)} wytrap records from {args.wytrap_records}")
    t0 = time.time()
    target = write_target_file(args.model, args.vocab, out_dir) if args.vocab else None
    clf = SpeciesNetClassifier(args.model, target_species_txt=str(target) if target else None)
    _log(f"classifier {clf.model_info.version} ({clf.model_info.type_}) loaded in "
         f"{time.time() - t0:.0f}s, {len(clf.labels)} labels")

    n_boxes = n_img = 0
    t0 = time.time()
    for rec in records:
        dets = rec.get("detections") or []
        image_path = rec["image_path"]
        new_dets: list[DetectionRecord] = []
        if dets:
            img = load_rgb_image(image_path)
            W, H = rec["image_size"]
            pre, keep = [], []
            for d in dets:
                x1, y1, x2, y2 = d["box_xyxy"]
                bb = BBox(xmin=x1 / W, ymin=y1 / H, width=(x2 - x1) / W, height=(y2 - y1) / H)
                p = clf.preprocess(img, bboxes=[bb]) if img is not None else None
                pre.append(p)
                keep.append(d)
            results = []
            for i in range(0, len(pre), args.batch_size):
                chunk = pre[i:i + args.batch_size]
                results.extend(clf.batch_predict([image_path] * len(chunk), chunk))
            for d, r in zip(keep, results):
                cls = r.get("classifications")
                if cls and target:
                    cls = masked_classifications(cls)
                if not cls:
                    topk = [{"common": "no cv result", "scientific": "", "score": 0.0}]
                else:
                    topk = _topk(cls["classes"], cls["scores"])
                top = topk[0]
                new_dets.append(DetectionRecord(
                    box_xyxy=d["box_xyxy"], det_score=d["det_score"], det_label=d["det_label"],
                    label=top["common"], fine_label=top["common"],
                    scientific_label=top["scientific"], cls_score=top["score"],
                    topk=topk[:args.topk], quality=d.get("quality", "ok"),
                    quality_reason=d.get("quality_reason", ""), scale="tight",
                    scale_scores={"tight": top["score"]}, cross_scale_agree=True,
                    lineage=top.get("lineage", {}),
                ))
                n_boxes += 1
        out_rec = ImageRecord(image_path=image_path, image_size=rec["image_size"],
                              detections=new_dets, error=rec.get("error"))
        rel = Path(image_path)
        out_path = out_dir / rel.parent.name / (rel.stem + ".json")
        save_record(out_rec, out_path)
        append_jsonl(out_rec, jsonl)
        n_img += 1
        if n_img % 50 == 0 or n_img == len(records):
            _log(f"{n_img}/{len(records)} images, {n_boxes} boxes, "
                 f"{n_img / (time.time() - t0):.1f} img/s")
    _write_manifest(out_dir, "classifier", args, n_img, n_boxes)
    return 0


# ----------------------------------------------------------------------------
# arm 2: full SpeciesNet ensemble
# ----------------------------------------------------------------------------

def run_ensemble_arm(args) -> int:
    from speciesnet import SpeciesNet
    from PIL import Image

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "all_records.jsonl"
    if jsonl.exists():
        jsonl.unlink()
    images_dir = Path(args.images)
    files = sorted(str(p) for p in images_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))
    _log(f"{len(files)} images under {images_dir}")
    if not files:
        raise SystemExit(f"no images found under {images_dir}")

    t0 = time.time()
    target = write_target_file(args.model, args.vocab, out_dir) if args.vocab else None
    model = SpeciesNet(args.model, components="all", geofence=not args.no_geofence,
                       target_species_txt=str(target) if target else None)
    _log(f"SpeciesNet loaded in {time.time() - t0:.0f}s (geofence={not args.no_geofence}, "
         f"candidate set={'vocab' if target else 'all labels'})")
    raw_json = out_dir / "speciesnet_predictions.json"
    # SpeciesNet treats an existing predictions_json as a resume checkpoint
    # and skips every image already in it, which silently reused stale
    # results when the same output dir was run with different boxes.
    if raw_json.exists():
        raw_json.unlink()
    t0 = time.time()
    if not args.detections_from and not target:
        preds = model.predict(filepaths=files, country=args.country, admin1_region=args.admin1,
                              batch_size=args.batch_size, progress_bars=False,
                              predictions_json=str(raw_json))
        if preds is None:
            preds = json.loads(raw_json.read_text())
    else:
        if args.detections_from:
            # Supplied detections (wytrap boxes) in SpeciesNet's detector output
            # format, highest conf first: the always-crop classifier crops to bboxes[0].
            dd: dict[str, dict] = {}
            for line in open(args.detections_from):
                if not line.strip():
                    continue
                r = json.loads(line)
                W, H = r["image_size"]
                dets = sorted(r.get("detections") or [], key=lambda d: -d["det_score"])
                dd[r["image_path"]] = {"filepath": r["image_path"], "detections": [
                    {"category": "1", "label": "animal", "conf": float(d["det_score"]),
                     "bbox": [d["box_xyxy"][0] / W, d["box_xyxy"][1] / H,
                              (d["box_xyxy"][2] - d["box_xyxy"][0]) / W,
                              (d["box_xyxy"][3] - d["box_xyxy"][1]) / H]}
                    for d in dets]}
            for f in files:
                dd.setdefault(f, {"filepath": f, "detections": []})
            _log(f"using {sum(len(v['detections']) for v in dd.values())} supplied boxes "
                 f"from {args.detections_from} instead of MDv5a")
        else:
            det = model.detect(filepaths=files, progress_bars=False)
            dd = {p["filepath"]: p for p in det["predictions"]}
            _log(f"SpeciesNet MDv5a: {sum(len(v.get('detections') or []) for v in dd.values())} boxes")
        cls = model.classify(filepaths=files, detections_dict=dd, country=args.country,
                             admin1_region=args.admin1, batch_size=args.batch_size,
                             progress_bars=False)
        cls_dict = {}
        for p in cls["predictions"]:
            if target and p.get("classifications"):
                p = {**p, "classifications": masked_classifications(p["classifications"])}
            cls_dict[p["filepath"]] = p
        preds = model.ensemble_from_past_runs(
            filepaths=files, classifications_dict=cls_dict, detections_dict=dd,
            country=args.country, admin1_region=args.admin1, progress_bars=False,
            predictions_json=str(raw_json))
        if preds is None:
            preds = json.loads(raw_json.read_text())
        for p in preds["predictions"]:
            p.setdefault("detections", dd[p["filepath"]].get("detections") or [])
    _log(f"predict done in {time.time() - t0:.0f}s, raw output at {raw_json}")

    n_boxes = 0
    rollups = {}
    for p in preds["predictions"]:
        fp = p["filepath"]
        with Image.open(fp) as im:
            W, H = im.size
        pred_label = p.get("prediction", "")
        name, sci = speciesnet_label_to_names(pred_label) if pred_label else ("", "")
        rollups[p.get("prediction_source", "?")] = rollups.get(p.get("prediction_source", "?"), 0) + 1
        cls = p.get("classifications") or {}
        topk = _topk(cls.get("classes", []), cls.get("scores", []))[:args.topk]
        dets: list[DetectionRecord] = []
        for d in p.get("detections") or []:
            if d.get("label") != "animal":
                continue
            x, y, w, h = d["bbox"]
            box = (int(round(x * W)), int(round(y * H)),
                   int(round((x + w) * W)), int(round((y + h) * H)))
            quality, reason = assess_box_quality(box, (W, H))
            dets.append(DetectionRecord(
                box_xyxy=list(box), det_score=float(d["conf"]), det_label="animal",
                label=name, fine_label=name, scientific_label=sci,
                cls_score=float(p.get("prediction_score", 0.0)), topk=topk,
                quality=quality, quality_reason=reason, scale="ensemble",
                scale_scores={"ensemble": float(p.get("prediction_score", 0.0))},
                cross_scale_agree=True, lineage=sn_lineage(pred_label) if pred_label else {},
            ))
            n_boxes += 1
        rec = ImageRecord(image_path=fp, image_size=[W, H], detections=dets,
                          error="; ".join(p["failures"]) if p.get("failures") else None)
        rel = Path(fp)
        save_record(rec, out_dir / rel.parent.name / (rel.stem + ".json"))
        append_jsonl(rec, jsonl)
    _log(f"converted {len(preds['predictions'])} predictions, {n_boxes} animal boxes; "
         f"prediction_source counts: {rollups}")
    _write_manifest(out_dir, "ensemble", args, len(preds["predictions"]), n_boxes,
                    extra={"prediction_source_counts": rollups})
    return 0


def _write_manifest(out_dir: Path, arm: str, args, n_img: int, n_boxes: int,
                    extra: dict | None = None) -> None:
    m = {"arm": arm, "model": args.model, "images": n_img, "boxes": n_boxes,
         "args": {k: v for k, v in vars(args).items() if k != "func"}}
    if extra:
        m.update(extra)
    (out_dir / "speciesnet_manifest.json").write_text(json.dumps(m, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="arm", required=True)
    c = sub.add_parser("classifier", help="SpeciesNet classifier on wytrap boxes")
    c.add_argument("--wytrap-records", required=True, help="all_records.jsonl of a wytrap run")
    c.add_argument("--output", required=True)
    c.add_argument("--model", default=DEFAULT_MODEL)
    c.add_argument("--batch-size", type=int, default=32)
    c.add_argument("--topk", type=int, default=5)
    c.add_argument("--vocab", default=None,
                   help="taxon vocabulary CSV: restrict SpeciesNet to the shared candidate set")
    c.set_defaults(func=run_classifier_arm)
    e = sub.add_parser("ensemble", help="full SpeciesNet (MDv5a + classifier + geofence)")
    e.add_argument("--images", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--model", default=DEFAULT_MODEL)
    e.add_argument("--detections-from", default=None,
                   help="wytrap all_records.jsonl whose boxes replace SpeciesNet's MDv5a")
    e.add_argument("--country", default="USA")
    e.add_argument("--admin1", default="ID")
    e.add_argument("--no-geofence", action="store_true")
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--topk", type=int, default=5)
    e.add_argument("--vocab", default=None,
                   help="taxon vocabulary CSV: restrict SpeciesNet to the shared candidate set "
                        "before its roll-up (geofence then has nothing left to remove)")
    e.set_defaults(func=run_ensemble_arm)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
