"""Run any AddaxAI model-zoo classifier on the boxes of a wytrap run.

AddaxAI ships every zoo model as a Hugging Face repo containing the weights,
`classes.csv`, `taxonomy.csv`, and an `inference.py` with a `ModelInference`
class (load_model / get_crop / get_classification, optionally get_tensor /
classify_batch). AddaxAI loads that file dynamically and so do we, so the
crop and preprocessing are exactly what AddaxAI Connect applies.

Output is wytrap-format records (same images, same boxes, same quality tags
as the source run; only the labels change), scored by
scripts/eval_image_level.py. Model classes are mapped to the Idaho vocabulary
through the repo's taxonomy.csv (helpers.taxonomy_to_idaho).

Usage:
    python scripts/run_addax_model_arm.py \\
        --model Addax-Data-Science/WUSA-SDZWA-v1 \\
        --wytrap-records /path/output-wytrap/all_records.jsonl \\
        --output /path/output-wusa [--batch-size 32] [--topk 5]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in (REPO_ROOT, REPO_ROOT / "wytrap"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

from helpers.helpers import taxonomy_to_idaho  # noqa: E402
from helpers.vocab import Vocab  # noqa: E402
from wytrap.io import DetectionRecord, ImageRecord, append_jsonl, save_record  # noqa: E402

WEIGHT_EXTS = (".pt", ".pth", ".onnx", ".pb", ".h5", ".safetensors", ".tflite")


def _log(msg: str) -> None:
    print(f"[addax {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model_dir(repo_id: str) -> Path:
    if Path(repo_id).is_dir():
        return Path(repo_id)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id))


def load_inference(model_dir: Path):
    """Mirror AddaxAI's classification_worker.load_inference_class."""
    inf_py = model_dir / "inference.py"
    if not inf_py.exists():
        raise FileNotFoundError(f"{model_dir} has no inference.py (not an AddaxAI zoo model?)")
    # The checkpoint AddaxAI passes as model_path is the one NOT named inside
    # inference.py: auxiliary files the script opens by name (e.g. the
    # torchvision ImageNet backbone SWUSA-SDZWA-v3 ships) are excluded.
    script_lines = inf_py.read_text(encoding="utf-8", errors="ignore").splitlines()

    def opened_by_name(fname: str) -> bool:
        return any(fname in ln and "model_dir" in ln and "/" in ln for ln in script_lines)

    candidates = [p for p in model_dir.iterdir() if p.suffix.lower() in WEIGHT_EXTS]
    weights = sorted((p for p in candidates if not opened_by_name(p.name)),
                     key=lambda p: p.stat().st_size, reverse=True) or \
        sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)
    if not weights:
        raise FileNotFoundError(f"no weight file in {model_dir}")
    spec = importlib.util.spec_from_file_location(f"addax_model_{model_dir.name}", inf_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    inf = mod.ModelInference(model_dir, weights[0])
    inf.load_model()
    return inf, weights[0]


def load_label_map(model_dir: Path) -> tuple[dict[str, tuple[str, str]], dict[str, dict]]:
    """model class code -> (idaho label, scientific), plus code -> lineage dict."""
    out: dict[str, tuple[str, str]] = {}
    lineages: dict[str, dict] = {}
    tax = model_dir / "taxonomy.csv"
    if tax.exists():
        with open(tax, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = row.get("model_class", "")
                sci = " ".join(x for x in (row.get("genus", ""), row.get("species", "")) if x)
                lineages[code] = {k: row.get(k, "") for k in ("class", "order", "family", "genus", "species")}
                out[code] = (taxonomy_to_idaho(row.get("class", ""), row.get("order", ""),
                                               row.get("family", ""), row.get("genus", ""),
                                               row.get("species", ""), code.replace("_", " ")),
                             sci or code)
    cls = model_dir / "classes.csv"
    if cls.exists():
        with open(cls, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = row.get("code") or row.get("class") or ""
                if code and code not in out:
                    common = row.get("common", code.replace("_", " "))
                    out[code] = (taxonomy_to_idaho(common=common), row.get("species", common))
    return out, lineages


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF repo id (Addax-Data-Science/...) or local dir")
    ap.add_argument("--wytrap-records", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--vocab", default=None,
                    help="taxon vocabulary CSV: mask the softmax to classes inside it and renormalise")
    args = ap.parse_args(argv)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "all_records.jsonl"
    if jsonl.exists():
        jsonl.unlink()

    t0 = time.time()
    model_dir = load_model_dir(args.model)
    inf, weights = load_inference(model_dir)
    label_map, lineages = load_label_map(model_dir)
    allowed: set[str] | None = None
    if args.vocab:
        vocab = Vocab.load(args.vocab)
        allowed = vocab.candidate_classes(lineages)
        _log(f"vocab mask: {len(allowed)}/{len(label_map)} classes kept; masked: "
             f"{sorted(set(label_map) - allowed)}")
    batched = hasattr(inf, "get_tensor") and hasattr(inf, "classify_batch")
    _log(f"{args.model}: {weights.name}, {len(label_map)} classes, "
         f"batched={batched}, loaded in {time.time() - t0:.0f}s on {getattr(inf, 'device', '?')}")
    _log("class -> eval label: " + ", ".join(
        f"{c}->{lab}" for c, (lab, _) in sorted(label_map.items()) if lab != c))

    records = [json.loads(l) for l in open(args.wytrap_records) if l.strip()]
    n_img = n_boxes = 0
    t0 = time.time()
    for rec in records:
        dets = rec.get("detections") or []
        new_dets: list[DetectionRecord] = []
        if dets:
            with Image.open(rec["image_path"]) as im:
                image = im.convert("RGB")
            W, H = image.size
            crops, keep = [], []
            for d in dets:
                x1, y1, x2, y2 = d["box_xyxy"]
                bbox = (x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H)
                try:
                    crops.append(inf.get_crop(image, bbox))
                    keep.append(d)
                except ValueError:
                    continue
            results: list[list] = []
            if batched:
                for i in range(0, len(crops), args.batch_size):
                    chunk = crops[i:i + args.batch_size]
                    batch = np.stack([inf.get_tensor(c) for c in chunk])
                    results.extend(inf.classify_batch(batch))
            else:
                results = [inf.get_classification(c) for c in crops]
            for d, probs in zip(keep, results):
                if allowed is not None:
                    probs = [(c, p) for c, p in probs if c in allowed]
                    z = sum(p for _, p in probs) or 1.0
                    probs = [(c, p / z) for c, p in probs]
                ranked = sorted(probs, key=lambda x: -x[1])[:args.topk]
                topk = []
                for code, score in ranked:
                    lab, sci = label_map.get(code, (str(code).replace("_", " "), str(code)))
                    topk.append({"common": lab, "scientific": sci, "score": float(score),
                                 "lineage": lineages.get(code, {})})
                top = topk[0]
                new_dets.append(DetectionRecord(
                    box_xyxy=d["box_xyxy"], det_score=d["det_score"], det_label=d["det_label"],
                    label=top["common"], fine_label=top["common"],
                    scientific_label=top["scientific"], cls_score=top["score"], topk=topk,
                    quality=d.get("quality", "ok"), quality_reason=d.get("quality_reason", ""),
                    scale="tight", scale_scores={"tight": top["score"]}, cross_scale_agree=True,
                    lineage=lineages.get(ranked[0][0], {}),
                ))
                n_boxes += 1
        out_rec = ImageRecord(image_path=rec["image_path"], image_size=rec["image_size"],
                              detections=new_dets, error=rec.get("error"))
        rel = Path(rec["image_path"])
        save_record(out_rec, out_dir / rel.parent.name / (rel.stem + ".json"))
        append_jsonl(out_rec, jsonl)
        n_img += 1
        if n_img % 50 == 0 or n_img == len(records):
            _log(f"{n_img}/{len(records)} images, {n_boxes} boxes, {n_img / (time.time() - t0):.1f} img/s")
    (out_dir / "addax_manifest.json").write_text(json.dumps({
        "model": args.model, "weights": weights.name, "model_dir": str(model_dir),
        "wytrap_records": args.wytrap_records, "images": n_img, "boxes": n_boxes,
        "vocab": args.vocab, "allowed_classes": sorted(allowed) if allowed is not None else None,
        "label_map": {k: v[0] for k, v in label_map.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
