"""Any AddaxAI model-zoo classifier (Western USA SDZWA, Southwest USA, ...).

AddaxAI ships every zoo model as a Hugging Face repo containing the weights,
`classes.csv`, `taxonomy.csv`, and an `inference.py` with a `ModelInference`
class (load_model / get_crop / get_classification, optionally get_tensor /
classify_batch). AddaxAI loads that file dynamically and so do we, so the
crop and preprocessing are exactly what AddaxAI Connect applies.

Model classes are mapped to display labels through the repo's taxonomy.csv
(`taxonomy.taxonomy_to_idaho`), and the same taxonomy drives vocabulary
masking: the softmax is renormalised over the classes inside the vocabulary.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from wytrap.classifiers.base import BoxClassifier, BoxInput, BoxResult
from wytrap.taxonomy import taxonomy_to_idaho

WEIGHT_EXTS = (".pt", ".pth", ".onnx", ".pb", ".h5", ".safetensors", ".tflite")


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
    """model class code -> (display label, scientific), plus code -> lineage dict."""
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


class AddaxClassifier(BoxClassifier):
    name = "addax"

    def __init__(self, model: str, vocab=None, topk: int = 5, device: str = "auto",
                 batch_size: int = 32, skip_bad: bool = False, **_):
        super().__init__(vocab=vocab, topk=topk, device=device)
        self.model = model
        self.batch_size = batch_size
        self.skip_bad = skip_bad
        self.model_dir = load_model_dir(model)
        self._inf, self.weights = load_inference(self.model_dir)
        self.label_map, self._lineages = load_label_map(self.model_dir)
        self.allowed = self.candidate_classes()
        self._batched = hasattr(self._inf, "get_tensor") and hasattr(self._inf, "classify_batch")

    def class_lineages(self) -> dict[str, dict]:
        return self._lineages

    def describe(self) -> dict:
        d = super().describe()
        d.update({"model": self.model, "weights": self.weights.name, "n_classes": len(self.label_map),
                  "candidate_set": sorted(self.allowed) if self.allowed is not None else None,
                  "label_map": {k: v[0] for k, v in self.label_map.items()}})
        return d

    def classify_image(self, image: Image.Image, image_path: str,
                       boxes: list[BoxInput]) -> list[BoxResult | None]:
        idx = self.animal_indices(boxes, self.skip_bad)
        results: list[BoxResult | None] = [None] * len(boxes)
        crops, kept = [], []
        for i in idx:
            try:
                crops.append(self._inf.get_crop(image, self.normalised(boxes[i].box_xyxy, image.size)))
                kept.append(i)
            except ValueError:        # degenerate box the model's cropper rejects
                continue
        if not crops:
            return results
        probs_list: list[list] = []
        if self._batched:
            for s in range(0, len(crops), self.batch_size):
                chunk = crops[s:s + self.batch_size]
                batch = np.stack([self._inf.get_tensor(c) for c in chunk])
                probs_list.extend(self._inf.classify_batch(batch))
        else:
            probs_list = [self._inf.get_classification(c) for c in crops]
        for i, probs in zip(kept, probs_list):
            if self.allowed is not None:
                probs = [(c, p) for c, p in probs if c in self.allowed]
                z = sum(p for _, p in probs) or 1.0
                probs = [(c, p / z) for c, p in probs]
            ranked = sorted(probs, key=lambda x: -x[1])[: self.topk]
            topk = []
            for code, score in ranked:
                lab, sci = self.label_map.get(code, (str(code).replace("_", " "), str(code)))
                topk.append({"common": lab, "scientific": sci, "score": float(score),
                             "lineage": self._lineages.get(code, {})})
            if not topk:
                continue
            top = topk[0]
            results[i] = BoxResult(label=top["common"], scientific=top["scientific"], score=top["score"],
                                   topk=topk, lineage=top["lineage"], scale_scores={"tight": top["score"]})
        return results
