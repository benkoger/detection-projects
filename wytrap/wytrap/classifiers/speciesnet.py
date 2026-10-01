"""SpeciesNet (Google / CameraTrapAI) as a wytrap classifier, two ways.

SpeciesNetClassifier     the classifier network on each MegaDetector box:
                         crop the way its always-crop model expects, softmax
                         over its 2,498 labels (or the vocabulary's candidate
                         set via SpeciesNet's own target-species mechanism),
                         top-k per box. Isolates "which classifier is better".

SpeciesNetEnsemble       SpeciesNet as shipped: classifier, then taxonomy
                         roll-up (a species call below threshold becomes its
                         genus / family / ...) and geofence (country, admin1).
                         One label per image, written onto every animal box.
                         Our boxes replace its MegaDetector v5a, so everything
                         up to detection is shared with the other classifiers.

Labels are 'uuid;class;order;family;genus;species;common'; the display
label comes from `taxonomy.speciesnet_names`, which maps by lineage onto the
evaluation classes (odocoileus -> deer) and otherwise keeps the common name.
Weights come from the Hugging Face mirror AddaxAI Connect uses, so no Kaggle
credentials are needed.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from wytrap.classifiers.base import BoxClassifier, BoxInput, BoxResult
from wytrap.taxonomy import speciesnet_lineage, speciesnet_names

DEFAULT_MODEL = "hf:Addax-Data-Science/SPECIESNET-v4-0-2-A"
MD_CATEGORY = {"animal": "1", "person": "2", "vehicle": "3"}


def _topk(classes: list[str], scores: list[float], k: int) -> list[dict]:
    out = []
    for c, s in list(zip(classes, scores))[:k]:
        name, sci = speciesnet_names(c)
        out.append({"common": name, "scientific": sci, "score": float(s),
                    "lineage": speciesnet_lineage(c)})
    return out


def masked_classifications(cls: dict, k: int = 5) -> dict:
    """Softmax over SpeciesNet's target_logits (the candidate set) and return a
    classifications dict in SpeciesNet's own format, top-k only."""
    labels = cls.get("target_classes") or []
    logits = np.asarray(cls.get("target_logits") or [], dtype=np.float64)
    if not len(labels) or not len(logits):
        return cls
    p = np.exp(logits - logits.max())
    p /= p.sum()
    order = np.argsort(-p)[:k]
    return {"classes": [labels[i] for i in order], "scores": [float(p[i]) for i in order]}


class _SpeciesNetBase(BoxClassifier):
    def __init__(self, vocab=None, topk: int = 5, device: str = "auto",
                 model: str | None = None, batch_size: int = 32, **_):
        super().__init__(vocab=vocab, topk=topk, device=device)
        from speciesnet.utils import ModelInfo
        self.model = model or DEFAULT_MODEL
        self.batch_size = batch_size
        with open(ModelInfo(self.model).classifier_labels, encoding="utf-8") as f:
            self.labels = [l.strip() for l in f if l.strip()]
        self._lineages = {l: speciesnet_lineage(l) for l in self.labels}
        self.target_file: Path | None = None
        cand = self.candidate_classes()
        if cand is not None:
            # SpeciesNet restricts itself through a target-species file; its
            # classifier then also returns logits over just those labels.
            self.target_file = Path(tempfile.mkdtemp(prefix="wytrap_sn_")) / "targets.txt"
            self.target_file.write_text("\n".join(l for l in self.labels if l in cand) + "\n")
            self.n_targets = sum(1 for l in self.labels if l in cand)

    def class_lineages(self) -> dict[str, dict]:
        return self._lineages

    def describe(self) -> dict:
        d = super().describe()
        d.update({"model": self.model, "n_labels": len(self.labels),
                  "candidate_set": getattr(self, "n_targets", None)})
        return d


class SpeciesNetClassifier(_SpeciesNetBase):
    name = "speciesnet"

    def __init__(self, vocab=None, topk: int = 5, device: str = "auto",
                 model: str | None = None, batch_size: int = 32, skip_bad: bool = False, **_):
        super().__init__(vocab=vocab, topk=topk, device=device, model=model, batch_size=batch_size)
        from speciesnet import SpeciesNetClassifier as _Clf
        self.skip_bad = skip_bad
        self._clf = _Clf(self.model, target_species_txt=str(self.target_file) if self.target_file else None)

    def classify_image(self, image: Image.Image, image_path: str,
                       boxes: list[BoxInput]) -> list[BoxResult | None]:
        from speciesnet import BBox
        idx = self.animal_indices(boxes, self.skip_bad)
        results: list[BoxResult | None] = [None] * len(boxes)
        if not idx:
            return results
        pre = []
        for i in idx:
            x, y, w, h = self.normalised(boxes[i].box_xyxy, image.size)
            pre.append(self._clf.preprocess(image, bboxes=[BBox(xmin=x, ymin=y, width=w, height=h)]))
        preds = []
        for s in range(0, len(pre), self.batch_size):
            chunk = pre[s:s + self.batch_size]
            # batch_predict keys results by filepath, so each box gets its own key
            keys = [f"{image_path}#box{s + j}" for j in range(len(chunk))]
            preds.extend(self._clf.batch_predict(keys, chunk))
        for i, r in zip(idx, preds):
            cls = r.get("classifications")
            if cls and self.target_file:
                cls = masked_classifications(cls, self.topk)
            if not cls:
                results[i] = BoxResult(label="no cv result", source="failure")
                continue
            topk = _topk(cls["classes"], cls["scores"], self.topk)
            top = topk[0]
            results[i] = BoxResult(label=top["common"], scientific=top["scientific"],
                                   score=top["score"], topk=topk, lineage=top["lineage"],
                                   scale_scores={"tight": top["score"]})
        return results


class SpeciesNetEnsemble(_SpeciesNetBase):
    name = "speciesnet-ensemble"

    def __init__(self, vocab=None, topk: int = 5, device: str = "auto",
                 model: str | None = None, batch_size: int = 8,
                 country: str = "USA", admin1: str | None = "WY", geofence: bool = True, **_):
        super().__init__(vocab=vocab, topk=topk, device=device, model=model, batch_size=batch_size)
        from speciesnet import SpeciesNet
        self.country, self.admin1, self.geofence = country, admin1, geofence
        self._model = SpeciesNet(self.model, components="all", geofence=geofence,
                                 target_species_txt=str(self.target_file) if self.target_file else None)
        self._scratch = Path(tempfile.mkdtemp(prefix="wytrap_sn_ens_"))

    def describe(self) -> dict:
        d = super().describe()
        d.update({"country": self.country, "admin1": self.admin1, "geofence": self.geofence})
        return d

    def classify_image(self, image: Image.Image, image_path: str,
                       boxes: list[BoxInput]) -> list[BoxResult | None]:
        results: list[BoxResult | None] = [None] * len(boxes)
        if not boxes:
            return results
        W, H = image.size
        # Our boxes in SpeciesNet's detector-output format, highest confidence
        # first: its always-crop classifier crops to detections[0].
        order = sorted(range(len(boxes)), key=lambda i: -boxes[i].det_score)
        dets = []
        for i in order:
            b = boxes[i]
            dets.append({"category": MD_CATEGORY.get(b.det_label, "1"), "label": b.det_label,
                         "conf": float(b.det_score), "bbox": list(self.normalised(b.box_xyxy, (W, H)))})
        dd = {image_path: {"filepath": image_path, "detections": dets}}
        cls = self._model.classify(filepaths=[image_path], detections_dict=dd,
                                   country=self.country, admin1_region=self.admin1,
                                   batch_size=self.batch_size, progress_bars=False)
        cls_dict = {}
        for p in cls["predictions"]:
            if self.target_file and p.get("classifications"):
                p = {**p, "classifications": masked_classifications(p["classifications"], self.topk)}
            cls_dict[p["filepath"]] = p
        # ensemble_from_past_runs writes to predictions_json and treats an
        # existing file as a resume checkpoint, so give it a fresh one.
        raw = self._scratch / "pred.json"
        if raw.exists():
            raw.unlink()
        preds = self._model.ensemble_from_past_runs(
            filepaths=[image_path], classifications_dict=cls_dict, detections_dict=dd,
            country=self.country, admin1_region=self.admin1, progress_bars=False,
            predictions_json=str(raw))
        if preds is None:
            preds = json.loads(raw.read_text())
        p = preds["predictions"][0]
        pred_label = p.get("prediction", "")
        name, sci = speciesnet_names(pred_label) if pred_label else ("", "")
        c = p.get("classifications") or {}
        topk = _topk(c.get("classes", []), c.get("scores", []), self.topk)
        score = float(p.get("prediction_score", 0.0))
        for i, b in enumerate(boxes):
            if b.det_label != "animal":
                continue
            results[i] = BoxResult(label=name, scientific=sci, score=score, topk=topk,
                                   lineage=speciesnet_lineage(pred_label) if pred_label else {},
                                   scale="ensemble", scale_scores={"ensemble": score},
                                   source=p.get("prediction_source", ""))
        return results
