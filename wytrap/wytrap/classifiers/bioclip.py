"""BioCLIP 2 zero-shot species classifier.

BioCLIP scores a crop against text prompts, one per candidate species. The
prompts are scientific (binomial) names, which the model was trained on and
which beat common names on fine-grained taxonomy. The candidate list comes
from a vocabulary (its member-species prompts), a wytrap species list
(`species_lists.py`), or a species file.

Multi-scale: each box is classified at three crops (tight, padded 2x, the
whole frame) and the highest-scoring scale wins. Close-ups and far shots
each get a scale that suits them, and disagreement between scales flags an
over-confident call on an uninformative crop.

Prior correction: a per-prompt log-space bias (from `wytrap calibrate`) is
subtracted before ranking. The uncorrected log-probs of every prompt are kept
on the record (`prompt_logp`) so the correction can be re-estimated offline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from PIL import Image

from wytrap.classifiers.base import BoxClassifier, BoxInput, BoxResult
from wytrap.species_lists import Species, load_species

# pybioclip pulls BioCLIP-2 from this HuggingFace repo by default.
BIOCLIP2_REPO_ID = "imageomics/bioclip-2"
BIOCLIP2_PROBE_FILES = ("open_clip_model.safetensors", "open_clip_config.json")


def bioclip_cache_status(repo_id: str = BIOCLIP2_REPO_ID) -> dict:
    """Inspect the local HF cache to report whether BioCLIP-2 weights are present."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return {"status": "unknown", "cache_dir": None,
                "missing": list(BIOCLIP2_PROBE_FILES), "cached_paths": {}}

    cached, missing = {}, []
    for fname in BIOCLIP2_PROBE_FILES:
        path = try_to_load_from_cache(repo_id=repo_id, filename=fname)
        if path is None:
            missing.append(fname)
        else:
            cached[fname] = str(path)
    status = "cached" if not missing else ("partial" if cached else "missing")
    return {"status": status, "cache_dir": HF_HUB_CACHE,
            "missing": missing, "cached_paths": cached}


@dataclass
class TopKEntry:
    common: str
    scientific: str
    score: float

    def to_dict(self) -> dict:
        return {"common": self.common, "scientific": self.scientific, "score": self.score}


@dataclass
class Classification:
    """One crop's result."""
    fine_label: str            # common name of top-1
    scientific_label: str      # binomial of top-1 (what BioCLIP saw)
    score: float
    topk: list[TopKEntry] = field(default_factory=list)
    # log-probability of every prompt, in `prompts` order, BEFORE any
    # prompt-bias correction (so calibration can be re-estimated offline)
    all_logp: list[float] = field(default_factory=list)


def crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = box
    W, H = image.size
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(x1 + 1, min(x2, W))
    y2 = max(y1 + 1, min(y2, H))
    return image.crop((x1, y1, x2, y2))


def pad_box(box: tuple[int, int, int, int], image_size: tuple[int, int],
            factor: float) -> tuple[int, int, int, int]:
    """Center-expand a box by `factor`, clamped to the image. A box that is
    already nearly full-frame stays about the same size, so the padded pass
    becomes a no-op there, which is right."""
    x1, y1, x2, y2 = box
    W, H = image_size
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * factor, (y2 - y1) * factor
    nx1 = max(0, int(round(cx - bw / 2)))
    ny1 = max(0, int(round(cy - bh / 2)))
    nx2 = min(W, int(round(cx + bw / 2)))
    ny2 = min(H, int(round(cy + bh / 2)))
    return (nx1, ny1, max(nx1 + 1, nx2), max(ny1 + 1, ny2))


class BioCLIPClassifier(BoxClassifier):
    name = "bioclip"

    def __init__(self, vocab=None, topk: int = 5, device: str = "auto",
                 model: str | None = None, species: str | Sequence = "wyoming_all",
                 prompt_bias: str | Path | dict | None = None,
                 multiscale: bool = True, multiscale_pad: float = 2.0,
                 skip_bad: bool = False, **_):
        """`vocab` wins over `species` when both are given: the candidate set is
        the vocabulary's prompts. `model` is an open_clip name for pybioclip
        (default BioCLIP 2). `prompt_bias` is a path to prompt_bias.json or
        a {prompt: bias} dict."""
        super().__init__(vocab=vocab, topk=topk, device=device)
        from bioclip.predict import CustomLabelsClassifier

        if self.vocab is not None:
            self.species: list[Species] = load_species(self.vocab.species())
        else:
            self.species = load_species(species if isinstance(species, (str, list)) else list(species))
        self.species_source = self.vocab_path or (species if isinstance(species, str) else "custom list")
        self.multiscale = multiscale
        self.multiscale_pad = multiscale_pad
        self.skip_bad = skip_bad
        self._prompts = [s["scientific"] for s in self.species]
        self._sci_to_common = {s["scientific"]: s["common"] for s in self.species}
        kw = {"device": self.device}
        if model:
            kw["model_str"] = model
        self._classifier = CustomLabelsClassifier(self._prompts, **kw)

        if isinstance(prompt_bias, (str, Path)):
            import json
            self.prompt_bias_path = str(prompt_bias)
            bias = json.loads(Path(prompt_bias).read_text())
            bias = bias.get("bias", bias)   # accept {"bias": {...}} or a flat map
        else:
            self.prompt_bias_path = None
            bias = prompt_bias or {}
        # Per-prompt log-space bias subtracted before ranking (prior
        # correction). Keyed by scientific name; prompts it lacks get 0.
        self.prompt_bias = [float(bias.get(p, 0.0)) for p in self._prompts]

    # ---- metadata ----------------------------------------------------------
    @property
    def prompts(self) -> list[str]:
        return list(self._prompts)

    def common_names(self) -> dict[str, str]:
        return dict(self._sci_to_common)

    def describe(self) -> dict:
        d = super().describe()
        d.update({"species": self.species_source, "n_prompts": len(self._prompts),
                  "prompt_bias": self.prompt_bias_path, "multiscale": self.multiscale,
                  "multiscale_pad": self.multiscale_pad})
        return d

    # ---- crops -------------------------------------------------------------
    def classify_crops(self, crops: Sequence[Image.Image]) -> list[Classification]:
        if not crops:
            return []
        preds = self._classifier.predict(list(crops))
        return self._unpack(preds, len(crops))

    def _unpack(self, preds: list[dict], n_images: int) -> list[Classification]:
        per_image: list[dict[str, float]] = [{} for _ in range(n_images)]
        n_species = len(self._prompts)
        for idx, p in enumerate(preds):
            img_idx = idx // n_species
            if img_idx >= n_images:
                break
            per_image[img_idx][p["classification"]] = float(p["score"])

        out: list[Classification] = []
        for probs in per_image:
            logp = [math.log(max(probs.get(pr, 0.0), 1e-12)) for pr in self._prompts]
            if any(self.prompt_bias):
                adj = [lp - b for lp, b in zip(logp, self.prompt_bias)]
                m = max(adj)
                z = sum(math.exp(a - m) for a in adj)
                ranked = [math.exp(a - m) / z for a in adj]
            else:
                ranked = [probs.get(pr, 0.0) for pr in self._prompts]
            items = sorted(zip(self._prompts, ranked), key=lambda kv: kv[1], reverse=True)
            top = [TopKEntry(self._sci_to_common.get(sci, sci), sci, score)
                   for sci, score in items[: self.topk]]
            if top:
                out.append(Classification(fine_label=top[0].common, scientific_label=top[0].scientific,
                                          score=top[0].score, topk=top,
                                          all_logp=[round(x, 4) for x in logp]))
            else:
                out.append(Classification(fine_label="", scientific_label="", score=0.0))
        return out

    # ---- the interface -----------------------------------------------------
    def classify_image(self, image: Image.Image, image_path: str,
                       boxes: list[BoxInput]) -> list[BoxResult | None]:
        idx = self.animal_indices(boxes, self.skip_bad)
        results: list[BoxResult | None] = [None] * len(boxes)
        if not idx:
            return results
        W, H = image.size
        tight = [crop(image, boxes[i].box_xyxy) for i in idx]
        cls_tight = self.classify_crops(tight)
        if self.multiscale:
            padded = [crop(image, pad_box(boxes[i].box_xyxy, (W, H), self.multiscale_pad)) for i in idx]
            cls_padded = self.classify_crops(padded)
            # The whole-frame pass is computed once and shared by every box.
            cls_full = self.classify_crops([image])[0]
        for k, i in enumerate(idx):
            ct = cls_tight[k]
            scale_scores = {"tight": ct.score}
            winner_name, winner = "tight", ct
            agree = True
            plogp = {"tight": ct.all_logp}
            if self.multiscale:
                cp = cls_padded[k]
                scale_scores["padded"] = cp.score
                scale_scores["full"] = cls_full.score
                for sname, c in (("padded", cp), ("full", cls_full)):
                    if c.score > winner.score:
                        winner_name, winner = sname, c
                # All three scales naming the same species is the reassuring
                # case; a 0.99 on one scale that the others contradict is the
                # over-confident-on-nothing pattern.
                agree = ct.fine_label == cp.fine_label == cls_full.fine_label
                plogp["padded"] = cp.all_logp
                plogp["full"] = cls_full.all_logp
            results[i] = BoxResult(
                label=winner.fine_label, scientific=winner.scientific_label, score=winner.score,
                topk=[t.to_dict() for t in winner.topk], scale=winner_name,
                scale_scores={k2: round(v, 4) for k2, v in scale_scores.items()},
                cross_scale_agree=agree, prompt_logp=plogp)
        return results
