"""BioCLIP-2 zero-shot species classifier wrapper."""

from dataclasses import dataclass, field
from typing import Sequence

from PIL import Image

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

    if not missing:
        status = "cached"
    elif cached:
        status = "partial"
    else:
        status = "missing"

    return {"status": status, "cache_dir": HF_HUB_CACHE,
            "missing": missing, "cached_paths": cached}


@dataclass
class TopKEntry:
    common: str
    scientific: str
    score: float

    def to_dict(self) -> dict:
        return {"common": self.common, "scientific": self.scientific,
                "score": self.score}


@dataclass
class Classification:
    fine_label: str            # common name of top-1 (back-compat)
    scientific_label: str      # binomial of top-1 (what BioCLIP saw)
    score: float
    topk: list[TopKEntry] = field(default_factory=list)
    # log-probability of every prompt, in Classifier.prompts order, BEFORE any
    # prompt-bias correction (so calibration can be re-estimated offline)
    all_logp: list[float] = field(default_factory=list)


@dataclass
class MultiScaleClassification:
    """Result of classifying one detection at multiple crop scales.

    `chosen` is whichever scale's Classification scored highest. `scale` is
    the name of that scale ("tight" / "padded" / "full"). `scale_scores`
    holds top-1 scores for every scale considered, useful for analysis.
    """
    chosen: Classification
    scale: str
    scale_scores: dict[str, float]


class Classifier:
    """Thin wrapper around pybioclip.CustomLabelsClassifier.

    Holds the species list and cached text embeddings; classifies PIL crops.
    Accepts either Species dicts (preferred) or plain strings (legacy).
    """

    def __init__(self, species: Sequence, topk: int = 5,
                 device: str = "auto", prompt_bias: dict[str, float] | None = None):
        from bioclip.predict import CustomLabelsClassifier

        self.species: list[Species] = load_species(list(species))
        self.topk = topk
        self.device = self._resolve_device(device)

        # BioCLIP sees scientific (binomial) names — they're what the model
        # was trained on and consistently outperform common names on
        # fine-grained taxonomy.
        self._prompts = [s["scientific"] for s in self.species]
        self._sci_to_common = {s["scientific"]: s["common"] for s in self.species}
        self._classifier = CustomLabelsClassifier(self._prompts, device=self.device)
        # Per-prompt log-space bias subtracted before ranking (prior
        # correction). Keyed by scientific name; missing prompts get 0.
        self.prompt_bias = [float((prompt_bias or {}).get(p, 0.0)) for p in self._prompts]

    @property
    def prompts(self) -> list[str]:
        return list(self._prompts)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def classify(self, crop: Image.Image) -> Classification:
        return self.classify_batch([crop])[0]

    def classify_batch(self, crops: Sequence[Image.Image]) -> list[Classification]:
        if not crops:
            return []
        preds = self._classifier.predict(list(crops))
        return self._unpack(preds, len(crops))

    def _unpack(self, preds: list[dict], n_images: int) -> list[Classification]:
        import math
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
            items = list(zip(self._prompts, ranked))
            items.sort(key=lambda kv: kv[1], reverse=True)
            top = items[: self.topk]
            topk_entries = [
                TopKEntry(common=self._sci_to_common.get(sci, sci),
                          scientific=sci,
                          score=score)
                for sci, score in top
            ]
            if topk_entries:
                head = topk_entries[0]
                out.append(Classification(
                    fine_label=head.common,
                    scientific_label=head.scientific,
                    score=head.score,
                    topk=topk_entries,
                    all_logp=[round(x, 4) for x in logp],
                ))
            else:
                out.append(Classification(
                    fine_label="", scientific_label="", score=0.0, topk=[]
                ))
        return out
