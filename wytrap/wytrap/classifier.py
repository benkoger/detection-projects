"""BioCLIP-2 zero-shot species classifier wrapper."""

from dataclasses import dataclass
from typing import Sequence

from PIL import Image


# pybioclip pulls BioCLIP-2 from this HuggingFace repo by default.
BIOCLIP2_REPO_ID = "imageomics/bioclip-2"
BIOCLIP2_PROBE_FILES = ("open_clip_model.safetensors", "open_clip_config.json")


def bioclip_cache_status(repo_id: str = BIOCLIP2_REPO_ID) -> dict:
    """Inspect the local HF cache to report whether BioCLIP-2 weights are present.

    Returns a dict with keys:
      - status: 'cached' | 'partial' | 'missing' | 'unknown'
      - cache_dir: which HF cache directory was checked
      - missing: list of probe filenames not found in the cache
      - cached_paths: dict of probe_filename -> absolute local path (for hits)
    """
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
class Classification:
    fine_label: str            # whatever BioCLIP returned (e.g. "coyote")
    score: float
    topk: list[tuple[str, float]]


class Classifier:
    """Thin wrapper around pybioclip.CustomLabelsClassifier.

    Holds the species list and cached text embeddings; classifies PIL crops.
    """

    def __init__(self, species: Sequence[str], topk: int = 5):
        from bioclip.predict import CustomLabelsClassifier

        self.species = list(species)
        self.topk = topk
        self._classifier = CustomLabelsClassifier(self.species)

    def classify(self, crop: Image.Image) -> Classification:
        return self.classify_batch([crop])[0]

    def classify_batch(self, crops: Sequence[Image.Image]) -> list[Classification]:
        if not crops:
            return []
        # pybioclip groups predictions by input image when given a list.
        # It returns a flat list of {classification, score, ...} sorted by
        # score; for batched inputs each image's preds are contiguous.
        preds = self._classifier.predict(list(crops))
        return self._unpack(preds, len(crops))

    def _unpack(self, preds: list[dict], n_images: int) -> list[Classification]:
        # pybioclip emits len(species) entries per image. Group and rank.
        per_image = [[] for _ in range(n_images)]
        n_species = len(self.species)
        for idx, p in enumerate(preds):
            img_idx = idx // n_species
            if img_idx >= n_images:
                break
            per_image[img_idx].append((p["classification"], float(p["score"])))

        out: list[Classification] = []
        for items in per_image:
            items.sort(key=lambda kv: kv[1], reverse=True)
            top = items[: self.topk]
            fine, score = top[0] if top else ("", 0.0)
            out.append(Classification(fine_label=fine, score=score, topk=top))
        return out
