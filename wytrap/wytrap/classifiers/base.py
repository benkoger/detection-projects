"""The classifier interface every species model implements.

A classifier receives one image and all of MegaDetector's boxes in it, and
returns one result per box (or None for boxes it leaves unclassified, such
as people and vehicles). Giving it the whole image rather than pre-cut
crops lets each model crop and pad the way it was trained: BioCLIP looks
at three scales, SpeciesNet's always-crop model pads its own square, the
AddaxAI zoo models ship their own `get_crop`.

Vocabulary restriction is the same for every classifier: `Vocab.candidate_classes`
decides which of the model's classes form the shared candidate set, and the
model renormalises its probabilities over those. Implementations expose their
classes' lineages through `class_lineages()` so the rule can be applied.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from wytrap.vocab import Vocab


@dataclass
class BoxInput:
    box_xyxy: tuple[int, int, int, int]   # absolute pixels
    det_score: float
    det_label: str = "animal"             # "animal" | "person" | "vehicle"
    quality: str = "ok"                   # wytrap box-quality tag


@dataclass
class BoxResult:
    """The classification half of a DetectionRecord."""
    label: str                            # display label (common name or node)
    scientific: str = ""
    score: float = 0.0
    topk: list[dict] = field(default_factory=list)   # {common, scientific, score[, lineage]}
    lineage: dict = field(default_factory=dict)       # ranks of the top-1, when known
    scale: str = "tight"                  # crop scale that won (BioCLIP) / "ensemble"
    scale_scores: dict = field(default_factory=dict)
    cross_scale_agree: bool = True
    prompt_logp: dict = field(default_factory=dict)   # BioCLIP: scale -> all prompt log-probs
    source: str = ""                      # e.g. SpeciesNet "classifier+rollup_to_genus"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class BoxClassifier(ABC):
    """Base class. Subclasses set `name` and implement `classify_image`."""

    name: str = "base"

    def __init__(self, vocab: str | Path | Vocab | None = None, topk: int = 5,
                 device: str = "auto"):
        self.vocab: Vocab | None = (Vocab.load(vocab) if isinstance(vocab, (str, Path))
                                    else vocab)
        self.vocab_path = str(vocab) if isinstance(vocab, (str, Path)) else None
        self.topk = topk
        self.device = resolve_device(device)

    # ---- what subclasses provide -------------------------------------------
    @abstractmethod
    def classify_image(self, image: Image.Image, image_path: str,
                       boxes: list[BoxInput]) -> list[BoxResult | None]:
        """One result per box, in order; None where the box is not classified."""

    def class_lineages(self) -> dict[str, dict]:
        """Model class code -> lineage dict, for vocabulary masking. Empty if
        the model's classes carry no taxonomy (then no masking is possible)."""
        return {}

    def describe(self) -> dict:
        """Settings worth recording in the run's manifest."""
        return {"classifier": self.name, "vocab": self.vocab_path, "topk": self.topk,
                "device": self.device}

    # ---- shared helpers ----------------------------------------------------
    def candidate_classes(self) -> set[str] | None:
        """The model's classes inside the vocabulary, or None when unrestricted."""
        if self.vocab is None:
            return None
        lins = self.class_lineages()
        return self.vocab.candidate_classes(lins) if lins else None

    @staticmethod
    def animal_indices(boxes: list[BoxInput], skip_bad: bool = False) -> list[int]:
        """Indices of the boxes a species model should look at."""
        return [i for i, b in enumerate(boxes)
                if b.det_label == "animal" and not (skip_bad and b.quality != "ok")]

    @staticmethod
    def normalised(box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[float, float, float, float]:
        """xyxy pixels -> (x, y, w, h) fractions, the convention of SpeciesNet
        and the AddaxAI models."""
        x1, y1, x2, y2 = box
        W, H = size
        return x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H
