"""Species classifiers and the registry `wytrap --classifier` picks from.

    bioclip              BioCLIP 2 zero-shot on a species list or a vocabulary's
                         prompts; multi-scale crops; optional prior correction
    speciesnet           SpeciesNet's classifier on each box (no roll-up)
    speciesnet-ensemble  SpeciesNet as shipped: classifier + taxonomy roll-up
                         + geofence, one label per image
    addax                any AddaxAI model-zoo repo (--model Addax-Data-Science/WUSA-SDZWA-v1)
    none                 detection only

Each takes `vocab=` (a taxonomy/*.csv path) to restrict itself to the
shared candidate set, and `model=` where a checkpoint choice exists.
"""

from __future__ import annotations

from wytrap.classifiers.base import BoxClassifier, BoxInput, BoxResult

CLASSIFIERS = ("bioclip", "speciesnet", "speciesnet-ensemble", "addax", "none")


def build_classifier(kind: str, model: str | None = None, vocab: str | None = None,
                     device: str = "auto", topk: int = 5, **opts) -> BoxClassifier | None:
    """Instantiate a classifier by name. `opts` are passed to the constructor
    (BioCLIP: species, prompt_bias, multiscale, multiscale_pad;
    SpeciesNet ensemble: country, admin1, geofence; all: batch_size)."""
    kind = (kind or "none").lower()
    if kind == "none":
        return None
    if kind == "bioclip":
        from wytrap.classifiers.bioclip import BioCLIPClassifier
        return BioCLIPClassifier(vocab=vocab, device=device, topk=topk, model=model, **opts)
    if kind == "speciesnet":
        from wytrap.classifiers.speciesnet import SpeciesNetClassifier
        return SpeciesNetClassifier(vocab=vocab, device=device, topk=topk, model=model, **opts)
    if kind == "speciesnet-ensemble":
        from wytrap.classifiers.speciesnet import SpeciesNetEnsemble
        return SpeciesNetEnsemble(vocab=vocab, device=device, topk=topk, model=model, **opts)
    if kind == "addax":
        if not model:
            raise ValueError("--classifier addax needs --model <HF repo id or local dir>, "
                             "e.g. Addax-Data-Science/WUSA-SDZWA-v1")
        from wytrap.classifiers.addax import AddaxClassifier
        return AddaxClassifier(model, vocab=vocab, device=device, topk=topk, **opts)
    raise ValueError(f"unknown classifier {kind!r}; choose from {CLASSIFIERS}")


__all__ = ["BoxClassifier", "BoxInput", "BoxResult", "CLASSIFIERS", "build_classifier"]
