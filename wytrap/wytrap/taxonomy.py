"""Label-mapping tables shared by the classifiers and the evaluator.

Two kinds of mapping live here:

* `IDAHO_EVAL_MERGES` — string merges from a classifier's common name onto
  the coarser labels of the LILA Idaho Camera Traps set (mule deer -> deer).
  Used by `wytrap eval` when no vocabulary is given.
* `taxonomy_to_idaho()` — the same mapping driven by lineage, for classifiers
  whose labels carry taxonomy (SpeciesNet, AddaxAI zoo models). Looked up in
  order (genus, species) -> genus -> family -> order; anything unmatched keeps
  the model's own common name, lower-cased.

SpeciesNet label strings ('uuid;class;order;family;genus;species;common')
are parsed by `speciesnet_lineage()` / `speciesnet_names()`.

With a vocabulary (`wytrap.vocab`) predictions resolve by lineage instead,
and these tables only supply display names.
"""

from __future__ import annotations

# Idaho classes with no counterpart in wyoming_all: "cattle" and "horse"
# (no livestock prompts). scripts/ai4wy/species_idaho.txt adds them; with
# wyoming_all they are counted as classification errors, which is the
# honest open-set result.
IDAHO_EVAL_MERGES = {
    # deer
    "mule deer": "deer",
    "white-tailed deer": "deer",
    "Odocoileus": "deer",
    # bear
    "black bear": "bear",
    "grizzly bear": "bear",
    "Ursidae": "bear",
    # canids: Idaho separates wolf / coyote / fox / domestic dog
    "gray wolf": "wolf",
    "red fox": "fox",
    "swift fox": "fox",
    "gray fox": "fox",
    # felids
    "cougar": "mountain lion",
    # lagomorphs (GT uses both "lagomorph" and "rabbit")
    "rabbit": "lagomorph",
    "snowshoe hare": "lagomorph",
    "white-tailed jackrabbit": "lagomorph",
    "Nuttall's cottontail": "lagomorph",
    "American pika": "lagomorph",
    # mustelids etc.
    "striped skunk": "skunk",
    # squirrels
    "red squirrel": "squirrel",
    "fox squirrel": "squirrel",
    "northern flying squirrel": "squirrel",
    "Columbian ground squirrel": "squirrel",
    "golden-mantled ground squirrel": "squirrel",
    "thirteen-lined ground squirrel": "squirrel",
    "Uinta ground squirrel": "squirrel",
    "Wyoming ground squirrel": "squirrel",
    "rock squirrel": "squirrel",
    "least chipmunk": "squirrel",
    "Sciuridae": "squirrel",
    # birds
    "wild turkey": "turkey",
    "dusky grouse": "grouse",
    "ruffed grouse": "grouse",
    "greater sage-grouse": "grouse",
    "sharp-tailed grouse": "grouse",
    # livestock (only present when using species_idaho.txt)
    "domestic cattle": "cattle",
    "domestic horse": "horse",
}


IDAHO_SPECIES_TO_CLASS = {
    ("canis", "lupus"): "wolf", ("canis", "lupis"): "wolf",   # WUSA typo
    ("canis", "latrans"): "coyote", ("canis", "familiaris"): "domestic dog",
    ("lynx", "rufus"): "bobcat", ("homo", "sapiens"): "human",
}
IDAHO_GENUS_TO_CLASS = {
    "odocoileus": "deer", "cervus": "elk", "alces": "moose",
    "antilocapra": "pronghorn", "ovis": "bighorn sheep",
    "ursus": "bear", "vulpes": "fox", "urocyon": "fox",
    "puma": "mountain lion", "mephitis": "skunk", "spilogale": "skunk",
    "bos": "cattle", "equus": "horse", "meleagris": "turkey",
    "dendragapus": "grouse", "bonasa": "grouse", "tympanuchus": "grouse",
    "centrocercus": "grouse", "lagopus": "grouse", "falcipennis": "grouse",
}
IDAHO_FAMILY_TO_CLASS = {
    "leporidae": "lagomorph", "sciuridae": "squirrel", "ursidae": "bear",
    "mephitidae": "skunk",
}
IDAHO_ORDER_TO_CLASS = {"lagomorpha": "lagomorph"}


def taxonomy_to_idaho(cls: str = "", order: str = "", family: str = "",
                      genus: str = "", species: str = "", common: str = "") -> str:
    """Map one taxon (any fields may be empty) to an Idaho eval label."""
    g, s = (genus or "").lower().strip(), (species or "").lower().strip()
    f, o = (family or "").lower().strip(), (order or "").lower().strip()
    if (g, s) in IDAHO_SPECIES_TO_CLASS:
        return IDAHO_SPECIES_TO_CLASS[(g, s)]
    if g in IDAHO_GENUS_TO_CLASS:
        return IDAHO_GENUS_TO_CLASS[g]
    if f in IDAHO_FAMILY_TO_CLASS:
        return IDAHO_FAMILY_TO_CLASS[f]
    if o in IDAHO_ORDER_TO_CLASS:
        return IDAHO_ORDER_TO_CLASS[o]
    return (common or "").lower().strip()


# ---- SpeciesNet label strings ------------------------------------------------

def speciesnet_lineage(label: str) -> dict[str, str]:
    """'uuid;class;order;family;genus;species;common' -> rank dict."""
    parts = label.split(";")
    if len(parts) != 7:
        return {}
    _, cls, order, family, genus, species, _ = parts
    return {"class": cls, "order": order, "family": family, "genus": genus, "species": species}


def speciesnet_names(label: str) -> tuple[str, str]:
    """(display name, scientific) for a SpeciesNet label string."""
    parts = label.split(";")
    if len(parts) != 7:
        return label, label
    _, cls, order, family, genus, species, common = parts
    sci = " ".join(p for p in (genus, species) if p) or common
    return taxonomy_to_idaho(cls, order, family, genus, species, common), sci
