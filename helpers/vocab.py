"""Taxon-node evaluation vocabulary and resolver.

A vocabulary (taxonomy/*.csv) is a list of labelled taxon nodes. Each row is
one member taxon of a label, given by its GBIF-backbone lineage; the node's
rank is the deepest filled rank. A prediction resolves to a label when its
lineage matches a member at that member's rank (and agrees on the ranks
above it). The resolver reports the relation between prediction and node:

    exact    prediction is at the node's rank            (mule deer -> deer? no: finer)
    finer    prediction is a descendant of the node      (mule deer -> deer)
    coarser  prediction is an ancestor of the node       (Cervidae -> deer)
    none     prediction lies outside every node

Predictions reach the resolver either as a lineage dict (SpeciesNet labels,
AddaxAI taxonomy.csv rows) or as a scientific binomial (BioCLIP prompts),
which is looked up in the nodes' prompt tables first and then by genus.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

RANKS = ("class", "order", "family", "genus", "species")


@dataclass
class Member:
    label: str
    lineage: dict[str, str]           # rank -> canonical name (lower-cased)
    rank: str                         # deepest filled rank
    prompts: list[tuple[str, str]] = field(default_factory=list)  # (binomial, common)


@dataclass
class Vocab:
    members: list[Member]
    labels: list[str]
    binomial_index: dict[str, Member]  # lower-cased binomial -> member

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        members: list[Member] = []
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(l for l in f if not l.startswith("#"))]
        for r in rows:
            lin = {k: (r.get(k) or "").strip().lower() for k in RANKS}
            if lin["species"] and lin["genus"]:
                lin["species"] = f"{lin['genus']} {lin['species']}"  # store binomial
            rank = next(k for k in reversed(RANKS) if lin[k])
            prompts = []
            for p in (r.get("prompts") or "").split(";"):
                p = p.strip()
                if not p:
                    continue
                sci, _, com = p.partition("|")
                prompts.append((sci.strip(), (com or sci).strip()))
            members.append(Member(r["label"].strip(), lin, rank, prompts))
        labels = list(dict.fromkeys(m.label for m in members))
        idx = {sci.lower(): m for m in members for sci, _ in m.prompts}
        return cls(members, labels, idx)

    # ---- resolution -------------------------------------------------------
    def resolve_lineage(self, lin: dict[str, str]) -> tuple[str | None, str]:
        """Return (label, relation) for a lineage dict (any ranks may be missing)."""
        lin = {k: (lin.get(k) or "").strip().lower() for k in RANKS}
        if lin["species"] and lin["genus"] and " " not in lin["species"]:
            lin["species"] = f"{lin['genus']} {lin['species']}"
        pred_rank = next((k for k in reversed(RANKS) if lin[k]), None)
        if pred_rank is None:
            return None, "none"
        # A species we list as a prompt resolves directly, even when the
        # lineage lacks the node's rank (e.g. no 'order' for a lagomorph).
        if lin["species"] in self.binomial_index:
            m = self.binomial_index[lin["species"]]
            return m.label, ("exact" if m.rank == "species" else "finer")
        best: tuple[str | None, str] = (None, "none")
        for m in self.members:
            node_rank = m.rank
            # finer or exact: prediction has the node's rank filled and it matches
            if lin[node_rank] and lin[node_rank] == m.lineage[node_rank] and \
                    all(not lin[k] or not m.lineage[k] or lin[k] == m.lineage[k]
                        for k in RANKS[:RANKS.index(node_rank)]):
                rel = "exact" if pred_rank == node_rank else "finer"
                return m.label, rel
            # coarser: prediction rank is above the node and the node sits under it
            if RANKS.index(pred_rank) < RANKS.index(node_rank) and \
                    m.lineage[pred_rank] == lin[pred_rank] and best[0] is None:
                best = (m.label, "coarser")
        return best

    def resolve_binomial(self, sci: str) -> tuple[str | None, str]:
        """Resolve a scientific name (binomial or genus) via prompt tables, then genus."""
        s = (sci or "").strip().lower()
        if not s:
            return None, "none"
        if s in self.binomial_index:
            m = self.binomial_index[s]
            return m.label, ("exact" if m.rank == "species" else "finer")
        parts = s.split()
        lin = {"genus": parts[0]}
        if len(parts) >= 2:
            lin["species"] = s
        return self.resolve_lineage(lin)

    def resolve(self, lineage: dict | None = None, scientific: str | None = None,
                common: str | None = None) -> tuple[str | None, str]:
        """Try lineage, then scientific name, then a common name equal to a label."""
        if lineage:
            lab, rel = self.resolve_lineage(lineage)
            if lab:
                return lab, rel
        if scientific:
            lab, rel = self.resolve_binomial(scientific)
            if lab:
                return lab, rel
        c = (common or "").strip().lower()
        if c in self.labels:
            return c, "exact"
        return None, "none"

    def pred_rank_name(self, lineage: dict | None, scientific: str | None = None) -> tuple[str, str] | None:
        """Deepest (rank, name) of a prediction, for ancestor checks."""
        lin = {k: ((lineage or {}).get(k) or "").strip().lower() for k in RANKS}
        if not any(lin.values()) and scientific:
            parts = scientific.strip().lower().split()
            if parts:
                lin["genus"] = parts[0]
                if len(parts) > 1:
                    lin["species"] = " ".join(parts[:2])
        if lin["species"] and lin["genus"] and " " not in lin["species"]:
            lin["species"] = f"{lin['genus']} {lin['species']}"
        for k in reversed(RANKS):
            if lin[k]:
                return k, lin[k]
        return None

    def is_ancestor(self, rank: str, name: str, label: str) -> bool:
        """True if taxon (rank, name) contains every member of `label`'s node."""
        ms = [m for m in self.members if m.label == label]
        return bool(ms) and all(m.lineage.get(rank) == name for m in ms) and \
            all(RANKS.index(rank) <= RANKS.index(m.rank) for m in ms)

    # ---- candidate lists for masking ---------------------------------------
    def bioclip_species_lines(self) -> list[str]:
        seen, out = set(), []
        for m in self.members:
            for sci, com in m.prompts:
                if sci not in seen:
                    seen.add(sci)
                    out.append(f"{sci} | {com}")
        return out

    def allowed(self, lineage: dict) -> bool:
        lab, rel = self.resolve_lineage(lineage)
        return lab is not None and rel in ("exact", "finer")

    def candidate_classes(self, lineages: dict[str, dict]) -> set[str]:
        """The subset of a model's classes that forms the shared candidate set.

        Applied identically to every model so all face the same decision:
          * a species-level class is kept if its binomial is a listed regional
            species (a vocabulary prompt); otherwise only if the model has no
            listed species for that node at all, so a coarse regional model's
            stand-in class (e.g. one cottontail for all rabbits) is not lost;
          * a class at the node's own rank or above it within the node is kept
            ("odocoileus species", "sciuridae family");
          * a class between the node and species level (a genus under a family
            node) is kept only if it contains a listed species, so
            "tamiasciurus species" stays and African squirrel genera go.
        """
        listed_genera = {b.split()[0] for b in self.binomial_index}
        info = {}
        for code, lin in lineages.items():
            n = {k: (lin.get(k) or "").strip().lower() for k in RANKS}
            if n["species"] and n["genus"] and " " not in n["species"]:
                n["species"] = f"{n['genus']} {n['species']}"
            lab, rel = self.resolve_lineage(n)
            if lab is None or rel not in ("exact", "finer"):
                continue
            info[code] = (n, lab, rel)
        has_listed = {lab for n, lab, _ in info.values() if n["species"] in self.binomial_index}
        keep = set()
        for code, (n, lab, rel) in info.items():
            if n["species"]:
                if n["species"] in self.binomial_index or lab not in has_listed:
                    keep.add(code)
            elif rel == "exact":
                keep.add(code)
            elif n["genus"] and n["genus"] in listed_genera:
                keep.add(code)
        return keep
