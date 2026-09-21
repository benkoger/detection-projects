"""Derive per-model candidate lists from a taxon-node vocabulary.

Writes, next to the vocabulary CSV:
    <vocab>.bioclip.txt      wytrap species file (one "Binomial | common" per node prompt)
    <vocab>.speciesnet.txt   SpeciesNet target-species file (full label strings whose
                             lineage falls inside a node), if --speciesnet-labels is given

and prints, for each AddaxAI zoo model given with --zoo, which of its classes
are inside the vocabulary (the mask run_addax_model_arm.py --vocab applies).

Usage:
    python scripts/build_vocab_lists.py taxonomy/idaho_vocab.csv \\
        [--speciesnet-labels /path/to/labels.txt | --speciesnet-model hf:Addax-Data-Science/SPECIESNET-v4-0-2-A] \\
        [--zoo Addax-Data-Science/WUSA-SDZWA-v1 ...]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from helpers.vocab import Vocab  # noqa: E402


def speciesnet_lineage(label: str) -> dict[str, str]:
    parts = label.split(";")
    if len(parts) != 7:
        return {}
    _, cls, order, family, genus, species, _ = parts
    return {"class": cls, "order": order, "family": family, "genus": genus, "species": species}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vocab")
    ap.add_argument("--speciesnet-labels", default=None)
    ap.add_argument("--speciesnet-model", default=None, help="hf:... or kaggle:... to fetch labels")
    ap.add_argument("--zoo", nargs="*", default=[])
    args = ap.parse_args(argv)

    vpath = Path(args.vocab)
    v = Vocab.load(vpath)
    lines = v.bioclip_species_lines()
    out = vpath.with_suffix(".bioclip.txt")
    out.write_text("# generated from %s: one prompt per vocabulary member species\n" % vpath.name
                   + "\n".join(lines) + "\n")
    print(f"[vocab] {len(v.labels)} labels, {len(v.members)} member taxa, "
          f"{len(lines)} BioCLIP prompts -> {out}")

    labels_path = args.speciesnet_labels
    if args.speciesnet_model and not labels_path:
        from speciesnet.utils import ModelInfo
        labels_path = ModelInfo(args.speciesnet_model).classifier_labels
    if labels_path:
        with open(labels_path, encoding="utf-8") as f:
            all_labels = [l.strip() for l in f if l.strip()]
        cand = v.candidate_classes({lab: speciesnet_lineage(lab) for lab in all_labels})
        keep = [lab for lab in all_labels if lab in cand]
        out = vpath.with_suffix(".speciesnet.txt")
        out.write_text("\n".join(keep) + "\n")
        by_label: dict[str, int] = {}
        for lab in keep:
            n, _ = v.resolve_lineage(speciesnet_lineage(lab))
            by_label[n] = by_label.get(n, 0) + 1
        print(f"[vocab] SpeciesNet targets: {len(keep)} labels -> {out}")
        print("        per node:", by_label)
        for lab in keep:
            print("          ", lab.split(";")[-1])
        missing = [l for l in v.labels if l not in by_label]
        if missing:
            print("        nodes with no SpeciesNet label:", missing)

    for repo in args.zoo:
        from huggingface_hub import snapshot_download
        d = Path(repo) if Path(repo).is_dir() else Path(snapshot_download(repo))
        with open(d / "taxonomy.csv", newline="", encoding="utf-8") as f:
            lins = {r["model_class"]: {k: r.get(k, "") for k in ("class", "order", "family", "genus", "species")}
                    for r in csv.DictReader(f)}
        cand = v.candidate_classes(lins)
        allowed = sorted(c for c in lins if c in cand)
        dropped = sorted(c for c in lins if c not in cand)
        print(f"[vocab] {repo}: {len(allowed)} classes inside vocabulary, {len(dropped)} masked")
        print("        kept:", allowed)
        print("        masked:", dropped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
