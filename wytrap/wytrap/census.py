"""Tally what a run (or several) saw, per camera, and compare it with a
vocabulary: which labels appear that the vocabulary lacks, which vocabulary
nodes never appear. The audit step for an unknown species distribution:
run a classifier unrestricted, census it, revise the vocabulary.

One image counts once, by its headline box (highest det_score among
quality-ok animal boxes) when that box's cls_score is at least --min-score.

Writes <out>/census.csv (camera x label counts), census_by_family.csv (the
same rolled up to family, the stable view when species calls between
look-alikes are shaky) and census.json; prints per-camera top labels with
an "outside vocabulary" mark, then the family table.

Usage:
    wytrap census --out /data/output/census \\
        --run C1N=/data/output/C1N_Shirley/speciesnet-open \\
        --run C3S=/data/output/C3S_Shirley/speciesnet-open \\
        --vocab taxonomy/wyoming_vocab.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from wytrap.vocab import Vocab


def headline(dets: list[dict]) -> dict | None:
    ok = [d for d in dets if d.get("det_label", "animal") == "animal"
          and d.get("quality", "ok") == "ok" and d.get("label") not in ("", "skipped")]
    return max(ok, key=lambda d: d["det_score"]) if ok else None


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR",
                    help="a run folder (its all_records.jsonl); NAME is the camera column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", default=None, help="vocabulary CSV to audit against")
    ap.add_argument("--min-score", type=float, default=0.5,
                    help="headline box cls_score floor for counting (default 0.5)")
    ap.add_argument("--top", type=int, default=25, help="labels printed per camera")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    return run(ap.parse_args(argv))


def run(args: argparse.Namespace) -> int:
    vocab = Vocab.load(args.vocab) if args.vocab else None
    counts: dict[str, Counter] = {}
    low_conf: Counter = Counter()
    sci_of: dict[str, str] = {}
    lineage_of: dict[str, dict] = {}
    for spec in args.run:
        name, d = spec.split("=", 1)
        c = Counter()
        n_img = n_animal = 0
        with open(Path(d) / "all_records.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                n_img += 1
                h = headline(r.get("detections") or [])
                if not h:
                    continue
                n_animal += 1
                if h["cls_score"] < args.min_score:
                    low_conf[name] += 1
                    continue
                c[h["label"]] += 1
                sci_of.setdefault(h["label"], h.get("scientific_label", ""))
                if h.get("lineage"):
                    lineage_of.setdefault(h["label"], h["lineage"])
        counts[name] = c
        print(f"[census] {name}: {n_img} images, {n_animal} with an animal, "
              f"{sum(c.values())} counted at cls_score >= {args.min_score}, "
              f"{low_conf[name]} below it")

    # Family level: species calls between look-alikes are the noisiest part of
    # any classifier, and roll-ups land at genus or family anyway, so this is
    # the stable view of what is present.
    def family_of(lab: str) -> str:
        lin = lineage_of.get(lab) or {}
        return (lin.get("family") or lin.get("order") or lin.get("class") or "").lower() \
            or f"({lab})"
    fam_counts = {n: Counter() for n in counts}
    for n, c in counts.items():
        for lab, k in c.items():
            fam_counts[n][family_of(lab)] += k
    families = sorted({f for c in fam_counts.values() for f in c},
                      key=lambda f: -sum(c[f] for c in fam_counts.values()))
    fam_members = defaultdict(set)
    for c in counts.values():
        for lab in c:
            fam_members[family_of(lab)].add(lab)

    labels = sorted({l for c in counts.values() for l in c}, key=lambda l: -sum(c[l] for c in counts.values()))
    status = {}
    for lab in labels:
        if vocab is None:
            status[lab] = ""
        else:
            node, rel = vocab.resolve(lineage=lineage_of.get(lab), scientific=sci_of.get(lab), common=lab)
            status[lab] = f"{node} ({rel})" if node else "OUTSIDE VOCABULARY"

    for name, c in counts.items():
        print(f"\n[census] {name}")
        for lab, n in c.most_common(args.top):
            mark = f"   <- {status[lab]}" if status.get(lab, "").startswith("OUTSIDE") else ""
            print(f"    {n:6d}  {lab:<32} {sci_of.get(lab, ''):<30}{mark}")

    for name, c in fam_counts.items():
        print(f"\n[census] {name} by family")
        for fam, n in c.most_common(args.top):
            members = ", ".join(sorted(fam_members[fam], key=lambda l: -counts[name][l])[:4])
            print(f"    {n:6d}  {fam:<20} {members}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "census_by_family.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "labels", *counts.keys(), "total"])
        for fam in families:
            row = [fam_counts[n][fam] for n in counts]
            w.writerow([fam, "; ".join(sorted(fam_members[fam])), *row, sum(row)])
    with open(out / "census.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "scientific", "vocab_node", *counts.keys(), "total"])
        for lab in labels:
            row = [counts[n][lab] for n in counts]
            w.writerow([lab, sci_of.get(lab, ""), status[lab], *row, sum(row)])
    summary = {"min_score": args.min_score, "vocab": args.vocab,
               "counts": {n: dict(c.most_common()) for n, c in counts.items()},
               "counts_by_family": {n: dict(c.most_common()) for n, c in fam_counts.items()},
               "below_min_score": dict(low_conf),
               "outside_vocabulary": [l for l in labels if status[l].startswith("OUTSIDE")]}
    if vocab is not None:
        seen_nodes = {status[l].split(" (")[0] for l in labels if not status[l].startswith("OUTSIDE")}
        summary["vocab_nodes_never_seen"] = [n for n in vocab.labels if n not in seen_nodes]
        print(f"\n[census] outside the vocabulary: {summary['outside_vocabulary'] or 'none'}")
        print(f"[census] vocabulary nodes never seen: {summary['vocab_nodes_never_seen'] or 'none'}")
    (out / "census.json").write_text(json.dumps(summary, indent=2))
    print(f"[census] wrote {out}/census.csv, census_by_family.csv and census.json")
    return 0
