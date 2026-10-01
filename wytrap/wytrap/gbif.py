"""Build a vocabulary from GBIF occurrence records: every camera-detectable
species recorded inside a region, with its lineage, as taxonomy/*.csv.

The region is a bounding box (default: Wyoming). GBIF is asked, per class
(Mammalia, Aves), which species have at least --min-records occurrences
with coordinates inside it since --since; each species' lineage and common
name then come from the GBIF backbone. "Camera-detectable" is a fixed rule
on the lineage, not a judgement per species:

    mammals  everything except bats, shrews and moles, and the small rodent
             families (mice, voles, pocket mice, jumping mice, pocket gophers)
             -- muskrat and beaver are kept by name
    birds    only families a trap on the ground registers: grouse and
             turkey, quail, corvids, raptors, vultures, owls, cranes, herons,
             waterfowl, pigeons

Domestic animals have few GBIF records and are added from a fixed list.

One node per species by default (--rank species); --rank genus or family
collapses siblings into one node named after the taxon, with the species
kept as prompts. Hierarchical evaluation and SpeciesNet's roll-up work the
same either way, so the rank is only a choice of what to report.

Usage:
    wytrap gbif --out taxonomy/wyoming_vocab.csv
    wytrap gbif --out taxonomy/site.csv --bbox -107.9,41.4,-107.3,42.0 --min-records 5
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

API = "https://api.gbif.org/v1"
CLASS_KEYS = {"Mammalia": 359, "Aves": 212}
# living records only: Wyoming's Eocene fossil beds would otherwise add
# sixty extinct families
BASIS = ["HUMAN_OBSERVATION", "MACHINE_OBSERVATION", "OBSERVATION", "PRESERVED_SPECIMEN",
         "MATERIAL_SAMPLE", "LIVING_SPECIMEN", "OCCURRENCE"]
WYOMING_BBOX = (-111.06, 41.0, -104.05, 45.0)

# ---- the camera-detectable rule --------------------------------------------
MAMMAL_EXCLUDE_ORDERS = {"chiroptera", "eulipotyphla", "soricomorpha"}
MAMMAL_EXCLUDE_FAMILIES = {"cricetidae", "muridae", "heteromyidae", "dipodidae", "zapodidae",
                           "geomyidae", "soricidae", "talpidae", "vespertilionidae"}
MAMMAL_KEEP_SPECIES = {"ondatra zibethicus"}        # muskrat: Cricetidae but very visible
BIRD_KEEP_FAMILIES = {"phasianidae", "odontophoridae", "corvidae", "accipitridae", "cathartidae",
                      "falconidae", "pandionidae", "strigidae", "tytonidae", "gruidae", "ardeidae",
                      "anatidae", "columbidae"}
# Bird families collapse to one node each (a trap sees "an owl", not which
# owl); game birds stay at species because sage-grouse vs turkey matters.
BIRD_FAMILY_LABEL = {"accipitridae": "hawk or eagle", "pandionidae": "hawk or eagle",
                     "falconidae": "falcon", "cathartidae": "vulture",
                     "strigidae": "owl", "tytonidae": "owl", "corvidae": "corvid",
                     "anatidae": "waterfowl", "gruidae": "crane", "ardeidae": "heron",
                     "columbidae": "dove"}
# GBIF backbone splits or renames folded back together: synonym -> (accepted
# binomial, common). The synonym stays on the row as an extra prompt, since
# classifiers may know the species by either name.
SYNONYMS = {"cervus elaphus": ("Cervus canadensis", "elk"),
            "alces americanus": ("Alces alces", "moose"),
            "martes caurina": ("Martes americana", "american marten"),
            "mustela vison": ("Neogale vison", "american mink"),
            "anas carolinensis": ("Anas crecca", "green-winged teal")}
# Species GBIF cannot count for us, included regardless of records:
# sensitive species whose coordinates are withheld (ferret), genera the
# backbone resolves only to genus level (Neogale: long-tailed weasel, mink),
# and rare residents with a handful of records (wolverine, lynx).
ALWAYS_INCLUDE = [  # (class, order, family, genus, species, common, [aliases])
    ("Mammalia", "Carnivora", "Mustelidae", "Mustela", "nigripes", "black-footed ferret", []),
    ("Mammalia", "Carnivora", "Mustelidae", "Mustela", "frenata", "long-tailed weasel", ["Neogale frenata"]),
    ("Mammalia", "Carnivora", "Mustelidae", "Neogale", "vison", "american mink", ["Mustela vison"]),
    ("Mammalia", "Carnivora", "Mustelidae", "Gulo", "gulo", "wolverine", []),
    ("Mammalia", "Carnivora", "Felidae", "Lynx", "canadensis", "canada lynx", []),
]
DOMESTIC = [  # (class, order, family, genus, species, common)
    ("Mammalia", "Artiodactyla", "Bovidae", "Bos", "taurus", "domestic cattle"),
    ("Mammalia", "Artiodactyla", "Bovidae", "Ovis", "aries", "domestic sheep"),
    ("Mammalia", "Artiodactyla", "Bovidae", "Capra", "hircus", "domestic goat"),
    ("Mammalia", "Perissodactyla", "Equidae", "Equus", "caballus", "horse"),
    ("Mammalia", "Carnivora", "Canidae", "Canis", "familiaris", "domestic dog"),
    ("Mammalia", "Carnivora", "Felidae", "Felis", "catus", "domestic cat"),
]


def _get(path: str, **params) -> dict:
    url = f"{API}/{path}" + ("?" + urllib.parse.urlencode(params, safe="(),", doseq=True) if params else "")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "wytrap"}),
                                        timeout=60) as r:
                return json.load(r)
        except Exception as e:                 # GBIF rate-limits and hiccups
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
            last = e
    raise RuntimeError(last)


def species_counts(class_key: int, bbox: tuple, since: int, state: str | None = None) -> dict[int, int]:
    """speciesKey -> occurrence count: records with coordinates inside bbox,
    or (when `state` is given) records labelled with that state whether or
    not they carry coordinates, whichever is larger. GBIF withholds the
    coordinates of sensitive species (black-footed ferret), so the bbox
    alone would miss them."""
    x1, y1, x2, y2 = bbox
    poly = f"POLYGON(({x1} {y1},{x2} {y1},{x2} {y2},{x1} {y2},{x1} {y1}))"   # counter-clockwise
    years = f"{since},{time.localtime().tm_year}"
    d = _get("occurrence/search", geometry=poly, classKey=class_key, hasCoordinate="true",
             year=years, occurrenceStatus="PRESENT", basisOfRecord=BASIS,
             facet="speciesKey", facetLimit=3000, limit=0)
    counts = {int(c["name"]): c["count"] for c in d["facets"][0]["counts"]}
    if state:
        d = _get("occurrence/search", stateProvince=state, classKey=class_key, year=years,
                 occurrenceStatus="PRESENT", basisOfRecord=BASIS,
                 facet="speciesKey", facetLimit=3000, limit=0)
        for c in d["facets"][0]["counts"]:
            k = int(c["name"])
            counts[k] = max(counts.get(k, 0), c["count"])
    return counts


def species_info(key: int) -> dict:
    d = _get(f"species/{key}")
    common = d.get("vernacularName")
    if not common:
        v = _get(f"species/{key}/vernacularNames", limit=100)
        eng = [x["vernacularName"] for x in v.get("results", []) if x.get("language") == "eng"]
        common = eng[0] if eng else d.get("canonicalName")
    return {"class": d.get("class", ""), "order": d.get("order", ""), "family": d.get("family", ""),
            "genus": d.get("genus", ""), "species": d.get("species") or d.get("canonicalName", ""),
            "common": (common or "").lower()}


def detectable(info: dict) -> bool:
    cls, order, fam = (info[k].lower() for k in ("class", "order", "family"))
    sp = info["species"].lower()
    if not (info["genus"] and sp and " " in sp):
        return False
    if "/" in info["common"]:          # eBird "slash" taxa (snow/ross's goose): not a species
        return False
    if cls == "mammalia":
        if sp in MAMMAL_KEEP_SPECIES:
            return True
        return order not in MAMMAL_EXCLUDE_ORDERS and fam not in MAMMAL_EXCLUDE_FAMILIES
    if cls == "aves":
        return fam in BIRD_KEEP_FAMILIES
    return False


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out", required=True, help="vocabulary CSV to write")
    ap.add_argument("--bbox", default=",".join(map(str, WYOMING_BBOX)),
                    help="west,south,east,north in decimal degrees (default: Wyoming)")
    ap.add_argument("--since", type=int, default=2000, help="earliest record year (default 2000)")
    ap.add_argument("--state", default="Wyoming",
                    help="also count records labelled with this state, coordinates or not "
                         "(default Wyoming; 'none' to use the bbox only)")
    ap.add_argument("--min-records", type=int, default=3,
                    help="occurrences needed to list a mammal (default 3)")
    ap.add_argument("--bird-min-records", type=int, default=200,
                    help="occurrences needed to list a bird (default 200; birders record "
                         "vagrants that a trap will never see)")
    ap.add_argument("--rank", choices=["species", "genus", "family"], default="species",
                    help="mammal node rank: one label per species (default), or siblings collapsed")
    ap.add_argument("--bird-rank", choices=["species", "family"], default="family",
                    help="bird node rank (default family: owl, hawk, waterfowl...; game birds "
                         "always stay at species)")
    ap.add_argument("--no-domestic", action="store_true", help="leave out the domestic animals")
    ap.add_argument("--name", default="Wyoming", help="region name for the file header")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    return run(ap.parse_args(argv))


def run(args: argparse.Namespace) -> int:
    bbox = tuple(float(x) for x in args.bbox.split(","))
    domestic_names = {f"{g} {s}".lower(): common for _, _, _, g, s, common in DOMESTIC}
    rows: list[dict] = []
    excluded: list[tuple[str, str, int]] = []
    state = None if args.state.lower() == "none" else args.state
    for cls, key in CLASS_KEYS.items():
        floor = args.min_records if cls == "Mammalia" else args.bird_min_records
        counts = species_counts(key, bbox, args.since, state)
        kept = {k: n for k, n in counts.items() if n >= floor}
        print(f"[gbif] {cls}: {len(counts)} species recorded, {len(kept)} with >= {floor} records")
        for k, n in sorted(kept.items(), key=lambda kv: -kv[1]):
            info = species_info(k)
            info["aliases"] = []
            syn = SYNONYMS.get(info["species"].lower())
            if syn:
                info["aliases"] = [info["species"]]
                info["species"], info["common"] = syn
                info["genus"] = syn[0].split()[0]
            if info["species"].lower() in domestic_names:
                info["common"] = domestic_names[info["species"].lower()]
            if detectable(info):
                dup = next((r for r in rows if r["species"] == info["species"]), None)
                if dup:                          # merged synonym
                    dup["records"] += n
                    dup["aliases"] = sorted(set(dup["aliases"] + info["aliases"]))
                else:
                    rows.append({**info, "records": n})
            elif cls == "Mammalia":
                excluded.append((info["species"], info["common"], n))
            time.sleep(0.05)
    have = {r["species"].lower() for r in rows}
    for c, o, f, g, s, common, aliases in ALWAYS_INCLUDE:
        if f"{g} {s}".lower() not in have:
            rows.append({"class": c, "order": o, "family": f, "genus": g, "species": f"{g} {s}",
                         "common": common, "records": 0, "aliases": aliases})
            print(f"[gbif] always-include: {common}")
    if not args.no_domestic:
        have = {r["species"].lower() for r in rows}
        for c, o, f, g, s, common in DOMESTIC:
            if f"{g} {s}".lower() not in have:
                rows.append({"class": c, "order": o, "family": f, "genus": g, "species": f"{g} {s}",
                             "common": common, "records": 0, "aliases": []})
    print(f"[gbif] {len(rows)} camera-detectable species")
    print("[gbif] mammals recorded but excluded by the rule:",
          ", ".join(f"{c or s} ({n})" for s, c, n in excluded) or "none")

    # ---- nodes: mammals at --rank, birds at --bird-rank (game birds at species)
    def node_key(r: dict) -> tuple[str, str]:
        """(group key, label)"""
        fam = r["family"].lower()
        if r["class"].lower() == "aves" and args.bird_rank == "family" and fam in BIRD_FAMILY_LABEL:
            lab = BIRD_FAMILY_LABEL[fam]
            return lab, lab
        if args.rank == "species" or r["class"].lower() == "aves":
            return r["species"], (r["common"] or r["species"].lower())
        return r[args.rank], r[args.rank].lower()
    groups: dict = defaultdict(list)
    label_of: dict = {}
    for r in rows:
        k, lab = node_key(r)
        groups[k].append(r)
        label_of[k] = lab
    # duplicate common names (GBIF occasionally) get the binomial appended
    seen: dict[str, int] = defaultdict(int)
    for k in groups:
        seen[label_of[k]] += 1
    for k in groups:
        if seen[label_of[k]] > 1:
            label_of[k] = f"{label_of[k]} ({k.lower()})"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {args.name} vocabulary from GBIF occurrences (bbox {args.bbox}, since {args.since},",
        f"# >= {args.min_records} records), one node per {args.rank}; generated {time.strftime('%Y-%m-%d')} by",
        f"# `wytrap gbif`. Camera-detectable rule and domestic additions: wytrap/gbif.py.",
        "# Edit freely; the GBIF record count per species is in the .records.json beside this file.",
        "label,class,order,family,genus,species,prompts",
    ]
    order_key = lambda r: (r["class"], r["order"], r["family"], r["genus"], r["species"])
    for k, members in sorted(groups.items(), key=lambda kv: order_key(kv[1][0])):
        # a label spanning two families (owl: Strigidae + Tytonidae) gets one
        # row per family, which the vocabulary format allows
        by_family = defaultdict(list)
        for m in members:
            by_family[m["family"]].append(m)
        for fam, ms in sorted(by_family.items()):
            m0 = ms[0]
            if len(ms) == 1 and args.rank == "species" and (m0["class"].lower() != "aves" or k == m0["species"]):
                lin = [m0["class"], m0["order"], fam, m0["genus"], m0["species"].split()[-1]]
            elif len({m["genus"] for m in ms}) == 1 and args.rank != "family":
                lin = [m0["class"], m0["order"], fam, m0["genus"], ""]
            else:
                lin = [m0["class"], m0["order"], fam, "", ""]
            prompts = ";".join(f"{m['species']}|{m['common'] or m['species']}"
                               + "".join(f";{a}|{m['common']}" for a in m.get("aliases", []))
                               for m in ms)
            lines.append(f"{label_of[k]},{','.join(lin)},{prompts}")
    out.write_text("\n".join(lines) + "\n")
    # occurrence counts beside the CSV: a rough abundance prior, and the
    # evidence behind each row when the list is edited by hand
    out.with_suffix(".records.json").write_text(json.dumps(
        {r["species"]: {"common": r["common"], "family": r["family"], "records": r["records"]}
         for r in sorted(rows, key=lambda r: -r["records"])}, indent=1))
    fam = sorted({r["family"] for r in rows})
    print(f"[gbif] wrote {out}: {len(groups)} nodes over {len(fam)} families")
    print("[gbif] families:", ", ".join(fam))
    return 0
