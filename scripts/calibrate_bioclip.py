"""Zero-shot prior correction for BioCLIP (logit adjustment), label-free.

BioCLIP's softmax over a prompt list carries a built-in prior: some prompts
score high on almost any crop and absorb uninformative ones. This estimates
each prompt's baseline as its mean log-probability over a pool of unlabelled
crops and subtracts it (Wang et al. 2022, DebiasPL; Menon et al. 2021, logit
adjustment), separately for each crop scale (tight / padded / full).

Leave-location-out: the biases applied to images at camera location L are
estimated only from crops at other locations, so no image helps set its own
correction. Ground-truth labels are never read.

Input: a wytrap run whose records carry `prompt_logp` (wytrap >= this change)
and `prompts.json`. Output: a new folder of wytrap-format records with
recalibrated labels and a re-picked winning scale, scored by
scripts/eval_image_level.py as usual, plus `prompt_bias.json` estimated from
ALL locations for use with `wytrap detect --prompt-bias` at deployment.

Optional regional prior: --target-prior prior.json ({scientific: weight})
adds log(weight) back after removing the baseline (default: uniform).

Usage:
    python scripts/calibrate_bioclip.py --pred /path/output-bioclip-... \\
        --output /path/output-bioclip-...-calib [--min-det 0.2] [--quality ok|all]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "wytrap"))
from wytrap.io import DetectionRecord, ImageRecord, append_jsonl, save_record  # noqa: E402

SCALES = ("tight", "padded", "full")


def location_of(image_path: str) -> str:
    m = re.search(r"(loc_\d+)", image_path)
    return m.group(1) if m else Path(image_path).parent.name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="wytrap output folder with prompt_logp")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-det", type=float, default=0.2,
                    help="boxes below this det_score are left out of the bias pool")
    ap.add_argument("--quality", choices=["ok", "all"], default="all",
                    help="which boxes form the bias pool (default all: the pool is unlabelled "
                         "crops as a deployed camera would produce them)")
    ap.add_argument("--target-prior", default=None)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args(argv)

    pred = Path(args.pred)
    pj = json.loads((pred / "prompts.json").read_text())
    prompts = pj["prompts"]
    common = dict(pj.get("common", {}))
    # common names come from the records' topk (scientific -> common)
    recs = [json.loads(l) for l in open(pred / "all_records.jsonl") if l.strip()]
    for r in recs:
        for d in r["detections"]:
            for e in d.get("topk", []):
                common.setdefault(e["scientific"], e["common"])
    missing = sum(1 for r in recs for d in r["detections"]
                  if d.get("fine_label") and not d.get("prompt_logp"))
    if missing:
        raise SystemExit(f"{missing} classified boxes lack prompt_logp: re-run wytrap "
                         f"with the version that records all prompt scores")

    log_prior = np.zeros(len(prompts))
    if args.target_prior:
        w = json.loads(Path(args.target_prior).read_text())
        v = np.array([float(w.get(p, 1.0)) for p in prompts])
        log_prior = np.log(v / v.sum())

    # ---- pool: (location, scale) -> list of logp vectors
    pool: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        loc = location_of(r["image_path"])
        seen_full = False
        for d in r["detections"]:
            if d["det_score"] < args.min_det or not d.get("prompt_logp"):
                continue
            if args.quality == "ok" and d.get("quality", "ok") != "ok":
                continue
            for sc in SCALES:
                v = d["prompt_logp"].get(sc)
                if not v:
                    continue
                if sc == "full":          # one full-image vector per image
                    if seen_full:
                        continue
                    seen_full = True
                pool[loc][sc].append(v)
    locs = sorted(pool)
    sums = {sc: np.zeros(len(prompts)) for sc in SCALES}
    counts = {sc: 0 for sc in SCALES}
    per_loc_sum = {loc: {} for loc in locs}
    for loc in locs:
        for sc in SCALES:
            arr = np.asarray(pool[loc][sc]) if pool[loc][sc] else np.zeros((0, len(prompts)))
            per_loc_sum[loc][sc] = (arr.sum(0) if len(arr) else np.zeros(len(prompts)), len(arr))
            sums[sc] += per_loc_sum[loc][sc][0]
            counts[sc] += len(arr)

    def bias_excluding(loc: str, sc: str) -> np.ndarray:
        s, n = per_loc_sum.get(loc, {}).get(sc, (np.zeros(len(prompts)), 0))
        tot, cnt = sums[sc] - s, counts[sc] - n
        if cnt == 0:
            return np.zeros(len(prompts))
        b = tot / cnt
        return b - b.mean()          # only relative bias matters

    # ---- deployment bias from all locations (tight scale is the main one;
    # all three are saved)
    full_bias = {sc: ((sums[sc] / counts[sc]) - (sums[sc] / counts[sc]).mean()).tolist()
                 if counts[sc] else [0.0] * len(prompts) for sc in SCALES}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt_bias.json").write_text(json.dumps({
        "bias": dict(zip(prompts, [round(x, 4) for x in full_bias["tight"]])),
        "bias_by_scale": {sc: dict(zip(prompts, [round(x, 4) for x in v]))
                          for sc, v in full_bias.items()},
        "pool_counts": counts, "locations": len(locs),
        "source": str(pred)}, indent=1))
    order = np.argsort(-np.array(full_bias["tight"]))
    print("[calib] pool:", counts, "over", len(locs), "locations")
    print("[calib] largest baseline pulls (tight scale, log-prob relative to mean):")
    for i in order[:8]:
        print(f"         {full_bias['tight'][i]:+.2f}  {prompts[i]} ({common.get(prompts[i], '')})")
    print("[calib] weakest:")
    for i in order[-4:]:
        print(f"         {full_bias['tight'][i]:+.2f}  {prompts[i]} ({common.get(prompts[i], '')})")

    # ---- recalibrate every record
    jsonl = out / "all_records.jsonl"
    if jsonl.exists():
        jsonl.unlink()
    changed = total = 0
    for r in recs:
        loc = location_of(r["image_path"])
        dets = []
        for d in r["detections"]:
            plp = d.get("prompt_logp") or {}
            if not plp:
                dets.append(DetectionRecord(**d))
                continue
            best = None
            scale_scores = {}
            tops = {}
            for sc in SCALES:
                v = plp.get(sc)
                if not v:
                    continue
                a = np.asarray(v) - bias_excluding(loc, sc) + log_prior
                p = np.exp(a - a.max())
                p /= p.sum()
                i = int(p.argmax())
                scale_scores[sc] = round(float(p[i]), 4)
                tops[sc] = prompts[i]
                if best is None or p[i] > best[1][best[2]]:
                    best = (sc, p, i)
            sc, p, i = best
            idx = np.argsort(-p)[:args.topk]
            topk = [{"common": common.get(prompts[j], prompts[j]), "scientific": prompts[j],
                     "score": round(float(p[j]), 4)} for j in idx]
            new_label = common.get(prompts[i], prompts[i])
            total += 1
            changed += new_label != d.get("fine_label")
            dets.append(DetectionRecord(**{**d,
                "label": new_label, "fine_label": new_label, "scientific_label": prompts[i],
                "cls_score": round(float(p[i]), 4), "topk": topk, "scale": sc,
                "scale_scores": scale_scores,
                "cross_scale_agree": len(set(tops.values())) == 1}))
        rec = ImageRecord(image_path=r["image_path"], image_size=r["image_size"],
                          detections=dets, error=r.get("error"))
        rel = Path(r["image_path"])
        save_record(rec, out / rel.parent.name / (rel.stem + ".json"))
        append_jsonl(rec, jsonl)
    print(f"[calib] relabelled {changed} of {total} classified boxes -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
