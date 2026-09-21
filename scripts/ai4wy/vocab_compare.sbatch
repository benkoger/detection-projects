#!/bin/bash
# Every classifier restricted to ONE shared candidate set, derived from a
# taxon-node vocabulary, on the same images and the same detector boxes,
# scored hierarchically by the same eval. Also re-scores the open-set runs
# already on disk with the same vocabulary, so both regimes sit side by side.
#
#   sbatch scripts/ai4wy/vocab_compare.sbatch
#
# Env-vars:
#   DATA_DIR   default /project/uwyo-0007/data/idaho-20pc
#   VOCAB      default taxonomy/idaho_vocab.csv
#   DETECTOR   default MDV1000-redwood (what AddaxAI Connect runs)
#   DET_THRESHOLD default 0.20 (inference floor; eval sweeps above it)
#   ZOO        space-separated AddaxAI zoo repos
#   SN_MODEL   default hf:Addax-Data-Science/SPECIESNET-v4-0-2-A
#SBATCH --job-name=wytrap-vocab
#SBATCH --account=uwyo-0007
#SBATCH --partition=gp-1,gp-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/wytrap-vocab-%j.out
#SBATCH --error=logs/wytrap-vocab-%j.err

set -euo pipefail
umask 002
mkdir -p logs

PROJECT_ROOT="${PROJECT_ROOT:-/project/uwyo-0007/software}"
VENV="${VENV:-$PROJECT_ROOT/.venv-wytrap}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/models/torch}"

DATA_DIR="${DATA_DIR:-/project/uwyo-0007/data/idaho-20pc}"
VOCAB="${VOCAB:-taxonomy/idaho_vocab.csv}"
DETECTOR="${DETECTOR:-MDV1000-redwood}"
SN_MODEL="${SN_MODEL:-hf:Addax-Data-Science/SPECIESNET-v4-0-2-A}"
ZOO="${ZOO:-Addax-Data-Science/WUSA-SDZWA-v1 Addax-Data-Science/SWUSA-SDZWA-v3}"
VTAG=$(basename "$VOCAB" .csv)
DTAG=$(echo "$DETECTOR" | tr '[:upper:]' '[:lower:]')

source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"
echo "node: $(hostname)  vocab: $VOCAB  detector: $DETECTOR"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

EVAL() {  # $1 = prediction dir
    python scripts/eval_image_level.py --labels "$DATA_DIR/labels.json" --pred "$1" \
        --out "$1/eval-$VTAG" --vocab "$VOCAB" \
        --det-threshold "${DET_THRESHOLD_CLS:-0.50}" --det-sweep 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
}

echo "===== candidate lists ====="
python scripts/build_vocab_lists.py "$VOCAB" --speciesnet-model "$SN_MODEL" --zoo $ZOO

echo "===== BioCLIP 2 (wytrap) on the vocabulary prompts ====="
OUT_BC="$DATA_DIR/output-bioclip-$VTAG-$DTAG"
# Records from before per-prompt scores were recorded cannot be calibrated;
# start that run over (resume would otherwise keep them).
if [[ -d "$OUT_BC" && ! -f "$OUT_BC/prompts.json" ]]; then
    echo "removing $OUT_BC (predates prompt_logp)"; rm -rf "$OUT_BC"
fi
wytrap detect --input "$DATA_DIR/images" --output "$OUT_BC" \
    --species "${VOCAB%.csv}.bioclip.txt" --detector "$DETECTOR" \
    --det-threshold "${DET_THRESHOLD:-0.20}" --device cuda --recursive --resume \
    --jsonl "$OUT_BC/all_records.jsonl"
EVAL "$OUT_BC"
BOXES="$OUT_BC/all_records.jsonl"   # every other arm reuses these boxes

echo "===== BioCLIP 2 + prior correction (leave-location-out, label-free) ====="
OUT_BCC="$OUT_BC-calib"
python scripts/calibrate_bioclip.py --pred "$OUT_BC" --output "$OUT_BCC"
EVAL "$OUT_BCC"

echo "===== SpeciesNet classifier, candidate set ====="
OUT_SNC="$DATA_DIR/output-speciesnet-clf-$VTAG-$DTAG"
python scripts/run_speciesnet_arms.py classifier --wytrap-records "$BOXES" \
    --output "$OUT_SNC" --model "$SN_MODEL" --vocab "$VOCAB"
EVAL "$OUT_SNC"

echo "===== SpeciesNet classifier, open set (all 2498 labels) ====="
OUT_SNO="$DATA_DIR/output-speciesnet-clf-open-$DTAG"
python scripts/run_speciesnet_arms.py classifier --wytrap-records "$BOXES" \
    --output "$OUT_SNO" --model "$SN_MODEL"
EVAL "$OUT_SNO"

echo "===== SpeciesNet classifier + roll-up, candidate set ====="
OUT_SNE="$DATA_DIR/output-speciesnet-ens-$VTAG-$DTAG"
python scripts/run_speciesnet_arms.py ensemble --images "$DATA_DIR/images" \
    --output "$OUT_SNE" --model "$SN_MODEL" --vocab "$VOCAB" --detections-from "$BOXES"
EVAL "$OUT_SNE"

ZOO_OUTS=()
for M in $ZOO; do
    T=$(basename "$M" | tr '[:upper:]' '[:lower:]')
    OUT="$DATA_DIR/output-addax-$T-$VTAG-$DTAG"
    echo "===== $M, candidate set ====="
    python scripts/run_addax_model_arm.py --model "$M" --wytrap-records "$BOXES" \
        --output "$OUT" --vocab "$VOCAB"
    EVAL "$OUT"
    ZOO_OUTS+=("$(basename "$M") [vocab]=$OUT")
done

echo "===== re-scoring the open-set runs on disk with $VOCAB ====="
OPEN=()
# (the earlier output-speciesnet-clf-* dirs predate the per-box keying fix
#  and are replaced by $OUT_SNO above)
for d in output-species_idaho-redwood020 \
         output-speciesnet-ens-species_idaho-redwood020 output-speciesnet-ens \
         output-addax-wusa-sdzwa-v1-species_idaho-redwood020 output-addax-swusa-sdzwa-v3-species_idaho-redwood020 \
         output-wyoming_all; do
    if [[ -f "$DATA_DIR/$d/all_records.jsonl" ]]; then
        EVAL "$DATA_DIR/$d" > /dev/null
        OPEN+=("$d [open]=$DATA_DIR/$d")
    fi
done

python - "$VTAG" \
    "BioCLIP 2 [vocab prompts]=$OUT_BC" "BioCLIP 2 + prior correction [vocab]=$OUT_BCC" \
    "SpeciesNet clf [vocab]=$OUT_SNC" "SpeciesNet clf [open]=$OUT_SNO" \
    "SpeciesNet clf+rollup [vocab]=$OUT_SNE" "${ZOO_OUTS[@]}" "${OPEN[@]}" <<'PY'
import json, sys, pathlib
vtag = sys.argv[1]
print(f"\n===== shared candidate set ({vtag}), hierarchical scoring, det floor 0.50 =====")
print(f"{'run':>58} {'n':>4} {'top1':>6} {'hier':>6} {'rollup':>7} {'outside':>8} {'top3':>6}  acc@cov>=0.8")
for arg in sys.argv[2:]:
    name, d = arg.split("=", 1)
    f = pathlib.Path(d) / f"eval-{vtag}" / "metrics.json"
    if not f.exists():
        print(f"{name:>58}  (no metrics)"); continue
    c = json.loads(f.read_text())["classification"]
    h = c.get("hierarchical", {}); acc = c.get("accuracy", {})
    cov = [r for r in c.get("accuracy_vs_coverage", []) if r["coverage"] >= 0.8]
    best = max(cov, key=lambda r: r["top1_accuracy"]) if cov else None
    print(f"{name:>58} {c.get('animal_images_with_detection', 0):4d} {acc.get('top1', 0):6.3f} "
          f"{h.get('top1_or_consistent_rollup', 0):6.3f} {h.get('rollup_rate', 0):7.3f} "
          f"{h.get('outside_vocabulary_rate', 0):8.3f} {acc.get('top3', 0):6.3f}  "
          + (f"{best['top1_accuracy']:.3f} @ {best['coverage']:.2f}" if best else "-"))
print("\ntop1 = correct node; hier = correct node or roll-up to an ancestor of it;")
print("rollup = share predicted above node rank; outside = share outside the vocabulary.")
PY
echo "[vocab_compare] done"
