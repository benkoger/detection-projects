#!/bin/bash
# Every classifier on the same images, the same MegaDetector boxes and the
# same candidate set (a taxon-node vocabulary), scored by the same evaluator.
#
#   sbatch scripts/ai4wy/compare_classifiers.sbatch
#
# Env-vars:
#   DATA_DIR      default /project/uwyo-0007/data/idaho-100pc (images/ + labels.json)
#   VOCAB         default taxonomy/idaho_vocab.csv
#   DETECTOR      default MDV1000-redwood (what AddaxAI Connect runs)
#   DET_THRESHOLD default 0.20 (record floor; the eval sweeps above it)
#   ZOO           space-separated AddaxAI zoo repos
#   ADMIN1        geofence state for the SpeciesNet ensemble (default ID)
#SBATCH --job-name=wytrap-compare
#SBATCH --account=uwyo-0007
#SBATCH --partition=gp-1,gp-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/wytrap-compare-%j.out
#SBATCH --error=logs/wytrap-compare-%j.err

set -euo pipefail
umask 002
mkdir -p logs

PROJECT_ROOT="${PROJECT_ROOT:-/project/uwyo-0007/software}"
VENV="${VENV:-$PROJECT_ROOT/.venv-wytrap}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/models/torch}"

DATA_DIR="${DATA_DIR:-/project/uwyo-0007/data/idaho-100pc}"
VOCAB="${VOCAB:-taxonomy/idaho_vocab.csv}"
DETECTOR="${DETECTOR:-MDV1000-redwood}"
ZOO="${ZOO:-Addax-Data-Science/WUSA-SDZWA-v1 Addax-Data-Science/SWUSA-SDZWA-v3}"
ADMIN1="${ADMIN1:-ID}"
VTAG=$(basename "$VOCAB" .csv)
DTAG=$(echo "$DETECTOR" | tr '[:upper:]' '[:lower:]')

source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"
echo "node: $(hostname)  vocab: $VOCAB  detector: $DETECTOR  data: $DATA_DIR"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

EVAL() {  # $1 = run dir
    wytrap eval --labels "$DATA_DIR/labels.json" --pred "$1" --out "$1/eval-$VTAG" \
        --vocab "$VOCAB" --det-threshold "${DET_THRESHOLD_CLS:-0.50}" \
        --det-sweep 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
}

echo "===== candidate lists ====="
wytrap vocab "$VOCAB" --speciesnet-model hf:Addax-Data-Science/SPECIESNET-v4-0-2-A --zoo $ZOO

echo "===== MegaDetector + BioCLIP 2 on the vocabulary prompts ====="
OUT_BC="$DATA_DIR/output-bioclip-$VTAG-$DTAG"
wytrap detect --input "$DATA_DIR/images" --output "$OUT_BC" --classifier bioclip \
    --vocab "$VOCAB" --detector "$DETECTOR" --det-threshold "${DET_THRESHOLD:-0.20}" \
    --keep-labels animal --device cuda
EVAL "$OUT_BC"
BOXES="$OUT_BC/all_records.jsonl"     # every other classifier reuses these boxes

echo "===== BioCLIP 2 + prior correction (leave-location-out, label-free) ====="
OUT_BCC="$OUT_BC-calib"
wytrap calibrate --pred "$OUT_BC" --output "$OUT_BCC"
EVAL "$OUT_BCC"

echo "===== SpeciesNet classifier, candidate set ====="
OUT_SNC="$DATA_DIR/output-speciesnet-$VTAG-$DTAG"
wytrap classify --records "$BOXES" --output "$OUT_SNC" --classifier speciesnet --vocab "$VOCAB"
EVAL "$OUT_SNC"

echo "===== SpeciesNet classifier, open set (all labels) ====="
OUT_SNO="$DATA_DIR/output-speciesnet-open-$DTAG"
wytrap classify --records "$BOXES" --output "$OUT_SNO" --classifier speciesnet
EVAL "$OUT_SNO"

echo "===== SpeciesNet + roll-up + geofence, candidate set ====="
OUT_SNE="$DATA_DIR/output-speciesnet-ens-$VTAG-$DTAG"
wytrap classify --records "$BOXES" --output "$OUT_SNE" --classifier speciesnet-ensemble \
    --vocab "$VOCAB" --admin1 "$ADMIN1"
EVAL "$OUT_SNE"

ROWS=("BioCLIP 2 [vocab]=$OUT_BC" "BioCLIP 2 + prior correction [vocab]=$OUT_BCC"
      "SpeciesNet [vocab]=$OUT_SNC" "SpeciesNet [open]=$OUT_SNO" "SpeciesNet + roll-up [vocab]=$OUT_SNE")
for M in $ZOO; do
    T=$(basename "$M" | tr '[:upper:]' '[:lower:]')
    OUT="$DATA_DIR/output-addax-$T-$VTAG-$DTAG"
    echo "===== $M, candidate set ====="
    wytrap classify --records "$BOXES" --output "$OUT" --classifier addax --model "$M" --vocab "$VOCAB"
    EVAL "$OUT"
    ROWS+=("$(basename "$M") [vocab]=$OUT")
done

python - "$VTAG" "${ROWS[@]}" <<'PY'
import json, sys, pathlib
vtag = sys.argv[1]
print(f"\n===== shared candidate set ({vtag}), hierarchical scoring, det floor 0.50 =====")
print(f"{'run':>44} {'n':>4} {'top1':>6} {'hier':>6} {'rollup':>7} {'outside':>8} {'top3':>6}  acc@cov>=0.8")
for arg in sys.argv[2:]:
    name, d = arg.split("=", 1)
    f = pathlib.Path(d) / f"eval-{vtag}" / "metrics.json"
    if not f.exists():
        print(f"{name:>44}  (no metrics)"); continue
    c = json.loads(f.read_text())["classification"]
    h = c.get("hierarchical", {}); acc = c.get("accuracy", {})
    cov = [r for r in c.get("accuracy_vs_coverage", []) if r["coverage"] >= 0.8]
    best = max(cov, key=lambda r: r["top1_accuracy"]) if cov else None
    print(f"{name:>44} {c.get('animal_images_with_detection', 0):4d} {acc.get('top1', 0):6.3f} "
          f"{h.get('top1_or_consistent_rollup', 0):6.3f} {h.get('rollup_rate', 0):7.3f} "
          f"{h.get('outside_vocabulary_rate', 0):8.3f} {acc.get('top3', 0):6.3f}  "
          + (f"{best['top1_accuracy']:.3f} @ {best['coverage']:.2f}" if best else "-"))
print("\ntop1 = correct node; hier = correct node or roll-up to an ancestor of it;")
print("rollup = share predicted above node rank; outside = share outside the vocabulary.")
PY
echo "[compare_classifiers] done"
