#!/bin/bash
# Inference only, no labels: MegaDetector once, then the three classifiers
# (BioCLIP 2 with prior correction, SpeciesNet with roll-up, Western USA
# SDZWA) on the same boxes, restricted to one shared candidate set, merged
# into one side-by-side table per image and per box.
#
# One folder per job. For a tree of camera folders use deploy_submit.sh,
# which submits this as a Slurm array with one task per camera folder.
#
#   IMAGES=/project/uwyo-0007/data/wysoundscape/images/CAM01 \
#       sbatch scripts/ai4wy/deploy_infer.sbatch
#
# Env-vars:
#   IMAGES        folder of images (walked recursively)          [required,
#                 or FOLDERS_FILE + SLURM_ARRAY_TASK_ID, see deploy_submit.sh]
#   OUT_ROOT      where run folders go; default <IMAGES parent>/output-<basename>
#                 (or OUT_ROOT_BASE/<basename> when set by deploy_submit.sh)
#   VOCAB         default taxonomy/wyoming_vocab.csv
#   DETECTOR      default MDV1000-redwood (what AddaxAI Connect runs)
#   DET_THRESHOLD default 0.20 (record floor; merge_predictions --min-det raises it)
#   PROMPT_BIAS   BioCLIP prior-correction file; default: the one estimated
#                 on Idaho. "none" disables; "self" re-estimates on this folder
#                 (needs several camera sub-folders to be meaningful).
#   SN_MODEL      default hf:Addax-Data-Science/SPECIESNET-v4-0-2-A
#   ZOO           default Addax-Data-Science/WUSA-SDZWA-v1
#SBATCH --job-name=wytrap-deploy
#SBATCH --account=uwyo-0007
#SBATCH --partition=gp-1,gp-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/wytrap-deploy-%A_%a.out
#SBATCH --error=logs/wytrap-deploy-%A_%a.err

set -euo pipefail
umask 002
mkdir -p logs

PROJECT_ROOT="${PROJECT_ROOT:-/project/uwyo-0007/software}"
VENV="${VENV:-$PROJECT_ROOT/.venv-wytrap}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/models/torch}"

if [[ -z "${IMAGES:-}" && -n "${FOLDERS_FILE:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    IMAGES=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$FOLDERS_FILE")
fi
[[ -n "${IMAGES:-}" && -d "$IMAGES" ]] || { echo "IMAGES must be an existing folder (got '${IMAGES:-}')"; exit 2; }
IMAGES=$(realpath "$IMAGES")
NAME=$(basename "$IMAGES")
# array tasks (deploy_submit.sh) land in OUT_ROOT_BASE/<camera>; a single
# job defaults to a sibling folder of IMAGES
OUT_ROOT="${OUT_ROOT:-${OUT_ROOT_BASE:+$OUT_ROOT_BASE/$NAME}}"
OUT_ROOT="${OUT_ROOT:-$(dirname "$IMAGES")/output-$NAME}"
VOCAB="${VOCAB:-taxonomy/wyoming_vocab.csv}"
DETECTOR="${DETECTOR:-MDV1000-redwood}"
DET_THRESHOLD="${DET_THRESHOLD:-0.20}"
SN_MODEL="${SN_MODEL:-hf:Addax-Data-Science/SPECIESNET-v4-0-2-A}"
ZOO="${ZOO:-Addax-Data-Science/WUSA-SDZWA-v1}"
PROMPT_BIAS="${PROMPT_BIAS:-/project/uwyo-0007/data/idaho-100pc/output-bioclip-idaho_vocab-mdv1000-redwood-calib/prompt_bias.json}"
VTAG=$(basename "$VOCAB" .csv)
DTAG=$(echo "$DETECTOR" | tr '[:upper:]' '[:lower:]')

source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"
mkdir -p "$OUT_ROOT"
echo "node: $(hostname)  images: $IMAGES  out: $OUT_ROOT  vocab: $VOCAB  detector: $DETECTOR"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "===== candidate lists ====="
python scripts/build_vocab_lists.py "$VOCAB" --speciesnet-model "$SN_MODEL" --zoo $ZOO

echo "===== MegaDetector ($DETECTOR) + BioCLIP 2 ====="
OUT_BC="$OUT_ROOT/bioclip-$VTAG-$DTAG"
BIAS_ARGS=()
if [[ "$PROMPT_BIAS" != "none" && "$PROMPT_BIAS" != "self" ]]; then
    [[ -f "$PROMPT_BIAS" ]] || { echo "PROMPT_BIAS file not found: $PROMPT_BIAS"; exit 2; }
    BIAS_ARGS=(--prompt-bias "$PROMPT_BIAS")
fi
wytrap detect --input "$IMAGES" --output "$OUT_BC" \
    --species "${VOCAB%.csv}.bioclip.txt" --detector "$DETECTOR" \
    --det-threshold "$DET_THRESHOLD" --keep-labels animal,person,vehicle \
    --device cuda --recursive --resume "${BIAS_ARGS[@]}" \
    --jsonl "$OUT_BC/all_records.jsonl"
BOXES="$OUT_BC/all_records.jsonl"   # every other arm reuses these boxes
BC_ARM="$OUT_BC"
if [[ "$PROMPT_BIAS" == "self" ]]; then
    echo "===== BioCLIP 2 prior correction estimated on this folder ====="
    BC_ARM="$OUT_BC-calib"
    python scripts/calibrate_bioclip.py --pred "$OUT_BC" --output "$BC_ARM"
fi

echo "===== SpeciesNet + roll-up, candidate set ====="
OUT_SN="$OUT_ROOT/speciesnet-ens-$VTAG-$DTAG"
python scripts/run_speciesnet_arms.py ensemble --images "$IMAGES" \
    --output "$OUT_SN" --model "$SN_MODEL" --vocab "$VOCAB" --detections-from "$BOXES"

ARMS=("bioclip=$BC_ARM" "speciesnet=$OUT_SN")
for M in $ZOO; do
    T=$(basename "$M" | tr '[:upper:]' '[:lower:]')
    OUT="$OUT_ROOT/addax-$T-$VTAG-$DTAG"
    echo "===== $M, candidate set ====="
    python scripts/run_addax_model_arm.py --model "$M" --wytrap-records "$BOXES" \
        --output "$OUT" --vocab "$VOCAB"
    ARMS+=("${T%%-*}=$OUT")
done

echo "===== merge ====="
MERGE_ARGS=()
for a in "${ARMS[@]}"; do MERGE_ARGS+=(--arm "$a"); done
python scripts/merge_predictions.py --out "$OUT_ROOT/merged" "${MERGE_ARGS[@]}"
echo "[deploy_infer] done: $OUT_ROOT/merged/images.csv"
