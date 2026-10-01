#!/bin/bash
# Inference only, no labels: MegaDetector once, then the three classifiers
# (BioCLIP 2 with prior correction, SpeciesNet with roll-up and geofence,
# Western USA SDZWA) on the same boxes and the same candidate set, merged
# into one side-by-side table per image and per box.
#
# One folder per job. For a tree of camera folders use deploy_submit.sh,
# which submits this as a Slurm array with one task per camera folder.
#
#   IMAGES=/project/uwyo-0007/data/CameraTrap_test/CAM01 \
#       sbatch scripts/ai4wy/deploy_infer.sbatch
#
# Env-vars:
#   IMAGES        folder of images (walked recursively)          [required,
#                 or FOLDERS_FILE + SLURM_ARRAY_TASK_ID, see deploy_submit.sh]
#   OUT_ROOT      where run folders go; default <IMAGES parent>/output-<basename>
#                 (or OUT_ROOT_BASE/<basename> when set by deploy_submit.sh)
#   VOCAB         default taxonomy/wyoming_vocab.csv
#   DETECTOR      default MDV1000-redwood (what AddaxAI Connect runs)
#   DET_THRESHOLD default 0.20 (record floor; every box above it is kept on disk)
#   MIN_DET       detector floor for the merged tables (default 0.5; raise or lower
#                 later with `wytrap merge --min-det` without re-running anything)
#   BOXES_FROM    all_records.jsonl of an earlier run on the same images: reuse
#                 its MegaDetector boxes and only re-run the classifiers (the
#                 cheap way to try a revised vocabulary)
#   PROMPT_BIAS   BioCLIP prior-correction file; default: the one estimated
#                 on Idaho. "none" disables; "self" re-estimates on this folder
#                 (needs several camera sub-folders to be meaningful).
#   ZOO           default Addax-Data-Science/WUSA-SDZWA-v1
#   ADMIN1        geofence state for the SpeciesNet ensemble (default WY)
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
OUT_ROOT="${OUT_ROOT:-${OUT_ROOT_BASE:+$OUT_ROOT_BASE/$NAME}}"
OUT_ROOT="${OUT_ROOT:-$(dirname "$IMAGES")/output-$NAME}"
VOCAB="${VOCAB:-taxonomy/wyoming_vocab.csv}"
DETECTOR="${DETECTOR:-MDV1000-redwood}"
DET_THRESHOLD="${DET_THRESHOLD:-0.20}"
ZOO="${ZOO:-Addax-Data-Science/WUSA-SDZWA-v1}"
ADMIN1="${ADMIN1:-WY}"
PROMPT_BIAS="${PROMPT_BIAS:-/project/uwyo-0007/data/idaho-100pc/output-bioclip-idaho_vocab-mdv1000-redwood-calib/prompt_bias.json}"
VTAG=$(basename "$VOCAB" .csv)
DTAG=$(echo "$DETECTOR" | tr '[:upper:]' '[:lower:]')

source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"
mkdir -p "$OUT_ROOT"
echo "node: $(hostname)  images: $IMAGES  out: $OUT_ROOT  vocab: $VOCAB  detector: $DETECTOR"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "===== MegaDetector ($DETECTOR) + BioCLIP 2 ====="
OUT_BC="$OUT_ROOT/bioclip-$VTAG-$DTAG"
BIAS_ARGS=()
if [[ "$PROMPT_BIAS" != "none" && "$PROMPT_BIAS" != "self" ]]; then
    [[ -f "$PROMPT_BIAS" ]] || { echo "PROMPT_BIAS file not found: $PROMPT_BIAS"; exit 2; }
    BIAS_ARGS=(--prompt-bias "$PROMPT_BIAS")
fi
if [[ -n "${BOXES_FROM:-}" ]]; then
    [[ -f "$BOXES_FROM" ]] || { echo "BOXES_FROM not found: $BOXES_FROM"; exit 2; }
    echo "reusing boxes from $BOXES_FROM (no detection pass)"
    wytrap classify --records "$BOXES_FROM" --output "$OUT_BC" --classifier bioclip \
        --vocab "$VOCAB" --device cuda "${BIAS_ARGS[@]}"
else
    wytrap detect --input "$IMAGES" --output "$OUT_BC" --classifier bioclip --vocab "$VOCAB" \
        --detector "$DETECTOR" --det-threshold "$DET_THRESHOLD" --device cuda "${BIAS_ARGS[@]}"
fi
BOXES="$OUT_BC/all_records.jsonl"     # every other classifier reuses these boxes
BC_ARM="$OUT_BC"
if [[ "$PROMPT_BIAS" == "self" ]]; then
    echo "===== BioCLIP 2 prior correction estimated on this folder ====="
    BC_ARM="$OUT_BC-calib"
    wytrap calibrate --pred "$OUT_BC" --output "$BC_ARM"
fi

echo "===== SpeciesNet + roll-up + geofence ====="
OUT_SN="$OUT_ROOT/speciesnet-ens-$VTAG-$DTAG"
wytrap classify --records "$BOXES" --output "$OUT_SN" --classifier speciesnet-ensemble \
    --vocab "$VOCAB" --admin1 "$ADMIN1"

ARMS=(--arm "bioclip=$BC_ARM" --arm "speciesnet=$OUT_SN")
for M in $ZOO; do
    T=$(basename "$M" | tr '[:upper:]' '[:lower:]')
    OUT="$OUT_ROOT/addax-$T-$VTAG-$DTAG"
    echo "===== $M ====="
    wytrap classify --records "$BOXES" --output "$OUT" --classifier addax --model "$M" --vocab "$VOCAB"
    ARMS+=(--arm "${T%%-*}=$OUT")
done

echo "===== merge ====="
wytrap merge --out "$OUT_ROOT/merged" --min-det "${MIN_DET:-0.5}" "${ARMS[@]}"
echo "[deploy_infer] done: $OUT_ROOT/merged/images.csv"
