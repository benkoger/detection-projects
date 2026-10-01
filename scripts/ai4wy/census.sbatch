#!/bin/bash
# Species audit for cameras whose deploy_infer job has run: SpeciesNet
# UNRESTRICTED (all 2,498 labels, only its state geofence) on the boxes the
# deploy job already produced, then one census table of what each camera
# saw and which of those labels the vocabulary lacks.
#
#   sbatch scripts/ai4wy/census.sbatch                       # every camera with a deploy run
#   CAMERAS=shirley sbatch scripts/ai4wy/census.sbatch       # only cameras matching (case-insensitive)
#
# To queue it behind running deploy jobs:
#   sbatch --dependency=afterany:$(squeue -u $USER -h -n wytrap-deploy -o %i | paste -sd:) \
#       scripts/ai4wy/census.sbatch
#
# Env-vars:
#   DATA_DIR   default /project/uwyo-0007/data/CameraTrap_test (ingested layout)
#   CAMERAS    substring filter on camera names (default: all)
#   VOCAB      default taxonomy/wyoming_vocab.csv (the one the deploy run used)
#   DETECTOR   default MDV1000-redwood (to find the deploy run's records)
#   ADMIN1     geofence state (default WY)
#   MIN_SCORE  census counts headline boxes at or above this cls_score (default 0.5)
#   MIN_DET    ... and at or above this detector score (default 0.5)
#SBATCH --job-name=wytrap-census
#SBATCH --account=uwyo-0007
#SBATCH --partition=gp-1,gp-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/wytrap-census-%j.out
#SBATCH --error=logs/wytrap-census-%j.err

set -euo pipefail
umask 002
mkdir -p logs

PROJECT_ROOT="${PROJECT_ROOT:-/project/uwyo-0007/software}"
VENV="${VENV:-$PROJECT_ROOT/.venv-wytrap}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/models/torch}"

DATA_DIR="${DATA_DIR:-/project/uwyo-0007/data/CameraTrap_test}"
CAMERAS="${CAMERAS:-}"
VOCAB="${VOCAB:-taxonomy/wyoming_vocab.csv}"
DETECTOR="${DETECTOR:-MDV1000-redwood}"
ADMIN1="${ADMIN1:-WY}"
VTAG=$(basename "$VOCAB" .csv)
DTAG=$(echo "$DETECTOR" | tr '[:upper:]' '[:lower:]')
OUT="$DATA_DIR/output"

source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"
echo "node: $(hostname)  data: $DATA_DIR  cameras: ${CAMERAS:-all}  vocab: $VOCAB"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# ---- which cameras: those with a finished deploy run (its records file exists)
RUNS=()
for d in "$OUT"/*/; do
    CAM=$(basename "$d")
    [[ -n "$CAMERAS" && "${CAM,,}" != *"${CAMERAS,,}"* ]] && continue
    BOXES="$d/bioclip-$VTAG-$DTAG/all_records.jsonl"
    [[ -f "$BOXES" ]] || { echo "[census] $CAM: no deploy records yet, skipping"; continue; }

    # ---- SpeciesNet on the deploy run's boxes, no vocabulary: whatever it
    # can name, it names. The geofence still drops species not known in WY.
    echo "===== $CAM: SpeciesNet unrestricted ====="
    wytrap classify --records "$BOXES" --output "$d/speciesnet-open" \
        --classifier speciesnet-ensemble --admin1 "$ADMIN1" --device cuda
    RUNS+=(--run "$CAM=$d/speciesnet-open")
done
[[ ${#RUNS[@]} -gt 0 ]] || { echo "[census] no cameras to audit under $OUT"; exit 1; }

# ---- the census: per-camera label counts, marked against the vocabulary
echo "===== census ====="
wytrap census --out "$OUT/census${CAMERAS:+-$CAMERAS}" --vocab "$VOCAB" \
    --min-score "${MIN_SCORE:-0.5}" --min-det "${MIN_DET:-0.5}" "${RUNS[@]}"

# ---- the combined tables and the agreement summary over every camera
# that has a merged/ folder (all of them, if deploy_cameras.sh ran this)
echo "===== combined tables ====="
wytrap merge --combine "$OUT"
echo "[census] done"
