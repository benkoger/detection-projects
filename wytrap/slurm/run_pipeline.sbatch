#!/bin/bash
#SBATCH --job-name=wytrap
#SBATCH --account=wildimageproc
#SBATCH --partition=beartooth-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/wytrap-%j.out
#SBATCH --error=logs/wytrap-%j.err

# Submit with:
#   INPUT_DIR=/path/to/photos OUTPUT_DIR=/path/to/results sbatch run_pipeline.sbatch
#
# Optional env-vars:
#   SPECIES=wyoming_mammals          # default: wyoming_all
#   DET_THRESHOLD=0.2
#   BATCH_SIZE=8
#   EDGE_MARGIN_FRAC=0.01
#   MIN_BOX_AREA_FRAC=0.005
#   MAX_ASPECT_RATIO=5.0
#   SKIP_BAD=1                       # opt in to --skip-classification-when-bad
#   EXTRA_ARGS="--recursive --resume"
#
# Persistent log: $OUTPUT_DIR/wytrap.log is written in addition to logs/wytrap-$JOBID.out,
# so the run survives after SLURM cleanup.

set -euo pipefail
mkdir -p logs

# Shared model cache so each researcher doesn't re-download MegaDetector + BioCLIP weights.
export HF_HOME="${HF_HOME:-/project/wildimageproc/software/models}"
export TORCH_HOME="${TORCH_HOME:-/project/wildimageproc/software/models/torch}"

VENV="${VENV:-/project/wildimageproc/software/.wildimageproc-venv}"
source "$VENV/bin/activate"

cd "$SLURM_SUBMIT_DIR"

: "${INPUT_DIR:?INPUT_DIR must be set}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

# Toggle --skip-classification-when-bad via SKIP_BAD=1.
SKIP_BAD_FLAG=""
if [[ "${SKIP_BAD:-0}" == "1" ]]; then
    SKIP_BAD_FLAG="--skip-classification-when-bad"
fi

wytrap detect \
    --input  "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --species "${SPECIES:-wyoming_all}" \
    --det-threshold "${DET_THRESHOLD:-0.2}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --edge-margin-frac "${EDGE_MARGIN_FRAC:-0.01}" \
    --min-box-area-frac "${MIN_BOX_AREA_FRAC:-0.005}" \
    --max-aspect-ratio "${MAX_ASPECT_RATIO:-5.0}" \
    --device cuda \
    --recursive \
    --resume \
    $SKIP_BAD_FLAG \
    ${EXTRA_ARGS:-}

echo "[wytrap.sbatch] complete. Persistent log: $OUTPUT_DIR/wytrap.log"
