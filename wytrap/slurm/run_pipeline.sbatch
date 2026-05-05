#!/bin/bash
#SBATCH --job-name=wytrap
#SBATCH --account=wildimageproc
#SBATCH --partition=beartooth-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
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
#   NO_TILE=1                        # disable SAHI sliced detection (default: tiled)
#   TILE_SIZE=480                    # tile edge length in px (default: 480)
#   TILE_OVERLAP=0.2                 # fractional overlap between adjacent tiles
#   EXTRA_ARGS="--recursive --resume"
#
# Persistent log: $OUTPUT_DIR/wytrap.log is written in addition to logs/wytrap-$JOBID.out,
# so the run survives after SLURM cleanup.

set -euo pipefail
mkdir -p logs

# Shared model cache so each researcher doesn't re-download MegaDetector + BioCLIP weights.
export INPUT_DIR="/project/wildimageproc/software/annotations/YNP-BisonGraze/images"
export OUTPUT_DIR="/project/wildimageproc/omartin9/detection-projects/wytrap/output-pipeline"
export SPECIES="ynp_testbed"

VENV="${VENV:-/project/wildimageproc/omartin9/.venv-wytrap-dev}"
source "$VENV/bin/activate"

cd "$SLURM_SUBMIT_DIR"

: "${INPUT_DIR:?INPUT_DIR must be set}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

# Toggle --skip-classification-when-bad via SKIP_BAD=1.
SKIP_BAD_FLAG=""
if [[ "${SKIP_BAD:-0}" == "1" ]]; then
    SKIP_BAD_FLAG="--skip-classification-when-bad"
fi

# Toggle --no-tile via NO_TILE=1.
NO_TILE_FLAG=""
if [[ "${NO_TILE:-0}" == "1" ]]; then
    NO_TILE_FLAG="--no-tile"
fi

wytrap detect \
    --input  "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --species "${SPECIES:-wyoming_all}" \
    --det-threshold "${DET_THRESHOLD:-0.10}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --edge-margin-frac "${EDGE_MARGIN_FRAC:-0.01}" \
    --min-box-area-frac "${MIN_BOX_AREA_FRAC:-0.005}" \
    --max-aspect-ratio "${MAX_ASPECT_RATIO:-5.0}" \
    --tile-size "${TILE_SIZE:-480}" \
    --tile-overlap "${TILE_OVERLAP:-0.2}" \
    --device cuda \
    --recursive \
    --resume \
    $SKIP_BAD_FLAG \
    $NO_TILE_FLAG \
    ${EXTRA_ARGS:-}

echo "[wytrap.sbatch] complete. Persistent log: $OUTPUT_DIR/wytrap.log"
