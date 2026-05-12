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
#   DET_THRESHOLD=0.50               # MegaDetector confidence cutoff (default 0.50)
#                                    # Records produced at this floor let eval
#                                    # simulate any det_threshold >= floor via
#                                    # scripts/eval_pipeline.py --sweep-det
#                                    # without re-running inference.
#   BATCH_SIZE=8
#   MIN_PIXEL_SIDE=60                # drop boxes with min(w,h) < 60 px
#   BORDER_OVERLAP_TRUNC=0.20        # mark as 'truncated' if perim overlap >= 20%
#   MAX_ASPECT_RATIO=8.0             # drop slivers above 8:1 aspect
#   SKIP_BAD=1                       # opt in to --skip-classification-when-bad
#   TILE=1                           # opt in to SAHI sliced detection
#                                    # (default: OFF — tiling hurt P/R on YNP-BisonGraze)
#   TILE_SIZE=480                    # tile edge length in px (default: 480)
#   TILE_OVERLAP=0.2                 # fractional overlap between adjacent tiles
#   NO_MULTISCALE=1                  # disable multi-scale BioCLIP (default: on)
#   MULTISCALE_PAD=2.0               # padded-crop expansion factor
#   GT=/path/to/combined.json        # default: YNP-BisonGraze combined.json
#                                    # below — set this to a different GT for
#                                    # other deployments
#   EVAL=0                           # set to 0 to skip eval (inference only)
#   CLS_MIN_CONF=0.70                # eval-side confidence threshold (default 0.70)
#   QUALITY=ok                       # eval-side quality filter (ok|all)
#   EXTRA_ARGS="--recursive --resume"
#
# Persistent log: $OUTPUT_DIR/wytrap.log is written in addition to logs/wytrap-$JOBID.out,
# so the run survives after SLURM cleanup.

set -euo pipefail
mkdir -p logs

# Shared model cache so each researcher doesn't re-download MegaDetector + BioCLIP weights.
export INPUT_DIR="/project/wildimageproc/software/annotations/YNP-BisonGraze/images"
export OUTPUT_DIR="/project/wildimageproc/omartin9/detection-projects/wytrap/output-pipeline"
export GT="${GT:-/project/wildimageproc/software/annotations/YNP-BisonGraze/combined.json}"
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

# Tiling is OFF by default in wytrap (SAHI hurt both precision and recall
# on YNP-BisonGraze in our tiny-eval A/B). Opt back in with TILE=1.
TILE_FLAG=""
if [[ "${TILE:-0}" == "1" ]]; then
    TILE_FLAG="--tile"
fi

# Toggle --no-multiscale via NO_MULTISCALE=1.
NO_MULTISCALE_FLAG=""
if [[ "${NO_MULTISCALE:-0}" == "1" ]]; then
    NO_MULTISCALE_FLAG="--no-multiscale"
fi

wytrap detect \
    --input  "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --species "${SPECIES:-wyoming_all}" \
    --det-threshold "${DET_THRESHOLD:-0.50}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --min-pixel-side "${MIN_PIXEL_SIDE:-60}" \
    --border-overlap-truncated "${BORDER_OVERLAP_TRUNC:-0.20}" \
    --max-aspect-ratio "${MAX_ASPECT_RATIO:-8.0}" \
    --tile-size "${TILE_SIZE:-480}" \
    --tile-overlap "${TILE_OVERLAP:-0.2}" \
    --multiscale-pad "${MULTISCALE_PAD:-2.0}" \
    --device cuda \
    --recursive \
    --resume \
    $SKIP_BAD_FLAG \
    $TILE_FLAG \
    $NO_MULTISCALE_FLAG \
    ${EXTRA_ARGS:-}

echo "[wytrap.sbatch] inference complete. Persistent log: $OUTPUT_DIR/wytrap.log"

# ===== eval (auto, unless EVAL=0) =====
if [[ "${EVAL:-1}" == "1" && -f "$GT" ]]; then
    echo "===== eval ====="
    T0=$(date +%s)
    python scripts/eval_pipeline.py \
        --gt   "$GT" \
        --pred "$OUTPUT_DIR" \
        --quality "${QUALITY:-ok}" \
        --cls-min-confidence "${CLS_MIN_CONF:-0.70}"
    echo "[wytrap.sbatch] eval took $(($(date +%s) - T0))s. "\
"artifacts in $OUTPUT_DIR/eval/"
elif [[ "${EVAL:-1}" != "1" ]]; then
    echo "[wytrap.sbatch] EVAL=0 set; skipping eval."
else
    echo "[wytrap.sbatch] GT not found at '$GT'; skipping eval."
fi
