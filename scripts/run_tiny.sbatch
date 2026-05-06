#!/bin/bash
#SBATCH --job-name=wytrap-tiny
#SBATCH --account=wildimageproc
#SBATCH --partition=beartooth-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/wytrap-tiny-%j.out
#SBATCH --error=logs/wytrap-tiny-%j.err

# Run the full wytrap pipeline (inference + eval) on a tiny stratified
# subset for fast A/B iteration. Submit with:
#
#   TINY=/path/to/tiny sbatch scripts/run_tiny.sbatch
#
# Where TINY is the --out dir produced by scripts/build_tiny_eval.py
# (contains images/ and combined.json). About 10 minutes end-to-end on
# ~250 images.
#
# Optional env-vars (override the wytrap defaults for this run):
#   SPECIES=ynp_testbed
#   DET_THRESHOLD=0.10
#   TILE_SIZE=480 TILE_OVERLAP=0.2
#   TILE=1 NO_MULTISCALE=1 SKIP_BAD=1
#   CLS_MIN_CONF=0.30                 # eval-side threshold
#   QUALITY=ok|all                    # eval-side quality filter
#
# This is a one-shot run: existing $TINY/output is wiped first so the
# inference output is always fresh. Use it for A/B comparisons of
# species lists, model variants, NMS thresholds, etc.

set -euo pipefail
mkdir -p logs

export HF_HOME="${HF_HOME:-/project/wildimageproc/software/models}"
export TORCH_HOME="${TORCH_HOME:-/project/wildimageproc/software/models/torch}"

VENV="${VENV:-/project/wildimageproc/omartin9/.venv-wytrap-dev}"
source "$VENV/bin/activate"

cd "$SLURM_SUBMIT_DIR"

: "${TINY:?TINY must be set (path to dir built by build_tiny_eval.py)}"
TINY_IMAGES="$TINY/images"
TINY_GT="$TINY/combined.json"
# OUTPUT_NAME lets you run multiple experiments side-by-side under the
# same TINY dir without trampling each other:
#   OUTPUT_NAME=output_with_tile TILE=1            sbatch run_tiny.sbatch
#   OUTPUT_NAME=output_det065   DET_THRESHOLD=0.65 sbatch run_tiny.sbatch
TINY_OUTPUT="$TINY/${OUTPUT_NAME:-output}"

[[ -d "$TINY_IMAGES" ]] || { echo "missing $TINY_IMAGES"; exit 2; }
[[ -f "$TINY_GT" ]]     || { echo "missing $TINY_GT"; exit 2; }

# Wipe previous run so we always evaluate the latest pipeline behavior on
# the same image set. (For A/B you usually want fresh inference each time.)
rm -rf "$TINY_OUTPUT"
mkdir -p "$TINY_OUTPUT"

# Optional toggles for inference.
SKIP_BAD_FLAG=""
if [[ "${SKIP_BAD:-0}" == "1" ]]; then SKIP_BAD_FLAG="--skip-classification-when-bad"; fi
# Tiling is OFF by default; TILE=1 opts back in.
TILE_FLAG=""
if [[ "${TILE:-0}" == "1" ]]; then TILE_FLAG="--tile"; fi
NO_MULTISCALE_FLAG=""
if [[ "${NO_MULTISCALE:-0}" == "1" ]]; then NO_MULTISCALE_FLAG="--no-multiscale"; fi

echo "===== inference ($(date)) ====="
T0=$(date +%s)
wytrap detect \
    --input  "$TINY_IMAGES" \
    --output "$TINY_OUTPUT" \
    --species "${SPECIES:-ynp_testbed}" \
    --det-threshold "${DET_THRESHOLD:-0.10}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --edge-margin-frac "${EDGE_MARGIN_FRAC:-0.01}" \
    --min-box-area-frac "${MIN_BOX_AREA_FRAC:-0.005}" \
    --max-aspect-ratio "${MAX_ASPECT_RATIO:-5.0}" \
    --tile-size "${TILE_SIZE:-480}" \
    --tile-overlap "${TILE_OVERLAP:-0.2}" \
    --multiscale-pad "${MULTISCALE_PAD:-2.0}" \
    --device cuda \
    --recursive \
    $SKIP_BAD_FLAG $TILE_FLAG $NO_MULTISCALE_FLAG \
    ${EXTRA_ARGS:-}

T1=$(date +%s)
echo "inference took $((T1 - T0))s"

echo "===== eval ($(date)) ====="
T2=$(date +%s)
python scripts/eval_pipeline.py \
    --gt   "$TINY_GT" \
    --pred "$TINY_OUTPUT" \
    --iou "${IOU:-0.5}" \
    --quality "${QUALITY:-ok}" \
    --top-ks "${TOP_KS:-1,3,5}" \
    --cls-min-confidence "${CLS_MIN_CONF:-0.30}"
T3=$(date +%s)
echo "eval took $((T3 - T2))s"

cat <<EOF
==========================================================
tiny run complete
  inference: $((T1 - T0))s
  eval     : $((T3 - T2))s
  total    : $((T3 - T0))s

  artifacts in:
    $TINY_OUTPUT
    $TINY_OUTPUT/eval/metrics.json
    $TINY_OUTPUT/eval/confusion_matrix_committed.png
    $TINY_OUTPUT/eval/species_timeseries_strict.html
==========================================================
EOF
