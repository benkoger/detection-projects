#!/bin/bash
#SBATCH --job-name=wytrap-eval
#SBATCH --account=wildimageproc
#SBATCH --partition=beartooth
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/wytrap-eval-%j.out
#SBATCH --error=logs/wytrap-eval-%j.err

# Compare wytrap pipeline outputs against COCO ground truth.
#
# Submit with:
#   GT=/path/to/combined.json \
#   PRED=/path/to/output-pipeline \
#     sbatch scripts/eval_pipeline.sbatch
#
# Optional env-vars:
#   OUT=/path/to/eval                # default: $PRED/eval
#   IOU=0.5
#   QUALITY=ok                       # 'ok' (default) or 'all'
#   NO_MERGE=1                       # disable GT label merging
#   TOP_KS=1,3,5
#   CLS_MIN_CONF=0.30                # below this cls_score, count as abstained

set -euo pipefail
mkdir -p logs

VENV="${VENV:-/project/wildimageproc/software/.wildimageproc-venv}"
source "$VENV/bin/activate"

cd "$SLURM_SUBMIT_DIR"

: "${GT:?GT must be set (path to COCO json with ground-truth)}"
: "${PRED:?PRED must be set (path to wytrap output folder)}"

NO_MERGE_FLAG=""
if [[ "${NO_MERGE:-0}" == "1" ]]; then
    NO_MERGE_FLAG="--no-merge"
fi

OUT_FLAG=""
if [[ -n "${OUT:-}" ]]; then
    OUT_FLAG="--out $OUT"
fi

python scripts/eval_pipeline.py \
    --gt   "$GT" \
    --pred "$PRED" \
    $OUT_FLAG \
    --iou "${IOU:-0.5}" \
    --quality "${QUALITY:-ok}" \
    --top-ks "${TOP_KS:-1,3,5}" \
    --cls-min-confidence "${CLS_MIN_CONF:-0.30}" \
    $NO_MERGE_FLAG

echo "[wytrap-eval.sbatch] done. Metrics in ${OUT:-$PRED/eval}/"
