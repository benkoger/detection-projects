#!/bin/bash
#SBATCH --job-name=wytrap-ocr-dates
#SBATCH --account=wildimageproc
#SBATCH --partition=beartooth
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/wytrap-ocr-dates-%j.out
#SBATCH --error=logs/wytrap-ocr-dates-%j.err

# OCR burned-in timestamps from camera-trap JPEGs to recover capture dates.
#
# Submit with:
#   IMAGES=/path/to/images OUT=/path/to/file_to_date.json \
#     sbatch scripts/ocr_burnin_dates.sbatch
#
# Optional env-vars:
#   LIMIT=100                    # cap files for a sanity-check run
#   WORKERS=16                   # match --cpus-per-task above
#   TOP_FRAC=0.12 BOTTOM_FRAC=0.12

set -euo pipefail
mkdir -p logs

VENV="${VENV:-/project/wildimageproc/omartin9/.venv-wytrap-dev}"
source "$VENV/bin/activate"

cd "$SLURM_SUBMIT_DIR"

: "${IMAGES:?IMAGES must be set}"
: "${OUT:?OUT must be set (path to file_to_date.json)}"

LIMIT_FLAG=""
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_FLAG="--limit $LIMIT"
fi

# Tesseract on Beartooth: try the module first, fall back to the venv-local
# binary if the module isn't available.
module load tesseract 2>/dev/null || true

python scripts/ocr_burnin_dates.py \
    --images "$IMAGES" \
    --out    "$OUT" \
    --workers "${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}" \
    --top-frac "${TOP_FRAC:-0.12}" \
    --bottom-frac "${BOTTOM_FRAC:-0.12}" \
    --resume \
    $LIMIT_FLAG

echo "[ocr-dates.sbatch] done. Output: $OUT"
