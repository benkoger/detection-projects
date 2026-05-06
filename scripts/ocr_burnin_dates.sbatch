#!/bin/bash
#SBATCH --job-name=wytrap-ocr-dates
#SBATCH --account=wildimageproc
#SBATCH --partition=beartooth-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/wytrap-ocr-dates-%j.out
#SBATCH --error=logs/wytrap-ocr-dates-%j.err

# OCR burned-in timestamps from camera-trap JPEGs to recover capture dates.
# Uses EasyOCR on GPU by default — pure pip install, no system binaries.
#
# Submit with:
#   IMAGES=/path/to/images OUT=/path/to/file_to_date.json \
#     sbatch scripts/ocr_burnin_dates.sbatch
#
# Optional env-vars:
#   LIMIT=100                    # cap files for a sanity-check run
#   ENGINE=tesseract             # default: easyocr
#   WORKERS=16                   # tesseract only; ignored for easyocr
#   TOP_FRAC=0.12 BOTTOM_FRAC=0.12

set -euo pipefail
mkdir -p logs

# Shared HF cache so EasyOCR's recognition model is reused across users.
export HF_HOME="${HF_HOME:-/project/wildimageproc/software/models}"
export TORCH_HOME="${TORCH_HOME:-/project/wildimageproc/software/models/torch}"
# EasyOCR caches its model under ~/.EasyOCR by default; redirect to project
# storage so other researchers inherit the cached weights.
export EASYOCR_MODULE_PATH="${EASYOCR_MODULE_PATH:-/project/wildimageproc/software/models/easyocr}"

VENV="${VENV:-/project/wildimageproc/omartin9/.venv-wytrap-dev}"
source "$VENV/bin/activate"
IMAGES="/project/wildimageproc/software/annotations/YNP-BisonGraze/images" 
OUT="/project/wildimageproc/omartin9/detection-projects/wytrap/output-pipeline/file_to_date.json"

cd "$SLURM_SUBMIT_DIR"

: "${IMAGES:?IMAGES must be set}"
: "${OUT:?OUT must be set (path to file_to_date.json)}"

LIMIT_FLAG=""
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_FLAG="--limit $LIMIT"
fi

python scripts/ocr_burnin_dates.py \
    --images "$IMAGES" \
    --out    "$OUT" \
    --engine "${ENGINE:-easyocr}" \
    --workers "${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}" \
    --top-frac "${TOP_FRAC:-0.12}" \
    --bottom-frac "${BOTTOM_FRAC:-0.12}" \
    --resume \
    $LIMIT_FLAG

echo "[ocr-dates.sbatch] done. Output: $OUT"
