#!/bin/bash
#SBATCH --job-name=wytrap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/wytrap-%j.out
#SBATCH --error=logs/wytrap-%j.err

# Submit with: INPUT_DIR=/path/to/photos OUTPUT_DIR=/path/to/results sbatch run_pipeline.sbatch
#
# Optional env-vars:
#   SPECIES=wyoming_mammals          # default: wyoming_all
#   DET_THRESHOLD=0.2
#   BATCH_SIZE=8
#   EXTRA_ARGS="--recursive --resume"

set -euo pipefail
mkdir -p logs

# Shared model cache so each researcher doesn't re-download MegaDetector + BioCLIP weights.
# The project owner sets these up once with world-readable perms.
export HF_HOME="${HF_HOME:-/project/wildimageproc/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/project/wildimageproc/cache/torch}"

VENV="${VENV:-/project/wildimageproc/software/.wildimageproc-venv}"
source "$VENV/bin/activate"

: "${INPUT_DIR:?INPUT_DIR must be set}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

wytrap detect \
    --input  "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --species "${SPECIES:-wyoming_all}" \
    --det-threshold "${DET_THRESHOLD:-0.2}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --device cuda \
    --recursive \
    --resume \
    ${EXTRA_ARGS:-}
