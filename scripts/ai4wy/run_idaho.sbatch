#!/bin/bash
# wytrap (MegaDetector v6 + BioCLIP 2) over the Idaho Camera Traps subset on
# GannettPeak (ai4wy). Grace Hopper nodes: aarch64 + H100.
#
# Submit from the repo root:
#   sbatch scripts/ai4wy/run_idaho.sbatch
#
# Optional env-vars:
#   INPUT_DIR   default /project/uwyo-0007/data/idaho-subset/images
#   OUTPUT_DIR  default /project/uwyo-0007/data/idaho-subset/output-wytrap
#   SPECIES     default wyoming_all; or a species file, e.g.
#               SPECIES=scripts/ai4wy/species_idaho.txt (closed-set arm)
#   LABELS      default $INPUT_DIR/../labels.json (from fetch_idaho_subset.py)
#   EVAL=0      skip scripts/eval_image_level.py after inference
#   SEQ_LEVEL=1 evaluate per sequence (use with --whole-sequences subsets)
#   AGG=vote    image-level label aggregation (default max_score)
#   DET_THRESHOLD=0.50  NO_MULTISCALE=1  TILE=1  SKIP_BAD=1  EXTRA_ARGS="..."
#
# Two-arm experiment (open-set vs closed-set species list):
#   OUTPUT_DIR=.../output-wyoming_all sbatch scripts/ai4wy/run_idaho.sbatch
#   SPECIES=scripts/ai4wy/species_idaho.txt OUTPUT_DIR=.../output-idaho \
#       sbatch scripts/ai4wy/run_idaho.sbatch
#
# Check your account name with `sacctmgr show assoc user=$USER format=account`
# and the walltime ceiling with `sinfo -o "%P %l"` (24h was rejected in the
# ARCC docs, so this asks for 8h).
#SBATCH --job-name=wytrap-idaho
#SBATCH --account=uwyo-0007
#SBATCH --partition=ai4wy-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/wytrap-idaho-%j.out
#SBATCH --error=logs/wytrap-idaho-%j.err

set -euo pipefail
mkdir -p logs

PROJECT_ROOT="${PROJECT_ROOT:-/project/uwyo-0007/software}"
VENV="${VENV:-$PROJECT_ROOT/.venv-wytrap}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/models/torch}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # weights were prefetched on a login node

INPUT_DIR="${INPUT_DIR:-/project/uwyo-0007/data/idaho-subset/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/project/uwyo-0007/data/idaho-subset/output-wytrap}"
LABELS="${LABELS:-$(dirname "$INPUT_DIR")/labels.json}"

source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"

echo "node: $(hostname)  arch: $(uname -m)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

SKIP_BAD_FLAG="";      [[ "${SKIP_BAD:-0}" == "1" ]]      && SKIP_BAD_FLAG="--skip-classification-when-bad"
TILE_FLAG="";          [[ "${TILE:-0}" == "1" ]]          && TILE_FLAG="--tile"
NO_MULTISCALE_FLAG=""; [[ "${NO_MULTISCALE:-0}" == "1" ]] && NO_MULTISCALE_FLAG="--no-multiscale"

wytrap detect \
    --input  "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --species "${SPECIES:-wyoming_all}" \
    --det-threshold "${DET_THRESHOLD:-0.50}" \
    --device cuda \
    --recursive \
    --resume \
    --jsonl "$OUTPUT_DIR/all_records.jsonl" \
    $SKIP_BAD_FLAG $TILE_FLAG $NO_MULTISCALE_FLAG ${EXTRA_ARGS:-}

echo "[run_idaho] inference done. per-image JSON under $OUTPUT_DIR, log at $OUTPUT_DIR/wytrap.log"

if [[ "${EVAL:-1}" == "1" && -f "$LABELS" ]]; then
    SEQ_FLAG=""; [[ "${SEQ_LEVEL:-0}" == "1" ]] && SEQ_FLAG="--sequence-level"
    python scripts/eval_image_level.py \
        --labels "$LABELS" --pred "$OUTPUT_DIR" \
        --det-threshold "${DET_THRESHOLD:-0.50}" \
        --agg "${AGG:-max_score}" --quality "${QUALITY:-ok}" $SEQ_FLAG
    echo "[run_idaho] eval written to $OUTPUT_DIR/eval/"
elif [[ "${EVAL:-1}" == "1" ]]; then
    echo "[run_idaho] LABELS not found at $LABELS; skipping eval"
fi
