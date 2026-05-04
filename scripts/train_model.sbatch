#!/bin/bash
#SBATCH --job-name=wytrap-train
#SBATCH --account=wildimageproc
#SBATCH --gres=gpu:1
#SBATCH --partition=beartooth-gpu
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

# Submit with:
#   sbatch scripts/train_model.sbatch
# Optional env-vars (override at submit time):
#   NUM_EPOCHS=30 sbatch scripts/train_model.sbatch
#   README="..." sbatch scripts/train_model.sbatch
#   RESEARCH_PROJECT=YNP-BisonGraze sbatch scripts/train_model.sbatch
#
# To chain training -> eval, use scripts/train_then_eval.sh instead.

set -euo pipefail
mkdir -p logs

# Shared model + HF caches so collaborators reuse downloads.
export HF_HOME="${HF_HOME:-/project/wildimageproc/software/models}"
export TORCH_HOME="${TORCH_HOME:-/project/wildimageproc/software/models/torch}"

VENV="${VENV:-/project/wildimageproc/software/.wildimageproc-venv}"
source "$VENV/bin/activate"

cd "$(dirname "$0")/.."

# Use the SLURM job ID as the run folder name so eval can find it via $RUN_ID.
python scripts/train_model.py \
    --research-project "${RESEARCH_PROJECT:-YNP-BisonGraze}" \
    --num-epochs "${NUM_EPOCHS:-10}" \
    --readme "${README:-Training run via SLURM}" \
    --run-id "${SLURM_JOB_ID}"

echo "[train.sbatch] training complete. Run folder: \$MODEL_PATH/runs/${SLURM_JOB_ID}"
