#!/bin/bash
#SBATCH --job-name=cameratrap-eval
#SBATCH --account=wildimageproc
#SBATCH --partition=beartooth-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

# Submit with:
#   RUN_ID=<train_jobid> sbatch scripts/eval_model.sbatch
# Or, to chain after training, use scripts/train_then_eval.sh.
#
# Optional env-vars:
#   WEIGHTS=model-epoch-9.pth          # default: final_model.pth
#   SCORE_THRESHOLD=0.5
#   IOU_THRESHOLD=0.5

set -euo pipefail
mkdir -p logs

export HF_HOME="${HF_HOME:-/project/wildimageproc/software/models}"
export TORCH_HOME="${TORCH_HOME:-/project/wildimageproc/software/models/torch}"

VENV="${VENV:-/project/wildimageproc/software/.wildimageproc-venv}"
source "$VENV/bin/activate"

cd "$SLURM_SUBMIT_DIR"

: "${RUN_ID:?RUN_ID must be set (the run folder name = the train job id)}"

python scripts/eval_model.py \
    --weights "${WEIGHTS:-final_model.pth}" \
    --score-threshold "${SCORE_THRESHOLD:-0.5}" \
    --iou-threshold "${IOU_THRESHOLD:-0.5}"

echo "[eval.sbatch] eval complete. Results in \$MODEL_PATH/runs/${RUN_ID}/eval/"
