#!/usr/bin/env bash
# Submit the training job, then submit the eval job depending on it.
#
# Default usage (most common):
#   scripts/train_then_eval.sh
#
# Override anything via env-vars if needed, e.g.:
#   NUM_EPOCHS=10 scripts/train_then_eval.sh
#   RESEARCH_PROJECT=high-elevation scripts/train_then_eval.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# ---- Defaults baked in so the bare command above just works ----
TIMESTAMP="$(date +%Y-%m-%d-%H%M%S)"
: "${NUM_EPOCHS:=30}"
: "${README:=full retrain on cleaned splits ${TIMESTAMP}}"
: "${RESEARCH_PROJECT:=YNP-BisonGraze}"
: "${WEIGHTS:=final_model.pth}"
: "${SCORE_THRESHOLD:=0.5}"
: "${IOU_THRESHOLD:=0.5}"

mkdir -p logs

# Submit train. sbatch prints "Submitted batch job NNNN".
TRAIN_OUTPUT=$(sbatch \
    --export=ALL,NUM_EPOCHS="$NUM_EPOCHS",README="$README",RESEARCH_PROJECT="$RESEARCH_PROJECT" \
    scripts/train_model.sbatch)
echo "$TRAIN_OUTPUT"
TRAIN_JOB_ID=$(echo "$TRAIN_OUTPUT" | awk '{print $NF}')

# The train script uses $SLURM_JOB_ID as the run folder name. Pass that same
# id to eval as RUN_ID, and chain with --dependency=afterok so eval only
# fires if training exits cleanly.
EVAL_OUTPUT=$(sbatch \
    --dependency=afterok:"$TRAIN_JOB_ID" \
    --export=ALL,RUN_ID="$TRAIN_JOB_ID",WEIGHTS="$WEIGHTS",SCORE_THRESHOLD="$SCORE_THRESHOLD",IOU_THRESHOLD="$IOU_THRESHOLD" \
    scripts/eval_model.sbatch)
echo "$EVAL_OUTPUT"
EVAL_JOB_ID=$(echo "$EVAL_OUTPUT" | awk '{print $NF}')

cat <<EOF

============================================================
Submitted:
  train  : job ${TRAIN_JOB_ID}
  eval   : job ${EVAL_JOB_ID}  (chained --dependency=afterok)
  README : ${README}
  epochs : ${NUM_EPOCHS}
  project: ${RESEARCH_PROJECT}

Run folder (after train starts):
  \$MODEL_PATH/runs/${TRAIN_JOB_ID}

------------------------------------------------------------
Tail the live train log:
  tail -F logs/train-${TRAIN_JOB_ID}.out

Tail train errors:
  tail -F logs/train-${TRAIN_JOB_ID}.err

When training finishes, eval logs appear at:
  tail -F logs/eval-${EVAL_JOB_ID}.out

Inspect after-the-fact (full log preserved in run folder):
  less \$MODEL_PATH/runs/${TRAIN_JOB_ID}/train.log
  less \$MODEL_PATH/runs/${TRAIN_JOB_ID}/eval/eval.log

------------------------------------------------------------
Queue status:
  squeue -u \$USER
  squeue -j ${TRAIN_JOB_ID},${EVAL_JOB_ID}

To cancel both jobs:
  scancel ${TRAIN_JOB_ID} ${EVAL_JOB_ID}
============================================================
EOF
