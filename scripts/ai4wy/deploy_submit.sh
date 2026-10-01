#!/bin/bash
# Submit deploy_infer.sbatch as a Slurm array over a data folder laid out by
# `wytrap ingest`: one task per camera folder under DATA_DIR/images, at most
# MAX_PARALLEL tasks at once. Each task writes to DATA_DIR/output/<camera>/.
#
#   scripts/ai4wy/deploy_submit.sh /project/uwyo-0007/data/CameraTrap_test
#
# Other env-vars (VOCAB, DETECTOR, PROMPT_BIAS, ...) pass through to the job.
# Re-running after failures is safe: wytrap resumes, the other arms are fast.
#
# When the array is done, one table for everything:
#   wytrap merge --combine DATA_DIR/output
set -euo pipefail
DATA_DIR=$(realpath "${1:?usage: deploy_submit.sh DATA_DIR (with images/ from wytrap ingest)}")
ROOT="$DATA_DIR/images"
OUT="${2:-$DATA_DIR/output}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
[[ -d "$ROOT" ]] || { echo "$ROOT not found: run 'wytrap ingest --source ... --out $DATA_DIR' first"; exit 1; }
mkdir -p "$OUT" logs

LIST="$OUT/folders.txt"
find "$ROOT" -mindepth 1 -maxdepth 1 -type d ! -name '.*' | sort > "$LIST"
N=$(wc -l < "$LIST" | tr -d ' ')
[[ "$N" -gt 0 ]] || { echo "no camera folders under $ROOT"; exit 1; }
echo "$N camera folders under $ROOT -> $OUT (max $MAX_PARALLEL at once)"
sbatch --array="0-$((N - 1))%$MAX_PARALLEL" \
    --export=ALL,FOLDERS_FILE="$LIST",OUT_ROOT_BASE="$OUT" \
    scripts/ai4wy/deploy_infer.sbatch
