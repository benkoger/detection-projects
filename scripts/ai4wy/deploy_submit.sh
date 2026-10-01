#!/bin/bash
# Submit deploy_infer.sbatch as a Slurm array, one task per immediate
# sub-folder (camera) of IMAGES_ROOT, at most MAX_PARALLEL tasks at once.
# Each task writes to OUT_ROOT/<camera>/.
#
#   scripts/ai4wy/deploy_submit.sh /project/uwyo-0007/data/wysoundscape/images \
#       [/project/uwyo-0007/data/wysoundscape/output]
#
# Other env-vars (VOCAB, DETECTOR, PROMPT_BIAS, ...) pass through to the job.
# Re-running after failures is safe: wytrap resumes, the other arms are fast.
#
# When the array is done, one table for everything:
#   wytrap merge --combine <OUT_ROOT>
set -euo pipefail
ROOT=$(realpath "${1:?usage: deploy_submit.sh IMAGES_ROOT [OUT_ROOT]}")
OUT="${2:-$(dirname "$ROOT")/output}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
mkdir -p "$OUT" logs

LIST="$OUT/folders.txt"
find "$ROOT" -mindepth 1 -maxdepth 1 -type d ! -name '.*' | sort > "$LIST"
N=$(wc -l < "$LIST" | tr -d ' ')
[[ "$N" -gt 0 ]] || { echo "no sub-folders under $ROOT"; exit 1; }
echo "$N camera folders under $ROOT -> $OUT (max $MAX_PARALLEL at once)"
sbatch --array="0-$((N - 1))%$MAX_PARALLEL" \
    --export=ALL,FOLDERS_FILE="$LIST",OUT_ROOT_BASE="$OUT" \
    scripts/ai4wy/deploy_infer.sbatch
