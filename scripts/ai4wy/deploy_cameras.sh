#!/bin/bash
# The whole deployment in one command: inference on every matching camera
# folder (one Slurm job each, in parallel), then — once all of them have
# finished — the unrestricted SpeciesNet audit, the census against the
# vocabulary, the combined tables and the agreement summary.
#
#   scripts/ai4wy/deploy_cameras.sh DATA_DIR [CAMERA_FILTER]
#
#   scripts/ai4wy/deploy_cameras.sh /project/uwyo-0007/data/CameraTrap_test shirley
#   scripts/ai4wy/deploy_cameras.sh /project/uwyo-0007/data/CameraTrap_test          # every camera
#
# DATA_DIR must have the `wytrap ingest` layout (images/<camera>/). Results:
#   DATA_DIR/output/<camera>/{bioclip-*,speciesnet-ens-*,addax-*,speciesnet-open,merged}/
#   DATA_DIR/output/all_images.csv, all_boxes.csv, summary.json      the combined tables
#   DATA_DIR/output/census[-filter]/census.csv, census_by_family.csv the vocabulary audit
#
# Re-running is safe: finished images are skipped (resume). To start over,
# move DATA_DIR/output aside first. Env-vars (VOCAB, DETECTOR, MIN_DET, ...)
# pass through to the jobs; see deploy_infer.sbatch and census.sbatch.
set -euo pipefail
DATA_DIR=$(realpath "${1:?usage: deploy_cameras.sh DATA_DIR [CAMERA_FILTER]}")
FILTER="${2:-}"
OUT="$DATA_DIR/output"
[[ -d "$DATA_DIR/images" ]] || { echo "$DATA_DIR/images not found: run 'wytrap ingest' first"; exit 1; }
cd "$(dirname "$0")/../.."          # repo root, so taxonomy/ and logs/ resolve
mkdir -p logs "$OUT"

JOBS=()
for d in "$DATA_DIR"/images/*/; do
    CAM=$(basename "$d")
    [[ -n "$FILTER" && "${CAM,,}" != *"${FILTER,,}"* ]] && continue
    JID=$(IMAGES="$d" OUT_ROOT_BASE="$OUT" sbatch --parsable scripts/ai4wy/deploy_infer.sbatch)
    echo "submitted $JID  $CAM  ($(find "$d" -type f | wc -l | tr -d ' ') files)"
    JOBS+=("$JID")
done
[[ ${#JOBS[@]} -gt 0 ]] || { echo "no camera folders match '$FILTER' under $DATA_DIR/images"; exit 1; }

DEP=$(IFS=:; echo "${JOBS[*]}")
JID=$(DATA_DIR="$DATA_DIR" CAMERAS="$FILTER" sbatch --parsable --dependency=afterany:$DEP \
      scripts/ai4wy/census.sbatch)
echo "submitted $JID  census + combined tables, runs after ${#JOBS[@]} inference job(s)"
echo "watch:   squeue -u \$USER"
echo "results: $OUT/summary.json, $OUT/census${FILTER:+-$FILTER}/census.csv"
