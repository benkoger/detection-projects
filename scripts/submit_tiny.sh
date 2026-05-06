#!/usr/bin/env bash
# Submit a tiny run with named experiment + arbitrary env-var overrides.
#
# Usage:
#   scripts/submit_tiny.sh <experiment-name> [VAR=val ...]
#
# Examples:
#   scripts/submit_tiny.sh full_only NO_TILE=1
#   scripts/submit_tiny.sh det065   DET_THRESHOLD=0.65
#   scripts/submit_tiny.sh both     DET_THRESHOLD=0.65 NO_TILE=1
#
# Output goes to $TINY/output_<experiment-name>/ . The wrapper sets
# OUTPUT_NAME for you so the experiment subdir is named consistently.
#
# Required env: TINY (or pass as TINY=...). Defaults to the standard tiny
# dir under this checkout if unset and that path exists.

set -euo pipefail

DEFAULT_TINY="/project/wildimageproc/omartin9/detection-projects/wytrap/tiny"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <experiment-name> [VAR=val ...]" >&2
    echo "example: $0 det065 DET_THRESHOLD=0.65" >&2
    exit 1
fi

EXPERIMENT="$1"; shift

: "${TINY:=$DEFAULT_TINY}"
if [[ ! -d "$TINY" ]]; then
    echo "ERROR: TINY=$TINY does not exist. Build it first with " \
         "scripts/build_tiny_eval.py, or pass TINY=/path/to/tiny." >&2
    exit 2
fi

# Collect all the override pairs into a comma-joined string for --export.
EXPORTS="ALL,TINY=$TINY,OUTPUT_NAME=output_$EXPERIMENT"
for arg in "$@"; do
    if [[ "$arg" != *=* ]]; then
        echo "ERROR: '$arg' is not VAR=val" >&2
        exit 1
    fi
    EXPORTS="$EXPORTS,$arg"
done

echo "submitting experiment '$EXPERIMENT'"
echo "  TINY        : $TINY"
echo "  output dir  : $TINY/output_$EXPERIMENT"
echo "  exports     : $EXPORTS"

cd "$(dirname "$0")/.."
sbatch --export="$EXPORTS" scripts/run_tiny.sbatch
