#!/bin/bash
# Run several detector configurations back to back in ONE job on an
# already-fetched Idaho subset, evaluate each, print a comparison table.
# One GPU for ~20 min beats four jobs each waiting for a node.
#
# Submit from the repo root:
#   sbatch scripts/ai4wy/detector_sweep.sbatch
#
# Env-vars:
#   DATA_DIR   default /project/uwyo-0007/data/idaho-20pc (must contain images/ and labels.json)
#   SPECIES    default scripts/ai4wy/species_idaho.txt
#   CONFIGS    one config per line, "tag | extra wytrap detect flags".
#              Default below: 0.20 floor baseline, 1920 px, 2560 px, native tiling.
#   DET_THRESHOLD_CLS  eval classification floor (default 0.50)
#
# Partition: both ai4wy partitions so Slurm takes whichever frees first.
#SBATCH --job-name=wytrap-detsweep
#SBATCH --account=uwyo-0007
#SBATCH --partition=gp-1,gp-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/wytrap-detsweep-%j.out
#SBATCH --error=logs/wytrap-detsweep-%j.err

set -euo pipefail
umask 002
mkdir -p logs

PROJECT_ROOT="${PROJECT_ROOT:-/project/uwyo-0007/software}"
VENV="${VENV:-$PROJECT_ROOT/.venv-wytrap}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/models/torch}"
export HF_HUB_OFFLINE=1

DATA_DIR="${DATA_DIR:-/project/uwyo-0007/data/idaho-20pc}"
SPECIES="${SPECIES:-scripts/ai4wy/species_idaho.txt}"
CONFIGS="${CONFIGS:-
det020   | --det-threshold 0.20
sz1920   | --det-threshold 0.20 --det-imgsz 1920
sz2560   | --det-threshold 0.20 --det-imgsz 2560
tile1280 | --det-threshold 0.20 --tile --tile-size 1280
}"

source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"

echo "node: $(hostname)  partition: $SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
[[ -s "$TORCH_HOME/hub/checkpoints/MDV6-yolov9-e-1280.pt" ]] && DETECTOR=MDV6-yolov9-e || DETECTOR=MDV1000-redwood
echo "detector: $DETECTOR   data: $DATA_DIR   species: $SPECIES"

TAGS=()
while IFS='|' read -r tag flags; do
    tag=$(echo "$tag" | xargs); flags=$(echo "$flags" | xargs)
    [[ -z "$tag" ]] && continue
    OUT="$DATA_DIR/output-$(basename "$SPECIES" .txt)-$tag"
    echo "===== $tag: wytrap detect $flags -> $OUT ====="
    T0=$(date +%s)
    wytrap detect --input "$DATA_DIR/images" --output "$OUT" \
        --species "$SPECIES" --detector "$DETECTOR" --device cuda \
        --recursive --resume --jsonl "$OUT/all_records.jsonl" $flags
    echo "[sweep] $tag inference took $(($(date +%s) - T0))s"
    python scripts/eval_image_level.py --labels "$DATA_DIR/labels.json" --pred "$OUT" \
        --det-threshold "${DET_THRESHOLD_CLS:-0.50}" \
        --det-sweep 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
    TAGS+=("$tag")
done <<< "$CONFIGS"

python - "$DATA_DIR" "$(basename "$SPECIES" .txt)" "${TAGS[@]}" <<'PY'
import json, sys, pathlib
data, sp = pathlib.Path(sys.argv[1]), sys.argv[2]
print("\n===== detector sweep summary (det floor 0.50 for P/R; classification on quality-ok boxes) =====")
print(f"{'tag':>10} {'P':>6} {'R':>6} {'F1':>6} {'top1':>6} {'squirrel':>9} {'lagomorph':>10}  FP rate by negative type")
for tag in sys.argv[3:]:
    m = json.loads((data / f"output-{sp}-{tag}" / "eval" / "metrics.json").read_text())
    d = next(r for r in m["detection"] if abs(r["det_threshold"] - 0.5) < 1e-6)
    c = m["classification"]; pc = c.get("per_class", {})
    fp = {k: v for k, v in d["fp_rate_by_negative_type"].items() if v}
    def rec(k):
        r = pc.get(k, {}).get("recall"); return f"{r:.2f}" if r is not None else "   -"
    print(f"{tag:>10} {d['precision']:6.3f} {d['recall']:6.3f} {d['f1']:6.3f} "
          f"{c.get('accuracy', {}).get('top1', 0):6.3f} {rec('squirrel'):>9} {rec('lagomorph'):>10}  {fp}")
PY
