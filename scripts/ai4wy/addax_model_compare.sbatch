#!/bin/bash
# Run AddaxAI model-zoo classifiers on the boxes of a wytrap run and score
# them with the same eval. Default: Western USA (SDZWA) and Southwest USA v3.
#
#   sbatch scripts/ai4wy/addax_model_compare.sbatch
#
# Env-vars:
#   DATA_DIR        default /project/uwyo-0007/data/idaho-20pc
#   WYTRAP_RECORDS  default $DATA_DIR/output-species_idaho-det020/all_records.jsonl
#   MODELS          space-separated HF repo ids (default below)
#   DET_THRESHOLD_CLS default 0.50
#SBATCH --job-name=wytrap-addaxzoo
#SBATCH --account=uwyo-0007
#SBATCH --partition=gp-1,gp-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/wytrap-addaxzoo-%j.out
#SBATCH --error=logs/wytrap-addaxzoo-%j.err

set -euo pipefail
umask 002
mkdir -p logs

PROJECT_ROOT="${PROJECT_ROOT:-/project/uwyo-0007/software}"
VENV="${VENV:-$PROJECT_ROOT/.venv-wytrap}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/models/torch}"

DATA_DIR="${DATA_DIR:-/project/uwyo-0007/data/idaho-20pc}"
WYTRAP_RECORDS="${WYTRAP_RECORDS:-$DATA_DIR/output-species_idaho-det020/all_records.jsonl}"
MODELS="${MODELS:-Addax-Data-Science/WUSA-SDZWA-v1 Addax-Data-Science/SWUSA-SDZWA-v3}"

source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"
echo "node: $(hostname)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
[[ -f "$WYTRAP_RECORDS" ]] || { echo "missing $WYTRAP_RECORDS"; exit 2; }

OUTS=()
for M in $MODELS; do
    TAG=$(basename "$M" | tr '[:upper:]' '[:lower:]')
    OUT="$DATA_DIR/output-addax-$TAG"
    echo "===== $M -> $OUT ====="
    python scripts/run_addax_model_arm.py --model "$M" --wytrap-records "$WYTRAP_RECORDS" --output "$OUT"
    python scripts/eval_image_level.py --labels "$DATA_DIR/labels.json" --pred "$OUT" \
        --det-threshold "${DET_THRESHOLD_CLS:-0.50}" --det-sweep 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
    OUTS+=("$M=$OUT")
done

python - "$(dirname "$WYTRAP_RECORDS")" "$DATA_DIR" "${OUTS[@]}" <<'PY'
import json, sys, pathlib
wy, data = sys.argv[1], pathlib.Path(sys.argv[2])
rows = [("wytrap (BioCLIP 2)", wy)]
for extra, label in [("output-speciesnet-clf", "SpeciesNet raw clf, same boxes"),
                     ("output-speciesnet-ens-mdv6", "SpeciesNet ensemble, same boxes"),
                     ("output-speciesnet-ens", "SpeciesNet ensemble (MDv5a)")]:
    if (data / extra / "eval" / "metrics.json").exists():
        rows.append((label, str(data / extra)))
rows += [(a.split("=", 1)[0].split("/")[-1], a.split("=", 1)[1]) for a in sys.argv[3:]]
print("\n===== classifier comparison (det floor 0.50; classification on quality-ok boxes) =====")
print(f"{'run':>34} {'n_cls':>6} {'top1':>6} {'top3':>6}  acc@cov>=0.8")
for name, d in rows:
    m = json.loads((pathlib.Path(d) / "eval" / "metrics.json").read_text())
    c = m["classification"]; acc = c.get("accuracy", {})
    cov = [r for r in c.get("accuracy_vs_coverage", []) if r["coverage"] >= 0.8]
    best = max(cov, key=lambda r: r["top1_accuracy"]) if cov else None
    print(f"{name:>34} {c.get('animal_images_with_detection', 0):6d} {acc.get('top1', 0):6.3f} {acc.get('top3', 0):6.3f}  "
          + (f"{best['top1_accuracy']:.3f} @ cov {best['coverage']:.2f}" if best else "-"))
PY
echo "[addax_model_compare] done"
