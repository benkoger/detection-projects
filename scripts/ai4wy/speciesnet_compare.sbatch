#!/bin/bash
# BioCLIP 2 vs SpeciesNet on the same Idaho subset, scored by the same eval.
#
#   arm clf: SpeciesNet classifier on the MegaDetector v6 boxes of an existing
#            wytrap run (default: the det020 sweep output). Only the classifier
#            differs from wytrap.
#   arm ens: SpeciesNet as shipped (MDv5a + classifier + geofence USA/ID).
#   arm ens6: SpeciesNet classifier + roll-up + geofence on the MDv6 boxes of
#            the wytrap run, i.e. AddaxAI Connect's pipeline shape with the
#            same detector as wytrap. Differs from wytrap only after detection.
#
# Submit from the repo root after scripts/ai4wy/setup_env.sh has installed
# speciesnet (re-run it after pulling):
#   sbatch scripts/ai4wy/speciesnet_compare.sbatch
#
# Env-vars:
#   DATA_DIR        default /project/uwyo-0007/data/idaho-20pc
#   WYTRAP_RECORDS  default $DATA_DIR/output-species_idaho-det020/all_records.jsonl
#   SN_MODEL        default hf:Addax-Data-Science/SPECIESNET-v4-0-2-A
#   COUNTRY=USA ADMIN1=ID   NO_GEOFENCE=1 to disable geofencing in the ensemble arm
#   DET_THRESHOLD_CLS default 0.50 (eval classification floor)
#SBATCH --job-name=wytrap-speciesnet
#SBATCH --account=uwyo-0007
#SBATCH --partition=gp-1,gp-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/wytrap-speciesnet-%j.out
#SBATCH --error=logs/wytrap-speciesnet-%j.err

set -euo pipefail
umask 002
mkdir -p logs

PROJECT_ROOT="${PROJECT_ROOT:-/project/uwyo-0007/software}"
VENV="${VENV:-$PROJECT_ROOT/.venv-wytrap}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/models/torch}"

DATA_DIR="${DATA_DIR:-/project/uwyo-0007/data/idaho-20pc}"
WYTRAP_RECORDS="${WYTRAP_RECORDS:-$DATA_DIR/output-species_idaho-det020/all_records.jsonl}"
SN_MODEL="${SN_MODEL:-hf:Addax-Data-Science/SPECIESNET-v4-0-2-A}"
GEO_FLAG=""; [[ "${NO_GEOFENCE:-0}" == "1" ]] && GEO_FLAG="--no-geofence"

source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"
echo "node: $(hostname)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
[[ -f "$WYTRAP_RECORDS" ]] || { echo "missing $WYTRAP_RECORDS (run the detector sweep first)"; exit 2; }
# Name outputs after the source run so different detectors never share a folder.
SRC_TAG=$(basename "$(dirname "$WYTRAP_RECORDS")" | sed 's/^output-//')
echo "source boxes: $SRC_TAG ($WYTRAP_RECORDS)"

echo "===== arm clf: SpeciesNet classifier on wytrap boxes ====="
OUT_CLF="$DATA_DIR/output-speciesnet-clf-$SRC_TAG"
python scripts/run_speciesnet_arms.py classifier \
    --wytrap-records "$WYTRAP_RECORDS" --output "$OUT_CLF" --model "$SN_MODEL"
python scripts/eval_image_level.py --labels "$DATA_DIR/labels.json" --pred "$OUT_CLF" \
    --det-threshold "${DET_THRESHOLD_CLS:-0.50}" --det-sweep 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9

echo "===== arm ens: full SpeciesNet ensemble ====="
OUT_ENS="$DATA_DIR/output-speciesnet-ens${NO_GEOFENCE:+-nogeo}"
python scripts/run_speciesnet_arms.py ensemble \
    --images "$DATA_DIR/images" --output "$OUT_ENS" --model "$SN_MODEL" \
    --country "${COUNTRY:-USA}" --admin1 "${ADMIN1:-ID}" $GEO_FLAG
python scripts/eval_image_level.py --labels "$DATA_DIR/labels.json" --pred "$OUT_ENS" \
    --det-threshold "${DET_THRESHOLD_CLS:-0.50}" --det-sweep 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9

echo "===== arm ens6: SpeciesNet ensemble on the wytrap run's boxes ($SRC_TAG) ====="
OUT_ENS6="$DATA_DIR/output-speciesnet-ens-$SRC_TAG${NO_GEOFENCE:+-nogeo}"
python scripts/run_speciesnet_arms.py ensemble \
    --images "$DATA_DIR/images" --output "$OUT_ENS6" --model "$SN_MODEL" \
    --detections-from "$WYTRAP_RECORDS" \
    --country "${COUNTRY:-USA}" --admin1 "${ADMIN1:-ID}" $GEO_FLAG
python scripts/eval_image_level.py --labels "$DATA_DIR/labels.json" --pred "$OUT_ENS6" \
    --det-threshold "${DET_THRESHOLD_CLS:-0.50}" --det-sweep 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9

python - "$DATA_DIR" "$(dirname "$WYTRAP_RECORDS")" "$OUT_CLF" "$OUT_ENS" "$OUT_ENS6" "$SRC_TAG" <<'PY'
import json, sys, pathlib
data = pathlib.Path(sys.argv[1]); tag = sys.argv[6]
runs = [(f"wytrap BioCLIP 2 [{tag}]", sys.argv[2]), (f"SpeciesNet raw clf [{tag} boxes]", sys.argv[3]),
        ("SpeciesNet ensemble (own MDv5a)", sys.argv[4]), (f"SpeciesNet ensemble [{tag} boxes]", sys.argv[5])]
print("\n===== BioCLIP 2 vs SpeciesNet (det floor 0.50; classification on quality-ok boxes) =====")
print(f"{'run':>40} {'P':>6} {'R':>6} {'F1':>6} {'n_cls':>6} {'top1':>6} {'top3':>6}  acc@cov>=0.8")
for name, d in runs:
    m = json.loads((pathlib.Path(d) / "eval" / "metrics.json").read_text())
    det = next(r for r in m["detection"] if abs(r["det_threshold"] - 0.5) < 1e-6)
    c = m["classification"]; acc = c.get("accuracy", {})
    cov = [r for r in c.get("accuracy_vs_coverage", []) if r["coverage"] >= 0.8]
    best = max(cov, key=lambda r: r["top1_accuracy"]) if cov else None
    print(f"{name:>40} {det['precision']:6.3f} {det['recall']:6.3f} {det['f1']:6.3f} "
          f"{c.get('animal_images_with_detection', 0):6d} {acc.get('top1', 0):6.3f} {acc.get('top3', 0):6.3f}  "
          + (f"{best['top1_accuracy']:.3f} @ cov {best['coverage']:.2f} (conf>={best['cls_min_conf']})" if best else "-"))
PY
echo "[speciesnet_compare] done"
