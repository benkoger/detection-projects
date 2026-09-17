#!/bin/bash
# One job: fetch an Idaho Camera Traps subset, run wytrap on it for each
# species-list arm, evaluate each arm. Runs entirely on a compute node.
#
# Submit from the repo root:
#   sbatch scripts/ai4wy/fetch_and_run_idaho.sbatch
#
# Env-vars (all optional):
#   DATA_DIR      default /project/uwyo-0007/data/idaho-<PER_CLASS>pc
#   PER_CLASS     default 20   images (or sequences with WHOLE_SEQ=1) per class
#   NEGATIVES     default 10   camera-problem hard negatives per label
#   WHOLE_SEQ=1   fetch whole sequences and evaluate per sequence
#   ARMS          default "wyoming_all scripts/ai4wy/species_idaho.txt"
#                 space-separated species lists; each gets its own output dir
#   DETECTOR    default auto: MDV6-yolov9-e if cached, else MDV1000-redwood
#   RUN_TAG     suffix for the output folders (output-<arm>-<RUN_TAG>) so a
#               rerun with different settings does not --resume into old results
#   DET_THRESHOLD      inference floor (default 0.50); boxes below it are never
#                      recorded. Run at 0.20 once to let the eval sweep simulate
#                      any higher floor without re-running inference.
#   DET_THRESHOLD_CLS  floor used for the classification section of the eval
#                      (default 0.50, independent of the inference floor)
#   DET_SWEEP          eval sweep thresholds (default 0.2..0.9)
#   DET_IMGSZ   detector input resolution, long side px (default native 1280;
#               try 1920 or 2560 for small animals)
#   TILE=1      SAHI tiling; TILE_SIZE default 1280 here (wytrap's own default
#               of 480 upscales each tile to 1280 and hurt on YNP)
#   AGG=max_score|vote  QUALITY=ok|all  EXTRA_ARGS="..."
#
# Needs outbound HTTPS from the compute node (LILA on Azure blob storage,
# Zenodo and Hugging Face for weights if not yet cached). The job checks this
# first and exits 3 if it is blocked; then fetch on a login node instead and
# use run_idaho.sbatch.
#SBATCH --job-name=wytrap-idaho-full
#SBATCH --account=uwyo-0007
#SBATCH --partition=gp-1,gp-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/wytrap-idaho-full-%j.out
#SBATCH --error=logs/wytrap-idaho-full-%j.err

set -euo pipefail
umask 002
mkdir -p logs

PROJECT_ROOT="${PROJECT_ROOT:-/project/uwyo-0007/software}"
VENV="${VENV:-$PROJECT_ROOT/.venv-wytrap}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/models/torch}"

PER_CLASS="${PER_CLASS:-20}"
NEGATIVES="${NEGATIVES:-10}"
DATA_DIR="${DATA_DIR:-/project/uwyo-0007/data/idaho-${PER_CLASS}pc}"
ARMS="${ARMS:-wyoming_all scripts/ai4wy/species_idaho.txt}"

source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"

echo "node: $(hostname)  arch: $(uname -m)  cpus: $SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# DETECTOR: auto -> MDv6 yolov9-e when its checkpoint is cached, else MDv1000
# redwood from the Hugging Face mirror (see wytrap detect --detector).
DETECTOR="${DETECTOR:-auto}"
if [[ "$DETECTOR" == "auto" ]]; then
    if [[ -s "$TORCH_HOME/hub/checkpoints/MDV6-yolov9-e-1280.pt" ]]; then
        DETECTOR="MDV6-yolov9-e"
    else
        DETECTOR="MDV1000-redwood"
    fi
fi
echo "detector: $DETECTOR"

# ---- 0. network check
if ! curl -sSf --max-time 20 -o /dev/null -r 0-0 \
        "https://lilawildlife.blob.core.windows.net/lila-wildlife/idaho-camera-traps/idaho-camera-traps.json.zip"; then
    echo "[job] no outbound HTTPS from $(hostname). Fetch on a login node with"
    echo "      scripts/fetch_idaho_subset.py, then submit scripts/ai4wy/run_idaho.sbatch."
    exit 3
fi
echo "[job] outbound HTTPS ok"

# ---- 1. weights: only fetch what the resolved detector actually needs
if python - "$DETECTOR" <<'PY'
import os, sys
from huggingface_hub import try_to_load_from_cache
from wytrap.classifier import bioclip_cache_status
from wytrap.detector import Detector
det = sys.argv[1]
if det in Detector.HF_WEIGHTS:
    repo, fname = Detector.HF_WEIGHTS[det]
    ok_det = try_to_load_from_cache(repo_id=repo, filename=fname) is not None
else:
    ok_det = os.path.getsize(os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints",
                                          "MDV6-yolov9-e-1280.pt")) > 10_000_000 \
             if os.path.exists(os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints",
                                            "MDV6-yolov9-e-1280.pt")) else False
ok_cls = bioclip_cache_status()["status"] == "cached"
print(f"[job] weights cached: detector={ok_det} bioclip2={ok_cls}")
sys.exit(0 if ok_det and ok_cls else 1)
PY
then
    echo "[job] skipping prefetch"
else
    # MDV6_TRIES=0 skips Zenodo entirely when the job is going to use redwood.
    MDV6_TRIES=$([[ "$DETECTOR" == MDV6* ]] && echo 5 || echo 0) \
        bash scripts/ai4wy/prefetch_weights.sh
fi

# ---- 2. data
WHOLE_FLAG=""; SEQ_FLAG=""
if [[ "${WHOLE_SEQ:-0}" == "1" ]]; then WHOLE_FLAG="--whole-sequences"; SEQ_FLAG="--sequence-level"; fi
T0=$(date +%s)
python scripts/fetch_idaho_subset.py \
    --out "$DATA_DIR" --per-class "$PER_CLASS" --negatives-per-class "$NEGATIVES" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" $WHOLE_FLAG
echo "[job] fetch took $(($(date +%s) - T0))s -> $DATA_DIR"

# ---- 3. inference + eval, one arm at a time
for SPECIES in $ARMS; do
    ARM_NAME=$(basename "$SPECIES" .txt)
    OUT="$DATA_DIR/output-$ARM_NAME${RUN_TAG:+-$RUN_TAG}"
    echo "===== arm: $SPECIES -> $OUT ====="
    T0=$(date +%s)
    TILE_FLAG=""; [[ "${TILE:-0}" == "1" ]] && TILE_FLAG="--tile --tile-size ${TILE_SIZE:-1280}"
    IMGSZ_FLAG=""; [[ -n "${DET_IMGSZ:-}" ]] && IMGSZ_FLAG="--det-imgsz $DET_IMGSZ"
    wytrap detect \
        --input "$DATA_DIR/images" --output "$OUT" \
        --species "$SPECIES" --det-threshold "${DET_THRESHOLD:-0.50}" $TILE_FLAG $IMGSZ_FLAG \
        --detector "$DETECTOR" --device cuda --recursive --resume \
        --jsonl "$OUT/all_records.jsonl" ${EXTRA_ARGS:-}
    echo "[job] inference ($ARM_NAME) took $(($(date +%s) - T0))s"
    python scripts/eval_image_level.py \
        --labels "$DATA_DIR/labels.json" --pred "$OUT" \
        --det-threshold "${DET_THRESHOLD_CLS:-0.50}" \
        --det-sweep "${DET_SWEEP:-0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}" \
        --agg "${AGG:-max_score}" --quality "${QUALITY:-ok}" $SEQ_FLAG
done

# ---- 4. side-by-side summary
python - "$DATA_DIR" "${RUN_TAG:-}" $ARMS <<'PY'
import json, sys, pathlib
data = pathlib.Path(sys.argv[1]); tag = sys.argv[2]
print("\n===== summary =====")
for sp in sys.argv[3:]:
    name = f"output-{pathlib.Path(sp).stem}" + (f"-{tag}" if tag else "")
    m = json.loads((data / name / "eval" / "metrics.json").read_text())
    best = max(m["detection"], key=lambda r: r["f1"])
    c = m["classification"]
    acc = c.get("accuracy", {})
    print(f"{pathlib.Path(sp).stem:>16}: det F1 {best['f1']:.3f}@{best['det_threshold']:.2f}  "
          f"cls n={c.get('animal_images_with_detection', 0)}  "
          + "  ".join(f"{k} {v:.3f}" for k, v in acc.items()))
PY
echo "[job] done. results under $DATA_DIR/output-*/eval/"
