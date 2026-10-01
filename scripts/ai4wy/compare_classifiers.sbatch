#!/bin/bash
# Every classifier on the same images, the same MegaDetector boxes and the
# same candidate set (a taxon-node vocabulary), scored by the same evaluator.
#
#   sbatch scripts/ai4wy/compare_classifiers.sbatch
#
# Env-vars:
#   DATA_DIR      default /project/uwyo-0007/data/idaho-100pc (images/ + labels.json)
#   VOCAB         default taxonomy/idaho_vocab.csv
#   DETECTOR      default MDV1000-redwood (what AddaxAI Connect runs)
#   DET_THRESHOLD default 0.20 (record floor; the eval sweeps above it)
#   ZOO           space-separated AddaxAI zoo repos
#   ADMIN1        geofence state for the SpeciesNet ensemble (default ID)
#SBATCH --job-name=wytrap-compare
#SBATCH --account=uwyo-0007
#SBATCH --partition=gp-1,gp-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/wytrap-compare-%j.out
#SBATCH --error=logs/wytrap-compare-%j.err

set -euo pipefail
umask 002
mkdir -p logs

PROJECT_ROOT="${PROJECT_ROOT:-/project/uwyo-0007/software}"
VENV="${VENV:-$PROJECT_ROOT/.venv-wytrap}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/models/torch}"

DATA_DIR="${DATA_DIR:-/project/uwyo-0007/data/idaho-100pc}"
VOCAB="${VOCAB:-taxonomy/idaho_vocab.csv}"
DETECTOR="${DETECTOR:-MDV1000-redwood}"
ZOO="${ZOO:-Addax-Data-Science/WUSA-SDZWA-v1 Addax-Data-Science/SWUSA-SDZWA-v3}"
ADMIN1="${ADMIN1:-ID}"
VTAG=$(basename "$VOCAB" .csv)
DTAG=$(echo "$DETECTOR" | tr '[:upper:]' '[:lower:]')

source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"
echo "node: $(hostname)  vocab: $VOCAB  detector: $DETECTOR  data: $DATA_DIR"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# ---------------------------------------------------------------------------
# EVAL: score one run folder against the Idaho labels. Called after every
# classifier below so each gets identical treatment.
#   --labels        labels.json from fetch_idaho_subset.py (one label per image,
#                   no boxes)
#   --pred          the run folder; the evaluator reads its all_records.jsonl
#   --out           metrics.json, per_image.csv, confusion matrix go here; the
#                   folder is named after the vocabulary so re-scoring the same
#                   run with another vocabulary does not overwrite it
#   --vocab         resolve predictions to vocabulary nodes by lineage (mule deer
#                   -> deer) and report the hierarchical score alongside top-1
#   --det-threshold the box-confidence floor for the classification section:
#                   only boxes at or above it (and tagged quality "ok") vote for
#                   the image's label
#   --det-sweep     presence/absence precision and recall are recomputed at
#                   each of these floors, so one run shows the whole trade-off
# ---------------------------------------------------------------------------
EVAL() {  # $1 = run dir
    wytrap eval --labels "$DATA_DIR/labels.json" --pred "$1" --out "$1/eval-$VTAG" \
        --vocab "$VOCAB" --det-threshold "${DET_THRESHOLD_CLS:-0.50}" \
        --det-sweep 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
}

# ---------------------------------------------------------------------------
# Step 0: show what the vocabulary means for each model. Nothing downstream
# depends on this (every classifier derives its own candidate set when it
# loads); it writes <vocab>.bioclip.txt / <vocab>.speciesnet.txt and prints
# which SpeciesNet labels and which zoo classes fall inside the vocabulary,
# so the log records exactly which classes each model was allowed to choose.
# ---------------------------------------------------------------------------
echo "===== candidate lists ====="
wytrap vocab "$VOCAB" --speciesnet-model hf:Addax-Data-Science/SPECIESNET-v4-0-2-A --zoo $ZOO

# ---------------------------------------------------------------------------
# Step 1: the only pass that runs the detector. MegaDetector finds every
# animal box, each box is tagged for quality (ok / low_pixels / truncated /
# thin), and BioCLIP 2 scores it against the vocabulary's species prompts at
# three crop scales (tight, 2x padded, whole frame), keeping the best scale.
#   --classifier bioclip  BioCLIP 2 zero-shot
#   --vocab               its prompt list is the vocabulary's member species
#   --detector            which MegaDetector checkpoint
#   --det-threshold 0.20  record boxes down to 0.20 confidence; the evaluator
#                         sweeps higher floors from these records, so no second
#                         detector pass is ever needed
#   --keep-labels animal  Idaho's public set has no people or vehicles, and the
#                         evaluator counts only animal boxes
# The run writes all_records.jsonl (boxes + BioCLIP labels), prompts.json
# (the prompt order, needed by calibrate) and the uncorrected log-probability
# of every prompt on every box, which is what the prior correction uses.
# ---------------------------------------------------------------------------
echo "===== MegaDetector + BioCLIP 2 on the vocabulary prompts ====="
OUT_BC="$DATA_DIR/output-bioclip-$VTAG-$DTAG"
wytrap detect --input "$DATA_DIR/images" --output "$OUT_BC" --classifier bioclip \
    --vocab "$VOCAB" --detector "$DETECTOR" --det-threshold "${DET_THRESHOLD:-0.20}" \
    --keep-labels animal --device cuda
EVAL "$OUT_BC"
# Every classifier from here on is given these boxes, so the comparison is
# classifier-only: same images, same crops, same quality tags.
BOXES="$OUT_BC/all_records.jsonl"

# ---------------------------------------------------------------------------
# Step 2: BioCLIP 2 prior correction (logit adjustment), no labels used.
# Some prompts win on almost any crop ("least chipmunk" absorbed small blurry
# boxes). calibrate estimates each prompt's mean log-probability over all the
# unlabelled crops of the run, leaving out the camera location being
# corrected, subtracts it, and re-ranks every box. The output is a new run
# folder with the corrected labels plus prompt_bias.json, which deploy jobs
# pass to `wytrap detect --prompt-bias`. No GPU work: it only re-reads the
# per-prompt scores step 1 recorded.
# ---------------------------------------------------------------------------
echo "===== BioCLIP 2 + prior correction (leave-location-out, label-free) ====="
OUT_BCC="$OUT_BC-calib"
wytrap calibrate --pred "$OUT_BC" --output "$OUT_BCC"
EVAL "$OUT_BCC"

# ---------------------------------------------------------------------------
# Step 3: SpeciesNet's classifier network on each box, restricted.
# `classify` re-reads the step 1 records, crops each animal box the way
# SpeciesNet's always-crop model expects, and keeps the top-k over the
# labels inside the vocabulary (47 of 2,498 on Idaho; the restriction goes
# through SpeciesNet's own target-species mechanism, which returns logits
# over just those labels, then a softmax). No roll-up, no geofence: this
# row isolates "how good is the classifier network itself".
# ---------------------------------------------------------------------------
echo "===== SpeciesNet classifier, candidate set ====="
OUT_SNC="$DATA_DIR/output-speciesnet-$VTAG-$DTAG"
wytrap classify --records "$BOXES" --output "$OUT_SNC" --classifier speciesnet --vocab "$VOCAB"
EVAL "$OUT_SNC"

# ---------------------------------------------------------------------------
# Step 4: the same network with no restriction, all 2,498 labels. The gap
# between this row and step 3 is what the candidate set buys. The evaluator
# reports how many of its predictions fall outside the vocabulary ("outside")
# and how many land on a coarser taxon ("rollup").
# ---------------------------------------------------------------------------
echo "===== SpeciesNet classifier, open set (all labels) ====="
OUT_SNO="$DATA_DIR/output-speciesnet-open-$DTAG"
wytrap classify --records "$BOXES" --output "$OUT_SNO" --classifier speciesnet
EVAL "$OUT_SNO"

# ---------------------------------------------------------------------------
# Step 5: SpeciesNet as shipped, except for the detector. Per image, the
# classifier output goes through SpeciesNet's ensemble: if the species call
# is below its confidence threshold it rolls up to genus, family, order...
# until one is confident enough, and the geofence (--admin1, a US state)
# drops species not known to occur there. One label per image, written onto
# every animal box; the 'source' field records which rule produced it
# (classifier, classifier+rollup_to_genus, ...). The candidate set is applied
# before the roll-up. This is the configuration closest to what AddaxAI
# Connect runs, and the row to read for "never wrong" behaviour.
# ---------------------------------------------------------------------------
echo "===== SpeciesNet + roll-up + geofence, candidate set ====="
OUT_SNE="$DATA_DIR/output-speciesnet-ens-$VTAG-$DTAG"
wytrap classify --records "$BOXES" --output "$OUT_SNE" --classifier speciesnet-ensemble \
    --vocab "$VOCAB" --admin1 "$ADMIN1"
EVAL "$OUT_SNE"

# ---------------------------------------------------------------------------
# Step 6: each AddaxAI zoo model (default Western USA and Southwest USA
# SDZWA) on the same boxes. The repo's own inference.py does the cropping
# and preprocessing, exactly as AddaxAI does; its taxonomy.csv maps classes
# onto vocabulary nodes and decides which classes are masked out (the
# softmax is renormalised over the rest).
# ---------------------------------------------------------------------------
ROWS=("BioCLIP 2 [vocab]=$OUT_BC" "BioCLIP 2 + prior correction [vocab]=$OUT_BCC"
      "SpeciesNet [vocab]=$OUT_SNC" "SpeciesNet [open]=$OUT_SNO" "SpeciesNet + roll-up [vocab]=$OUT_SNE")
for M in $ZOO; do
    T=$(basename "$M" | tr '[:upper:]' '[:lower:]')
    OUT="$DATA_DIR/output-addax-$T-$VTAG-$DTAG"
    echo "===== $M, candidate set ====="
    wytrap classify --records "$BOXES" --output "$OUT" --classifier addax --model "$M" --vocab "$VOCAB"
    EVAL "$OUT"
    ROWS+=("$(basename "$M") [vocab]=$OUT")
done

# ---------------------------------------------------------------------------
# Step 7: one table. Reads each run's eval-<vocab>/metrics.json and prints:
#   n        animal images with at least one box at the 0.50 floor
#   top1     image label == vocabulary node of the ground truth
#   hier     top1, or a roll-up to a taxon that contains the true node
#            (SpeciesNet saying "Odocoileus species" for a mule deer counts)
#   rollup   share of images predicted above the node's rank
#   outside  share predicted outside the vocabulary altogether
#   top3     true node among the three best labels
#   acc@cov  top-1 accuracy on the most confident 80%+ of images, and the
#            coverage at which it was measured: what a review queue that
#            auto-accepts confident calls would see
# ---------------------------------------------------------------------------
python - "$VTAG" "${ROWS[@]}" <<'PY'
import json, sys, pathlib
vtag = sys.argv[1]
print(f"\n===== shared candidate set ({vtag}), hierarchical scoring, det floor 0.50 =====")
print(f"{'run':>44} {'n':>4} {'top1':>6} {'hier':>6} {'rollup':>7} {'outside':>8} {'top3':>6}  acc@cov>=0.8")
for arg in sys.argv[2:]:
    name, d = arg.split("=", 1)
    f = pathlib.Path(d) / f"eval-{vtag}" / "metrics.json"
    if not f.exists():
        print(f"{name:>44}  (no metrics)"); continue
    c = json.loads(f.read_text())["classification"]
    h = c.get("hierarchical", {}); acc = c.get("accuracy", {})
    cov = [r for r in c.get("accuracy_vs_coverage", []) if r["coverage"] >= 0.8]
    best = max(cov, key=lambda r: r["top1_accuracy"]) if cov else None
    print(f"{name:>44} {c.get('animal_images_with_detection', 0):4d} {acc.get('top1', 0):6.3f} "
          f"{h.get('top1_or_consistent_rollup', 0):6.3f} {h.get('rollup_rate', 0):7.3f} "
          f"{h.get('outside_vocabulary_rate', 0):8.3f} {acc.get('top3', 0):6.3f}  "
          + (f"{best['top1_accuracy']:.3f} @ {best['coverage']:.2f}" if best else "-"))
print("\ntop1 = correct node; hier = correct node or roll-up to an ancestor of it;")
print("rollup = share predicted above node rank; outside = share outside the vocabulary.")
PY
echo "[compare_classifiers] done"
