# Running wytrap on ai4wy (GannettPeak)

GannettPeak is ARCC's AI cluster: 36 Grace Hopper nodes (ARM Neoverse V2 CPUs,
H100 GPUs), Slurm partitions `ai4wy-1` (1 GPU/node) and `ai4wy-2` (2 GPUs/node),
login node `ai4wy-log2` or the Open OnDemand portal at https://ai4wy.arcc.uwyo.edu.
Every job needs `--account`. Nothing CUDA-related works without `--gres=gpu:N`.

Because the nodes are **aarch64**, the old beartooth venv cannot be reused.
These scripts build a fresh one with uv under `/project/uwyo-0007/software`.

## One-time setup (login node)

```bash
cd /project/uwyo-0007/software/detection-projects
bash scripts/ai4wy/setup_env.sh          # uv + venv + deps + import check (~5 min)
bash scripts/ai4wy/prefetch_weights.sh   # MegaDetector v6 + BioCLIP 2 into /project/uwyo-0007/software/models
```

## Pull the Idaho Camera Traps test subset (login node)

```bash
source /project/uwyo-0007/software/.venv-wytrap/bin/activate
python scripts/fetch_idaho_subset.py \
    --out /project/uwyo-0007/data/idaho-subset --per-class 100 --workers 16
```

Defaults sample up to 100 single-label images per class, at most one frame per
sequence, across 23 classes that overlap the Wyoming species lists plus
`empty`. That is roughly 2,200 images at about 1.6 MB each, so 3 to 4 GB. Use `--per-class 20` for a first 10-minute smoke test.
`labels.json` carries image-level ground truth; the LILA dataset has no boxes.

For evaluation-grade subsets, add hard negatives and whole sequences:

```bash
python scripts/fetch_idaho_subset.py \
    --out /project/uwyo-0007/data/idaho-seq --per-class 60 --whole-sequences \
    --negatives-per-class 40
```

`--whole-sequences` downloads every frame of each sampled sequence (counts
then refer to sequences, about 1.5 frames each), which lets the eval score
detection per sequence and softens the "sequence label on an empty frame"
noise. `--negatives-per-class` adds images labelled only with camera problems
(snow on lens, foggy lens, ...) as hard negatives for the false-positive rate.

## Everything in one job (preferred)

```bash
sbatch scripts/ai4wy/fetch_and_run_idaho.sbatch                 # 20 per class, both arms
PER_CLASS=100 NEGATIVES=40 WHOLE_SEQ=1 sbatch scripts/ai4wy/fetch_and_run_idaho.sbatch
```

The job checks outbound HTTPS from the compute node, caches weights if
needed, fetches the subset into `/project/uwyo-0007/data/idaho-<N>pc`, runs
wytrap once per species list (`wyoming_all` and `species_idaho.txt` by
default), evaluates each, and prints a side-by-side summary at the end of
the `.out` log. If the node has no network the job exits with code 3 and the
two-step path below applies.

## Two-step path (fetch on login node, run on compute)

```bash
sbatch scripts/ai4wy/run_idaho.sbatch                 # open-set: wyoming_all
SPECIES=scripts/ai4wy/species_idaho.txt \
    OUTPUT_DIR=/project/uwyo-0007/data/idaho-subset/output-idaho \
    sbatch scripts/ai4wy/run_idaho.sbatch             # closed-set: Idaho classes only
tail -f logs/wytrap-idaho-<jobid>.out
```

Each job runs `wytrap detect` then `scripts/eval_image_level.py`. Inference
outputs: one JSON per image, `all_records.jsonl`, `wytrap.log`. Eval outputs
under `<OUTPUT_DIR>/eval/`: `metrics.json`, `per_image.csv`,
`confusion_matrix.csv` and `.png`, `eval.log`.

The Idaho labels are per sequence with no boxes, so the box-matching
`scripts/eval_pipeline.py` does not apply. The image-level eval reports:

- detection as presence/absence over a det_score sweep, with false-positive
  rate per negative type (empty, human, vehicle, camera problems) and per
  location;
- classification given the image at a fixed det_score floor: top-1/3/5
  accuracy, accuracy versus coverage over a cls_score floor, per-class
  precision/recall, confusion matrix, and splits by day/night and by wytrap
  quality tag. Predictions and labels both pass through
  `helpers.IDAHO_EVAL_MERGES` (mule deer and white-tailed deer both become
  "deer", and so on).

The two-arm run above is the first experiment worth reading: the accuracy gap
between `wyoming_all` (148 prompts, includes species Idaho never sees) and
`species_idaho.txt` (25 prompts) is the cost of open-set confusion, which
bears directly on how long the I-80 species list should be. Note that
`wyoming_all` has no livestock prompts, so Idaho's cattle and horse images can
only be scored correctly in the closed-set arm.

Re-run eval without re-running inference:

```bash
python scripts/eval_image_level.py --labels .../labels.json --pred .../output-wytrap \
    --agg vote --sequence-level --quality all
```

## Jupyter

`setup_env.sh` registers the venv as a kernel named "wytrap (ai4wy)" in
`~/.local/share/jupyter/kernels/wytrap`. Pick it in JupyterLab from the Open
OnDemand portal. Kernel specs are per user, so each person runs once:

```bash
/project/uwyo-0007/software/.venv-wytrap/bin/python -m ipykernel install --user \
    --name wytrap --display-name "wytrap (ai4wy)"
```

## Sharing with the project group

`/project/uwyo-0007` directories are group `uwyo-0007` with setgid, so new
files inherit the group. `setup_env.sh` sets `umask 002` so they are also
group-writable. If the venv was built before that line existed, fix it once:

```bash
chmod -R g+rwX /project/uwyo-0007/software
git -C /project/uwyo-0007/software/detection-projects config core.sharedRepository group
```

## Detector choice and the Zenodo problem

PytorchWildlife downloads MegaDetector v6 from Zenodo only, and Zenodo goes
down for hours at a time (it did during setup). `wytrap detect --detector`
now also accepts `MDV1000-redwood`, `MDV5a` and `MDV5b`, which come from the
Hugging Face mirror `agentmorris/megadetector` and load through
PytorchWildlife's v5 class. `prefetch_weights.sh` tries Zenodo a few times,
then always caches redwood. Both sbatch scripts default to `DETECTOR=auto`:
MDv6 yolov9-e when its checkpoint is cached, otherwise MDv1000 redwood. Set
`DETECTOR=...` explicitly to pin one, and record which detector a run used
(it is printed at the top of the job log and in `wytrap.log`) since the two
are not interchangeable when comparing numbers. Redwood is also what AddaxAI
Connect runs, so it is the more relevant baseline for the I-80 deployment.

## Gotchas

- The setup and download steps need outbound network. Run them on a login node.
  Jobs run with `HF_HUB_OFFLINE=1` and read weights from the shared cache.
- Torch wheels from PyPI carry CUDA 13 and need driver 580+. If
  `torch.cuda.is_available()` is false in a job, check `nvidia-smi` in the job
  log and, if the driver is older, reinstall torch from the cu128 index:
  `uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision`.
- Wall-time: 24 h requests were rejected in the ARCC examples. Use 8 h chunks
  and `--resume`.
