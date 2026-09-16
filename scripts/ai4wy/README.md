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

## Run

```bash
sbatch scripts/ai4wy/run_idaho.sbatch
tail -f logs/wytrap-idaho-<jobid>.out
```

Outputs: one JSON per image under `output-wytrap/`, plus `all_records.jsonl`
and `wytrap.log`. Because the Idaho labels are image-level, the box-matching
`scripts/eval_pipeline.py` does not apply. Compare the top-1 species per image
against `labels.json` instead.

## Gotchas

- The setup and download steps need outbound network. Run them on a login node.
  Jobs run with `HF_HUB_OFFLINE=1` and read weights from the shared cache.
- Torch wheels from PyPI carry CUDA 13 and need driver 580+. If
  `torch.cuda.is_available()` is false in a job, check `nvidia-smi` in the job
  log and, if the driver is older, reinstall torch from the cu128 index:
  `uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision`.
- Wall-time: 24 h requests were rejected in the ARCC examples. Use 8 h chunks
  and `--resume`.
