# Running wytrap on ai4wy (GannettPeak)

ai4wy is ARCC's GannettPeak cluster: Grace Hopper nodes (aarch64 CPUs,
H100-class GPUs), Slurm partitions `gp-1` (1 GPU/node) and `gp-2` (2
GPUs/node), account `uwyo-0007`. Everything below runs from the repo root at
`/project/uwyo-0007/software/detection-projects`. The pipeline itself is the
`wytrap` command; `wytrap/README.md` explains what it does and how to extend
it. These job scripts only decide which images, which classifiers, and which
Slurm resources.

## One-time setup (login node)

```bash
cd /project/uwyo-0007/software/detection-projects
bash scripts/ai4wy/setup_env.sh          # uv venv at /project/uwyo-0007/software/.venv-wytrap
bash scripts/ai4wy/prefetch_weights.sh   # MegaDetector + BioCLIP 2 into /project/uwyo-0007/software/models
```

`setup_env.sh` installs `requirements-ai4wy.txt` (PytorchWildlife, pybioclip,
speciesnet, torch for CUDA 13) and `wytrap` in editable mode, and registers a
Jupyter kernel named "wytrap (ai4wy)". SpeciesNet and the AddaxAI zoo models
download on first use from Hugging Face; compute nodes have outbound HTTPS.

The group directory is setgid and every script sets `umask 002`, so files
written by one member stay writable by the others.

## Jobs

| script | what it does | typical time |
|---|---|---|
| `fetch_and_run_idaho.sbatch` | fetch a LILA Idaho subset on the compute node, run BioCLIP 2, evaluate | 20 min at `PER_CLASS=20` |
| `run_idaho.sbatch` | BioCLIP 2 + eval on an already-fetched subset | 10 min |
| `detector_sweep.sbatch` | several detector settings on one subset, one table | 1 h |
| `compare_classifiers.sbatch` | every classifier on the same boxes and candidate set, one table | 30 min at `PER_CLASS=100` |
| `deploy_infer.sbatch` | unlabelled folder: detector once, three classifiers, merged table | 2 h per 20k images |
| `deploy_submit.sh` | `deploy_infer` as a Slurm array, one task per camera folder | |

All take their settings as environment variables, documented in each file's
header. Submit from the repo root so `logs/` and `taxonomy/` resolve:

```bash
PER_CLASS=100 NEGATIVES=40 sbatch scripts/ai4wy/fetch_and_run_idaho.sbatch
DATA_DIR=/project/uwyo-0007/data/idaho-100pc sbatch scripts/ai4wy/compare_classifiers.sbatch
```

Each job is a handful of `wytrap` calls; read the script to see them, and
copy the lines into an interactive session (`salloc --account=uwyo-0007
--partition=gp-1,gp-2 --gres=gpu:1`) to try something new.

## Evaluation data: LILA Idaho Camera Traps

`scripts/fetch_idaho_subset.py` pulls a class-balanced subset (one frame per
sequence by default) plus camera-problem negatives, writing
`<out>/images/loc_XXXX_im_NNNNNN.jpg` and `<out>/labels.json`. Labels are
per sequence and carry no boxes, so `wytrap eval` scores presence/absence
and image-level species. Humans, vehicles and dogs are absent from the
public set, and the "other" co-label on rare species is ignored. Species
present: deer, elk, moose, pronghorn, bighorn sheep, cattle, wolf, coyote,
fox, bear, mountain lion, bobcat, skunk, lagomorphs, squirrels, turkey,
grouse.

Caveat for every number: SpeciesNet and the Western USA model were trained
largely on LILA data, very likely including these cameras; BioCLIP 2 was
not. Idaho measures the supervised models on familiar data.

## Results so far (idaho-100pc, 982 animal images, redwood boxes, shared candidate set)

| classifier | top-1 | hierarchical | top-3 |
|---|---|---|---|
| SpeciesNet, restricted | 0.960 | 0.960 | 0.985 |
| SpeciesNet + roll-up, restricted | 0.942 | 0.976 | 0.984 |
| SpeciesNet, all 2,498 labels | 0.945 | 0.954 | 0.983 |
| Western USA SDZWA, restricted | 0.929 | 0.929 | 0.978 |
| BioCLIP 2 + prior correction | 0.841 | 0.841 | 0.928 |
| BioCLIP 2 | 0.833 | 0.833 | 0.906 |

Detector: MDv1000 redwood at native 1280 px, floor 0.30, is the best value
(F1 0.89 for presence/absence). Larger input sizes and tiling trade recall
for false positives on these cameras.

## Deployment: unlabelled camera folders

The WySoundscape images live on MedicineBow and `/project` is not shared,
so they were copied to `/project/uwyo-0007/data/CameraTrap_test` (Globus
for hundreds of GB; rsync for a single camera). Then:

```bash
# one camera
IMAGES=/project/uwyo-0007/data/CameraTrap_test/<CAM> sbatch scripts/ai4wy/deploy_infer.sbatch

# every camera folder as an array (8 at once), then one table for all of them
scripts/ai4wy/deploy_submit.sh /project/uwyo-0007/data/CameraTrap_test /project/uwyo-0007/data/CameraTrap_output
wytrap merge --combine /project/uwyo-0007/data/CameraTrap_output
```

Each task writes `<out>/<camera>/merged/images.csv` (one headline label per
classifier per image, person and vehicle counts, consensus) and `boxes.csv`.
Budget about 4 img/s for BioCLIP 2 (detector included), 9 for SpeciesNet and
17 for the zoo model on one GPU, so roughly 35 GPU-hours per 300k images,
spread across the array. Tasks resume, so a killed one can be resubmitted.

## Gotchas

- Partitions are `gp-1`/`gp-2`; the ARCC docs' `ai4wy-1/2` do not exist.
- 24 h wall-time requests were rejected; the scripts ask for 2–8 h.
- Zenodo (MegaDetector v6 weights) is down for hours at a time. The default
  detector, MDv1000 redwood, comes from Hugging Face and needs no Zenodo.
- Keep `HF_HOME` and `TORCH_HOME` under `/project` (the scripts do), or the
  weights land in a home-directory quota.
- `uv` installs to `~/.local/bin`; `export PATH="$HOME/.local/bin:$PATH"`
  if the shell cannot find it.
