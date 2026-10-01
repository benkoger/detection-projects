# wytrap

A camera-trap pipeline for Wyoming: **MegaDetector** finds the animals,
**a classifier of your choice** names them, and every classifier is held to
the **same species list** so results are comparable. One command, one
record format, one evaluator.

This guide assumes you can run Python and have used a camera-trap model
before, but not that you know this code. It covers using wytrap first, then
how it is built, then how to extend it.

---

## 1. What it does

```
images ──► MegaDetector ──► boxes ──► classifier ──► records (JSON)
                             │                          │
                             │ wytrap classify          │ wytrap eval    (with labels)
                             └──► another classifier    │ wytrap merge   (without)
                                   on the same boxes    └──► tables
```

Three classifiers are built in, and they answer different questions:

| `--classifier` | what it is | when to use it |
|---|---|---|
| `bioclip` | BioCLIP 2, zero-shot: scores each crop against species *names*; no training on camera traps | species the supervised models never saw; quick experiments with a new list |
| `speciesnet` | Google's SpeciesNet classifier network, 2,498 labels, per box | the strongest per-crop classifier on North American data |
| `speciesnet-ensemble` | SpeciesNet as shipped: classifier + taxonomic roll-up + geofence; one label per image | production use: rarely wrong because it backs off to genus/family when unsure |
| `addax` | any model from the AddaxAI zoo (`--model Addax-Data-Science/WUSA-SDZWA-v1` is Western USA) | regional supervised models |
| `none` | detection only | |

On the Idaho Camera Traps benchmark (982 images, shared candidate set)
SpeciesNet restricted to the regional list reached 96% top-1, Western USA
93%, BioCLIP 2 84%. The ensemble's roll-up costs 2 points of species-level
accuracy and in exchange is almost never outright wrong (97.6% "correct or
a correct ancestor").

## 2. Install

On the ai4wy cluster everything is set up by `scripts/ai4wy/setup_env.sh`
(see `scripts/ai4wy/README.md`). Elsewhere:

```bash
pip install torch torchvision          # pick the build for your CUDA from pytorch.org
pip install -e wytrap                  # core: PytorchWildlife, pybioclip, Pillow, numpy
pip install "speciesnet>=5.0"          # optional, for the two SpeciesNet classifiers
pip install matplotlib                 # optional, for the confusion-matrix figure
```

Weights download on first use into the Hugging Face cache. Set `HF_HOME`
(and `TORCH_HOME`) to somewhere with space; on a cluster, somewhere shared.

## 3. Use it

### Arrange a raw camera dump

```bash
wytrap ingest --source /data/CameraTrap_raw --out /data/CameraTrap_test
```

makes `images/<camera>/` (one flat folder per camera; Reconyx sub-folder
names fold into the file name), `labels.json` with empty labels in the same
schema the evaluator reads, and `manifest.json` with per-camera counts, EXIF
date ranges and sequences (frames under a minute apart). Files are moved by
default; `--copy` or `--link` keep the source. The Idaho subsets from
`scripts/fetch_idaho_subset.py` have the same layout, so every later command
works the same on both.

### Detect and classify a folder

```bash
wytrap detect --input /data/cam01 --output /runs/cam01-bioclip \
    --classifier bioclip --vocab taxonomy/wyoming_vocab.csv
```

Walks the folder recursively, runs MegaDetector (MDv1000 "redwood" by
default), classifies every animal box with BioCLIP 2 restricted to the
species in the vocabulary, and writes:

```
/runs/cam01-bioclip/
  all_records.jsonl        one line per image: boxes + labels  <- most tools read this
  manifest.json            detector, classifier, every setting
  prompts.json             (BioCLIP only) the prompt list, for calibration
  wytrap.log               the console log, appended across resumes
  <mirrors the image tree>/IMG_0001.json   the same record, one file per image
```

It resumes by default: rerun the same command after a crash and finished
images are skipped. People and vehicles are recorded with MegaDetector's
label and not classified.

### Try another classifier on the same boxes

Detection is the slow part and should be shared. `classify` re-labels the
boxes of an earlier run:

```bash
wytrap classify --records /runs/cam01-bioclip/all_records.jsonl \
    --output /runs/cam01-speciesnet --classifier speciesnet-ensemble \
    --vocab taxonomy/wyoming_vocab.csv --admin1 WY

wytrap classify --records /runs/cam01-bioclip/all_records.jsonl \
    --output /runs/cam01-wusa --classifier addax --model Addax-Data-Science/WUSA-SDZWA-v1 \
    --vocab taxonomy/wyoming_vocab.csv
```

### Compare them

With no ground truth, put the runs side by side:

```bash
wytrap merge --out /runs/cam01-merged \
    --arm bioclip=/runs/cam01-bioclip --arm speciesnet=/runs/cam01-speciesnet --arm wusa=/runs/cam01-wusa
```

`images.csv` has one row per image with each classifier's headline label
and score, person and vehicle counts, and a `consensus` column filled when a
majority agree. `boxes.csv` is the same per box. Rows where the classifiers
disagree are the ones worth a human look.

With labels (a `labels.json` as written by `scripts/fetch_idaho_subset.py`):

```bash
wytrap eval --labels /data/idaho/labels.json --pred /runs/idaho-speciesnet \
    --vocab taxonomy/idaho_vocab.csv
```

reports presence/absence precision and recall swept over the detector
threshold, top-1/top-3 accuracy, a confusion matrix, per-class recall with
the top confusions, accuracy against coverage (how accurate the model is on
the fraction of images it is most sure about), day/night splits, and with a
vocabulary the hierarchical score (correct node, or a roll-up to a taxon
containing it).

### Audit the species list

Any vocabulary is a hypothesis about what the cameras see. To test it, run
a classifier *unrestricted* on the same boxes and count:

```bash
wytrap classify --records /runs/cam01-bioclip/all_records.jsonl --output /runs/cam01-open \
    --classifier speciesnet-ensemble --admin1 WY            # no --vocab
wytrap census --out /runs/census --run cam01=/runs/cam01-open --vocab taxonomy/wyoming_vocab.csv
```

`census.csv` has one row per label with counts per camera and whether the
label resolves to a vocabulary node; labels marked outside the vocabulary
with a real count are additions, nodes never seen are candidates to drop.

### Prior correction for BioCLIP

Zero-shot models have a built-in bias: some prompts win on any crop. For
BioCLIP 2 on Idaho, "least chipmunk" absorbed small blurry boxes. The
correction is label-free (it uses only the model's own scores on unlabelled
crops) and is estimated leave-location-out:

```bash
wytrap calibrate --pred /runs/idaho-bioclip --output /runs/idaho-bioclip-calib
```

writes recalibrated records and a `prompt_bias.json` to pass as
`--prompt-bias` on later `detect` runs. It is logit adjustment (Menon et
al. 2021) applied to zero-shot prompts, as in DebiasPL (Wang et al. 2022).

### The vocabulary

`taxonomy/*.csv` lists the species a deployment can see, as *taxon nodes*:

```
label,class,order,family,genus,species,prompts
deer,Mammalia,Artiodactyla,Cervidae,Odocoileus,,Odocoileus hemionus|mule deer;Odocoileus virginianus|white-tailed deer
elk,Mammalia,Artiodactyla,Cervidae,Cervus,canadensis,Cervus canadensis|elk
lagomorph,Mammalia,Lagomorpha,,,,Lepus americanus|snowshoe hare;Sylvilagus nuttallii|mountain cottontail
```

A node's rank is its deepest filled rank ("deer" is the genus, "lagomorph"
the order). `prompts` are the member species, which become BioCLIP's prompt
list, and the rule in `Vocab.candidate_classes` decides for *every* model
which of its classes count as inside the vocabulary: listed species, roll-up
labels at or above the node's rank, and genera that contain a listed
species. So SpeciesNet keeps "odocoileus species" and loses 120 African
squirrels, and Western USA keeps "chipmunk" because it is the only squirrel
it has. `wytrap vocab taxonomy/wyoming_vocab.csv --speciesnet-model hf:... --zoo ...`
prints exactly what each model keeps.

To change what the pipeline looks for, edit the CSV. Names are GBIF
backbone canonical names; a species BioCLIP has never seen the name of will
still get a prompt, just a poor one.

## 4. How it is built

```
wytrap/
  cli.py             argparse front end; one function per subcommand
  ingest.py          raw dump -> images/ + labels.json + manifest.json
  run.py             the per-image loop shared by detect and classify
  detector.py        MegaDetector wrapper (PytorchWildlife); HF-hosted v5/v1000, Zenodo v6
  classifiers/
    base.py          BoxClassifier interface, BoxInput / BoxResult, vocabulary masking
    bioclip.py       BioCLIP 2: prompts, three crop scales, prompt bias
    speciesnet.py    SpeciesNetClassifier (per box) and SpeciesNetEnsemble (per image)
    addax.py         loads a zoo repo's inference.py the way AddaxAI does
  vocab.py           Vocab: load the CSV, resolve a prediction to a node, candidate sets
  taxonomy.py        label-mapping tables (SpeciesNet label parsing, Idaho merges)
  io.py              DetectionRecord / ImageRecord and their JSON (de)serialisation
  calibrate.py       prior correction
  evaluate.py        image-level evaluation
  merge.py           side-by-side tables
  census.py          per-camera label counts audited against a vocabulary
  species_lists.py   built-in BioCLIP species lists (wyoming_all, ...)
```

**The record** (`io.py`) is the contract between everything. One
`ImageRecord` per image holds `DetectionRecord`s, each with the box, the
detector's score and label, a `quality` tag, and the classification fields:
`label`, `scientific_label`, `cls_score`, `topk`, `lineage`, `scale`,
`source`. Every classifier fills the same fields, so `eval` and `merge`
never need to know which model produced a run.

**Box quality** (`run.assess_box_quality`) tags each box `ok`, `low_pixels`
(short side under 60 px), `truncated` (a fifth of the perimeter on the frame
edge) or `thin`. Classifiers still run on bad boxes by default, but the
evaluator scores only `ok` ones and `merge` prefers them for the headline
label. On real cameras about half of all boxes are `low_pixels`: distant
animals the detector sees but no classifier can name.

**The classifier interface** (`classifiers/base.py`):

```python
class BoxClassifier:
    name = "..."
    def classify_image(self, image, image_path, boxes: list[BoxInput]) -> list[BoxResult | None]
    def class_lineages(self) -> dict[str, dict]      # optional: enables --vocab masking
    def describe(self) -> dict                        # goes into manifest.json
```

It gets the whole image and all boxes, not pre-cut crops, because each model
crops differently: BioCLIP looks at a tight crop, a 2x padded crop and the
whole frame and keeps the best; SpeciesNet pads to a square its own way; zoo
models ship a `get_crop`. Returning `None` for a box leaves it unclassified.
`animal_indices()` and `normalised()` are the helpers most implementations
need.

**Vocabulary masking** happens in the base class: if the subclass can say
what lineage each of its classes has, `candidate_classes()` returns the
allowed subset and the subclass renormalises over it (SpeciesNet does this
through its own target-species mechanism, which also gives it logits over
just those classes).

**The loop** (`run.py`): `process_folder` detects and classifies,
`reclassify` reads records and classifies; both call `classify_boxes`,
which turns `BoxResult`s into record fields, and `_run`, which owns
progress, ETA, resume, error capture and the summary. Per-image JSON paths
mirror the image tree below the images' common root, so two cameras whose
Reconyx folders are both called `100RECNX` never collide.

## 5. Extend it

### Add a classifier

1. Create `classifiers/mymodel.py` with a `BoxClassifier` subclass. Load
   weights in `__init__`, implement `classify_image`. If the model's classes
   have taxonomy, return it from `class_lineages()` and apply
   `self.candidate_classes()` so `--vocab` works.
2. Register it in `classifiers/__init__.py`: add the name to `CLASSIFIERS`
   and a branch in `build_classifier`.
3. Run it on an existing run's boxes with `wytrap classify --classifier mymodel`
   and score it with `wytrap eval`. No other file needs to change.

The AddaxAI zoo is the quickest source of new supervised models: any repo
with an `inference.py` works through `--classifier addax --model <repo>`
with no code at all.

### Fine-tune on your own crops

BioCLIP 2's image encoder is a good base for a linear probe: run `detect`,
take the `ok` boxes with verified labels, embed the crops with
`open_clip`, fit a logistic regression, and wrap it as a classifier whose
`classify_image` embeds and scores. A few dozen verified crops per class is
where that starts to pay off; the review queue of a deployment produces
them.

### Add a vocabulary

Copy `taxonomy/wyoming_vocab.csv`, edit rows, and check it with
`wytrap vocab`. The evaluator's hierarchical scoring and every classifier's
masking follow from the file; nothing is hard-coded to Idaho except the
`IDAHO_EVAL_MERGES` fallback used when no vocabulary is given.

### Change the detector

`detector.py` wraps PytorchWildlife. `HF_WEIGHTS` maps names to Hugging
Face files for the YOLOv5-family checkpoints; MDv6 variants come from Zenodo
through PytorchWildlife itself. `--det-imgsz` and `--tile` change the input
resolution and sliced inference; on full-frame camera-trap images neither
helped, but zoomed-out deployments may differ.

### Known limits

- SpeciesNet and Western USA were trained on LILA data, which includes the
  Idaho evaluation cameras. Their Idaho numbers are optimistic for new sites.
- BioCLIP 2's prior correction needs several camera locations to estimate;
  a single folder gives it nothing to leave out. Reuse a bias file from a
  larger run (`--prompt-bias`) or calibrate on the combined output.
- The SpeciesNet ensemble produces one label per image and writes it onto
  every animal box; mixed-species images get the dominant animal's label.
- Nothing here handles sequences or video; `wytrap eval --sequence-level`
  groups frames by the `seq_id` in the labels file, but inference is per
  image.
