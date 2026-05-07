import copy
import json
import os
import random
import re

from koger_detection.utils.json import rename_categories


# Default mapping that collapses predator and deer subcategories into broader
# classes. All other categories are mapped to themselves so rename_categories
# preserves them.
DEFAULT_SUPERCATEGORIES = {
    "Bear": "predator",
    "Canid": "predator",
    "badger": "predator",
    "Deer": "ungulate",
    "elk": "ungulate",
    "moose": "ungulate",
}


def assign_supercategories(coco, supercategories=None, default="other"):
    """Set the 'supercategory' field on each entry of coco['categories']."""
    if supercategories is None:
        supercategories = DEFAULT_SUPERCATEGORIES
    for cat in coco["categories"]:
        cat["supercategory"] = supercategories.get(cat["name"], default)
    return coco


DEFAULT_CATEGORY_MERGES = {
    "coyote": "Canid",
    "wolf": "Canid",
    "fox": "Canid",
    "grizzly bear": "Bear",
    "black bear": "Bear",
    "mule deer": "Deer",
    "white-tailed deer": "Deer",
}


# Eval-time-only merges for the YNP_TESTBED species list.
#
# Inference now uses species-level prompts (max-over-members strategy), so
# per-image JSONs preserve the actual species name (e.g. "yellow-bellied
# marmot"). At eval time we collapse both GT raw labels and predicted species
# names to the same vocabulary so the confusion matrix is readable.
#
# Includes everything in DEFAULT_CATEGORY_MERGES (predator/deer collapse)
# plus all member species expanded into the testbed list.
YNP_EVAL_MERGES = {
    **DEFAULT_CATEGORY_MERGES,

    # Canids (predicted species names not already covered above)
    "gray wolf": "Canid",
    "red fox": "Canid",
    "swift fox": "Canid",
    "gray fox": "Canid",

    # Rodents → "rodent"
    "yellow-bellied marmot": "rodent",
    "red squirrel": "rodent",
    "fox squirrel": "rodent",
    "Wyoming ground squirrel": "rodent",
    "Uinta ground squirrel": "rodent",
    "Columbian ground squirrel": "rodent",
    "golden-mantled ground squirrel": "rodent",
    "thirteen-lined ground squirrel": "rodent",
    "least chipmunk": "rodent",
    "meadow vole": "rodent",
    "deer mouse": "rodent",

    # Birds → "bird"
    "common raven": "bird",
    "American crow": "bird",
    "black-billed magpie": "bird",
    "Canada jay": "bird",
    "Steller's jay": "bird",
    "Clark's nutcracker": "bird",
    "ruffed grouse": "bird",
    "dusky grouse": "bird",
    "white-crowned sparrow": "bird",
    "wild turkey": "bird",
    "golden eagle": "bird",
    "bald eagle": "bird",
    "red-tailed hawk": "bird",
    "great horned owl": "bird",
    "American robin": "bird",
    "American woodcock": "bird",
    "house finch": "bird",
    "black-capped chickadee": "bird",
    "western meadowlark": "bird",
    "european starling": "bird"
}


def merge_categories(coco, category_merges=None, out_file=None):
    """Merge groups of category names into combined classes.

    Uses koger_detection.utils.json.rename_categories under the hood. Any
    category in the input not listed in `category_merges` is mapped to itself
    so it is preserved unchanged.

    Args:
        coco: either a path to a coco json file or an already-loaded coco dict
        category_merges: dict mapping old category name -> new category name.
            Defaults to DEFAULT_CATEGORY_MERGES (Canid / Bear / Deer).
        out_file: if given, save the resulting coco dict to this path.

    Returns the new coco dict.
    """
    if category_merges is None:
        category_merges = DEFAULT_CATEGORY_MERGES

    if isinstance(coco, str):
        with open(coco, "r") as f:
            coco = json.load(f)
    coco = copy.deepcopy(coco)

    full_mapping = {}
    for cat in coco["categories"]:
        name = cat["name"]
        full_mapping[name] = category_merges.get(name, name)

    valid_cat_ids = {cat["id"] for cat in coco["categories"]}
    orphans = [a for a in coco["annotations"] if a["category_id"] not in valid_cat_ids]
    if orphans:
        bad_ids = sorted({a["category_id"] for a in orphans})
        print(f"Warning: dropping {len(orphans)} annotations with unknown "
              f"category_id(s) {bad_ids} (not in coco['categories']).")
        coco["annotations"] = [a for a in coco["annotations"]
                               if a["category_id"] in valid_cat_ids]

    return rename_categories(coco, full_mapping, out_file=out_file)


# Drops every YNP_<number>A_ image (YNP_1A_, YNP_2A_, ..., YNP_12A_) while
# leaving B/C variants in place.
YNP_A_PATTERN = r"^YNP_\d+A_"


def remove_images_by_pattern(coco, pattern=YNP_A_PATTERN, out_file=None):
    """Remove all images (and their annotations) whose file_name matches a regex.

    Args:
        coco: either a path to a coco json file or an already-loaded coco dict
        pattern: regex matched with re.search against each image's file_name.
            Defaults to YNP_A_PATTERN, which removes any YNP_XA image.
        out_file: if given, save the resulting coco dict to this path.

    Returns the new coco dict.
    """
    if isinstance(coco, str):
        with open(coco, "r") as f:
            coco = json.load(f)
    coco = copy.deepcopy(coco)

    regex = re.compile(pattern)
    kept_images = [im for im in coco["images"] if not regex.search(im["file_name"])]
    kept_image_ids = {im["id"] for im in kept_images}
    kept_anns = [a for a in coco["annotations"] if a["image_id"] in kept_image_ids]

    for new_id, ann in enumerate(kept_anns):
        ann["id"] = new_id + 1

    coco["images"] = kept_images
    coco["annotations"] = kept_anns

    print(f"Removed images matching '{pattern}': "
          f"{len(kept_images)} images and {len(kept_anns)} annotations remain.")

    if out_file:
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(coco, f, indent=4, separators=(",", ": "))

    return coco


def rebalance_train_val(train_coco, val_coco, fraction_val=0.25, seed=0,
                        train_out=None, val_out=None):
    """Recombine train+val and resplit, hitting fraction_val while ensuring
    each category has at least one image in both splits.

    Args:
        train_coco, val_coco: either paths to coco json files or loaded dicts.
            Both must share the same `categories` list.
        fraction_val: target fraction of images that should land in val.
        seed: RNG seed for the shuffle.
        train_out, val_out: if given, save resulting splits to these paths.

    Returns (new_train_coco, new_val_coco).
    """
    if isinstance(train_coco, str):
        with open(train_coco, "r") as f:
            train_coco = json.load(f)
    if isinstance(val_coco, str):
        with open(val_coco, "r") as f:
            val_coco = json.load(f)

    if train_coco["categories"] != val_coco["categories"]:
        raise ValueError("train and val coco dicts must share the same categories")

    images = copy.deepcopy(train_coco["images"]) + copy.deepcopy(val_coco["images"])
    anns = copy.deepcopy(train_coco["annotations"]) + copy.deepcopy(val_coco["annotations"])

    # Dedupe images by id (train+val from a common source share an id space).
    seen = {}
    for im in images:
        seen.setdefault(im["id"], im)
    images = list(seen.values())

    img_cats = {}
    for ann in anns:
        img_cats.setdefault(ann["image_id"], set()).add(ann["category_id"])

    # Drop images with no annotations - they don't contribute to either split.
    images = [im for im in images if im["id"] in img_cats]

    rng = random.Random(seed)
    rng.shuffle(images)

    n_total = len(images)
    n_val_target = int(round(n_total * fraction_val))

    val_ids, train_ids = set(), set()

    cat_ids = sorted({c for cs in img_cats.values() for c in cs})
    cat_image_count = {c: 0 for c in cat_ids}
    for cs in img_cats.values():
        for c in cs:
            cat_image_count[c] += 1

    # Seed both splits with one image per category, starting with the rarest
    # categories so we don't accidentally lock them all into one split.
    for cat in sorted(cat_ids, key=lambda c: cat_image_count[c]):
        if cat_image_count[cat] < 2:
            print(f"Warning: category id {cat} has only "
                  f"{cat_image_count[cat]} image(s); cannot guarantee both splits.")
        val_has = any(cat in img_cats[i] for i in val_ids)
        train_has = any(cat in img_cats[i] for i in train_ids)
        for im in images:
            if val_has and train_has:
                break
            if im["id"] in val_ids or im["id"] in train_ids:
                continue
            if cat not in img_cats[im["id"]]:
                continue
            if not val_has:
                val_ids.add(im["id"])
                val_has = True
            elif not train_has:
                train_ids.add(im["id"])
                train_has = True

    # Distribute remaining images to hit the target val fraction.
    for im in images:
        if im["id"] in val_ids or im["id"] in train_ids:
            continue
        if len(val_ids) < n_val_target:
            val_ids.add(im["id"])
        else:
            train_ids.add(im["id"])

    def build(template, kept_ids):
        new = {k: copy.deepcopy(v) for k, v in template.items()
               if k not in ("images", "annotations")}
        id_to_im = {im["id"]: im for im in images}
        new["images"] = [id_to_im[i] for i in kept_ids]
        new["annotations"] = [a for a in anns if a["image_id"] in kept_ids]
        for new_id, ann in enumerate(new["annotations"]):
            ann["id"] = new_id + 1
        return new

    new_train = build(train_coco, train_ids)
    new_val = build(val_coco, val_ids)

    print(f"Rebalanced: {len(new_train['images'])} train / "
          f"{len(new_val['images'])} val "
          f"(val fraction = {len(new_val['images']) / n_total:.3f}, "
          f"target {fraction_val}).")

    for path, coco in ((train_out, new_train), (val_out, new_val)):
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump(coco, f, indent=4, separators=(",", ": "))

    return new_train, new_val
