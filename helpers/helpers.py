import copy
import json
import os
import re

from koger_detection.utils.json import rename_categories


# Default mapping that collapses predator and deer subcategories into broader
# classes. All other categories are mapped to themselves so rename_categories
# preserves them.
DEFAULT_CATEGORY_MERGES = {
    "coyote": "Canid",
    "wolf": "Canid",
    "fox": "Canid",
    "grizzly bear": "Bear",
    "black bear": "Bear",
    "mule deer": "Deer",
    "white-tailed deer": "Deer",
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
