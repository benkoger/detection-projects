from typing import Any, Sequence, Tuple, cast
import random

import albumentations as A
import albumentations.augmentations.crops.functional as fcrop

import numpy as np

class RandomCropWithBBox(A.crops.transforms.RandomCrop):
    """ Takes a random crop of fixed size making sure to contain at least one bounding box.
    """

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, tuple[int, int, int, int]]:
        
        # TODO: Make this a parameter of the class
        buffer = 200
        
        img = params["image"]
        image_height, image_width = img.shape[:2]

        if (image_height == self.height) and (image_width == self.width):
            crop_coords = fcrop.get_crop_coords(image_height, image_width, self.height, self.width, 0, 0)
            return {"crop_coords": crop_coords}

        if self.height > image_height or self.width > image_width:
            raise CropSizeError(
                f"Crop size (height, width) exceeds image dimensions (height, width):"
                f" {(self.height, self.width)} vs {img.shape[:2]}",
            )
        # Note, bbox coordinates are in range 0 to 1
        bbox = random.choice(params['bboxes'])
        
        # Choose location of left side of the crop
        # Farthest left crop could start and contain this box (in pixels)
        min_left = bbox[2] * image_width - self.width + buffer
        max_left = bbox[0] * image_width - buffer
        # Normalize so 0 is far left size and 1 is width minus width of crop
        min_left = min_left / (image_width - self.width)
        max_left = max_left / (image_width - self.width)
        # Must be between 0.0  and 1
        min_left = np.clip(min_left, 0.0, 0.999)
        max_left = np.clip(max_left, 0.0, 0.999)
        
        # Chose value for left side of crop from within range of min_left and max_left
        w_start = random.uniform(min_left, max_left)

        # Choose location of top of the crop
        # top most crop could start and contain this box (in pixels)
        min_height = bbox[3] * image_height - self.height + buffer
        max_height = bbox[1] * image_height - buffer
        # Normalize so 0 is far left size and 1 is width minus width of crop
        min_height = min_height / (image_height - self.height)
        max_height = max_height / (image_height - self.height)
        # Must be between 0.0  and 1
        min_height = np.clip(min_height, 0.0, 0.999)
        max_height = np.clip(max_height, 0.0, 0.999)
        # Chose value for left side of crop from within range of min_left and max_left
        h_start = random.uniform(min_height, max_height)

        crop_coords = fcrop.get_crop_coords(image_height, image_width, self.height, self.width, h_start, w_start)
        return {"crop_coords": crop_coords}

    @property
    def targets_as_params(self) -> list[str]:
        return ["image", "bboxes"]