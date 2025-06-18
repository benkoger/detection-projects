import copy
import json
import math
import os
import requests

import PIL
from PIL import Image

import numpy as np
import torch
import torchvision

def get_ious(dataset):
    """Return all non-zeros iou values for bboxes in image dataset."""

    all_ious = []
    for ind, item in enumerate(dataset):
        ious = torchvision.ops.box_iou(item[1]['boxes'],
                                       item[1]['boxes'])
        # Get the upper triangle of the matrix excluding the diagnal
        idx = torch.triu_indices(*ious.shape, offset=1)
        iou = ious[idx[0], idx[1]].flatten()

        all_ious.append(iou[iou!=0].numpy())
    return np.concatenate(all_ious)

def get_bounding_box_areas(coco_json):
    """ Return a list of the areas of all bounding boxes in the json file."""
    bbox_areas = []

    for ann in coco_json['annotations']:
        area = ann['bbox'][2] * ann['bbox'][3]
        bbox_areas.append(area)

    return bbox_areas

def scale_annotations_and_images(coco_json, scale_factor, image_folder, 
                                 new_annotation_file):
    """ Scale images and annotations by some factor.
    
    Args:
        coco_json: coco format dict
        scale_factor: how much to scale images and annotations
        image_folder: where images should be saved
        new_annotation_file: where json should be saved
    """
    coco_json = copy.deepcopy(coco_json)
    for ann in coco_json['annotations']:
        ann['bbox'] = [int(v * scale_factor) for v in ann['bbox']]
    
    for image_info in coco_json['images']:
        scaled_file = f"scaled-{scale_factor}-{image_info['file_name']}"
        image_info['file_name'] = scaled_file

        # Load image
        url = image_info['coco_url']
        im = Image.open(requests.get(url, stream=True).raw)
    
        scaled_width = int(math.ceil(im.size[0]*scale_factor))
        scaled_height = int(math.ceil(im.size[1]*scale_factor))
        
        image_info['width'] = scaled_width
        image_info['height'] = scaled_height

        if scale_factor < 1:
            resample = PIL.Image.LANCZOS
        else:
            resample = PIL.Image.BICUBIC
        im = im.resize((scaled_width, scaled_height), resample=resample) 
        path = os.path.join(image_folder, scaled_file)
        im.save(path) 
    
    with open(new_annotation_file, "w") as write_file:
        json.dump(coco_json, write_file, indent=4, 
                  separators=(',', ': '))