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