import numpy as np
from pycocotools.cocoeval import COCOeval, Params
import torch

import koger_detection.torchvision_reference.utils as utils
from koger_detection.torchvision_reference.coco_utils import get_coco_api_from_dataset
from koger_detection.torchvision_reference.engine import _get_iou_types
from koger_detection.torchvision_reference.coco_eval import CocoEvaluator

class CustomCocoParams(Params):
    #Allows custom evaulation parameters for COCO evaulation
    
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation (list of [min_area, max_area])
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    
    def __init__(self, iouType='segm', **kwargs):
        super().__init__(iouType)
        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)
            else:
                raise ValueError("unexpected kwarg value", key)
        
class CustomCocoEval(COCOeval):
    """Allows custom evaulation parameters for COCO evaulation"""
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm', **kwargs):
        super().__init__(cocoGt, cocoDt, iouType)
        self.params = CustomCocoParams(iouType, **kwargs) # parameters
        # These are empty lists when params is created 
        # COCOeval fills in when initializes, this is how they are initialized
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

        
    def summarize(self):
        print("Summarize cannot be run with custom parameters. Use COCOeval.")
        
class CustomCocoEvaluator(CocoEvaluator):
    """Allows custom evaulation parameters for COCO evaulation"""
    def __init__(self, coco_gt, iou_types, **kwargs):
        super().__init__(coco_gt, iou_types)
        
        for iou_type in iou_types:
            self.coco_eval[iou_type] = CustomCocoEval(coco_gt, 
                                                      iouType=iou_type, 
                                                      **kwargs)

def calc_precision_recall(predictor, data_loader, score_threshs=[0.2, 0.7]):
    """ Calculate precision recall for a trained model using Predictor.

    Similar to koger_detection.engine.evaluate but doesn't calculate loss and 
    uses Predictor. Assumes batch size of one.
    
    predictor: koger_detection.obj_det.predictor.Predictor
    dataloader: dataloader that returns image and labels
        Image is in format returned from PIL.Image.open()
    """
    if data_loader.batch_size != 1:
        print("Warning: expects batch size must be 1.")
        return None
    
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(predictor)
    
    iouThrs = [0.2, 0.5, 0.8]
    # Right now only designed for one areaRng value
    areaRng = [[0 ** 2, 1e5 ** 2]]
    maxDets = [350]
    
    coco_evaluator = CustomCocoEvaluator(coco, iou_types, 
                                         iouThrs=iouThrs,
                                         areaRng=areaRng,
                                         maxDets=maxDets
                                        )

    losses = []
    for batch_num, (images, targets) in enumerate(data_loader):

        outputs = predictor(images[0])

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in [outputs]]
        targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)
    
    # gather the stats from all processes

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # results in dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [iouThrs,recThrs,catIds,areaRng,maxDets] parameter dimensions (see above)
    #  precision  - [iouThrs x recThrs x catIds x areaRng x maxDets] precision for every evaluation setting
    #  recall     - [iouThrs x catIds x areaRng x maxDets] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    coco_eval = coco_evaluator.coco_eval['bbox'].eval
    # params actually used for evaluation
    params = coco_eval['params']
    precision = coco_eval['precision']
    
    obj_cats = data_loader.dataset.coco.dataset['categories']
    
    # default iou thresh values are 
    print(f"Using an iou thresholds of {params.iouThrs}")

    print(f"Using score thresholds {score_threshs}")
    
    area_rng_ind = 0
    print(f"Using area size range {params.areaRng[area_rng_ind]}")
    max_dets_ind = 0
    scores = coco_eval['scores']
    
    for cat_ind, cat in enumerate(obj_cats):
        for iou_ind, iou_p in enumerate(params.iouThrs):
            print(f"iou: {iou_p}")
            for score_thresh in score_threshs:
                print(f"score_thresh: {score_thresh}")
              
                above_thresh = np.argwhere(
                    scores[iou_ind, :, cat_ind, area_rng_ind, max_dets_ind] >= score_thresh
                )
                if len(above_thresh) == 0:
                    print(f"{cat['name']:8}: recall: {np.nan:1.2f}, precision: {np.nan:1.3f}")
                    continue
                max_ind = np.max(above_thresh)
                category = cat['name']
                recall = params.recThrs[max_ind]
                prec = precision[iou_ind, max_ind, cat_ind, area_rng_ind, max_dets_ind]

                print(f"{cat['name']:8}: recall: {recall:1.2f}, precision: {prec:1.3f}")

    return coco_evaluator