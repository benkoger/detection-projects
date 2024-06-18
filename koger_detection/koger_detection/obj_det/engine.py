import math
import os
import random
import time
from datetime import datetime

import cv2
from dotenv import load_dotenv
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor
from pycocotools.cocoeval import COCOeval, Params

import koger_detection.torchvision_reference.utils as utils
from koger_detection.obj_det.mydatasets import CocoDetection
from koger_detection.torchvision_reference.coco_eval import CocoEvaluator
from koger_detection.torchvision_reference.coco_utils import get_coco_api_from_dataset
from koger_detection.torchvision_reference.engine import _get_iou_types


def worker_init_fn(worker_id):
    np.random.seed(datetime.now().microsecond 
                   + worker_id * 1000000)
    random.seed(datetime.now().microsecond 
                   + worker_id * 1000000)
    

def get_detection_model(num_classes, **model_cfg):
    """ Load pretrained detection model.
    
    Args:
        num_classes: number of possible classes (counting background class)
            Note: num_classes can be read from model_cfg
        **model_cfg: possible useful arguments:
            - trainable_backbone_layers
            - box_detections_per_img
            - box_nms_thresh
            - fixed_size
            - max_size
            - min_size
    """
    if model_cfg['model_type'] == "bbox":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT",
            **model_cfg
        )
    elif model_cfg['model_type'] == "bbox_v2":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT",
            **model_cfg
        )
    elif model_cfg['model_type'] == "keypoints":
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
            **model_cfg
        )
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, writer=None, 
                    print_freq=50, scaler=None):
    # Based on https://github.com/pytorch/vision/blob/main/references/detection/engine.py
    
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    running_loss = 0
    for batch_num, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images, targets = data
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        running_loss += loss_value

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        if batch_num % 10 == 9:
            cum_batch_num = epoch * len(data_loader) + batch_num
            writer.add_scalar('Train/loss',
                                running_loss / 10,
                                cum_batch_num)
            running_loss = 0
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

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

            
def add_image_to_tensorboard(image, output, targets, writer, global_step, im_id):
    """ Add image to tensorboard with model output as dots.
    
    Args:
        image: image in format fed into pytorch model
        output: output from model for image
        writer: tensorboard writer
        global_step: global training step
        im_id: image identifier so tensorboard knows how to group with other saved images
    """
    boxes = output['boxes'].detach().to('cpu').numpy().astype(np.uint32)
    scores = output['scores'].detach().to('cpu').numpy()
    targets = targets['boxes'].numpy().astype(np.uint32)
    image = image.cpu().numpy().transpose(1, 2, 0).copy() # Copy makes circle work for unclear reasons
    boxes_drawn = 0
    min_score_thresh = .05 # Detected but low confident
    pred_score_thresh = .5 # Detected with confidence
    for box, score in zip(boxes, scores):
        
        if score < min_score_thresh:
            continue
        x = np.mean([box[0], box[2]])
        y = np.mean([box[1], box[3]])
        if score < pred_score_thresh: 
            cv2.circle(image, [int(x), int(y)], 3, (1.0, 0, 0), -1)
        else:
            cv2.circle(image, [int(x), int(y)], 3, (.04, .73, .71), -1)
        boxes_drawn += 1
    for box in targets:
        x = np.mean([box[0], box[2]])
        y = np.mean([box[1], box[3]])
        cv2.circle(image, [int(x), int(y)], 10, (1.0, 1.0, 1.0), 1)
    cv2.putText(image, f"score thresh: {pred_score_thresh}. Detections: {boxes_drawn}. Step: {global_step}", 
                org=[20, 50], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                color=[1.0, 1.0, 1.0], thickness=2, lineType=cv2.LINE_AA)
    writer.add_image(im_id, image, global_step, dataformats='HWC')

def add_images_to_tensorboard(images, outputs, targets, writer, global_step, batch_num):
    for im_ind, (image, output, target) in enumerate(zip(images, outputs, targets)):
        im_id = f"image {im_ind + batch_num}"
        add_image_to_tensorboard(image, output, target, writer, global_step, im_id)
    
    

# @torch.inference_mode()
def evaluate(model, data_loader, device, writer=None, step_num=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    
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
    for batch_num, (images, targets) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        
        model.train()
        with torch.no_grad():
            loss_dict = model(images, targets)
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()
            losses.append(loss_value)
        
        model.eval()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]
        model_time = time.time() - model_time
        if batch_num < 4:
            add_images_to_tensorboard(images, outputs, targets, writer, step_num, batch_num)

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    val_loss = np.mean(np.array(losses))
    writer.add_scalar('Val/loss', val_loss, step_num)

    running_loss = 0
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
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
    
    print(f"VAL LOSS: {val_loss}")
    
    # default iou thresh values are 
    print(f"Using an iou thresholds of {params.iouThrs}")

    score_threshs = [.2, .7]
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
                writer.add_scalar(f"Val/{category}/iou-{iou_p}/thr-{score_thresh}/precision", 
                                  prec, step_num)
                writer.add_scalar(f"Val/{category}/iou-{iou_p}/thr-{score_thresh}/recall", 
                                  recall, step_num)
                print(f"{cat['name']:8}: recall: {recall:1.2f}, precision: {prec:1.3f}")

    
    
    torch.set_num_threads(n_threads)
    
    
    
    
    return coco_evaluator, loss_value


def collate_fn(batch):
    return tuple(zip(*batch))


def train(cfg, model, optimizer, lr_scheduler, transform_train,
         transform_val):
    # transform_val must not change bounding boxes! Because read for coco eval seperately

    # train on the GPU or on the CPU, if a GPU is not available
    if torch.cuda.is_available():
        device = torch.device('cuda')  
    else:
        print("No GPU detected. Stopping.")
        return
    
    cfg_t = cfg['training']
    
    # use our dataset and defined transformations
    dataset = CocoDetection(cfg_t['image_folder'],
                            cfg_t['train_json_path'],
                            transform=transform_train)
    dataset_test = CocoDetection(cfg_t['image_folder'],
                                 cfg_t['val_json_path'],
                                 transform=transform_val)


    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg_t['batch_size'], shuffle=True, 
        num_workers=cfg_t['num_workers'], collate_fn=collate_fn,
        worker_init_fn=worker_init_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn, worker_init_fn=worker_init_fn)
    
    writer = SummaryWriter(cfg_t['run_folder'])
    
    # move model to the right device
    model.to(device)

    for epoch in range(cfg_t['num_epochs']):
        train_one_epoch(model, optimizer, data_loader, device, epoch, 
                        writer=writer, print_freq=50)
        
        if cfg_t['lr_scheduler']['name'] != "ReduceOnPlateau":
            # update the learning rate
            lr_scheduler.step()
        
        # evaluate on the test dataset
        if epoch % cfg_t['epochs_per_val'] == 0:
            _, loss = evaluate(model, data_loader_test, device=device,
                               writer=writer, step_num=(epoch+1)*len(data_loader))
            
            if cfg_t['lr_scheduler']['name'] == "ReduceOnPlateau":
                lr_scheduler.step(loss)
        
        # Can only save last learning rate this way because of RoP weirdness
        # in the pytorch implementation (no get_last_lr())
        if cfg_t['lr_scheduler']['name'] == "ReduceOnPlateau":
            # Only works if called after step
            writer.add_scalar('Learning rate', lr_scheduler._last_lr[0],
                              (epoch+1)*len(data_loader))
        else:
            writer.add_scalar('Learning rate', lr_scheduler.get_last_lr(),
                              (epoch+1)*len(data_loader))

    torch.save(model.state_dict(), 
               os.path.join(cfg_t['run_folder'], "final_model.pth"))
    print("That's it!")
