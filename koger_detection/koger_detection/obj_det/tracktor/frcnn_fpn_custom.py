import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import _default_anchorgen
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.transform import resize_boxes
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead
from torchvision.models.resnet import resnet50


from tracktor.frcnn_fpn import FRCNN_FPN

class FRCNN_FPN_CUSTOM(FRCNN_FPN):
    
    def __init__(self, model_type, num_classes, trainable_backbone_layers=3, **model_cfg):
        """
        Args:
            num_classes: number of possible classes (counting background class)
                Note: num_classes can be read from model_cfg
            trainable_backbone_layers: number of trainable (not frozen) layers starting from final block.
                Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
                Note: can be read from model_cfg
            **model_cfg: possible useful arguments:
                - trainable_backbone_layers
                - box_detections_per_img
                - box_nms_thresh
                - fixed_size
                - max_size
                - min_size
        """
        if model_type == "bbox_v2":
            # This is replicating much of 
            # torchvision.models.detection.fasterrcnn_resnet50_fpn_v2
            # (https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py)
            
            backbone = resnet50(weights=None, progress=True)
            backbone = _resnet_fpn_extractor(backbone, 
                                             trainable_backbone_layers, 
                                             norm_layer=nn.BatchNorm2d)
            rpn_anchor_generator = _default_anchorgen()
            rpn_head = RPNHead(backbone.out_channels, 
                               rpn_anchor_generator.num_anchors_per_location()[0], 
                               conv_depth=2)
            box_head = FastRCNNConvFCHead((backbone.out_channels, 7, 7), 
                                          [256, 256, 256, 256], [1024], 
                                          norm_layer=nn.BatchNorm2d
                                         )
        super(FRCNN_FPN, self).__init__(backbone, 
                                        num_classes=num_classes,
                                        rpn_anchor_generator=rpn_anchor_generator,
                                        rpn_head=rpn_head,
                                        box_head=box_head,
                                        **model_cfg,
                                       )
        
        # Values added in the FRCNN_FPN __init__ method
        # these values are cached to allow for feature reuse
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None
        
    def predict_boxes(self, boxes):
        """ This function needs to be changed from the default to handle models that can predict more than one class.
        Currently this is handeled by combining all non-background classes into a single class. In the future may
        make sense to preserve class identity for imporoved tracking especially in dense conditions.
        """
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)
 
        boxes = resize_boxes(boxes, self.original_image_sizes[0], self.preprocessed_images.image_sizes[0])
        proposals = [boxes]

        box_features = self.roi_heads.box_roi_pool(self.features, proposals, self.preprocessed_images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)

        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        
        # Here the function deviates from tracktor to deal with multiple classes
        # Changes are inline with how ROIHeads does postprocessing:
        # https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py
        
        # create labels for each prediction 
        # (Currently not used but helpful if want to explicitly use labels for tracking)
        num_classes = class_logits.shape[-1]
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(pred_scores)
        
        # remove predictions with the background label
        pred_boxes = pred_boxes[:, 1:]
        pred_scores = pred_scores[:, 1:]
        labels = labels[:, 1:]
        
        # Only take the highest scoring box per track 
        top_score_ind = torch.argmax(pred_scores, 1)
        box_inds = torch.arange(len(pred_boxes), device=device)
        pred_scores = pred_scores[box_inds, top_score_ind]
        pred_boxes = pred_boxes[box_inds, top_score_ind]
        labels = labels[box_inds, top_score_ind]

        pred_boxes = pred_boxes.detach()
        pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])
        pred_scores = pred_scores.detach()
        return pred_boxes, pred_scores
    
    
    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        # Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/generalized_rcnn.py
        # Just for inference
        # Goal is to reduce double call to backbone per image
        # add self.features, self.original_image_sizes, and self.prepocessed_images
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        self.original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            self.original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        
        self.preprocessed_images = images

        self.features = self.backbone(images.tensors)
        if isinstance(self.features, torch.Tensor):
            self.features = OrderedDict([("0", self.features)])
        proposals, proposal_losses = self.rpn(images, self.features, targets)
        detections, detector_losses = self.roi_heads(self.features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, self.original_image_sizes)  # type: ignore[operator]

        return detections
    