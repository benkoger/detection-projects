import numpy as np
import torch

from koger_detection.obj_det.engine import get_detection_model

class Predictor:
    """ Assumes images are passed as H x W x 3 with values from 0 to 255."""
    
    def __init__(self, cfg, model_weights_path=None, invert_color_channel=False, device=None):
        """
        Args:
            cfg: model config dictionary
            model_weights_path: full path to model weights to use
            invert_color_channel: if images will be changed from BGR to RGB or vice versa
        """
        self.invert_color_channel = invert_color_channel

        if model_weights_path is None:
            model_weights_path = cfg.pop("model_weights_pth")
        elif "model_weights_pth" in cfg:
            cfg.pop("model_weights_pth")

        self.model = get_detection_model(**cfg)
        self.model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    def _prep_one(self, image):
        """
        Prepare a single image for the detector.
        Input: HWC numpy array or torch.Tensor, 0..255
        Output: CHW float tensor on self.device, 0..1
        """
        if isinstance(image, torch.Tensor):
            x = image.to(dtype=torch.float32)
        elif isinstance(image, np.ndarray):
            x = torch.as_tensor(image.astype("float32"))
        else:
            raise ValueError("Expects torch.Tensor or numpy.ndarray")

        if self.invert_color_channel:
            x = x[:, :, [2, 1, 0]]

        # HWC → CHW
        x = x.permute(2, 0, 1).contiguous()
        x /= 255.0

        # Move to model device
        x = x.to(self.device, non_blocking=True)
        return x

    def __call__(self, images):
        """
        images: single image or list/tuple of images (HWC)
        returns: list of detection dicts
        """
        # Normalize to list
        if not isinstance(images, (list, tuple)):
            images = [images]

        imgs = [self._prep_one(im) for im in images]

        with torch.no_grad():
            outputs = self.model(imgs)  # list[dict], len == len(imgs)

        return outputs
