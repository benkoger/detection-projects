import torch

from koger_detection.obj_det.engine import get_detection_model

class Predictor:
    """ Assumes images are passed as H x W x 3 with values from 0 to 255."""
    
    def __init__(self, cfg, model_weights_path=None, rgb=True):
        """
        Args:
            cfg: model config dictionary
            model_weights_path: full path to model weights to use
            rgb: if images will be passed as RGB
        """
        self.rgb = rgb
        
        if model_weights_path is None:
            model_weights_pth = cfg.pop("model_weights_pth")
        elif "model_weights_pth" in cfg:
            cfg.pop("model_weights_pth")
            
        self.model = get_detection_model(**cfg)
        self.model.load_state_dict(torch.load(model_weights_pth))
        
        # train on the GPU or on the CPU, if a GPU is not available
        if torch.cuda.is_available():
            device = torch.device('cuda')  
        else:
            device = torch.device('cpu')
        
        self.model.to(device)
        self.model.eval()
        
    def __call__(self, image):
        if self.rgb:
            # image is loaded as RGB but needs to be in BGR
            image = image[:, :, ::-1]
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        if torch.cuda.is_available():
            image = image.cuda()
        image /= 255.0
        with torch.no_grad():
            return self.model([image])[0]