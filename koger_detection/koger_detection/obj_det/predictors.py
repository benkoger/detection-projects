import torch

from .engine import get_detection_model

class Predictor:
    def __init__(self, model_weights_pth, **cfg):
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
        # Assume image is loaded as RGB but needs to be in BGR
        # image = image[:, :, ::-1]
        # image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        # Moved to video dataset reader
        image = image.type('torch.FloatTensor')
        image = image.cuda()
        image /= 255.0
        with torch.no_grad():
            return self.model([image])[0]