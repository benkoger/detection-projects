import albumentations as A
import cv2
from imutils.video import FileVideoStream
import numpy as np
import time
import torch
import torchvision
import torch.utils.data as data


class CocoDetection(torchvision.datasets.CocoDetection):
    """ Coco dataloader that is compatible with FasterRcnn """
    
    def __init__(self, img_folder, ann_file, transform=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.transform = transform # Assumes using albulmentaion
        
    
    def __getitem__(self, idx):
        
        img, target = super(CocoDetection, self).__getitem__(idx) 
        image_id = self.ids[idx] 

        boxes = []
        labels = []
        area = []
        for ann in target:
            box = ann['bbox']
            boxes.append([box[0], box[1], box[0]+box[2], box[1]+box[3]]) 
            labels.append(ann['category_id'])
            area.append(ann['area'])

        if self.transform is not None:
            if not isinstance(self.transform, A.core.composition.Compose):
                print("Transform is expected to be a albumentations Compose object")
                return False

            transformed = self.transform(image=np.array(img), bboxes=boxes,
                                            class_labels=labels, area=area)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_labels = transformed['class_labels']
            transformed_area = transformed['area'] 
            
        else:
            transformed_image = np.array(img)
            transformed_bboxes = boxes
            transformed_labels = labels
            transformed_area = area

        # Allow images with no boxes
        if len(transformed_bboxes) == 0:
            t_boxes = torch.zeros((0, 4), dtype=torch.float32)
            t_labels = torch.zeros((0, 1), dtype=torch.int64)
            t_area = torch.zeros((0, 1), dtype=torch.float32)
            t_image_id = torch.tensor([image_id])
        
        else:
            # convert everything into a torch.Tensor
            t_boxes = torch.as_tensor(transformed_bboxes, dtype=torch.float32)
            t_labels = torch.as_tensor(transformed_labels, dtype=torch.int64)
            t_area = torch.as_tensor(transformed_area, dtype=torch.float32)
            t_image_id = torch.tensor([image_id])

        # suppose all instances are not crowd
        iscrowd = torch.zeros_like(t_labels)

        target = {}
        target['boxes'] = t_boxes
        target['labels'] = t_labels
        target['image_id'] = t_image_id
        target['area'] = t_area
        target['iscrowd'] = iscrowd

        return transformed_image, target 
    

def filterFrame(frame):
    if frame is not None:
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("FRAME")
            print(frame)
    return frame
    
class VideoDataset(torch.utils.data.IterableDataset):
    """ Used for inference on continous videos."""
    def __init__(self, file, to_float=False, preprocess=False):
        """
        Args:
            to_float: if True return frame as float from 0 to 1
        """
        super(VideoDataset).__init__()
        self.fvs = FileVideoStream(file, transform=filterFrame).start()
        time.sleep(1)
        self.preprocess = preprocess
        self.to_float = to_float
    
    def __iter__(self):
        return self
    
    def stop_stream(self):
        self.fvs.stop()
        raise StopIteration

    def __next__(self):
        if self.fvs.running():
            if not self.fvs.more():
                self.stop_stream()
            frame = self.fvs.read()
            if frame is None:
                print("Empty frame. Ending stream.")
                self.stop_stream()
            if self.preprocess:
                frame = torch.from_numpy(frame.transpose(2, 0, 1))
            if self.to_float:
                frame = frame.type('torch.FloatTensor')
                frame /= 255.0
            return frame     
        else:
            self.stop_stream()
            
        return iter(range(self.start, self.end))
    
    def stop(self):
        self.fvs.stop()
        print("FileVideoStream stopped.")

class ImageDataset(data.Dataset):
    """ Simple image dataset for inference on a list of image paths """

    def __init__(self, files, rgb=True):
        """
        Args:
            files: list of paths to image files
            rgb: If True return image as rgb, else return bgr
        """
        self.files = files
        self.rgb = rgb

    def __getitem__(self, idx):
        image = cv2.imread(self.files[idx])
        if image is None:
            print(f"Error loading {self.files[idx]}")
            # ideally cleaner way to handle this
            return {"image": np.zeros((2, 2, 3), dtype=np.uint8), "filename": self.files[idx]}
        if self.rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        return {"image": image, "filename": self.files[idx]}

    def __len__(self):
        return len(self.files)

    
        