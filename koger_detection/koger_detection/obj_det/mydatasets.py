import albumentations as A
import cv2
from imutils.video import FileVideoStream
import numpy as np
import time
import torch
import torchvision


class CocoDetection(torchvision.datasets.CocoDetection):
    """ Coco dataloader that is compatible with FasterRcnn """
    
    def __init__(self, img_folder, ann_file, transform=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        if not isinstance(transform, A.core.composition.Compose):
            print("Warning: Transform is expected to be a ",
                  "albumentations Compose object.")
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

        if not isinstance(self.transform, A.core.composition.Compose):
            print("Transform is expected to be a albumentations Compose object")
            return False

        has_boxes = False
        count = 0
        while not has_boxes:
            transformed = self.transform(image=np.array(img), bboxes=boxes,
                                         class_labels=labels, area=area)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_labels = transformed['class_labels']
            transformed_area = transformed['area'] 
            
            if len(transformed_bboxes) > 0:
                has_boxes = True
            if count > 100:
                if count % 10 == 0:
                    print("cant find boxes", count)
            count += 1

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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame
    
class VideoDataset(torch.utils.data.IterableDataset):
    """ Used for inference on continous videos."""
    def __init__(self, file, to_float=False):
        """
        Args:
            to_float: if True return frame as float from 0 to 1
        """
        super(VideoDataset).__init__()
        self.fvs = FileVideoStream(file, transform=filterFrame).start()
        time.sleep(1)
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
        