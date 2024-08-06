from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, faster_rcnn
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from torchvision.tv_tensors import BoundingBoxes

from torchvision.transforms import v2

import matplotlib.pyplot as plt

import PIL

import os
import xmltodict
import pandas as pd
import torch
import torchvision
import torch.nn as nn

# with open(path) as fd:
#     doc = xmltodict.parse(fd.read())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    
    dataset_dir = "dataset"
    data_loader = MaskDataset(dataset_dir)
    
    transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(45.)
    ])
    
    image, label = data_loader[0]
    
    image = image[0]
    label = label[0]
    
    new_image, new_boxes = transforms(image, label["boxes"])
    
    stop = 1
    
    for image, label in data_loader:
        stop = 1


class MaskDataset(torch.utils.data.Dataset):

    
    def __init__(self, dataset_dir, batch_size=8, transform=None, target_transform=None):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_dir = os.path.join(dataset_dir, "images")
        self.img_annot = os.path.join(dataset_dir, "annotations")
        self.images = os.listdir(self.img_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.label_map = {"without_mask": 1, "with_mask": 2, "mask_weared_incorrect": 3}


    def __len__(self):
        return int(len(self.images)/self.batch_size)


    def __getitem__(self, idx):
        
        images = []
        targets = []
        for ii in range(self.batch_size*idx, self.batch_size*(idx+1)):
            
            img_path = os.path.join(self.img_dir, self.images[ii])
            images.append(pil_to_tensor(PIL.Image.open(img_path).convert("RGB")).to(dtype=torch.float))
            
            img_label_path = os.path.join(self.img_annot, self.images[ii][:-3] + "xml")            
            ground_truth = xmltodict.parse(open(img_label_path).read())

            objects = [ground_truth["annotation"]["object"]] if type(ground_truth["annotation"]["object"]) is not list else ground_truth["annotation"]["object"]
            
            boxes = torch.tensor([[float(box["bndbox"]["xmin"]), float(box["bndbox"]["ymin"]), float(box["bndbox"]["xmax"]), float(box["bndbox"]["ymax"])] for box in objects], dtype=torch.float)
            labels = torch.tensor([self.label_map[label["name"]] for label in objects], dtype=torch.int64)
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            image_path = img_path
            label_path = img_label_path
            iscrowd = torch.zeros(boxes.shape[0], dtype=torch.int8)
            targets.append({"boxes": BoundingBoxes(boxes, format="XYXY", canvas_size=images[-1].shape[-2:]),
                            "labels": labels,
                            "area": areas,
                            "image_path": image_path,
                            "label_path": label_path,
                            "iscrowd": iscrowd})
            
        return images, targets
    
    
    
if __name__ == "__main__":
    main()