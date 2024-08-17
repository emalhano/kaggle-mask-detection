# -*- coding: utf-8 -*-
"""
Created on Sun May 19 18:58:04 2024

@author: eduardob
"""


from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, faster_rcnn
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

import os
import xmltodict
import pandas as pd
import torch
import torchvision
import torch.nn as nn

from torchvision.transforms import v2


from model_and_data.data_loader import MaskDataset
from model_and_data.model_functions import create_model, train_model


device = torch.device("cuda:0" if- torch.cuda.is_available() else "cpu")


# def main():
    
#     transforms = v2.Compose([
#         v2.RandomHorizontalFlip(p=0.5),
#         v2.RandomRotation(45.)
#     ])

#     model = create_model().to(device)
#     data_loader = MaskDataset("./dataset", batch_size=8, transform=)
#     train_model(model, data_loader)



model = create_model()

transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(45.)
])

data_loader = MaskDataset("./dataset", batch_size=8, transform=transforms)
# train_model(model, data_loader)

img, label = data_loader[3]

image = draw_bounding_boxes(img[0]/255., label[0]["boxes"])

im = to_pil_image(image)
im.show()

model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_channels=1024, num_classes=4)

for parameter in model.parameters():
    parameter.requires_grad = False
    
for parameter in model.roi_heads.parameters():    
    parameter.requires_grad = True
    

