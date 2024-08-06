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


from model_and_data.data_loader import MaskDataset
from model_and_data.model_functions import create_model, train_model


device = torch.device("cuda:0" if- torch.cuda.is_available() else "cpu")


def main():
    
    model = create_model().to(device)
    data_loader = MaskDataset("./dataset", batch_size=8, transform=)
    train_model(model, data_loader)




img = read_image("dataset/images/maksssksksss0.png")

# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = [preprocess(img)]

# Step 4: Use the model and visualize the prediction
prediction = model(batch)[0]
labels = [weights.meta["categories"][i] for i in prediction["labels"]]
box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                          labels=labels,
                          colors="red",
                          width=4, font_size=30)
im = to_pil_image(box.detach())
im.show()

model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_channels=1024, num_classes=4)

for parameter in model.parameters():
    parameter.requires_grad = False
    
for parameter in model.roi_heads.parameters():    
    parameter.requires_grad = True
    

