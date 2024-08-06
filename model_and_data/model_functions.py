# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:20:27 2024

@author: eduardob
"""

import torch
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

from data_loader import MaskDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def create_model():
    
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)

    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_channels=in_channels, num_classes=4)

    for parameter in model.parameters():
        parameter.requires_grad = False
        
    for parameter in model.roi_heads.parameters():
        parameter.requires_grad = True
        
    return model



def train_model(model, data_loader, epochs=15):
    
    my_optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        
        print(f"Epoch: {epoch:.0f}/{range(epochs)[-1]:.0f}")
        
        n=0
        model.train()
        for image, label in data_loader:
            
            for ii in range(len(image)):
                image[ii] = image[ii].to(device)
                label[ii]["boxes"] = label[ii]["boxes"].to(device)
                label[ii]["labels"] = label[ii]["labels"].to(device)
            
            n=n+1
            print(f"Batch: {n:.0f}/{len(data_loader):.0f}")
            
            loss = model(image, label)
            my_loss = loss["loss_classifier"] + loss["loss_box_reg"]
            
            print(f"Loss: {my_loss.item():.10f}")
            
            my_optim.zero_grad()
            my_loss.backward()
            my_optim.step()