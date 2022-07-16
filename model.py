from turtle import window_height
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', weights='ResNeXt101_32X8D_Weights.DEFAULT')
        self.fc = nn.Linear(1000, 3, bias=True)
    
    def forward(self, x):
        return self.fc(self.backbone(x))