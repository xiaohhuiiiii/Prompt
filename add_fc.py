import torch
import torch.nn as nn
import os
import torchvision.models as models


class FC_model(nn.Module):
    def __init__(self):
        super(FC_model, self).__init__()
        self.backbond = models.resnet50(pretrained=True)
        for param in self.backbond.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(1000, 100)
    def forward(self, x):
        output = self.backbond(x)
        output = self.fc(output)
        return output