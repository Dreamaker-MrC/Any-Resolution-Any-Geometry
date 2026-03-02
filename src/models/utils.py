import torch
import torch.nn.functional as F
import torch.nn as nn


class SimpleResizer(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height

    def forward(self, x):
        return F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True)
