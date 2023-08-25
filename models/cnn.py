# encoding: utf-8

"""
CNN architecture for training neural networks on
simple image processing tasks.

Author: Patrick Binder
Date: 25.08.2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    CNN for image processing tasks.
    """
    def __init__(self) -> None:
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)

        self.up_conv1 = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=2, padding=0, bias=False),
        )
        self.up_conv2 = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=2, padding=0, bias=False),
        )
        self.up_conv3 = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=2, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.up_conv1(x))
        x = F.relu(self.up_conv2(x))
        x = self.up_conv3(x)

        return x
