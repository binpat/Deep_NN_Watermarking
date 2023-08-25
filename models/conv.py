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


class Conv(nn.Module):
    """
    Extra simple CNN for deblurring images.
    """
    def __init__(self) -> None:
        super(Conv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x
