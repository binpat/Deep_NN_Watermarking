# This implementation is adapted from https://github.com/ZJZAC/Deep-Model-Watermarking

import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation,
                               dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation,
                               dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x+y)


class HidingRes(nn.Module):
    def __init__(self, in_c=1, out_c=1):
        super(HidingRes, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 128, 3, 1, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(128, affine=True)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(128, affine=True)
        self.conv3 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
        self.norm3 = nn.InstanceNorm2d(128, affine=True)

        self.res1 = ResidualBlock(128, dilation=2)

        self.deconv3 = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(128, 128, kernel_size=1),
            )
        self.norm4 = nn.InstanceNorm2d(128, affine=True)
        self.deconv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.norm5 = nn.InstanceNorm2d(128, affine=True)
        self.deconv1 = nn.Conv2d(128, out_c, 1)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = F.relu(self.norm2(self.conv2(y)))
        y = F.relu(self.norm3(self.conv3(y)))

        y = self.res1(y)

        y = F.relu(self.norm4(self.deconv3(y)))
        y = F.relu(self.norm5(self.deconv2(y)))
        y = F.relu(self.deconv1(y))

        return y
