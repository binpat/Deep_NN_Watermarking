# This implementation is adapted from https://github.com/ZJZAC/Deep-Model-Watermarking

import torch.nn as nn


def discriminator_block(in_filters, out_filters, stride, normalize):
    """Returns layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_filters))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, requires_grad=True):
        super(Discriminator, self).__init__()
        layers = []

        layers.extend(discriminator_block(in_channels, 128, 2, False))
        layers.extend(discriminator_block(128, 512, 2, True))

        layers.append(nn.Conv2d(512, 1, 3, 1, 1))

        layers.append(nn.MaxPool2d(kernel_size=4, stride=4))

        self.model = nn.Sequential(*layers)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, img):
        return self.model(img)
