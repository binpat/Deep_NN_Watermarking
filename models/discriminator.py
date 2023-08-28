"""
This implementation is adapted from https://github.com/ZJZAC/Deep-Model-Watermarking
The following MIT License applies to this file:

MIT License

Copyright (c) 2020 ZJZAC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
