# encoding: utf-8

"""
PyTorch Dataset implementation for training the model to be protected.

Author: Patrick Binder
Date: 25.08.2023
"""

from torch.utils.data import Dataset
from torchvision.io import read_image
from torch import Tensor

from typing import Tuple

import glob
import os


class BlurDataset(Dataset):
    """
    Blur dataset to be used for training model to be protected.
    """
    def __init__(self, blur_path: str, truth_path: str) -> None:
        self.blur_path = os.path.join(blur_path, '*.png')
        self.truth_path = os.path.join(truth_path, '*.png')

        self.blur_img_paths = glob.glob(self.blur_path, recursive=True)
        self.truth_img_paths = glob.glob(self.truth_path, recursive=True)

        if len(self.blur_img_paths) != len(self.truth_img_paths):
            raise ValueError(f'Blur and truth directories are expected to have the same number of images, '
                             f'but got {len(self.blur_img_paths)} images in blur path and '
                             f'{len(self.truth_img_paths)} images in truth path.')
        else:
            self.length = len(self.blur_img_paths)

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        blur_img_path = self.blur_img_paths[item]
        truth_img_path = self.truth_img_paths[item]

        if blur_img_path[-9:] != truth_img_path[-9:]:
            raise ValueError(f'Blur image and truth image are expected to have the same name,'
                             f'but got {blur_img_path[-9:]} for blur image and'
                             f'{truth_img_path[-9:]} for truth image.')

        blur_img = read_image(blur_img_path)
        truth_img = read_image(truth_img_path)

        return blur_img, truth_img

    def __len__(self) -> int:
        return self.length
