# encoding: utf-8

"""
PyTorch Dataset implementation for training the watermarking method.

Author: Patrick Binder
Date: 26.08.2023
"""

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import torch
import glob
import os


class WatermarkingDataset(Dataset):
    """
    Watermarking dataset to be used for the training of the to image watermarking models.
    """
    def __init__(self, data_path: str) -> None:
        """
        Images in data_path are expected to be preprocessed by preprocessing.py

        :param data_path: path to the directory where the images are saved
        """
        self.img_paths = glob.glob(os.path.join(data_path, '*'), recursive=True)
        self.length = len(self.img_paths)
        self.transform_img = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def __getitem__(self, item) -> torch.Tensor:
        """
        Loads image from dataset given an index and applies a transformation to a PyTorch Tensor

        :param item: index of the to be loaded image

        :returns:
            loaded image as PyTorch Tensor
        """
        with Image.open(self.img_paths[item]) as img:
            return self.transform_img(img)

    def __len__(self) -> int:
        return self.length
