# encoding: utf-8

"""
PyTorch Dataset implementation used for fine-tuning the
reconstruction network.

Author: Patrick Binder
Date: 27.08.2023
"""

from torch.utils.data import Dataset
from torchvision.io import read_image

from typing import List
from torch import Tensor

import glob
import os


def check_filename(filenames: List[str], length: int = -9) -> None:
    """
    Sanity check whether the correct images are loaded by comparing their filenames.

    :param filenames: list of filenames to be checked
    :param length: indices of the paths to be checked

    :raises:
        ValueError: if filenames differ at the checked indices
    """
    for i, filename in enumerate(filenames):
        if i != 0 and filenames[0][length] != filename[length]:
            raise ValueError(f'Loaded images are expected to have the same name, '
                             f'but got {filenames[0][length]} and {filename[length]}.')


class FinetuneDataset(Dataset):
    """
    Fine-tuning data set to be used for fine-tuning the reconstruction network.
    """
    def __init__(self, ds_paths: List[str]) -> None:
        """
        Images in all given paths are expected to be preprocessed by preprocessing.py

        :param ds_paths: paths to the datasets
        """
        self.paths: List[List[str]] = [glob.glob(os.path.join(path, '*.png'), recursive=True) for path in ds_paths]
        self.length: int = len(self.paths[0])

    def __getitem__(self, item: int) -> List[Tensor]:
        """
        Loads images from datasets given an index and applies transformations to PyTorch Tensors

        :param item: index of the images to be loaded

        :returns:
            loaded images as list of PyTorch Tensors
        """
        self.img_paths = [paths[item] for paths in self.paths]
        check_filename(filenames=self.img_paths)

        return [read_image(img_path) for img_path in self.img_paths]

    def __len__(self) -> int:
        return self.length
