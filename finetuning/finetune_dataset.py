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


# class FinetuneDataset(Dataset):
#     """
#     Fine-tuning data set to be used for fine-tuning the reconstruction network.
#     """
#     def __init__(self, wm_sm_path: str, wm_h_path: str, clear_blur_path: str, clear_deblur_path: str) -> None:
#         """
#         Images in all given paths are expected to be preprocessed by preprocessing.py
#
#         :param wm_sm_path: path to images watermarked and deblurred by SM
#         :param wm_h_path: path to images watermarked by H and deblurred by M
#         :param clear_blur_path: path to blurred images
#         :param clear_deblur_path: path to images deblurred by M
#         """
#         self.wm_sm_path: str = os.path.join(wm_sm_path, '*.png')
#         self.wm_h_path: str = os.path.join(wm_h_path, '*.png')
#         self.clear_blur_path: str = os.path.join(clear_blur_path, '*.png')
#         self.clear_deblur_path: str = os.path.join(clear_deblur_path, '*.png')
#
#         self.wm_sm_img_paths: List[str] = glob.glob(self.wm_sm_path, recursive=True)
#         self.wm_m_img_paths: List[str] = glob.glob(self.wm_h_path, recursive=True)
#         self.clear_blur_img_paths: List[str] = glob.glob(self.clear_blur_path, recursive=True)
#         self.clear_deblur_img_paths: List[str] = glob.glob(self.clear_deblur_path, recursive=True)
#
#         self.length: int = len(self.wm_sm_img_paths)
#
#     def __getitem__(self, item: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
#         """
#         Loads images from datasets given an index and applies transformations to PyTorch Tensors
#
#         :param item: index of the images to be loaded
#
#         :returns:
#             loaded images as tuple of PyTorch Tensors
#         """
#         wm_sm_img_path = self.wm_sm_img_paths[item]
#         wm_m_img_path = self.wm_m_img_paths[item]
#         clear_blur_img_path = self.clear_blur_img_paths[item]
#         clear_deblur_img_path = self.clear_deblur_img_paths[item]
#
#         # check whether all filenames are equal
#         check_filename(wm_sm_img_path, wm_m_img_path)
#         check_filename(wm_sm_img_path, clear_blur_img_path)
#         check_filename(wm_sm_img_path, clear_deblur_img_path)
#
#         wm_sm_img = read_image(wm_sm_img_path)
#         wm_m_img = read_image(wm_m_img_path)
#         clear_blur_img = read_image(clear_blur_img_path)
#         clear_deblur_img = read_image(clear_deblur_img_path)
#
#         return wm_sm_img, wm_m_img, clear_blur_img, clear_deblur_img
#
#     def __len__(self) -> int:
#         return self.length
