# encoding: utf-8

"""
Utils file for training neural networks.

Author: Patrick Binder
Date: 25.08.2023
"""


import matplotlib.pyplot as plt
import numpy as np

from argparse import Namespace

from typing import Dict, List, Tuple

from torchvision import transforms

import torch
import time
import os


def _image_grid(array: np.ndarray, n_cols: int) -> np.ndarray:
    """
    Helper function that reshapes the arrays for save_img_batch.

    :param array: array of images
    :param n_cols: number of columns, typically equivalent to batch size

    :return:
        (np.ndarray): reshaped array of images
    """
    index, channels, height, width = array.shape
    n_rows = index // n_cols

    img_grid = (array.reshape((n_rows, n_cols, height, width, channels))
                .swapaxes(1, 2)
                .reshape(height * n_rows, width * n_cols))

    return img_grid


def save_img_batch(opt: Namespace, imgs: List[torch.Tensor], batch: int, epoch: int, directory: str,
                   deblurring: bool = False) -> None:
    """
    Function to save images from current batch.

    Args:
        :param opt: parsed arguments passed to the main function
        :param imgs: list of images to be plotted
        :param batch: current batch
        :param epoch: current epoch
        :param directory: path where the visualization should be saved to
        :param deblurring: whether deblurring settings should be used for plotting
    """
    result = torch.cat(imgs, 0)

    grid = _image_grid(result.cpu().detach().numpy(), n_cols=opt.batch_size)
    path = os.path.join(directory, f'{epoch:03d}_{batch:02d}.png')

    if deblurring:
        plt.imsave(fname=path, arr=grid.astype(dtype=np.uint8), cmap='gray')
    else:
        plt.imsave(fname=path, arr=grid, cmap='gray', vmin=0, vmax=1)


def create_dirs(dir_names: List[str], main_dir: str) -> Dict[str, str]:
    """
    Creates various directories concerned with a training experiment to
    store information on the training process.

    :param dir_names: list of strings with directory names
    :param main_dir: main directory name

    :return:
        (dict): paths to the created directories
    """
    cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    os.makedirs(os.path.join(main_dir, 'experiments'), exist_ok=True)

    experiment_dir = 'experiments/experiment_' + cur_time
    directories = dict(zip(dir_names, [None]*len(dir_names)))
    for directory in directories.keys():
        directories[directory] = os.path.join(main_dir, experiment_dir, directory)
        os.makedirs(directories[directory], exist_ok=True)

    return directories


def get_device() -> torch.device:
    """
    Returns available GPU device for model training.

    :return:
        (torch.device): device used for training
    """
    return torch.device('mps') if torch.has_mps else \
        torch.device('cuda') if torch.cuda.is_available() else \
        torch.device('cpu')


def random_crop(imgs: List[torch.Tensor], crop_std: int, img_size: Tuple[int, int]) -> List[torch.Tensor]:
    """
    Applies random cropping and subsequent resizing to a list of images.

    :param imgs: list of images the processing is applied to
    :param crop_std: standard deviation of normal distribution
    :param img_size: size in pixels the images are resized to

    :return:
        (List[torch.Tensor): cropped and resized images
    """
    border_size = int(torch.normal(0, crop_std, (1, 1)).round().abs()[0, 0])
    border_transform = transforms.Compose([
        transforms.CenterCrop((img_size[0] - 5 * border_size, img_size[1] - 5 * border_size)),
        transforms.Resize(img_size, antialias=False),
    ])

    return [border_transform(img) for img in imgs]
