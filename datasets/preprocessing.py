# encoding: utf-8

"""
Script for preprocessing images such that they can be used for training.

Author: Patrick Binder
Date: 25.08.2023
"""

import argparse
from argparse import Namespace

import glob
import os

from PIL import Image

from torchvision import transforms
from torchvision.utils import save_image

from typing import Tuple


def pre_process(opt: Namespace) -> None:
    """
    Pre-processing step to prepare the data used for training.

    The images are resized, turned into grayscale and divided into
    train, validation and test datasets.

    +---dataset_name_truth: containing the pre-processed ground truth images
        |
        +---train_{train set size}
        |
        +---valid_{validation set size}
        |
        +---test_{test set size}

    Additionally, another dataset with artificially blurred images will
    be created.

    +---dataset_name_blurred: containing the ground truth images + gaussian blur
        |
        +---train_{train set size}
        |
        +---valid_{validation set size}
        |
        +---test_{test set size}

    All the images in the train, validation and test sets have a unique number, that is
    the same for the preprocessed ground truth image (in the safe_path_truth directory)
    as for the respective blurred image (in the safe_path_processed directory).

    Args:
        opt (Namespace): parsed arguments passed to the function
    """
    # get the paths of all the images to be processed
    img_paths = glob.glob(os.path.join(opt.data_path, '*'), recursive=True)

    if sum(opt.dataset_sizes) > len(img_paths):
        raise ValueError(f'Total number of images specified in dataset_sizes '
                         f'({sum(opt.dataset_sizes)}) exceeds number of images '
                         f'available in the provided dataset ({len(img_paths)}).')

    # names for the directories of datasets and splits
    ds_name = os.path.basename(opt.data_path)
    preprocessed_path = os.path.join(ds_name + '_truth')
    blurred_path = os.path.join(ds_name + '_blurred')
    split_names = [f'train_{opt.dataset_sizes[0]}',
                   f'valid_{opt.dataset_sizes[1]}',
                   f'test_{opt.dataset_sizes[2]}']

    # create all the directories to which the images will be saved
    for save_path in [preprocessed_path, blurred_path]:
        for split_name in split_names:
            os.makedirs(os.path.join(save_path, split_name), exist_ok=True)

    # torch transform step to set images to grayscale and resize them
    transform_truth = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Resize(size=opt.img_size, antialias=True),
    ])

    # torch transform step for blurring an image
    transform_blur = transforms.Compose([
        transforms.GaussianBlur(kernel_size=33, sigma=(1., 5.)),
    ])

    for i, img_path in enumerate(img_paths):
        with Image.open(img_path) as img:

            # depending on i, choose to which split it is saved
            if i < opt.dataset_sizes[0]:
                img_name = os.path.join(split_names[0], f'{i:05d}.png')
            elif i < sum(opt.dataset_sizes[:2]):
                img_name = os.path.join(split_names[1], f'{i:05d}.png')
            elif i < sum(opt.dataset_sizes):
                img_name = os.path.join(split_names[2], f'{i:05d}.png')

            # preprocess image and save it to the ground truth dataset
            img_proc = transform_truth(img)
            img_save_path_trans = os.path.join(preprocessed_path, img_name)
            save_image(img_proc, img_save_path_trans)

            # blur image and save it to the blurred dataset
            img_blur = transform_blur(img_proc)
            img_save_path_blur = os.path.join(blurred_path, img_name)
            save_image(img_blur, img_save_path_blur)

        # if enough images where processed stop the loop
        if i+1 == sum(opt.dataset_sizes):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Preprocessing',
        description='Converts images to grayscale, applies preprocessing and '
                    'splits them into train, validation and test datasets. '
                    'Further, a second folder is created with artificially '
                    'blurred images.'
    )

    # paths
    parser.add_argument('--data_path', type=str, default="../../VOC",
                        help='image folder')

    # sizes
    parser.add_argument('--img_size', type=Tuple[int, int], default=(256, 256),
                        help='size in pixels of the resulting image')
    parser.add_argument('--dataset_sizes', type=Tuple[int, int, int], default=(80, 32, 32),
                        help='number of pictures in the train, validation and test sets respectively')

    args = parser.parse_args()
    pre_process(opt=args)
