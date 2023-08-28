# encoding: utf-8

"""
Create a new dataset by applying one or more trained models on a dataset.

Author: Patrick Binder
Date: 26.08.2023
"""

import argparse
from argparse import Namespace

import torch
import glob
import sys
import os

from torchvision import transforms
from torchvision.utils import save_image

from typing import Tuple
from PIL import Image

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils.utils import get_device

from models.cnn import CNN
from models.conv import Conv
from models.unet import UNet
from models.ceilnet import HidingRes as CeilNet


def dataset_from_model(opt: Namespace) -> None:
    """
    Creates a new dataset in the same style of the input dataset
    by applying one or more trained models to the images.

    Takes a dataset as input, ideally in the style from preprocessing.py
    The preferred style looks therefore something like

    +---input_path: containing some pre-processed images
        |
        +---train_{train set size}
        |
        +---valid_{validation set size}
        |
        +---test_{test set size}

    The resulting dataset will mimic this style, i.e. the structure,
    the subdirectory names and the filenames of the images they contain
    will be the same.

    +---save_path: containing the models outputs
        |
        +---train_{train set size}
        |
        +---valid_{validation set size}
        |
        +---test_{test set size}

    :param opt: parsed arguments passed to the function
    """
    # initialize device and transforms for images and watermark
    device = get_device()

    transform_img = transforms.Compose([
        transforms.ToTensor()
    ])

    transform_watermark = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(opt.img_size, antialias=True),
    ])

    # load required models
    cnn, conv, hnet, watermark, rnet = None, None, None, None, None

    if opt.cnn_ckpt_path is not None:
        cnn = CNN().to(device).eval()
        cnn.load_state_dict(torch.load(opt.cnn_ckpt_path))

    if opt.conv_ckpt_path is not None:
        conv = Conv().to(device).eval()
        conv.load_state_dict(torch.load(opt.conv_ckpt_path))

    if opt.hnet_ckpt_path is not None:
        hnet = UNet(in_channels=2, n_classes=1, depth=4, padding=True, up_mode='upsample').to(device).eval()
        hnet.load_state_dict(torch.load(opt.hnet_ckpt_path))

        if opt.watermark_path is None:
            raise ValueError('Path to a watermark image is required if Hnet is applied.')

        with Image.open(opt.watermark_path) as img:
            watermark = transform_watermark(img).to(device).float()
            watermark = watermark[None, :, :, :]

    if opt.rnet_ckpt_path is not None:
        rnet = CeilNet(in_c=1, out_c=1).to(device).eval()
        rnet.load_state_dict(torch.load(opt.rnet_ckpt_path)).eval()

    # create directory and get subdirectories
    os.makedirs(opt.save_path, exist_ok=True)
    dirs = os.listdir(opt.input_path)
    if '.DS_Store' in dirs:
        dirs.remove('.DS_Store')

    # iterate over each subdirectory and its contents and apply the provided models
    for directory in dirs:
        os.makedirs(os.path.join(opt.save_path, directory), exist_ok=True)
        img_paths = glob.glob(os.path.join(opt.input_path, directory, '*.png'), recursive=True)

        for img_path in img_paths:
            with Image.open(img_path) as img:
                img = transform_img(img).to(device).float()
                img = img[None, :1, :, :]

                if cnn is not None:
                    img = cnn(img)

                if conv is not None:
                    img = conv(img)

                if hnet is not None:
                    img_wm = torch.cat([img, watermark], dim=1)
                    img = hnet(img_wm)

                if rnet is not None:
                    img = rnet(img)

                path = os.path.join(opt.save_path, directory, os.path.basename(img_path)[:-3] + 'png')
                save_image(img, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Create Dataset',
        description='Creates a new dataset by applying one or more provided '
                    'models on a given source dataset.'
    )

    # dataset paths
    parser.add_argument('--input_path', type=str,
                        help='input dataset to model')
    parser.add_argument('--save_path', type=str,
                        help='directory to save the new images to')

    # M - CNN
    parser.add_argument('--cnn_ckpt_path', type=str, nargs='?',
                        default=None,
                        help='[optional] path to saved CNN checkpoint')

    # SM - Conv
    parser.add_argument('--conv_ckpt_path', type=str, nargs='?',
                        default=None,
                        help='[optional] path to saved Conv checkpoint')

    # Hnet - UNet
    parser.add_argument('--hnet_ckpt_path', type=str, nargs='?',
                        default=None,
                        help='[optional] path to saved Hnet checkpoint')
    parser.add_argument('--watermark_path', type=str, nargs='?',
                        default='image_watermarking/watermarks/flower_rgb.png',
                        help='[optional] path to the watermark image, required if Hnet is applied')

    # Rnet - CEILNet
    parser.add_argument('--rnet_ckpt_path', type=str, nargs='?',
                        default=None,
                        help='[optional] path to saved Rnet checkpoint')

    # advanced settings
    parser.add_argument('--img_size', type=Tuple[int, int], default=(256, 256),
                        help='size in pixels of the resulting image')

    args = parser.parse_args()
    dataset_from_model(opt=args)
