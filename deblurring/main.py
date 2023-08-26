# encoding: utf-8

"""
Main file for training a simple deblurring model.

Author: Patrick Binder
Date: 25.08.2023
"""

import argparse
from argparse import Namespace

from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from typing import Tuple

import random
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils.utils import create_dirs, get_device, random_crop, save_img_batch
from utils.logger import Logger

from models.cnn import CNN
from models.conv import Conv

from blur_dataset import BlurDataset


def main(opt: Namespace) -> None:
    # get device on which the model is trained and prepare directories
    device = get_device()
    directories = create_dirs(['checkpoints', 'train_imgs', 'valid_imgs', 'logs'], 'deblurring')

    # prepare logger
    logger = Logger(log_path=os.path.join(directories['logs'], 'log.txt'))
    logger.log_arguments(opt)

    # initialize and load model
    model = CNN() if opt.model == 'cnn' else Conv()
    if opt.ckpt_path is not None:
        model.load_state_dict(torch.load(opt.ckpt_path))
    logger.log_model(model=model, ckpt_path=opt.ckpt_path)
    model.to(device)

    # fix seed if deterministic training is enabled
    if type(opt.seed) is int:
        logger.log(f'Deterministic training with seed {opt.seed}')
        torch.manual_seed(opt.seed)
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.use_deterministic_algorithms(True)

    # create datasets and dataloaders
    train_data = BlurDataset(
        blur_path=opt.blur_path_train,
        truth_path=opt.truth_path_train
    )

    valid_data = BlurDataset(
        blur_path=opt.blur_path_valid,
        truth_path=opt.truth_path_valid
    )

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=not isinstance(opt.seed, int))
    valid_loader = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=False)
    dataloaders = {'train': train_loader, 'valid': valid_loader}

    # specify criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True,
    )

    # start training
    for epoch in range(opt.n_epochs):
        logger.log(f'========== epoch {epoch:02d}/{opt.n_epochs} ==========')

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            plot_dir = directories['train_imgs'] if phase == 'train' else directories['valid_imgs']
            n_batches = len(dataloaders[phase])
            losses = 0

            # iterate over each batch in the datasets
            for i, batch in enumerate(pbar := tqdm(dataloaders[phase])):

                with torch.set_grad_enabled(phase == 'train'):
                    blur_imgs = batch[0].to(device).float()
                    truth_imgs = batch[1].to(device).float()

                    blur_imgs = blur_imgs[:, :1, :, :]
                    truth_imgs = truth_imgs[:, :1, :, :]

                    # apply random center crop and resizing if enabled
                    if type(opt.crop_std) == int:
                        blur_imgs, truth_imgs = random_crop(
                            imgs=[blur_imgs, truth_imgs],
                            crop_std=opt.crop_std,
                            img_size=opt.img_size
                        )

                    # get predictions and loss from this batch
                    pred = model(blur_imgs)
                    loss = criterion(pred, truth_imgs)

                    # perform training
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    losses += loss.detach().cpu()

                # report loss
                msg = f'[{epoch}/{opt.n_epochs}][{i}/{n_batches}][{phase}] loss: {losses / i+1:.4f}'
                logger.log(msg, write=i + 1 == n_batches)
                pbar.set_description(msg)

                # plot images
                if i % opt.plot_batch == 0:
                    save_img_batch(
                        opt=opt,
                        imgs=[blur_imgs, pred, truth_imgs],
                        batch=i,
                        epoch=epoch,
                        directory=plot_dir,
                        deblurring=True
                    )

        # update schedular and save model checkpoint
        lr_scheduler.step(losses / n_batches)
        torch.save(
            model.state_dict(),
            os.path.join(directories['checkpoints'], f'ckpt_epoch_{epoch:02d}')
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Deblurring',
        description='Trains a simple neural network on image deblurring.'
    )

    # training data sets
    parser.add_argument('--blur_path_train', type=str, default="datasets/VOC_blurred/train_80",
                        help='training data path to blurred images')
    parser.add_argument('--truth_path_train', type=str,
                        default="datasets/VOC_truth/train_80",
                        help='training data path to ground truth images')

    # validation data sets
    parser.add_argument('--blur_path_valid', type=str, default="datasets/VOC_blurred/valid_32",
                        help='validation data path to blurred images')
    parser.add_argument('--truth_path_valid', type=str,
                        default="datasets/VOC_blurred/valid_32",
                        help='validation data path to ground truth images')

    # train options
    parser.add_argument('--model', type=str, default='cnn',
                        help="model to be trained: either 'cnn' or 'conv'")
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size used for training and validation')
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs for training')
    parser.add_argument('--plot_batch', type=int, default=3,
                        help='plot images each plot_batch batches')
    parser.add_argument('--ckpt_path', type=str, nargs='?',
                        default=None,
                        help='checkpoint path to continue training')

    # advanced settings
    parser.add_argument('--seed', type=int, nargs='?', default=None,
                        help='fix a seed for training and enable deterministic training')
    parser.add_argument('--crop_std', type=int, nargs='?', default=None,
                        help='standard deviation for crop width, enables random center crop and resizing')
    parser.add_argument('--img_size', type=Tuple[int, int], default=(256, 256),
                        help='size in pixels of the resulting image')

    args = parser.parse_args()
    main(opt=args)
