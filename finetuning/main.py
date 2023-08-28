# encoding: utf-8

"""
Main file for finetuning the reconstruction network.

Author: Patrick Binder
Date: 27.08.2023
"""

import argparse
from argparse import Namespace

import torch
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from typing import Dict, Optional
from tqdm import tqdm
from PIL import Image

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils.utils import create_dirs, get_device, save_img_batch
from utils.logger import Logger

from models.ceilnet import HidingRes as CeilNet

from finetune_dataset import FinetuneDataset


def main(opt: Namespace) -> None:
    """
    Main file for finetuning the reconstruction network on the
    surrogate model predictions.

    :param opt: parsed arguments passed to the function
    """
    # get device on which the model is trained and prepare directories
    device = get_device()
    directories = create_dirs(['checkpoints', 'train_imgs', 'valid_imgs', 'logs'], 'finetuning')

    # prepare logger
    logger = Logger(log_path=os.path.join(directories['logs'], 'log.txt'))
    logger.log_arguments(opt)

    # initialize model and load checkpoint
    r_net = CeilNet(in_c=1, out_c=1).to(device)
    r_net.load_state_dict(torch.load(opt.ckpt_path))
    logger.log_model(model=r_net, ckpt_path=opt.ckpt_path)

    # create optimizer and scheduler
    optimizer = torch.optim.Adam(
        r_net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
    )

    lr_scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=5, verbose=True
    )

    criterion = nn.MSELoss()
    half_batch_size = int(opt.batch_size / 2)

    # create training and validation dataloader
    train_loader, valid_loader = None, None
    dataloaders: Dict[str, Optional[DataLoader]] = {'train': train_loader, 'valid': valid_loader}

    for phase in dataloaders.keys():
        if phase == 'train':
            ds_paths = [opt.wmark_sm_path_train, opt.wmark_h_path_train,
                        opt.clear_blurred_path_train, opt.clear_deblurred_path_train]
        else:
            ds_paths = [opt.wmark_sm_path_valid, opt.wmark_h_path_valid,
                        opt.clear_blurred_path_valid, opt.clear_deblurred_path_valid]

        dataloaders[phase] = DataLoader(
            dataset=FinetuneDataset(ds_paths),
            batch_size=opt.batch_size,
            shuffle=phase == 'train',
            num_workers=opt.num_workers,
            pin_memory=True
        )

    # prepare watermarks for training
    wm_trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    watermark = wm_trans(Image.open(opt.watermark)).repeat(opt.batch_size, 1, 1, 1).to(device)
    watermark_clear = wm_trans(Image.open(opt.watermark_clean)).repeat(opt.batch_size, 1, 1, 1).to(device)

    # start training
    for epoch in range(opt.n_epochs):
        logger.log(f'========== epoch {epoch:02d}/{opt.n_epochs} ==========')

        for phase in ['train', 'valid']:
            if phase == 'train':
                r_net.train()
            else:
                r_net.eval()

            plot_dir = directories['train_imgs'] if phase == 'train' else directories['valid_imgs']
            wm_sm_losses, wm_h_losses, cl_bl_losses, cl_de_losses, cons_losses = 0, 0, 0, 0, 0
            n_batches = len(dataloaders[phase])

            # iterate over each batch in the datasets
            for i, batch in enumerate(pbar := tqdm(dataloaders[phase])):
                wm_sm_img = batch[0].to(device).float()[:, :1, :, :]
                wm_h_img = batch[1].to(device).float()[:, :1, :, :]
                clear_blur_img = batch[2].to(device).float()[:, :1, :, :]
                clear_deblur_img = batch[3].to(device).float()[:, :1, :, :]

                with torch.set_grad_enabled(phase == 'train'):
                    # get reconstructions from images
                    wm_sm_rec = r_net(wm_sm_img)
                    wm_h_rec = r_net(wm_h_img)

                    cl_bl_rec = r_net(clear_blur_img)
                    cl_de_rec = r_net(clear_deblur_img)

                    # calculate losses
                    wm_sm_loss = criterion(wm_sm_rec, watermark)
                    wm_h_loss = criterion(wm_h_rec, watermark)

                    cl_bl_loss = criterion(cl_bl_rec, watermark_clear)
                    cl_de_loss = criterion(cl_de_rec, watermark_clear)

                    cons_loss = criterion(wm_sm_rec[:half_batch_size],
                                          wm_sm_rec[half_batch_size:2*half_batch_size])

                    loss = (wm_sm_loss + wm_h_loss + cl_bl_loss + cl_de_loss + cons_loss) * 10000

                    # perform training
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # track losses
                    wm_sm_losses += wm_sm_loss.detach().cpu()
                    wm_h_losses += wm_h_loss.detach().cpu()
                    cl_bl_losses += cl_bl_loss.detach().cpu()
                    cl_de_losses += cl_de_loss.detach().cpu()
                    cons_losses += cons_loss.detach().cpu()

                # report losses
                pbar.set_description(f'[{epoch}/{opt.n_epochs}][{i+1}/{n_batches}][{phase}] '
                                     f'wmark_sm_loss: {wm_sm_losses / (i+1):.4f} | '
                                     f'clear_loss: {cl_de_losses / (i+1):.4f}')

                logger.log(
                    f'[{epoch}/{opt.n_epochs}][{i+1}/{n_batches}][{phase}] '
                    f'wmark_sm_loss: {wm_sm_losses / (i+1):.4f} | '
                    f'wmark_h_loss: {wm_h_losses / (i+1):.4f} | '
                    f'clear_blur_loss: {cl_bl_losses / (i+1):.4f} | '
                    f'clear_deblur_loss: {cl_de_losses / (i+1):.4f} | '
                    f'cons_loss: {cons_losses / (i+1):.4f}',
                    write=i+1 == n_batches,
                )

                # plot images
                if (i+1) % opt.plot_batch == 0:
                    save_img_batch(
                        opt=opt,
                        imgs=[watermark, wm_sm_rec, wm_h_rec, cl_bl_rec, cl_de_rec],
                        batch=i+1,
                        epoch=epoch,
                        directory=plot_dir,
                        deblurring=False
                    )

        # update scheduler and save model checkpoint
        sum_loss = wm_sm_losses + wm_h_losses + cl_bl_losses + cl_de_losses + cons_losses
        lr_scheduler.step(sum_loss / n_batches)
        torch.save(
            r_net.state_dict(),
            os.path.join(directories['checkpoints'], f'ckpt_epoch_{epoch:02d}.pt')
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Reconstruction Network Finetuning',
        description='Finetunes the reconstruction network on the predictions '
                    'of a surrogate model SM.'
    )

    # training data sets
    parser.add_argument('--wmark_sm_path_train', type=str,
                        help='training data path to images deblurred by SM (with watermark)')
    parser.add_argument('--wmark_h_path_train', type=str,
                        help='training data path to images deblurred by M and watermarked by H')
    parser.add_argument('--clear_blurred_path_train', type=str,
                        help='training data path to blurred images (without watermark)')
    parser.add_argument('--clear_deblurred_path_train', type=str,
                        help='training data path to images deblurred by M (without watermark)')

    # validation data sets
    parser.add_argument('--wmark_sm_path_valid', type=str,
                        help='validation data path to images deblurred by SM (with watermark)')
    parser.add_argument('--wmark_h_path_valid', type=str,
                        help='validation data path to images deblurred by M and watermarked by H')
    parser.add_argument('--clear_blurred_path_valid', type=str,
                        help='validation data path to blurred images (without watermark)')
    parser.add_argument('--clear_deblurred_path_valid', type=str,
                        help='validation data path to images deblurred by M (without watermark)')

    # dataloader settings
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')

    # optimization settings
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam optimizer')

    # train options
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='number of epochs for training')
    parser.add_argument('--plot_batch', type=int, default=3,
                        help='each plot_batch batches the currently progresses images are plotted')
    parser.add_argument('--ckpt_path', type=str,
                        help='path to reconstruction network model training checkpoint, to continue training')

    # watermark image paths
    parser.add_argument('--watermark', type=str, default='image_watermarking/watermarks/flower_rgb.png',
                        help='path to the watermark to be embedded into images')
    parser.add_argument('--watermark_clean', type=str, default='image_watermarking/watermarks/clean.png',
                        help='path to the clean watermark')

    args = parser.parse_args()
    main(opt=args)
