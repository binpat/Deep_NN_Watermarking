# encoding: utf-8

"""
Learns an image watermarking method by training a hiding,
reconstruction and discriminator network.

Author: Patrick Binder
Date: 26.08.2023
"""

import argparse
from argparse import Namespace

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
import os

from typing import Dict, List, Tuple
from PIL import Image

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils.utils import create_dirs, get_device, save_img_batch
from utils.logger import Logger

from models.unet import UNet
from models.ceilnet import HidingRes as CeilNet
from models.discriminator import Discriminator

from watermarking_dataset import WatermarkingDataset


def main(opt: Namespace):
    """
    Main file for training the image watermarking models.

    :param opt: parsed arguments passed to the function
    """
    # get device on which the model is trained and prepare directories
    device = get_device()
    directories = create_dirs(['checkpoints', 'train_imgs', 'valid_imgs', 'logs'], 'image_watermarking')

    # prepare logger
    logger = Logger(log_path=os.path.join(directories['logs'], 'log.txt'))
    logger.log_arguments(opt)

    # create training and evaluation dataloader
    train_loader = DataLoader(
        dataset=WatermarkingDataset(opt.train_data),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset=WatermarkingDataset(opt.valid_data),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    dataloaders: Dict[str, DataLoader] = {'train': train_loader, 'valid': valid_loader}

    # initialize hiding, reconstruction and discriminator networks and their optimizers
    models: List[str] = ['h', 'r', 'd']
    ckpt_paths: Dict[str, str] = {'h': opt.h_net_ckpt, 'r': opt.r_net_ckpt, 'd': opt.d_net_ckpt}

    h_net = UNet(in_channels=2, n_classes=1, depth=4, padding=True, up_mode='upsample').to(device)
    r_net = CeilNet(in_c=1, out_c=1).to(device)
    d_net = Discriminator(in_channels=1).to(device)
    net: Dict[str, nn.Module] = {'h': h_net, 'r': r_net, 'd': d_net}

    optimizer_h, optimizer_r, optimizer_d = None, None, None
    optimizer: Dict[str, torch.optim.Adam] = {'h': optimizer_h, 'r': optimizer_r, 'd': optimizer_d}

    scheduler_h, scheduler_r, scheduler_d = None, None, None
    scheduler: Dict[str, ReduceLROnPlateau] = {'h': scheduler_h, 'r': scheduler_r, 'd': scheduler_d}

    for model in models:
        optimizer[model] = torch.optim.Adam(net[model].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        scheduler[model] = ReduceLROnPlateau(optimizer[model], mode='min', factor=0.2, patience=5, verbose=True)
        if ckpt_paths[model] is not None:
            net[model].load_state_dict(torch.load(ckpt_paths[model]))
        logger.log_model(model=net[model], ckpt_path=ckpt_paths[model])

    mse = nn.MSELoss()
    half_batch_size = int(opt.batch_size / 2)

    # prepare watermark for training
    wm_trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    watermark = wm_trans(Image.open(opt.watermark)).repeat(opt.batch_size, 1, 1, 1).to(device)
    watermark_clear = wm_trans(Image.open(opt.watermark_clean)).repeat(opt.batch_size, 1, 1, 1).to(device)

    # prepare truths for adversarial training
    patch = (1, opt.img_size[0] // 2 ** 4, opt.img_size[1] // 2 ** 4)
    adv_true = torch.from_numpy(np.ones((watermark.size(0), *patch), dtype=np.float32)).to(device)   # watermark free
    adv_fake = torch.from_numpy(np.zeros((watermark.size(0), *patch), dtype=np.float32)).to(device)  # watermarked

    # start training
    for epoch in range(opt.n_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                for model in models:
                    net[model].train()
            else:
                for model in models:
                    net[model].eval()

            plot_dir = directories['train_imgs'] if phase == 'train' else directories['valid_imgs']
            adv_losses, mse_losses, clean_losses, wm_losses, cons_losses = 0, 0, 0, 0, 0
            emb_ext_losses, dis_losses = 0, 0
            n_batches = len(dataloaders[phase])

            # iterate over each batch in the datasets
            for i, batch in enumerate(pbar := tqdm(dataloaders[phase])):
                img = batch.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    # calculate embedding loss
                    img_wm = net['h'](torch.cat([img, watermark], dim=1))
                    d_pred_wm = net['d'](img_wm)

                    adv_loss = mse(d_pred_wm, adv_true)
                    mse_loss = mse(img, img_wm)
                    emb_loss = opt.l_adv * adv_loss + opt.l_mse * mse_loss

                    # calculate extraction loss
                    img_rec = net['r'](img)
                    clean_loss = mse(img_rec, watermark_clear)

                    img_wm_rec = net['r'](img_wm)
                    wm_loss = mse(img_wm_rec, watermark)

                    cons_loss = mse(img_wm_rec[0:half_batch_size],
                                    img_wm_rec[half_batch_size:half_batch_size*2])

                    ext_loss = (clean_loss * 2 + wm_loss + cons_loss) * opt.l_mse
                    emb_ext_loss = emb_loss + ext_loss * opt.l

                    # calculate discriminator loss
                    d_pred_clean = net['d'](img)
                    dis_loss = mse(d_pred_wm.detach(), adv_fake) + mse(d_pred_clean, adv_true)

                    if phase == 'train':
                        # perform backprop on hiding and reconstruction networks
                        net['h'].zero_grad()
                        net['r'].zero_grad()

                        emb_ext_loss.backward()
                        optimizer['h'].step()
                        optimizer['r'].step()

                        # perform backprop on discriminator
                        net['d'].zero_grad()

                        dis_loss.backward()
                        optimizer['d'].step()

                    # track losses
                    emb_ext_losses += emb_ext_loss.detach().cpu()
                    dis_losses += dis_loss.detach().cpu()

                    adv_losses += adv_loss.detach().cpu()
                    mse_losses += mse_loss.detach().cpu()
                    clean_losses += clean_loss.detach().cpu()
                    wm_losses += wm_loss.detach().cpu()
                    cons_losses += cons_loss.detach().cpu()

                # report losses
                pbar.set_description(f'[{epoch}/{opt.n_epochs}][{i}/{n_batches}][{phase}] '
                                     f'emb_ext_loss: {emb_ext_losses/i:.4f} | dis_loss: {dis_losses/i:.4f}')

                logger.log(
                    f'[{epoch}/{opt.n_epochs}][{i}/{n_batches}][{phase}] '
                    f'emb_ext_loss: {emb_ext_losses / i:.4f} | dis_loss: {dis_losses / i:.4f} | '
                    f'adv_loss: {adv_losses / i:.4f} | mse_loss: {mse_losses / i:.4f} | '
                    f'clean_loss: {clean_losses / i:.4f} | wm_loss: {wm_losses / i:.4f} | '
                    f'cons_loss: {cons_losses / i:.4f}',
                    write=i+1 == n_batches,
                )

                # plot images
                if i % opt.plot_batch == 0:
                    save_img_batch(
                        opt=opt,
                        imgs=[watermark, img, img_rec, (img_wm - img)*50, img_wm, img_wm_rec],
                        batch=i,
                        epoch=epoch,
                        directory=plot_dir,
                        deblurring=False
                    )

        # update scheduler and save model checkpoints
        if phase == 'valid':
            scheduler['h'].step(emb_ext_loss/n_batches)
            scheduler['r'].step(emb_ext_loss/n_batches)
            scheduler['d'].step(dis_losses/n_batches)

            for model in models:
                torch.save(
                    net[model].state_dict(),
                    os.path.join(directories['checkpoints'], f'{model}_net_ckpt_epoch_{epoch:02d}.pt')
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Image Watermarking',
        description='Learns an image watermarking method by training a '
                    'hiding, reconstruction and discriminator network.'
    )

    # dataset paths
    parser.add_argument('--train_data', type=str, default='datasets/VOC_deblurred_conv/train_80',
                        help='training data path')
    parser.add_argument('--valid_data', type=str, default='datasets/VOC_deblurred_conv/valid_32',
                        help='validation data path')

    # dataloader settings
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='number of workers')

    # optimization settings
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam optimizer')

    # paths to continue training
    parser.add_argument('--h_net_ckpt', type=str, nargs='?',
                        default='image_watermarking/experiments/'
                                'experiment_2023-08-26-21_27_47/checkpoints/h_net_ckpt_epoch_49.pt',
                        help='[optional] hiding network checkpoint path to continue training')
    parser.add_argument('--r_net_ckpt', type=str, nargs='?',
                        default='image_watermarking/experiments/'
                                'experiment_2023-08-26-21_27_47/checkpoints/r_net_ckpt_epoch_49.pt',
                        help='[optional] reconstruction network checkpoint path to continue training.')
    parser.add_argument('--d_net_ckpt', type=str, nargs='?',
                        default='image_watermarking/experiments/'
                                'experiment_2023-08-26-21_27_47/checkpoints/d_net_ckpt_epoch_49.pt',
                        help='[optional] discriminator checkpoint path to continue training.')

    # training settings
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='number of epochs to perform training')
    parser.add_argument('--l_mse', type=float, default=10000,
                        help='weighting for losses calculated via the mse (except adversarial loss)')
    parser.add_argument('--l_adv', type=float, default=0.01,
                        help='weighting for the adversarial loss')
    parser.add_argument('--l', type=float, default=1/3,
                        help='weighting for the extraction loss when summed with embedding loss')
    parser.add_argument('--plot_batch', type=int, default=5,
                        help='plot images each plot_batch batches')

    # watermark image paths
    parser.add_argument('--watermark', type=str, default='image_watermarking/watermarks/flower_rgb.png',
                        help='path to the watermark to be embedded into images')
    parser.add_argument('--watermark_clean', type=str, default='image_watermarking/watermarks/clean.png',
                        help='path to the clean watermark')

    # advanced settings
    parser.add_argument('--img_size', type=Tuple[int, int], default=(256, 256),
                        help='size in pixels of the resulting image')

    args = parser.parse_args()
    main(opt=args)
