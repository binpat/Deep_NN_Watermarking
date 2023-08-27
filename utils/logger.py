# encoding: utf-8

"""
Logger class to log information about experiments like training performances.

Author: Patrick Binder
Date: 25.08.2023
"""

import time
import os

from typing import Optional
from argparse import Namespace

import torch.nn as nn


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Logger:
    """
    Logger for logging training performance like losses.
    """
    def __init__(self, log_path: str, console: bool = True) -> None:
        self.path = log_path
        self.console = console

        if os.path.exists(self.path):
            self.delete()

        self.create()

    def create(self) -> None:
        with open(self.path, 'w') as log_file:
            log_file.writelines(time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime()) + '\n')

    def delete(self) -> None:
        os.remove(self.path)

    def log(self, text: str, write: bool = True) -> None:
        with open(self.path, 'a') as log_file:
            log_file.writelines(text + '\n')

        if self.console and write:
            print(text)

    def log_model(self, model: nn.Module, ckpt_path: Optional[str]) -> None:
        self.log(str(model))
        self.log(f'Number of trainable parameters: {count_parameters(model)}')
        if ckpt_path is not None:
            self.log(f'Model parameters loaded from: {ckpt_path}')

    def log_arguments(self, args: Namespace) -> None:
        self.log('Arguments:', write=False)
        for arg in vars(args):
            self.log(f'\t{arg}: {getattr(args, arg)}', write=False)
