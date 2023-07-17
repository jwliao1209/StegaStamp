import argparse

import torch
import torch.nn as nn
from pytorch_lightning import Trainer

from src.dataset import LitDataLoader
from src.model import StegaStampModel, LitModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=20,
        help='Maximum epochs.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size.',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate.',
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay.',
    )
    parser.add_argument(
        '--accelerator',
        type=str,
        default="",
        help='Accelerator.',
    )
    parser.add_argument(
        '--device',
        type=int,
        default=[0],
        nargs='+',
        help='Device ID.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number workers.',
    )
    return parser.parse_args()


def train(args):
    optimizer = torch.optim.Adam(lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=5, gamma=0.8
    )
    model = StegaStampModel()
    lit_model = LitModel(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    lit_dataloader = LitDataLoader()

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
    )
    trainer.fit(lit_model, datamodule=lit_dataloader)
    return


if __name__ == "__main__":
    args = parse_arguments()
    train(args)
