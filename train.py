import argparse
import wandb

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.dataset import LitDataLoader
from src.model import StegaStampModel
from src.losses import StegaStampLoss
from src.pl_model import LitModel
from src.logger import LogPredictionsCallback


wandb.init(entity="citi2023", project="StegaStamp", name="stegastamp")
wandb_logger = WandbLogger(
    project="StegaStamp",
    log_model="all",  # log all new checkpoints during training
)


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
        default=1e-4,
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
        default="cuda",
        help='Accelerator.',
    )
    parser.add_argument(
        '--devices',
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
    model = StegaStampModel()
    criterion = StegaStampLoss(mse_weight=10, bce_weight=1)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr,
        # weight_decay=args.weight_decay
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer=optimizer, step_size=5, gamma=0.8
    # )
    lit_model = LitModel(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        # scheduler=scheduler,
    )
    lit_dataloader = LitDataLoader(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # checkpoint_callback = ModelCheckpoint(monitor='valid_loss', mode='min')
    log_predictions_callback = LogPredictionsCallback(wandb_logger=wandb_logger)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,
        accelerator=args.accelerator,
        devices=args.devices,
        # precision="16-mixed",
        callbacks=[
            # checkpoint_callback,
            log_predictions_callback,
        ],
        logger=wandb_logger,
    )
    trainer.fit(lit_model, datamodule=lit_dataloader)
    return


if __name__ == "__main__":
    args = parse_arguments()
    train(args)
    wandb.finish()
