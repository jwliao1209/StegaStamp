from pytorch_lightning.callbacks import Callback


class LogPredictionsCallback(Callback):
    def __init__(self, wandb_logger):
        self.wandb_logger = wandb_logger

    def on_validation_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx=0,
    ):
        if batch_idx == 0:
            images = [img for img in batch["image"]]
            captions = [f for f in batch["fingerprint"]]
            self.wandb_logger.log_image(
                key='sample_images',
                images=images,
                caption=captions
            )

        return
