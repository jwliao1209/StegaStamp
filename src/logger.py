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
    ):
        if batch_idx == 0:
            images = batch["image"][0].cpu().permute(1,2,0)
            outputs = outputs["outputs"]["encoder"][0].cpu().permute(1,2,0)
        
            # captions = [",".join(map(lambda x: str(x), f.tolist())) for f in batch["fingerprint"][0].cpu()]
            self.wandb_logger.log_image(
                key='sample_images',
                images=[images, outputs],
                # caption=captions,
            )

        return
