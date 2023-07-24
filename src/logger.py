import torch
from pytorch_lightning.callbacks import Callback


class LogPredictionsCallback(Callback):
    def __init__(self, wandb_logger, sample_num=8):
        self.wandb_logger = wandb_logger
        self.sample_num = sample_num
    
    @staticmethod
    def get_image_to_cpu_and_permute_dim(image):
        return image.cpu().permute(1, 2, 0)

    def on_validation_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
    ):
        if batch_idx == 0:
            images = []
            for i in range(self.sample_num):
                image = [batch["image"][i], outputs["outputs"]["encoder"][i]]
                images.append(
                    torch.cat(list(map(self.get_image_to_cpu_and_permute_dim, image)))
                )
            # captions = [",".join(map(lambda x: str(x), f.tolist())) for f in batch["fingerprint"][0].cpu()]
            self.wandb_logger.log_image(
                key='sample_images',
                images=images,
                # caption=captions,
            )

        return
