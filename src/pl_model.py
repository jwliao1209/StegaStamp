import torch
import pytorch_lightning as pl


class LitModel(pl.LightningModule):
    def __init__(self, model, criterion, optimizer, scheduler=None):
        super(LitModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_hyperparameters()

    def forward(self, **inputs):
        return self.model(**inputs)

    def configure_optimizers(self):
        return self.optimizer#, [scheduler]

    def compute_accuracy(self, predition, ground_truth):
        return ((predition > 0).long() == ground_truth).float().mean()

    def share_step(self, batch, prefix):
        outputs = self.forward(**batch)
        loss = self.criterion(batch, outputs)
        acc = self.compute_accuracy(outputs["decoder"], batch["fingerprint"])

        self.log(f"{prefix}/mse_loss", loss["mse"], prog_bar=False)
        self.log(f"{prefix}/bce_loss", loss["bce"], prog_bar=False)
        self.log(f"{prefix}/total_loss", loss["total"], prog_bar=True)
        self.log(f"{prefix}/acc", acc, prog_bar=True)

        return dict(inputs=batch, outputs=outputs, loss=loss, acc=acc)

    def training_step(self, batch, batch_idx):
        output_dict = self.share_step(batch, "train")
        self.log(
            f"mse_weight",
            torch.tensor(self.criterion.active_mse_weight), prog_bar=False
        )
        self.criterion.update_mse_weight(output_dict["acc"])
        return output_dict["loss"]["total"]

    def validation_step(self, batch, batch_idx):
        output_dict = self.share_step(batch, "valid")
        return dict(outputs=output_dict["outputs"])
    
    def test_step(self, batch, batch_idx):
        output_dict = self.share_step(batch, "test")
        return dict(outputs=output_dict["outputs"])
