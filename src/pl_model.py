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

    # def share_step(self, batch):
    #     images, labels = batch
    #     outputs = self.forward(images)
    #     loss = self.criterion(images, labels)
    #     return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = self.criterion(batch, outputs)
        acc = self.compute_accuracy(outputs["decoder"], batch["fingerprint"])

        self.log("train_loss/mse", loss["mse"], prog_bar=False)
        self.log("train_loss/bce", loss["bce"], prog_bar=True)
        self.log("train_loss/total", loss["total"], prog_bar=False)
        self.log("train_acc", acc, prog_bar=True)

        if self.criterion.steps_since_l2_loss_activated == -1:
            if acc > 0.9:
                self.criterion.steps_since_l2_loss_activated = 0
        else:
            self.criterion.steps_since_l2_loss_activated += 1

        return loss["total"]
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = self.criterion(batch, outputs)
        acc = self.compute_accuracy(outputs["decoder"], batch["fingerprint"])

        self.log("valid_loss/mse", loss["mse"], prog_bar=False)
        self.log("valid_loss/bce", loss["bce"], prog_bar=False)
        self.log("valid_loss/total", loss["total"], prog_bar=True)
        self.log("valid_acc", acc, prog_bar=True)
        return
    
    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = self.criterion(batch, outputs)
        acc = self.compute_accuracy(outputs["decoder"], batch["fingerprint"])

        self.log("test_loss/mse", loss["mse"], prog_bar=False)
        self.log("test_loss/bce", loss["bce"], prog_bar=False)
        self.log("test_loss/total", loss["total"], prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return dict(loss=loss["total"], acc=acc)
