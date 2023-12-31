import torch.nn as nn


class StegaStampLoss(nn.Module):
    def __init__(self, mse_weight, bce_weight):
        super(StegaStampLoss, self).__init__()
        self.mse_weight = mse_weight
        self.bce_weight = bce_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

        self.active_mse_weight = 0
        self.steps_since_mse_loss_activated = -1
        self.mse_loss_await = 1000
        self.mse_loss_ramp = 3000

    def update_mse_weight(self, acc):
        if acc > 0.9:
            self.steps_since_mse_loss_activated += 1

        self.active_mse_weight = min(
            max(0, self.mse_weight * (self.steps_since_mse_loss_activated - self.mse_loss_await) / self.mse_loss_ramp), self.mse_weight
        )
        return

    def forward(self, inputs, outputs):
        mse_loss = self.mse(inputs["image"], outputs["encoder"])
        bce_loss = self.bce(outputs["decoder"], inputs["fingerprint"])
        total_loss = self.active_mse_weight * mse_loss + self.bce_weight * bce_loss
        return dict(mse=mse_loss, bce=bce_loss, total=total_loss)
