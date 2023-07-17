import math

import torch
import torch.nn as nn
from torch.nn.functional import relu, sigmoid
from pytorch_lightning import LightningModule


# class DecoderBlock():
#     def __init__(self, in_channels, out_channels):
#         super(DecoderBlock, self).__init__()

#     def forward(self, x):


class StegaStampEncoder(nn.Module):
    def __init__(
        self,
        resolution=32,
        image_channel=1,
        fingerprint_size=100,
        return_residual=False,
    ):
        super(StegaStampEncoder, self).__init__()
        self.fingerprint_size = fingerprint_size
        self.image_channel = image_channel
        self.return_residual = return_residual
        self.secret_dense = nn.Linear(self.fingerprint_size, 16 * 16 * image_channel)

        log_resolution = int(math.log(resolution, 2))
        assert resolution == 2 ** log_resolution, f"Image resolution must be a power of 2, got {resolution}."

        self.fingerprint_upsample = nn.Upsample(scale_factor=(2**(log_resolution-4), 2**(log_resolution-4)))
        self.conv1 = nn.Conv2d(2 * image_channel, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)
        self.pad6 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up6 = nn.Conv2d(256, 128, 2, 1)
        self.upsample6 = nn.Upsample(scale_factor=(2, 2))
        self.conv6 = nn.Conv2d(128 + 128, 128, 3, 1, 1)
        self.pad7 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up7 = nn.Conv2d(128, 64, 2, 1)
        self.upsample7 = nn.Upsample(scale_factor=(2, 2))
        self.conv7 = nn.Conv2d(64 + 64, 64, 3, 1, 1)
        self.pad8 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up8 = nn.Conv2d(64, 32, 2, 1)
        self.upsample8 = nn.Upsample(scale_factor=(2, 2))
        self.conv8 = nn.Conv2d(32 + 32, 32, 3, 1, 1)
        self.pad9 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up9 = nn.Conv2d(32, 32, 2, 1)
        self.upsample9 = nn.Upsample(scale_factor=(2, 2))
        self.conv9 = nn.Conv2d(32 + 32 + 2 * image_channel, 32, 3, 1, 1)
        self.conv10 = nn.Conv2d(32, 32, 3, 1, 1)
        self.residual = nn.Conv2d(32, image_channel, 1)

    def forward(self, fingerprint, image):
        fingerprint = relu(self.secret_dense(fingerprint))
        fingerprint = fingerprint.view((-1, self.IMAGE_CHANNELS, 16, 16))
        fingerprint_enlarged = self.fingerprint_upsample(fingerprint)
        inputs = torch.cat([fingerprint_enlarged, image], dim=1)
        conv1 = relu(self.conv1(inputs))
        conv2 = relu(self.conv2(conv1))
        conv3 = relu(self.conv3(conv2))
        conv4 = relu(self.conv4(conv3))
        conv5 = relu(self.conv5(conv4))
        up6 = relu(self.up6(self.pad6(self.upsample6(conv5))))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = relu(self.conv6(merge6))
        up7 = relu(self.up7(self.pad7(self.upsample7(conv6))))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = relu(self.conv7(merge7))
        up8 = relu(self.up8(self.pad8(self.upsample8(conv7))))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = relu(self.conv8(merge8))
        up9 = relu(self.up9(self.pad9(self.upsample9(conv8))))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = relu(self.conv9(merge9))
        conv10 = relu(self.conv10(conv9))
        residual = self.residual(conv10)
        if not self.return_residual:
            residual = sigmoid(residual)
        return residual


class StegaStampDecoder(nn.Module):
    def __init__(self, resolution=32, image_channel=1, fingerprint_size=1):
        super(StegaStampDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(image_channel, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.ReLU(),
        )
        self.fc_input_size = resolution * resolution * 128 // 32 // 32, 512
        self.dense = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, fingerprint_size),
        )

    def forward(self, x):
        x = self.decoder(x)
        out = self.dense(x.view(-1, self.fc_input_size))
        return out


class StegaStampLoss(nn.Module):
    def __init__(self, mse_weight, bce_weight):
        self.mse_weight = mse_weight
        self.bce_weight = bce_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    # def update_mse_weight(self):
    #     return min(
    #         max(0, self.mse_weight * (steps_since_l2_loss_activated - args.l2_loss_await) / args.l2_loss_ramp),
    #         self.mse_weight
    #     )

    def forward(self, inputs, outputs):
        mse_loss = self.mse(inputs["images"], outputs["encoder"])
        bce_loss = self.bce(inputs["fingerprint"], outputs["decoder"])
        return self.mse_weight * mse_loss + self.bce_weight * bce_loss


class StegaStampModel(nn.Module):
    def __init__(self):
        super(StegaStampModel, self).__init__()
        self.encoder = StegaStampEncoder()
        self.decoder = StegaStampDecoder()
    
    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return dict(encoder=encoder_out, decoder=decoder_out)


class LitModel(LightningModule):
    def __init__(self, model, optimizer, scheduler):
        super(LitModel, self).__init__()
        self.model = model
        self.criterion = StegaStampLoss(mse_weight=10, bce_weight=1)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x):
        return self.model(x)

    def configure_optimizer(self):
        return [self.optimizer], [self.scheduler]
    
    # def share_step(self, batch):
    #     images, labels = batch
    #     outputs = self.forward(images)
    #     loss = self.criterion(images, labels)
    #     return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch["images"])
        loss = self.compute_loss(batch, outputs)
        return dict(loss=loss, inputs=batch, outputs=outputs)
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch["images"])
        loss = self.compute_loss(batch, outputs)
        return dict(loss=loss, inputs=batch, outputs=outputs)
    
    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch["images"])
        loss = self.compute_loss(batch, outputs)
        return dict(loss=loss, inputs=batch, outputs=outputs)
