# import torch
# import torch.nn as nn


# class DoubleConv(nn.Module):
#     """Two consecutive convolution blocks."""

#     def __init__(self, in_channels, out_channels):

#         super().__init__()

#         self.block = nn.Sequential(

#             nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),

#             nn.BatchNorm2d(out_channels),

#             nn.ReLU(inplace=True),

#             nn.Conv2d(
#                 out_channels,
#                 out_channels,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),

#             nn.BatchNorm2d(out_channels),

#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.block(x)


# class UNet(nn.Module):

#     def __init__(
#         self,
#         in_channels=3,
#         out_channels=1,
#     ):

#         super().__init__()

#         # -------------------------
#         # Encoder
#         # -------------------------

#         self.enc1 = DoubleConv(
#             in_channels,
#             64,
#         )

#         self.enc2 = DoubleConv(
#             64,
#             128,
#         )

#         self.enc3 = DoubleConv(
#             128,
#             256,
#         )

#         self.enc4 = DoubleConv(
#             256,
#             512,
#         )

#         self.pool = nn.MaxPool2d(
#             kernel_size=2,
#             stride=2,
#         )

#         # -------------------------
#         # Bottleneck
#         # -------------------------

#         self.bottleneck = DoubleConv(
#             512,
#             1024,
#         )

#         # -------------------------
#         # Decoder
#         # -------------------------

#         self.up4 = nn.ConvTranspose2d(
#             1024,
#             512,
#             kernel_size=2,
#             stride=2,
#         )

#         self.dec4 = DoubleConv(
#             1024,
#             512,
#         )

#         self.up3 = nn.ConvTranspose2d(
#             512,
#             256,
#             kernel_size=2,
#             stride=2,
#         )

#         self.dec3 = DoubleConv(
#             512,
#             256,
#         )

#         self.up2 = nn.ConvTranspose2d(
#             256,
#             128,
#             kernel_size=2,
#             stride=2,
#         )

#         self.dec2 = DoubleConv(
#             256,
#             128,
#         )

#         self.up1 = nn.ConvTranspose2d(
#             128,
#             64,
#             kernel_size=2,
#             stride=2,
#         )

#         self.dec1 = DoubleConv(
#             128,
#             64,
#         )

#         # -------------------------
#         # Output
#         # -------------------------

#         self.final = nn.Conv2d(
#             64,
#             out_channels,
#             kernel_size=1,
#         )

#     def forward(self, x):

#         # Encoder

#         e1 = self.enc1(x)

#         e2 = self.enc2(
#             self.pool(e1)
#         )

#         e3 = self.enc3(
#             self.pool(e2)
#         )

#         e4 = self.enc4(
#             self.pool(e3)
#         )

#         # Bottleneck

#         b = self.bottleneck(
#             self.pool(e4)
#         )

#         # Decoder

#         d4 = self.up4(b)

#         d4 = torch.cat(
#             [d4, e4],
#             dim=1,
#         )

#         d4 = self.dec4(d4)

#         d3 = self.up3(d4)

#         d3 = torch.cat(
#             [d3, e3],
#             dim=1,
#         )

#         d3 = self.dec3(d3)

#         d2 = self.up2(d3)

#         d2 = torch.cat(
#             [d2, e2],
#             dim=1,
#         )

#         d2 = self.dec2(d2)

#         d1 = self.up1(d2)

#         d1 = torch.cat(
#             [d1, e1],
#             dim=1,
#         )

#         d1 = self.dec1(d1)

#         return self.final(d1)

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):

        x = nn.functional.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class UNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        pretrained=True,
    ):
        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet18(weights=weights)

        # --------------------------------------------------
        # ResNet Encoder
        # --------------------------------------------------

        self.input_layer = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )

        self.maxpool = backbone.maxpool

        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        # --------------------------------------------------
        # Decoder
        # --------------------------------------------------

        self.decoder4 = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
        )

        self.decoder3 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
        )

        self.decoder2 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
        )

        self.decoder1 = DecoderBlock(
            in_channels=64,
            skip_channels=64,
            out_channels=64,
        )

        # Final upsampling:
        # 128 × 128 → 256 × 256

        self.final = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(
                64,
                out_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x):

        # --------------------------------------------------
        # Encoder
        # --------------------------------------------------

        x0 = self.input_layer(x)
        # [B, 64, 128, 128]

        x1 = self.maxpool(x0)
        x1 = self.encoder1(x1)
        # [B, 64, 64, 64]

        x2 = self.encoder2(x1)
        # [B, 128, 32, 32]

        x3 = self.encoder3(x2)
        # [B, 256, 16, 16]

        x4 = self.encoder4(x3)
        # [B, 512, 8, 8]

        # --------------------------------------------------
        # Decoder
        # --------------------------------------------------

        d4 = self.decoder4(x4, x3)
        # [B, 256, 16, 16]

        d3 = self.decoder3(d4, x2)
        # [B, 128, 32, 32]

        d2 = self.decoder2(d3, x1)
        # [B, 64, 64, 64]

        d1 = self.decoder1(d2, x0)
        # [B, 64, 128, 128]

        output = self.final(d1)
        # [B, 1, 256, 256]

        return output