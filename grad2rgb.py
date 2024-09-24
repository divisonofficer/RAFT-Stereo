import torch
import torch.nn as nn
from torchvision import models


class UNetResNet(nn.Module):
    def __init__(
        self, encoder_depth=34, pretrained=True, in_channels=6, out_channels=3
    ):
        """
        Initializes the UNetResNet model.

        Parameters:
        - encoder_depth (int): Depth of the ResNet encoder (e.g., 34, 50, 101).
        - pretrained (bool): Whether to use pretrained weights for the encoder.
        - in_channels (int): Number of input channels. Default is 6.
        - out_channels (int): Number of output channels. Default is 3.
        """
        super(UNetResNet, self).__init__()

        # Load a pre-trained ResNet as encoder
        self.encoder = getattr(models, f"resnet{encoder_depth}")(pretrained=pretrained)

        # Modify the first convolutional layer to accept 'in_channels' instead of 3
        original_conv = self.encoder.conv1
        new_conv = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # Initialize the new conv layer
        if pretrained:
            with torch.no_grad():
                if in_channels == 3:
                    # If input channels are 3, use the pretrained weights directly
                    new_conv.weight = original_conv.weight
                elif in_channels > 3:
                    # If input channels > 3, copy the pretrained weights for the first 3 channels
                    new_conv.weight[:, :3, :, :] = original_conv.weight
                    # Initialize the additional channels by copying the first 3 channels
                    new_conv.weight[:, 3:, :, :] = original_conv.weight[:, :3, :, :]
                else:
                    # If input channels < 3, copy the first 'in_channels' weights
                    new_conv.weight = original_conv.weight[:, :in_channels, :, :]
        else:
            # If not using pretrained weights, initialize the new conv layer normally
            nn.init.kaiming_normal_(
                new_conv.weight, mode="fan_out", nonlinearity="relu"
            )

        # Replace the encoder's conv1 with the new_conv
        self.encoder.conv1 = new_conv
        self.encoder_layers = list(self.encoder.children())
        # Remove the fully connected layer and avg pool from the encoder
        self.encoder = nn.Sequential(*self.encoder_layers[:-2])

        # Decoder layers with 5 upsampling steps to match encoder's downsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=2),
            nn.Sigmoid(),  # Assuming output is normalized between 0 and 1
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch, in_channels, height, width)

        Returns:
        - torch.Tensor: Output tensor of shape (batch, out_channels, height, width)
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
