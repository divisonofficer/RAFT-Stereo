import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.SyncBatchNorm(channels)  # Use SyncBatchNorm
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.SyncBatchNorm(channels)  # Use SyncBatchNorm

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        return out


# Updated LuminanceWeightNet with ResNet blocks
class LuminanceWeightNet(nn.Module):
    def __init__(self):
        super(LuminanceWeightNet, self).__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        # Residual blocks
        self.resblock1 = ResidualBlock(16)
        self.resblock2 = ResidualBlock(16)
        # Existing layers
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, rgb, nir):
        x = torch.cat([rgb, nir], dim=1)
        x = F.relu(self.conv1(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = F.relu(self.conv2(x))
        weight = torch.sigmoid(self.conv3(x))
        return weight


# Updated MNet with ResNet blocks
class MNet(nn.Module):
    def __init__(self):
        super(MNet, self).__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        # Residual blocks
        self.resblock1 = ResidualBlock(16)
        self.resblock2 = ResidualBlock(16)
        # Existing layers
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, rgb, nir):
        x = torch.cat([rgb, nir], dim=1)
        x = F.relu(self.conv1(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = F.relu(self.conv2(x))
        m = torch.tanh(self.conv3(x))
        return m


# Fusion model
class RGBNIRFusionNet(nn.Module):
    def __init__(self):
        super(RGBNIRFusionNet, self).__init__()
        self.luminance_weight_net = LuminanceWeightNet()
        self.m_net = MNet()
        self.m_net_2 = MNet()

    def forward(self, inputs):
        rgb = inputs[:, :3, :, :] / 255.0
        nir = inputs[:, 3:, :, :] / 255.0
        with torch.cuda.amp.autocast(True):
            # Convert RGB to YCrCb and extract luminance (Y)
            rgb_ycrcb = self.rgb_to_ycrcb(rgb)
            y_channel = rgb_ycrcb[:, 0:1, :, :]  # Extract luminance channel (Y)
            cr_channel = rgb_ycrcb[:, 1:2, :, :]  # Extract Cr channel
            cb_channel = rgb_ycrcb[:, 2:3, :, :]  # Extract Cb channel

            # Calculate luminance weight and m
            luminance_weight = self.luminance_weight_net(rgb, nir)
            m = self.m_net(rgb, nir)
            m2 = self.m_net_2(rgb, nir)

            # Calculate the new luminance channel
            y_fused = y_channel * luminance_weight + nir * (1 - luminance_weight)

            # Adjust Cr and Cb channels using m
            cr_fused = cr_channel * (1 + m)
            cb_fused = cb_channel * (1 + m2)

            # Combine Y, Cr, and Cb to form the new YCrCb image
            ycrcb_fused = torch.cat([y_fused, cr_fused, cb_fused], dim=1)

            # Convert the fused YCrCb back to RGB
            fusion = self.ycrcb_to_rgb(ycrcb_fused)

            return fusion * 255.0

    def rgb_to_ycrcb(self, rgb):
        # Assuming input is normalized to [0, 1], convert to YCrCb
        r = rgb[:, 0:1, :, :]
        g = rgb[:, 1:2, :, :]
        b = rgb[:, 2:3, :, :]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cr = (r - y) * 0.713 + 0.5
        cb = (b - y) * 0.564 + 0.5
        return torch.cat([y, cr, cb], dim=1)

    def ycrcb_to_rgb(self, ycrcb):
        # Convert YCrCb back to RGB
        y = ycrcb[:, 0:1, :, :]
        cr = ycrcb[:, 1:2, :, :]
        cb = ycrcb[:, 2:3, :, :]
        r = y + 1.403 * (cr - 0.5)
        g = y - 0.344 * (cb - 0.5) - 0.714 * (cr - 0.5)
        b = y + 1.773 * (cb - 0.5)
        return torch.cat([r, g, b], dim=1)
