import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(LocalAttentionModule, self).__init__()
        self.local_conv1 = nn.Conv2d(
            in_channels, in_channels // reduction, kernel_size=1
        )
        self.local_bn1 = nn.BatchNorm2d(in_channels // reduction)
        self.local_relu = nn.ReLU(inplace=True)
        self.local_conv2 = nn.Conv2d(
            in_channels // reduction, in_channels, kernel_size=1
        )
        self.local_bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        local_branch = self.local_conv1(x)
        if x.size(0) > 1:
            local_branch = self.local_bn1(local_branch)
        local_branch = self.local_relu(local_branch)
        local_branch = self.local_conv2(local_branch)
        if x.size(0) > 1:
            local_branch = self.local_bn2(local_branch)
        return local_branch


class GlobalAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(GlobalAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # Global average pooling branch
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # First branch
        self.global_conv1 = nn.Conv2d(
            in_channels, in_channels // reduction, kernel_size=1
        )
        self.global_bn1 = nn.BatchNorm2d(in_channels // reduction)
        self.global_relu = nn.ReLU(inplace=True)
        self.global_conv2 = nn.Conv2d(
            in_channels // reduction, in_channels, kernel_size=1
        )
        self.global_bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # Global average pooling branch
        avg_pool = self.global_avg_pool(x)

        # First branch

        global_branch = self.global_conv1(avg_pool)
        if x.size(0) > 1:
            global_branch = self.global_bn1(global_branch)

        global_branch = self.global_relu(global_branch)

        global_branch = self.global_conv2(global_branch)

        if x.size(0) > 1:
            global_branch = self.global_bn2(global_branch)
        return global_branch


class MultiScaleChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(MultiScaleChannelAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.local_attention = LocalAttentionModule(in_channels, reduction)
        self.global_attention = GlobalAttentionModule(in_channels, reduction)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.local_attention(x) + self.global_attention(x))

        return out


class AttentionFeatureFusion(nn.Module):
    def __init__(self, in_channels=128, reduction=16):
        super(AttentionFeatureFusion, self).__init__()

        self.attention_rgb = MultiScaleChannelAttentionModule(in_channels, reduction)
        self.attention_nir = MultiScaleChannelAttentionModule(in_channels, reduction)

        self.attention_fusion = MultiScaleChannelAttentionModule(in_channels, reduction)

    def forward(self, rgb, nir):
        # Apply the attention modules to the input features
        rgb_att = self.attention_rgb(rgb)
        nir_att = self.attention_nir(nir)

        # Concatenate the attention features
        att_add = rgb_att + nir_att

        rgb_att = 2 * rgb * rgb_att / att_add
        nir_att = 2 * nir * nir_att / att_add
        att_features = rgb_att + nir_att

        # Apply the attention fusion module
        att_fusion = self.attention_fusion(att_features)

        out = att_fusion * rgb_att + (1 - att_fusion) * nir_att

        return out


class ConcatFusion(nn.Module):
    def __init__(self, in_channels=128, reduction=16):
        super(ConcatFusion, self).__init__()

        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, 1)

    def forward(self, rgb, nir):
        return self.conv1(torch.cat((rgb, nir), dim=1))
