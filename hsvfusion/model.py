from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.extractor import BasicEncoder, ResidualBlock
from core.fusion import AttentionFeatureFusion, BAttentionFeatureFusion

from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from hsvfusion.utils import HSVRGB, RGBHSV, GuidedFilter


class HSVNet(torch.nn.Module):

    def __init__(self, args):
        super(HSVNet, self).__init__()

        self.encoder = BasicEncoder(downsample=2, output_dim=256)
        self.fusion = AttentionFeatureFusion(in_channels=256, reduction=4)
        self.channel_reduction = nn.Sequential(
            ResidualBlock(256, 128),
            ResidualBlock(128, 64),
            nn.Conv2d(64, 2, 3, padding=1),
            nn.ReLU(),
        )

        self.raft_stereo = RAFTStereo(args)
        self.hsv2rgb = HSVRGB()
        self.rgb2hsv = RGBHSV()
        self.gf = GuidedFilter()
        self.padder = None

    def forward(
        self, v: torch.Tensor, n: torch.Tensor, raft_stereo=True, att_out=False
    ):

        _, _, H, W = v.size()
        if self.padder is None:
            self.padder = InputPadder(v.size(), divis_by=32)

        v, n = self.padder.pad(v, n)

        hsv = self.rgb2hsv(v)

        hsv_input = (v / 255 * 2 - 1).contiguous()
        n_input = (n / 255 * 2 - 1).contiguous()

        hsv_fmap = self.encoder(hsv_input)
        nir_fmap = self.encoder(n_input.repeat(1, 3, 1, 1))
        fusion_fmap = self.fusion(hsv_fmap, nir_fmap)

        w = F.sigmoid(self.channel_reduction(fusion_fmap))

        w = F.interpolate(w, scale_factor=4, mode="bilinear", align_corners=False)

        rgb = self.hsv2rgb(
            torch.concat(
                [hsv[:, :1], hsv[:, 1:2], w[:, :1] * hsv[:, 2:3] + w[:, 1:] * n],
                dim=1,
            )
        )
        # rgb = self.gf(n, rgb, radius=5) * 255
        rgb = rgb[..., :H, :W]
        if raft_stereo:
            rgb_left, rgb_right = torch.split(rgb, W // 2, dim=-1)
            flow = self.raft_stereo(rgb_left, rgb_right, iters=5)
            return rgb, flow
        if att_out:
            return self.fusion(hsv_fmap, nir_fmap, debug_attention=True)
        return rgb

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(False)
                m.eval()

    def freeze_raft(
        self,
    ):
        for name, param in self.encoder.named_parameters():
            param.requires_grad_(False)
        for name, param in self.raft_stereo.named_parameters():
            param.requires_grad_(False)
        self.freeze_bn()
