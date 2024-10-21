from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.extractor import BasicEncoder
from core.fusion import AttentionFeatureFusion
from core.raft_stereo import RAFTStereo


class HSVNet(torch.nn.Module):

    def __init__(self, args):
        super(HSVNet, self).__init__()

        self.encoder = BasicEncoder(downsample=2, output_dim=256)
        self.fusion = AttentionFeatureFusion(in_channels=256, reduction=4)
        self.channel_reduction = nn.Conv2d(
            in_channels=256, out_channels=2, kernel_size=1
        )

        self.raft_stereo = RAFTStereo(args)

    def rgb_to_hsv(self, rgb: torch.Tensor):
        """
        RGB 텐서를 HSV 텐서로 변환합니다.

        입력:
            rgb: Tensor of shape (b, 3, h, w) with values in [0, 1]

        출력:
            hsv: Tensor of shape (b, 3, h, w) with H in [0, 360], S and V in [0, 1]
        """
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

        maxc, _ = rgb.max(dim=1)
        minc, _ = rgb.min(dim=1)
        delta = maxc - minc + 1e-10  # 작은 값을 더해 division by zero 방지

        # Hue 계산
        mask = delta > 0
        h = torch.zeros_like(maxc)

        # Red is max
        mask_r = (maxc == r) & mask
        h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360

        # Green is max
        mask_g = (maxc == g) & mask
        h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120) % 360

        # Blue is max
        mask_b = (maxc == b) & mask
        h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240) % 360

        # Saturation 계산
        s = torch.zeros_like(maxc)
        s[maxc != 0] = delta[maxc != 0] / maxc[maxc != 0]

        # Value 계산
        v = maxc

        hsv = torch.stack([h, s, v], dim=1)
        return hsv

    def hsv_to_rgb(self, hsv):
        """
        HSV 텐서를 RGB 텐서로 변환합니다.

        입력:
            hsv: Tensor of shape (b, 3, h, w)
                - H in [0, 360]
                - S in [0, 1]
                - V in [0, 1]

        출력:
            rgb: Tensor of shape (b, 3, h, w) with values in [0, 1]
        """
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]

        c = v * s  # 채도와 명도를 이용해 채도
        h_prime = h / 60.0  # Hue를 60으로 나눠 섹션 결정
        x = c * (1 - torch.abs((h_prime % 2) - 1))

        zero = torch.zeros_like(h)

        # 각 섹션에 따른 RGB 중간값 계산
        cond = (h_prime >= 0) & (h_prime < 1)
        r = torch.where(cond, c, torch.zeros_like(c))
        g = torch.where(cond, x, torch.zeros_like(x))
        b = torch.zeros_like(x)

        cond = (h_prime >= 1) & (h_prime < 2)
        r = torch.where(cond, x, r)
        g = torch.where(cond, c, g)

        cond = (h_prime >= 2) & (h_prime < 3)
        g = torch.where(cond, c, g)
        b = torch.where(cond, x, b)

        cond = (h_prime >= 3) & (h_prime < 4)
        g = torch.where(cond, x, g)
        b = torch.where(cond, c, b)

        cond = (h_prime >= 4) & (h_prime < 5)
        r = torch.where(cond, x, r)
        b = torch.where(cond, c, b)

        cond = (h_prime >= 5) & (h_prime < 6)
        r = torch.where(cond, c, r)
        b = torch.where(cond, x, b)

        m = v - c
        r = r + m
        g = g + m
        b = b + m

        rgb = torch.stack([r, g, b], dim=1)
        return rgb

    def guided_filter(self, I, p, radius=3, eps=1e-6):
        """
        Perform guided filtering on input images using PyTorch.

        Args:
            I (torch.Tensor): Guide image tensor of shape (batch_size, 1, height, width).
            p (torch.Tensor): Input image tensor to be filtered of shape (batch_size, 3, height, width).
            radius (int, optional): Radius of the window. Default is 15.
            eps (float, optional): Regularization parameter to avoid division by zero. Default is 1e-6.

        Returns:
            torch.Tensor: Filtered image tensor of shape (batch_size, 3, height, width).
        """
        # Ensure the guide image has a single channel
        assert (
            I.dim() == 4 and I.size(1) == 1
        ), "Guide image I must have shape (batch, 1, H, W)"
        # Ensure the input image has three channels
        assert (
            p.dim() == 4 and p.size(1) == 3
        ), "Input image p must have shape (batch, 3, H, W)"

        batch_size, _, height, width = I.size()
        window_size = (2 * radius + 1) ** 2

        # Define a box filter kernel
        # The kernel has shape (channels, 1, kernel_size, kernel_size) and is normalized
        # to compute the mean.
        def box_filter(x):
            # x: (batch, channels, height, width)
            channels = x.size(1)
            kernel = (
                torch.ones(
                    (channels, 1, 2 * radius + 1, 2 * radius + 1),
                    dtype=x.dtype,
                    device=x.device,
                )
                / window_size
            )
            return F.conv2d(
                x, weight=kernel, bias=None, stride=1, padding=radius, groups=channels
            )

        # Step 1: Compute mean of I, p, I*p, and I*I
        mean_I = box_filter(I)  # (batch, 1, H, W)
        mean_p = box_filter(p)  # (batch, 3, H, W)
        mean_Ip = box_filter(I * p)  # (batch, 3, H, W)
        mean_II = box_filter(I * I)  # (batch, 1, H, W)

        # Step 2: Compute covariance of (I, p) and variance of I
        cov_Ip = mean_Ip - mean_I * mean_p  # (batch, 3, H, W)
        var_I = mean_II - mean_I * mean_I  # (batch, 1, H, W)

        # Step 3: Compute a and b
        a = cov_Ip / (var_I + eps)  # (batch, 3, H, W)
        b = mean_p - a * mean_I  # (batch, 3, H, W)

        # Step 4: Compute mean of a and b
        mean_a = box_filter(a)  # (batch, 3, H, W)
        mean_b = box_filter(b)  # (batch, 3, H, W)

        # Step 5: Compute the output image
        q = mean_a * I + mean_b  # Broadcasting I from (batch,1,H,W) to (batch,3,H,W)

        # Optionally, clamp the output to valid image range
        q = torch.clamp(q, 0, 1)

        return q

    def forward(
        self, v: torch.Tensor, n: torch.Tensor, raft_stereo=True, att_out=False
    ):
        _, _, H, W = v.size()
        if v.max() > 1:
            v /= 255
            n /= 255
        multiple = 32
        pad_h = (multiple - H % multiple) if H % multiple != 0 else 0
        pad_w = (multiple - W % multiple) if W % multiple != 0 else 0
        v = F.pad(
            v,
            (0, pad_w, 0, pad_h),
        )
        n = F.pad(
            n,
            (0, pad_w, 0, pad_h),
        )

        hsv = self.rgb_to_hsv(v)
        hsv_input = (hsv * 2 - 1).contiguous()
        n_input = (n * 2 - 1).contiguous()

        hsv_fmap = self.encoder(hsv_input)
        nir_fmap = self.encoder(n_input.repeat(1, 3, 1, 1))
        fusion_fmap = self.fusion(hsv_fmap, nir_fmap)
        w = F.sigmoid(self.channel_reduction(fusion_fmap))
        w = F.interpolate(w, scale_factor=4, mode="bilinear", align_corners=False)
        # print(v.shape, n.shape, w.shape, hsv.shape)

        rgb = self.hsv_to_rgb(
            torch.concat(
                [hsv[:, :1], hsv[:, 1:2], w[:, :1] * hsv[:, 2:3] + w[:, 1:] * n], dim=1
            )
        )[:, :, :H, :W]
        rgb = self.guided_filter(n[:, :, :H, :W], rgb) * 255

        if raft_stereo:
            pd = 8
            rgb_left, rgb_right = torch.split(rgb, W // 2, dim=-1)
            rgb_left = rgb_left[:, :, :, : W // 2 - pd]
            rgb_right = rgb_right[:, :, :, pd:]
            flow = self.raft_stereo(rgb_left, rgb_right, test_mode=True)[1][
                :, :, :H, : W // 2 - pd
            ]
            return rgb, flow
        if att_out:
            return self.fusion(hsv_fmap, nir_fmap, debug_attention=True)
        return rgb

    def freeze_raft(
        self,
    ):
        for name, param in self.encoder.named_parameters():
            param.requires_grad_(False)
        for name, param in self.raft_stereo.named_parameters():
            param.requires_grad_(False)
        print("Raft Freezed")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        encoder_exists = (
            len([key for key in state_dict if "module.fnet.encoder." in key]) > 0
        )
        if not encoder_exists:
            keys = list(state_dict.keys())
            for key in keys:
                if "module.fnet." in key:
                    key_encoder = key.replace("module.fnet.", "module.encoder.")
                    state_dict[key_encoder] = state_dict[key]
        ret = super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        return ret

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        ret = super().load_state_dict(state_dict, strict, assign)
        if "encoder" not in state_dict:
            self.encoder.load_state_dict(state_dict, strict, assign)
        return ret
