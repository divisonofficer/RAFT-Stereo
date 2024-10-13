from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.extractor import BasicEncoder
from core.fusion import AttentionFeatureFusion


class HSVNet(torch.nn.Module):

    def __init__(self):
        super(HSVNet, self).__init__()

        self.encoder = BasicEncoder()
        self.fusion = AttentionFeatureFusion()
        self.linear = torch.nn.Linear(128, 1)

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

    def forward(self, v: torch.Tensor, n: torch.Tensor):
        if v.max() > 1:
            v /= 255
            n /= 255

        hsv = self.rgb_to_hsv(v)
        hsv_input = hsv * 2 - 1
        n_input = n * 2 - 1
        hsv_fmap = self.encoder(hsv_input)
        nir_fmap = self.encoder(n_input.repeat(1, 3, 1, 1))
        fusion_fmap = self.fusion(hsv_fmap, nir_fmap)
        w = F.sigmoid(self.linear(fusion_fmap))

        hsv[:, 2, :, :] = w * hsv[:, 2, :, :] + (1 - w) * n
        rgb = self.hsv_to_rgb(hsv)
        return rgb * 255

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
                    key_encoder = key.replace("module.fnet.", "encoder.")
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
