import torch.nn as nn
import torch
import torch.nn.functional as F


class GuidedFilter(nn.Module):
    def __init__(self):
        super(GuidedFilter, self).__init__()

    def forward(self, I, p, radius=3, eps=1e-6):
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


class RGBHSV(nn.Module):
    def __init__(self):
        super(RGBHSV, self).__init__()

    def forward(self, rgb: torch.Tensor):
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


class HSVRGB(nn.Module):
    def __init__(self):
        super(HSVRGB, self).__init__()

    def forward(self, hsv):
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
