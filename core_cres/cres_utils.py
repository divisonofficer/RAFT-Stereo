import torch
import torch.nn.functional as F
import numpy as np


def bilinear_sampler(
    img: torch.Tensor, coords: torch.Tensor, mode="bilinear", mask=False
):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]

    img = F.grid_sample(img, coords, mode=mode, align_corners=True)

    if mask:
        mask = (
            (coords[:, :, :, 0:1] < 0)
            | (coords[:, :, :, 0:1] > W - 1)
            | (coords[:, :, :, 1:2] < 0)
            | (coords[:, :, :, 1:2] > H - 1)
        )
        mask = ~mask
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    x_grid, y_grid = np.meshgrid(np.arange(wd), np.arange(ht))
    y_grid, x_grid = torch.tensor(y_grid, dtype=torch.float32), torch.tensor(
        x_grid, dtype=torch.float32
    )
    coords = torch.stack([x_grid, y_grid], dim=0)
    coords = coords.unsqueeze(0).repeat(batch, 1, 1, 1)
    return coords


def manual_pad(x, pady, padx):
    if pady > 0:
        u = x[:, :, 0:1, :].repeat(1, 1, pady, 1)
        d = x[:, :, -1:, :].repeat(1, 1, pady, 1)
        x = torch.cat([u, x, d], dim=2)
    if padx > 0:
        l = x[:, :, :, 0:1].repeat(1, 1, 1, padx)
        r = x[:, :, :, -1:].repeat(1, 1, 1, padx)
        x = torch.cat([l, x, r], dim=3)
    return x
