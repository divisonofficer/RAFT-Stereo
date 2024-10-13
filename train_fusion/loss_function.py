from typing import List
import torch
import torch.nn.functional as F
import numpy as np
from core.raft_stereo_fusion import RAFTStereoFusion
from train_fusion.ssim.utils import SSIM, warp


def ssim(x, y, channel=1):
    C1 = 0.01**2
    C2 = 0.03**2
    mu_x = F.avg_pool2d(x, 3, 1)
    mu_y = F.avg_pool2d(y, 3, 1)

    sigma_x = F.avg_pool2d(x**2, 3, 1) - mu_x**2
    sigma_y = F.avg_pool2d(y**2, 3, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return SSIM


def reproject_disparity(
    disparity_map: torch.Tensor, left_image: torch.Tensor, max_disparity=128
):
    batch_size, channels, height, width = left_image.shape
    # Create a mesh grid for pixel coordinates
    x_coords, y_coords = torch.meshgrid(
        torch.arange(width, device=left_image.device),
        torch.arange(height, device=left_image.device),
        indexing="xy",
    )

    x_coords = x_coords.unsqueeze(0).expand(batch_size, -1, -1).float()
    y_coords = y_coords.unsqueeze(0).expand(batch_size, -1, -1).float()

    # Compute the new x coordinates based on disparity
    disparity_map = F.pad(
        disparity_map, (1, 1, 1, 1), mode="constant", value=0
    )  # Pad to handle boundary
    disparity_map = F.interpolate(
        disparity_map, size=(height, width), mode="bilinear", align_corners=False
    )  # Resample disparity map

    # Convert disparity map to float type
    disparity_map = disparity_map.squeeze(1)

    x_new_coords = x_coords - disparity_map
    y_new_coords = y_coords

    # Create grid tensor with shape [N, H, W, 2]
    grid = torch.stack([x_new_coords, y_new_coords], dim=-1)

    # Normalize the grid to the range [-1, 1]
    grid = (
        2.0 * grid / torch.tensor([width - 1, height - 1], device=left_image.device)
        - 1.0
    )

    # Perform bilinear interpolation for the reprojected image
    reprojected_image = F.grid_sample(
        left_image, grid, mode="bilinear", align_corners=False
    )

    return reprojected_image


def warp_reproject_loss(
    flow_preds: List[torch.Tensor],
    img_left: torch.Tensor,
    img_right: torch.Tensor,
    loss_gamma=0.85,
    loss_beta=0.9,
):
    """Loss function defined over sequence of flow predictions"""
    flow_loss = torch.Tensor([0.0]).to(img_left.device)
    preds_cnt = len(flow_preds)
    img_left = img_left / 255.0
    img_right = img_right / 255.0
    # Apply ReLU to ensure disparity is non-negative
    for i, flow_pred in enumerate(flow_preds):
        warp_right = warp(img_left, flow_pred)
        mask = warp(torch.ones_like(img_left).to(img_left.device), flow_pred, "zeros")
        ssim_loss = SSIM()(warp_right, img_right)
        l1_loss = torch.abs(warp_right - img_right)
        loss = (ssim_loss * 0.85 + 0.15 * l1_loss.mean(1, True))[mask > 0]

        flow_loss += loss.mean() * (loss_beta ** (len(flow_preds) - i - 1))

        # reproject = reproject_disparity(flow_pred, img_left)
        # # Compute the main loss

        # ssim_loss = 1 - ssim(reproject, img_right, channel=img_right.shape[1]).mean()
        # # l1_loss = F.l1_loss(reproject, img_right)

        # flow_loss += ssim_loss * (loss_beta ** (preds_cnt - i - 1))

    return flow_loss, {
        "ssim_loss": ssim_loss[mask > 0].mean().item(),
        "l1_loss": l1_loss[mask > 0].mean().item(),
    }


def disparity_smoothness(disp, img):
    def gradient(pred):
        D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    def get_disparity_smoothness(disp, img):
        disp_gradients_x, disp_gradients_y = gradient(disp)
        image_gradients_x, image_gradients_y = gradient(img)

        # Ensure dimensions are consistent
        min_height = min(disp_gradients_x.shape[2], image_gradients_y.shape[2])
        min_width = min(disp_gradients_x.shape[3], image_gradients_y.shape[3])

        disp_gradients_x = disp_gradients_x[:, :, :min_height, :min_width]
        disp_gradients_y = disp_gradients_y[:, :, :min_height, :min_width]
        image_gradients_x = image_gradients_x[:, :, :min_height, :min_width]
        image_gradients_y = image_gradients_y[:, :, :min_height, :min_width]

        # Add epsilon to avoid zero values causing problems
        epsilon = 1e-6

        # Avoid overflow in exp by clamping the gradients
        weights_x = torch.exp(
            -torch.mean(torch.abs(image_gradients_x), 1, keepdim=True).clamp(
                min=-50, max=50
            )
        )
        weights_y = torch.exp(
            -torch.mean(torch.abs(image_gradients_y), 1, keepdim=True).clamp(
                min=-50, max=50
            )
        )

        # Calculate smoothness terms and ensure non-NaN/Inf results
        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        # Ensure no NaN or Inf in smoothness
        smoothness_x = torch.where(
            torch.isnan(smoothness_x), torch.zeros_like(smoothness_x), smoothness_x
        )
        smoothness_y = torch.where(
            torch.isnan(smoothness_y), torch.zeros_like(smoothness_y), smoothness_y
        )
        smoothness_x = torch.where(
            torch.isinf(smoothness_x), torch.zeros_like(smoothness_x), smoothness_x
        )
        smoothness_y = torch.where(
            torch.isinf(smoothness_y), torch.zeros_like(smoothness_y), smoothness_y
        )

        return torch.abs(smoothness_x) + torch.abs(smoothness_y)

    # Initialize disparity smoothness
    disp_smoothness = torch.Tensor([0.0]).to(disp[-1].device)
    weight = 0.8

    # Loop over disparity scales and calculate smoothness
    dis_len = len(disp)
    for idx, scaled_disp in enumerate(disp):
        disp_smoothness += (weight ** (dis_len - idx - 1)) * get_disparity_smoothness(
            scaled_disp, img
        ).mean()
        # Reduce weight for each scale

    return disp_smoothness


def self_supervised_loss(input, flow):
    image_viz_left, image_viz_right, image_nir_left, image_nir_right = input

    image_left = torch.concat([image_viz_left, image_nir_left], dim=1)
    image_right = torch.concat([image_viz_right, image_nir_right], dim=1)

    loss, metric = warp_reproject_loss(flow, image_left, image_right)

    disparity_smooth = disparity_smoothness(flow, image_left)

    loss += disparity_smooth
    metric["disp_smooth"] = disparity_smooth
    return loss.mean(), metric


def self_fm_loss(model: RAFTStereoFusion, input, flow):
    model.eval()
    image_viz_left, image_viz_right, image_nir_left, image_nir_right = input

    reproject_right = reproject_disparity(flow[-1], image_viz_left)
    reproject_nir_right = reproject_disparity(flow[-1], image_nir_left)

    fmap1, fmap2, _ = model.extract_feature_map(
        [image_viz_left, image_viz_right, image_nir_left, image_nir_right]
    )
    fmap1_reproject, fmap2_reproject, _ = model.extract_feature_map(
        [image_viz_left, reproject_right, image_nir_left, reproject_nir_right]
    )

    fmap2_loss = F.mse_loss(fmap2, fmap2_reproject)

    loss = fmap2_loss
    metric = {"fmap2_loss": fmap2_loss.item()}
    model.train()
    return loss, metric


def gt_loss(model, flow_gt, flow_preds, loss_gamma=0.9, max_flow=700):
    """Loss function defined over sequence of flow predictions"""

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    _, _, h, w = flow_preds[-1].shape
    flow_gt = -flow_gt[:, :, :h, :w]
    for i in range(n_predictions):
        assert (
            not torch.isnan(flow_preds[i]).any()
            and not torch.isinf(flow_preds[i]).any()
        )
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        if n_predictions > 1:
            i_weight = loss_gamma ** (n_predictions - i - 1)
        else:
            i_weight = 1.0
        # print(flow_preds[i].shape, flow_gt.shape)
        # i_loss = (flow_preds[i] - flow_gt[:, :, :h, :w]).abs()
        flow_gt_cropped = flow_gt[:, :, :h, :w]
        flow_pred_cropped = flow_preds[i][:, :, :h, :w]

        # 마스크 생성: 모든 채널이 0보다 큰 픽셀
        mask = flow_gt_cropped <= 0

        # 마스크된 L1 손실 계산
        i_loss = (
            F.l1_loss(flow_pred_cropped[mask], flow_gt_cropped[mask], reduction="mean")
            if mask.any()
            else torch.tensor(0.0, device=flow_gt.device)
        )

        flow_loss += i_weight * i_loss.mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics
