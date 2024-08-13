import torch
import torch.nn.functional as F
import numpy as np
from core.raft_stereo_fusion import RAFTStereoFusion


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
    flow_preds: torch.Tensor,
    img_left: torch.Tensor,
    img_right: torch.Tensor,
    loss_gamma=0.85,
    loss_beta=0.9,
):
    """Loss function defined over sequence of flow predictions"""
    flow_loss = 0.0
    preds_cnt = len(flow_preds)
    img_left = img_left / 255.0
    img_right = img_right / 255.0
    # Apply ReLU to ensure disparity is non-negative
    for i, flow_pred in enumerate(flow_preds):
        reproject = reproject_disparity(flow_pred, img_left)
        # Compute the main loss
        ssim_loss = 1 - ssim(reproject, img_right, channel=img_right.shape[1])
        l1_loss = F.l1_loss(reproject, img_right)
        flow_loss += ((1 - loss_gamma) * ssim_loss + loss_gamma * l1_loss) * (
            loss_beta ** (preds_cnt - i - 1)
        )

    return flow_loss, {
        "ssim_loss": ssim_loss.mean().item(),
        "l1_loss": l1_loss.mean().item(),
    }


def disparity_smoothness(disp, img):
    def gradient(pred):
        D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    def get_disparity_smoothness(disp, img):
        disp_gradients_x, disp_gradients_y = gradient(disp)
        image_gradients_x, image_gradients_y = gradient(img)
        min_height = min(disp_gradients_x.shape[2], image_gradients_y.shape[2])
        min_width = min(disp_gradients_x.shape[3], image_gradients_y.shape[3])

        disp_gradients_x = disp_gradients_x[:, :, :min_height, :min_width]
        disp_gradients_y = disp_gradients_y[:, :, :min_height, :min_width]
        image_gradients_x = image_gradients_x[:, :, :min_height, :min_width]
        image_gradients_y = image_gradients_y[:, :, :min_height, :min_width]
        weights_x = torch.exp(
            -torch.mean(torch.abs(image_gradients_x), 1, keepdim=True)
        )
        weights_y = torch.exp(
            -torch.mean(torch.abs(image_gradients_y), 1, keepdim=True)
        )
        # Adjust the gradients to have the same shape

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        return torch.abs(smoothness_x) + torch.abs(smoothness_y)

    disp_smoothness = 0
    weight = 1.0
    for scaled_disp in disp:
        disp_smoothness += weight * get_disparity_smoothness(scaled_disp, img)
        weight /= 2.3
    return disp_smoothness.mean()


def self_supervised_loss(model: RAFTStereoFusion, input, flow):
    image_viz_left, image_viz_right, image_nir_left, image_nir_right = input

    loss, metric = warp_reproject_loss(flow, image_viz_left, image_viz_right)
    loss2, metric_nir = warp_reproject_loss(flow, image_nir_left, image_nir_right)

    for k, v in metric_nir.items():
        metric[f"{k}_nir"] = v
    loss += loss2
    viz_smooth = disparity_smoothness(flow, image_viz_left)
    nir_smooth = disparity_smoothness(flow, image_nir_left)
    loss += viz_smooth + nir_smooth
    metric["viz_smooth"] = viz_smooth.item()
    metric["nir_smooth"] = nir_smooth.item()

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


def warp(im, disp, grid_size=2):
    theta = torch.Tensor(np.array([[1, 0, 0], [0, 1, 0]])).cuda()
    theta = theta.expand((disp.size()[0], 2, 3)).contiguous()
    grid = F.affine_grid(theta, disp.size())
    disp = disp.transpose(1, 2).transpose(2, 3)
    disp = torch.cat((disp, torch.zeros(disp.size()).cuda()), 3)

    grid = grid + 2 * disp
    sampled = F.grid_sample(im, grid)
    return sampled


def gt_loss(model, flow_gt, flow_preds, loss_gamma=0.9, max_flow=700):
    """Loss function defined over sequence of flow predictions"""

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0
    flow_gt = -flow_gt[0]

    for i in range(n_predictions):
        assert (
            not torch.isnan(flow_preds[i]).any()
            and not torch.isinf(flow_preds[i]).any()
        )
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
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
