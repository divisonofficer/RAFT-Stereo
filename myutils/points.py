from typing import Callable, Optional, Tuple
import numpy as np
import torch
import cv2
from myutils.image_process import disparity_image_edge_eval
from myutils.matrix import rmse_loss, mae_loss
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def transfrom_points(points: np.ndarray, transform_mtx: np.ndarray):
    """
    Transform points using a 4x4 transformation matrix
    Args:
        points (np.ndarray): 3D points to transform
        transform_mtx (np.ndarray): 4x4 transformation matrix
    Returns:
        np.ndarray: Transformed points
    """
    points = points.reshape(-1, 3)
    points = points[(points[:, 0] != 0) | (points[:, 1] != 0)]
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = transform_mtx @ points.T
    return points[:3].T


def transform_point_inverse(points: np.ndarray, transform_mtx: np.ndarray):
    """
    Transform points using a 4x4 transformation matrix
    Args:
        points (np.ndarray): 3D points to transform
        transform_mtx (np.ndarray): 4x4 transformation matrix
    Returns
        np.ndarray: Transformed points
    """
    transform_mtx = np.linalg.pinv(transform_mtx)
    return transfrom_points(points, transform_mtx)


def project_points_on_camera(
    points: np.ndarray,
    focal_length: float,
    cx: float,
    cy: float,
    image_width: float = 0,
    image_height: float = 0,
):
    """
    Project 3D points to 2D image plane
    Args:
        points (np.ndarray): 3D points to project
        focal_length (float): Focal length of the camera
        cx (float): Principal point x-coordinate
        cy (float): Principal point y-coordinate
        image_width (float): Image width, Optional
        image_height (float): Image height, Optional
    Returns:
        np.ndarray: Projected points
    """
    points[:, 0] = points[:, 0] * focal_length / points[:, 2] + cx
    points[:, 1] = points[:, 1] * focal_length / points[:, 2] + cy

    if image_width > 0 and image_height > 0:
        points = points[
            (points[:, 0] >= 0)
            & (points[:, 0] <= image_width - 1)
            & (points[:, 1] >= 0)
            & (points[:, 1] <= image_height - 1)
            & (points[:, 2] > 0)
        ]
    return points


def render_depth_map(
    points: np.ndarray, width: int = 0, height: int = 0, max_depth=10000
):
    if width == 0:
        width = int(points[:, 0].max()) + 1
    if height == 0:
        height = int(points[:, 1].max()) + 1
    canvas = np.zeros((height, width), dtype=np.uint8)

    for u, v, depth in points:
        depth = depth / max_depth * 255
        depth = np.clip(depth, 0, 255).asdtype(np.uint8)
        canvas[int(v), int(u)] = depth
    return canvas


def points_sampled_disparity(points: np.ndarray, disparity_map: np.ndarray):
    points = points[
        points[:, 1]
        < disparity_map.shape[0] & points[:, 0]
        < disparity_map.shape[1] & points[:, 1]
        >= 0 & points[:, 0]
        >= 0
    ]
    u, v, d = points.T
    d = disparity_map[v.astype(int), u.astype(int)]
    points[:, 2] = d
    return points


def points_sampled_disparity_loss(
    points: np.ndarray,
    disparity_map: np.ndarray,
    focal_length: float,
    baseline: float,
):
    points = points[
        points[:, 1]
        < disparity_map.shape[0] & points[:, 0]
        < disparity_map.shape[1] & points[:, 1]
        >= 0 & points[:, 0]
        >= 0
    ]
    points[:, 2] = focal_length * baseline / points[:, 2]
    points_sampled = points_sampled_disparity(points, disparity_map)
    return rmse_loss(points[:, 2], points_sampled[:, 2])


def lidar_points_to_disparity(
    points: np.ndarray,
    transform_mtx: np.ndarray,
    focal_length: float,
    baseline: float,
    cx: float,
    cy: float,
):
    points = transform_point_inverse(points, transform_mtx)
    points = project_points_on_camera(points, focal_length, cx, cy, 720, 540)
    points[:, 2] = focal_length * baseline / points[:, 2] - 1
    return points


def pad_lidar_points(lidar_projected_points, target_size=5000):
    current_size = len(lidar_projected_points)

    if current_size >= target_size:
        return lidar_projected_points[:target_size]

    # 필요한 포인트 수 계산
    needed = target_size - current_size

    # 기존 포인트에서 랜덤하게 샘플링 (복원 추출)
    # 샘플링할 포인트 수가 현재 포인트 수보다 많을 경우, 여러 번 반복할 수 있음
    # NumPy의 random.choice를 사용하여 인덱스를 랜덤하게 선택
    sampled_indices = np.random.choice(current_size, size=needed, replace=True)
    sampled_points = lidar_projected_points[sampled_indices]

    # 기존 포인트와 샘플링된 포인트를 결합
    padded_lidar_projected_points = np.concatenate(
        [lidar_projected_points, sampled_points], axis=0
    )

    return padded_lidar_projected_points


def combine_block(
    lidar_points: np.ndarray,
    disparity_rgb: np.ndarray,
    disparity_nir: np.ndarray,
    combined_disparity: np.ndarray,
    criteria: Callable[
        [np.ndarray, np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]]], bool
    ],
    blk_w=24,
    blk_h=24,
):
    width = disparity_rgb.shape[-1]
    height = disparity_rgb.shape[-2]
    n_blk_u = (width + blk_w - 1) // blk_w  # Ceiling division
    n_blk_v = (height + blk_h - 1) // blk_h  # Ceiling division
    u, v, z = lidar_points.T
    for blk_v_idx in range(n_blk_v):
        for blk_u_idx in range(n_blk_u):
            # Define the vertical block boundaries
            st_v = blk_v_idx * blk_h
            en_v = min((blk_v_idx + 1) * blk_h, height)
            st_u = blk_u_idx * blk_w
            en_u = min((blk_u_idx + 1) * blk_w, width)

            # Identify LiDAR points within the current vertical block
            in_block = (u >= st_u) & (u < en_u) & (v >= st_v) & (v < en_v)

            if not np.any(in_block):
                # No points in this vertical block; retain the horizontal-based disparity
                continue

            # Get the indices of points in the current block
            bu, bv, bz = lidar_points[in_block].T

            # Ensure u and v are within image bounds
            valid = (bu >= 0) & (bu < width) & (bv >= 0) & (bv < height)
            bu, bv, bz = np.stack([bu, bv, bz], axis=1)[valid].T

            critic = criteria(
                bu.astype(np.int32), bv.astype(np.int32), bz, (st_u, en_u, st_v, en_v)
            )

            # Choose the disparity map with lower loss for this block
            if critic:
                chosen_disparity = disparity_rgb[st_v:en_v, st_u:en_u]
            else:
                chosen_disparity = disparity_nir[st_v:en_v, st_u:en_u]

            # Assign the chosen disparity to the combined map
            combined_disparity[st_v:en_v, st_u:en_u] = chosen_disparity

    return combined_disparity


def combine_disparity_by_lidar(
    lidar_points: np.ndarray,
    disparity_rgb: np.ndarray,
    disparity_nir: np.ndarray,
    block_width=24,
    block_height=24,
):
    """
    Combine two disparity maps (RGB and NIR) using LiDAR points by processing both horizontal and vertical blocks.

    Parameters:
    - lidar_points: (N, 3) array of LiDAR points (u, v, z).
    - disparity_rgb: (H, W) disparity map from RGB.
    - disparity_nir: (H, W) disparity map from NIR.
    - block_width: Width of each block for horizontal processing.
    - block_height: Height of each block for vertical processing.

    Returns:
    - combined_disparity: (H, W) combined disparity map.

    """
    # Initialize combined disparity map with horizontal processing
    width = disparity_rgb.shape[-1]
    height = disparity_rgb.shape[-2]
    combined_disparity = np.zeros_like(disparity_rgb)

    def criteria(bu, bv, bz, block_bounds):
        if len(bu) == 0:
            return False
        return rmse_loss(disparity_rgb[bv, bu], bz) < rmse_loss(
            disparity_nir[bv, bu], bz
        )

    combined_disparity = combine_block(
        lidar_points,
        disparity_rgb,
        disparity_nir,
        combined_disparity,
        criteria,
        block_width,
        height,
    )
    combined_disparity = combine_block(
        lidar_points,
        disparity_rgb,
        disparity_nir,
        combined_disparity,
        criteria,
        block_width,
        block_height,
    )

    return combined_disparity


def combine_disparity_by_edge(
    lidar_points: np.ndarray,
    disparity_rgb: np.ndarray,
    disparity_nir: np.ndarray,
    image_rgb: np.ndarray,
    image_nir: np.ndarray,
    block_width=24,
    block_height=24,
):
    """
    Combine two disparity maps (RGB and NIR) using LiDAR points by processing both horizontal and vertical blocks.

    Parameters:
    - lidar_points: (N, 3) array of LiDAR points (u, v, z).
    - disparity_rgb: (H, W) disparity map from RGB.
    - disparity_nir: (H, W) disparity map from NIR.
    - block_width: Width of each block for horizontal processing.
    - block_height: Height of each block for vertical processing.

    Returns:
    - combined_disparity: (H, W) combined disparity map.

    """
    # Initialize combined disparity map with horizontal processing
    width = disparity_rgb.shape[-1]
    height = disparity_rgb.shape[-2]
    combined_disparity = np.zeros_like(disparity_rgb)

    def criteria(bu, bv, bz, block_bounds):
        st_u, en_u, st_v, en_v = block_bounds
        rgb_edge_eval = disparity_image_edge_eval(
            disparity_rgb[st_v:en_v, st_u:en_u], image_rgb[st_v:en_v, st_u:en_u]
        )
        nir_edge_eval = disparity_image_edge_eval(
            disparity_nir[st_v:en_v, st_u:en_u], image_nir[st_v:en_v, st_u:en_u]
        )
        rgb_rmse = rmse_loss(disparity_rgb[bv, bu], bz) if len(bu) > 0 else 1
        nir_rmse = rmse_loss(disparity_nir[bv, bu], bz) if len(bu) > 0 else 1
        min_rmse = min(rgb_rmse, nir_rmse)
        rgb_rmse -= min_rmse
        nir_rmse -= min_rmse

        rgb_loss_norm = rgb_rmse / (rgb_rmse + nir_rmse) + rgb_edge_eval / (
            rgb_edge_eval + nir_edge_eval + 1e-6
        )
        return rgb_loss_norm < 0.5

    combined_disparity = combine_block(
        lidar_points,
        disparity_rgb,
        disparity_nir,
        combined_disparity,
        criteria,
        block_width,
        height,
    )
    combined_disparity = combine_block(
        lidar_points,
        disparity_rgb,
        disparity_nir,
        combined_disparity,
        criteria,
        block_width,
        block_height,
    )

    return combined_disparity


def refine_disparity(
    disparity_map: torch.Tensor,
    image_left: torch.Tensor,
    image_right: torch.Tensor,
    disp_thresh=4.0,
):
    disparity_map = disparity_map.unsqueeze(0)
    image_left = image_left.unsqueeze(0)
    image_right = image_right.unsqueeze(0)
    reprojected_right = reproject_disparity(disparity_map, image_left)
    ssim_loss = ssim_torch(reprojected_right, image_right).mean(dim=1)

    # Create a mask for conditions where disparity_map <= 4.0 and ssim_loss >= 0.98
    ssim_loss = F.pad(ssim_loss, [0, 2, 0, 2])
    mask = (disparity_map <= disp_thresh) & (
        ssim_loss.unsqueeze(1) >= 0.98
    )  # Adjust shape of ssim_loss for broadcasting

    # Set the corresponding pixels in disparity_map to 0
    disparity_map[mask] = 0

    return disparity_map[0]


def refine_disparity_with_monodepth(disparity_map: np.ndarray, mono_depth: np.ndarray):
    mask = (mono_depth <= 1).astype(np.float32)
    mask = gaussian_filter(mask, 9)
    disparity_map = disparity_map * (1 - mask) + mono_depth * mask
    return disparity_map


def ssim_torch(x: torch.Tensor, y: torch.Tensor):
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


def compute_disparity_lidar_error(
    lidar_points: np.ndarray,
    transform_mtx: np.ndarray,
    disparity: np.ndarray,
    focal_length: float,
    baseline: float,
    cx: float,
    cy: float,
    cx2: float = 0,
):
    """
    Compute the RMSE error between LiDAR points and disparity map
    Args:
        lidar_points (np.ndarray): LiDAR points
        transform_mtx (np.ndarray): Transformation matrix
        disparity (np.ndarray): Disparity map
        focal_length (float): Focal length of the camera
        baseline (float): Baseline of the camera
        cx (float): Principal point x-coordinate
        cy (float): Principal point y-coordinate
        cx2 (float): Secondary principal point x-coordinate, Optional
    """
    lidar_points = transform_point_inverse(lidar_points, transform_mtx)
    lidar_projected_points = project_points_on_camera(
        lidar_points, focal_length, cx, cy, 720, 540
    )
    if cx2 == 0:
        cx2 = cx
    lidar_projected_points[:, 2] = (
        focal_length * baseline / lidar_projected_points[:, 2] - cx2 + cx
    )
    u, v, d = lidar_projected_points.T
    u = u.astype(np.int32)
    v = v.astype(np.int32)
    rmse = rmse_loss(disparity, d)
    return rmse
