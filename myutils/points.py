import numpy as np
import torch
import cv2
from myutils.matrix import rmse_loss, mae_loss
import torch.nn.functional as F


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


def combine_disparity_by_lidar(
    lidar_points: np.ndarray,
    disparity_rgb: np.ndarray,
    disparity_nir: np.ndarray,
    block_width=24,
):
    # Get image dimensions

    height, width = disparity_rgb.shape
    u, v, z = lidar_points.T
    u = u.astype(np.int32)
    v = v.astype(np.int32)

    num_blocks = (width + block_width - 1) // block_width  # Ceiling division

    # Initialize combined disparity map
    combined_disparity = np.zeros_like(disparity_rgb)

    for block_idx in range(num_blocks):
        # Define the horizontal range for the current block
        start_u = block_idx * block_width
        end_u = min((block_idx + 1) * block_width, width)

        # Identify LiDAR points within the current block
        in_block = (u >= start_u) & (u < end_u)

        if not np.any(in_block):
            # If no points in this block, default to disparity_rgb
            combined_disparity[:, start_u:end_u] = disparity_rgb[:, start_u:end_u]
            continue

        # Get the indices of points in the current block
        block_u = u[in_block]
        block_v = v[in_block]
        block_z = z[in_block]

        # Ensure u and v are within image bounds
        valid = (block_u >= 0) & (block_u < width) & (block_v >= 0) & (block_v < height)
        block_u = block_u[valid]
        block_v = block_v[valid]
        block_z = block_z[valid]

        if len(block_z) == 0:
            # No valid points after filtering
            combined_disparity[:, start_u:end_u] = disparity_rgb[:, start_u:end_u]
            continue

        # Sample disparity values from both maps
        sampled_rgb = disparity_rgb[block_v, block_u]
        sampled_nir = disparity_nir[block_v, block_u]

        # Compute RMSE loss for both disparity maps
        rgb_loss = rmse_loss(block_z, sampled_rgb)
        nir_loss = rmse_loss(block_z, sampled_nir)

        # Choose the disparity map with lower loss for this block
        if rgb_loss < nir_loss:
            chosen_disparity = disparity_rgb[:, start_u:end_u]
        else:
            chosen_disparity = disparity_nir[:, start_u:end_u]

        # Assign the chosen disparity to the combined map
        combined_disparity[:, start_u:end_u] = chosen_disparity
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
