import numpy as np
import torch
import cv2
from myutils.matrix import rmse_loss, mae_loss


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
