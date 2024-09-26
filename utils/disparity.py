import numpy as np
import cv2


def disparity_to_depth(disparity: np.ndarray, focal_length: float, baseline: float):
    disparity = disparity.astype(np.float32)
    depth = focal_length * baseline / disparity
    depth[depth < 0] = 0
    depth[np.isinf(depth)] = 0
    depth[np.isnan(depth)] = 0
    return depth


def disparity_color(
    disparity: np.ndarray, max_disparity=32, colormap=cv2.COLORMAP_MAGMA
):

    disparity = disparity / max_disparity
    disparity = np.clip(disparity, 0, 1)
    disparity = (disparity * 255).astype(np.uint8)
    disparity = cv2.applyColorMap(disparity, colormap)
    return disparity
