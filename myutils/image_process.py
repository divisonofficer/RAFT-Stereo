from typing import Tuple
import cv2
import numpy as np
import torch

from myutils.matrix import rmse_loss


def cv2toTensor(image: np.ndarray, batch_dim: bool = True):
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    image = image.transpose((2, 0, 1))
    image_tensor = torch.from_numpy(image).float()
    if batch_dim:
        image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def pixel_graident(img: np.ndarray):
    """
    get normalized pixel gradient of an image
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    return gradient_magnitude


def disparity_image_edge_eval(disparity: np.ndarray, image: np.ndarray):
    intensity = int(np.median(image))
    if intensity > 100:
        intensity = 100
    edge_img = cv2.Canny(image, intensity, intensity * 2)
    disp_gradient = pixel_graident(disparity)
    edge_disp = np.zeros_like(edge_img)
    edge_disp[disp_gradient > 2] = 255
    return rmse_loss(edge_img.astype(np.float32), edge_disp.astype(np.float32))


def read_image_pair(
    frame_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ret = []
    for path in ["rgb/left.png", "rgb/right.png", "nir/left.png", "nir/right.png"]:
        img = cv2.imread(
            f"{frame_path}/{path}",
            cv2.IMREAD_GRAYSCALE if "nir" in path else cv2.IMREAD_COLOR,
        )
        if not "nir" in path:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ret.append(img)
    return tuple(ret)
