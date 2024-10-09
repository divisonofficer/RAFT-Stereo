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


def disparity_image_edge(disparity: np.ndarray, image: np.ndarray):
    intensity = int(np.median(image))
    if intensity > 100:
        intensity = 100
    edge_img = cv2.Canny(image, intensity, intensity * 2)
    disp_gradient = pixel_graident(disparity)
    edge_disp = np.zeros_like(edge_img)
    edge_disp[disp_gradient > 2] = 255
    return edge_img.astype(np.float32) / 255, edge_disp.astype(np.float32) / 255


def disparity_image_edge_eval(disparity: np.ndarray, image: np.ndarray):
    edge_img, edge_disp = disparity_image_edge(disparity, image)
    return rmse_loss(edge_img, edge_disp)


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


def guided_filter(I: np.ndarray, p: np.ndarray, radius=15, eps=1e-6):
    # I: guide image (grayscale)
    # p: input image to be filtered (color)
    # radius: window radius
    # eps: regularization parameter

    # Step 1: Mean of I, p, I*p, and I^2

    mean_I = cv2.boxFilter(I, cv2.CV_64F, (radius, radius))
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (radius, radius))

    # Initialize output
    q = np.zeros_like(p)

    # Process each channel separately
    for c in range(p.shape[2]):
        mean_p = cv2.boxFilter(p[:, :, c], cv2.CV_64F, (radius, radius))
        mean_Ip = cv2.boxFilter(I * p[:, :, c], cv2.CV_64F, (radius, radius))

        # Step 2: Covariance of (I, p) and variance of I
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = mean_II - mean_I * mean_I

        # Step 3: Calculate a and b
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        # Step 4: Mean of a and b
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))

        # Step 5: Output q for channel c
        q[:, :, c] = np.clip(mean_a * I + mean_b, 0, 255)

    return q


def gamma_correction(img, gamma=0.5):
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(
        "uint8"
    )
    if img.ndim == 3 and img.shape[2] == 3:  # RGB 이미지인 경우
        channels = cv2.split(img)
        corrected_channels = [cv2.LUT(channel, look_up_table) for channel in channels]
        return cv2.merge(corrected_channels)
    else:  # 단일 채널 이미지인 경우
        return cv2.LUT(img, look_up_table)
