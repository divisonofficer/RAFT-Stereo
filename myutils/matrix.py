import numpy as np
import torch


def rmse_loss(src: np.ndarray, tar: np.ndarray):
    """
    Compute the Root Mean Squared Error (RMSE) between two images
    Args:
        src (np.ndarray): Source image
        tar (np.ndarray): Target image
    Returns:
        float: RMSE loss
    """
    return np.sqrt(np.mean((src - tar) ** 2))


def mae_loss(src: np.ndarray, tar: np.ndarray):
    """
    Compute the Mean Absolute Error (MAE) between two images
    Args:
        src (np.ndarray): Source image
        tar (np.ndarray): Target image
    Returns:
        float: MAE loss
    """
    return np.mean(np.abs(src - tar))
