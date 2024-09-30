import numpy as np
import torch


def cv2toTensor(image: np.ndarray, batch_dim: bool = True):
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    image = image.transpose((2, 0, 1))
    image_tensor = torch.from_numpy(image).float()
    if batch_dim:
        image_tensor = image_tensor.unsqueeze(0)
    return image_tensor
