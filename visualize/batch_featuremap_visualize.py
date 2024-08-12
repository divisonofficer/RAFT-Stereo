import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def featuremap_visualize(fmap, image):
    image = image.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    fmap = fmap.cpu().numpy().transpose(1, 2, 0)

    fmap -= fmap.min()
    fmap = fmap / fmap.max()
    fmap = np.mean(fmap, axis=2)

    fmap = fmap - fmap.min()
    fmap = fmap / fmap.max()
    wmap = fmap.copy().reshape(fmap.shape[0], fmap.shape[1], 1)
    fmap = (fmap * 255).astype(np.uint8)

    fmap = (cv2.applyColorMap(-fmap, cv2.COLORMAP_JET) * wmap).astype(np.uint8)

    fmap = cv2.resize(fmap, (image.shape[1], image.shape[0]))
    fmap = cv2.addWeighted(image, 0.7, fmap, 0.5, 0)

    return fmap


def batch_featuremap_visualize(input_batch, featuremap_batch):
    (left, right), (left_rgb, right_rgb), (left_nir, right_nir) = featuremap_batch
    (image0, image1, image2, image3) = input_batch[:4]

    batch_size = left.shape[0]
    fig, axs = plt.subplots(
        batch_size, 1, figsize=(20, 20 * batch_size)
    )  # 세로로 subplot을 배치
    if batch_size == 1:
        axs = [axs]

    custom_cmap = plt.get_cmap("jet", 256)
    mapped_colors = custom_cmap(np.linspace(0, 1, 256))[:, :3] * 255
    custom_cmap_array = np.array(mapped_colors, dtype=np.uint8)
    for i in range(batch_size):
        (
            left_img,
            right_img,
            left_rgb_img,
            right_rgb_img,
            left_nir_img,
            right_nir_img,
        ) = [
            featuremap_visualize(x[0][i], x[1][i])
            for x in [
                (left, image0),
                (right, image1),
                (left_rgb, image0),
                (right_rgb, image1),
                (left_nir, image2),
                (right_nir, image3),
            ]
        ]
        rgb = np.concatenate([left_rgb_img, right_rgb_img], axis=1)
        nir = np.concatenate([left_nir_img, right_nir_img], axis=1)
        fusion = np.concatenate([left_img, right_img], axis=1)
        im = axs[i].imshow(np.concatenate([rgb, nir, fusion], axis=0))
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap="jet"),
            ax=axs,
            orientation="vertical",
            fraction=0.02,
            pad=0.04,
        )

    plt.show()
