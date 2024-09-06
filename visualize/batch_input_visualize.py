import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def batch_input_visualize(input_batch, output_batch=None, disparity_max=64):
    norm = Normalize(vmin=0, vmax=disparity_max)
    image0, image1, image2, image3 = [x.cuda() for x in input_batch[1:5]]

    if len(input_batch) > 5:
        dis_batch_1, dis_batch_2 = [x.cuda() for x in input_batch[5:7]]
    else:
        dis_batch_1, dis_batch_2 = None, None

    batch_size = image0.size(0)
    fig, axs = plt.subplots(
        batch_size, 1, figsize=(20, 8 * batch_size)
    )  # 세로로 subplot을 배치
    if batch_size == 1:
        axs = [axs]
    for i in range(batch_size):
        image_left = image0[i].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        image_right = image1[i].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

        image_nir_left = image2[i].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

        image_nir_right = image3[i].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

        if image_nir_right.shape[2] == 1:
            image_nir_right = cv2.cvtColor(image_nir_right, cv2.COLOR_GRAY2BGR)
            image_nir_left = cv2.cvtColor(image_nir_left, cv2.COLOR_GRAY2BGR)

        rgb_plot = np.concatenate([image_left, image_right], axis=1)
        nir_plot = np.concatenate([image_nir_left, image_nir_right], axis=1)

        if dis_batch_1 is not None and dis_batch_2 is not None:
            dis1 = dis_batch_1[i].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            dis2 = dis_batch_2[i].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            dis1 = cv2.applyColorMap(
                np.clip(dis1 / disparity_max * 255, 0, 255).astype(np.uint8),
                cv2.COLORMAP_MAGMA,
            )
            dis2 = cv2.applyColorMap(
                np.clip(dis2 / disparity_max * 255, 0, 255).astype(np.uint8),
                cv2.COLORMAP_MAGMA,
            )
            rgb_plot = np.concatenate([rgb_plot, dis1], axis=1)
            nir_plot = np.concatenate([nir_plot, dis2], axis=1)

        disparity = None
        if output_batch is not None:
            disparity = -(output_batch.cpu().numpy())[i][0]

            print(disparity.min(), disparity.max(), np.median(disparity))

            disparity = np.clip(disparity, 0, disparity_max)

            disparity_color = cv2.applyColorMap(
                np.clip(disparity / disparity_max * 255, 0, 255).astype(np.uint8),
                cv2.COLORMAP_MAGMA,
            )

            # image_right_re = (
            #     reprojected_right[0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            # )
            # image_nir_right_re = (
            #     reprojected_right_nir[0]
            #     .cpu()
            #     .numpy()
            #     .transpose(1, 2, 0)
            #     .astype(np.uint8)
            # )

        plot_input = np.concatenate([rgb_plot, nir_plot], axis=0)
        if disparity is not None:
            plot_input = np.concatenate(
                [
                    plot_input,
                    cv2.resize(
                        disparity_color,
                        (image_left.shape[1] * 2, image_left.shape[0] * 2),
                    ),
                ],
                axis=1,
            )
        # image_disparity_single = inputs[0][0][0].replace("left","disparity_color")
        im = axs[i].imshow(
            plot_input, cmap="magma", norm=norm
        )  # 각 subplot에 이미지 출력
        # Adding the colorbar for each subplot
        cbar = fig.colorbar(
            im, ax=axs[i], orientation="vertical", fraction=0.02, pad=0.04
        )

    plt.tight_layout()
    plt.show()
