import numpy as np
import cv2
import matplotlib.pyplot as plt


def batch_input_visualize(input_batch, output_batch=None, disparity_max=64):

    image0, image1, image2, image3, dis_batch_1, dis_batch_2 = [
        x.cuda() for x in input_batch[1:]
    ]
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

        if image_nir_right.shape[2] == 1:
            image_nir_right = cv2.cvtColor(image_nir_right, cv2.COLOR_GRAY2BGR)
            image_nir_left = cv2.cvtColor(image_nir_left, cv2.COLOR_GRAY2BGR)

        rgb_plot = np.concatenate([image_left, image_right, dis1], axis=1)
        nir_plot = np.concatenate([image_nir_left, image_nir_right, dis2], axis=1)
        disparity = None
        if output_batch is not None:
            disparity = -(output_batch.cpu().numpy())[i][0]

            print(disparity.min(), disparity.max(), np.median(disparity))

            disparity_max = 64

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

        axs[i].imshow(plot_input)  # 각 subplot에 이미지 출력
        axs[i].axis("off")  # 축 제거

    plt.tight_layout()
    plt.show()
