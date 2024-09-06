import cv2
import numpy as np
from pfmread import read
from typing import Union, Optional


class NoiseGenerator:

    def add_read_noise(self, image, mean=0, std_dev=10):
        noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
        noisy_image = cv2.add(image.astype(np.float32), noise)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def add_shot_noise(self, image):
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy_image = np.random.poisson(image * vals) / float(vals)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def add_thermal_noise(self, image, mean=0, std_dev=20):
        noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
        noisy_image = cv2.add(image.astype(np.float32), noise)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def add_quantization_noise(self, image, bits=4):
        image = image >> (8 - bits)
        noisy_image = image << (8 - bits)
        return noisy_image

    def add_fixed_pattern_noise(self, image, intensity=10):
        pattern = np.random.normal(0, intensity, image.shape).astype(np.float32)
        noisy_image = cv2.add(image.astype(np.float32), pattern)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def add_dark_current_noise(self, image, mean=0, std_dev=5):
        noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
        noisy_image = cv2.add(image.astype(np.float32), noise)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def darker_image(self, img, alpha=1.0, beta=0, gamma=0.3):
        darkened_image = img * gamma
        darkened_image[0] = darkened_image[0] + 30
        alpha = 1.0  # Contrast control (1.0-3.0)
        beta = 0  # Brightness control (0-100)
        night_image = cv2.convertScaleAbs(darkened_image, alpha=alpha, beta=beta)
        return night_image

    def all_noise(
        self, img: np.ndarray, read_noise_dev=10, thermal_noise_dev=20, quanta_bits=6
    ):
        img = self.add_read_noise(img, std_dev=read_noise_dev)
        img = self.add_shot_noise(img)
        img = self.add_thermal_noise(img, std_dev=thermal_noise_dev)
        img = self.add_dark_current_noise(img)
        img = self.add_quantization_noise(img, quanta_bits)
        img = self.add_fixed_pattern_noise(img)
        return img

    def larger_intensity(self, img, min_intensity=128, max_intensity=255):
        img = img.astype(np.float32)
        img[img >= max_intensity] = max_intensity
        img = img - min_intensity
        img[img < 0] = 0
        img = img / (max_intensity - min_intensity) * 255
        return img.astype(np.uint8)

    def burnt_effect(self, image, threshold=200, intensity=50):
        image = image.astype(np.float32)

        # 각 픽셀에서 3채널 중 가장 높은 값을 찾음
        peak_brightness = np.max(image, axis=2)

        # 임계값 이상인 경우 마스크 생성
        mask = peak_brightness > threshold

        # 밝기 조정 값 계산
        adjustment = (
            (peak_brightness[mask] - threshold) / (255 - threshold)
        ) * intensity

        # 각 채널에 동일하게 조정값을 더함
        for i in range(3):
            image[mask, i] += adjustment

        # 이미지 범위를 0~255로 클리핑
        image = np.clip(image, 0, 255)

        return image.astype(np.uint8)

    def darken_shadows(self, image, shadow_threshold=100, shadow_intensity=50):
        image = image.astype(np.float32)

        # 어두운 부분 어둡게
        mask = image < shadow_threshold
        image[mask] -= shadow_intensity
        image = np.clip(image, 0, 255)

        return image.astype(np.uint8)

    def compute_disparity_gt_error(
        self,
        disparity_gt: Union[str, np.ndarray],
        disparity_est: Union[str, np.ndarray],
        apply_colormap: bool = True,
    ):
        if type(disparity_gt) is str:
            disparity_gt = read(disparity_gt)
        if type(disparity_est) is str:
            disparity_est = cv2.imread(
                disparity_est, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE
            )
        if (
            type(disparity_est) is not np.ndarray
            or type(disparity_gt) is not np.ndarray
        ):
            raise ValueError(
                "disparity_gt and disparity_est must be either a path or a numpy array"
            )
        diffs = np.abs(disparity_gt - disparity_est)
        diffs = np.log(diffs + 1)
        diffs = 255 - (np.clip(diffs, 0, 4) / 4 * 255).astype(np.uint8)
        if apply_colormap:
            return cv2.applyColorMap(diffs, cv2.COLORMAP_JET)
        return diffs

    def filter_image_burn(self, image):
        # image = self.all_noise(
        #     image, read_noise_dev=5, thermal_noise_dev=5, quanta_bits=8
        # )
        image = self.burnt_effect(image, threshold=64, intensity=700)
        image = self.darken_shadows(image, shadow_threshold=64, shadow_intensity=500)
        return image

    def filter_image_burn_light(self, image):
        image = self.all_noise(
            image, read_noise_dev=5, thermal_noise_dev=5, quanta_bits=6
        )
        image = self.burnt_effect(image, threshold=64, intensity=300)
        image = self.darken_shadows(image, shadow_threshold=64, shadow_intensity=400)
        return image

    def filter_image_dark(self, image):
        image = self.all_noise(
            image, read_noise_dev=10, thermal_noise_dev=10, quanta_bits=4
        )
        image = self.darker_image(image, alpha=1.0, beta=0, gamma=0.1)
        return image

    def filter_image_dark_high_gain(self, image):
        image = self.all_noise(
            image, read_noise_dev=20, thermal_noise_dev=30, quanta_bits=4
        )
        image = self.darker_image(image, alpha=1.0, beta=0, gamma=0.3)
        return image
