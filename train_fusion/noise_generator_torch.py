import torch
import torch.nn.functional as F
import cv2
from pfmread import read
from typing import Union

class NoiseGenerator:

    def add_read_noise(self, image, mean=0, std_dev=10):
        noise = torch.normal(mean, std_dev, size=image.shape).to(image.device)
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 255).to(torch.uint8)

    def add_shot_noise(self, image):
        vals = len(torch.unique(image))
        vals = 2 ** torch.ceil(torch.log2(torch.tensor(vals, dtype=torch.float32)))
        noisy_image = torch.poisson(image * vals) / float(vals)
        return torch.clamp(noisy_image, 0, 255).to(torch.uint8)

    def add_thermal_noise(self, image, mean=0, std_dev=20):
        noise = torch.normal(mean, std_dev, size=image.shape).to(image.device)
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 255).to(torch.uint8)

    def add_quantization_noise(self, image, bits=4):
        image = image >> (8 - bits)
        noisy_image = image << (8 - bits)
        return noisy_image

    def add_fixed_pattern_noise(self, image, intensity=10):
        pattern = torch.normal(0, intensity, size=image.shape).to(image.device)
        noisy_image = image + pattern
        return torch.clamp(noisy_image, 0, 255).to(torch.uint8)

    def add_dark_current_noise(self, image, mean=0, std_dev=5):
        noise = torch.normal(mean, std_dev, size=image.shape).to(image.device)
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 255).to(torch.uint8)

    def darker_image(self, img, alpha=1.0, beta=0, gamma=0.3):
        darkened_image = img * gamma
        darkened_image[0] = darkened_image[0] + 30
        darkened_image = torch.clamp(darkened_image * alpha + beta, 0, 255)
        return darkened_image.to(torch.uint8)

    def all_noise(
        self, img: torch.Tensor, read_noise_dev=10, thermal_noise_dev=20, quanta_bits=6
    ):
        img = self.add_read_noise(img, std_dev=read_noise_dev)
        img = self.add_shot_noise(img)
        img = self.add_thermal_noise(img, std_dev=thermal_noise_dev)
        img = self.add_dark_current_noise(img)
        img = self.add_quantization_noise(img, quanta_bits)
        img = self.add_fixed_pattern_noise(img)
        return img

    def larger_intensity(self, img, min_intensity=128, max_intensity=255):
        img = img.to(torch.float32)
        img[img >= max_intensity] = max_intensity
        img = img - min_intensity
        img[img < 0] = 0
        img = img / (max_intensity - min_intensity) * 255
        return img.to(torch.uint8)

    def burnt_effect(self, image, threshold=200, intensity=50):
        image = image.to(torch.float32)

        # 각 픽셀에서 3채널 중 가장 높은 값을 찾음
        peak_brightness = torch.max(image, dim=2)[0]

        # 임계값 이상인 경우 마스크 생성
        mask = peak_brightness > threshold

        # 밝기 조정 값 계산
        adjustment = ((peak_brightness - threshold) / (255 - threshold)) * intensity

        # mask를 image의 3채널에 맞게 확장
        mask = mask.unsqueeze(-1).expand_as(image)
        
        # adjustment를 image와 동일한 shape로 확장
        adjustment = adjustment.unsqueeze(-1).expand_as(image)

        # 각 채널에 동일하게 조정값을 더함
        image = torch.where(mask, image + adjustment, image)

        # 이미지 범위를 0~255로 클리핑
        image = torch.clamp(image, 0, 255)
        return image.to(torch.uint8)

    def darken_shadows(self, image, shadow_threshold=100, shadow_intensity=50):
        image = image.to(torch.float32)

        # 어두운 부분 어둡게
        mask = image < shadow_threshold
        image[mask] -= shadow_intensity
        image = torch.clamp(image, 0, 255)

        return image.to(torch.uint8)

    def compute_disparity_gt_error(
        self,
        disparity_gt: Union[str, torch.Tensor],
        disparity_est: Union[str, torch.Tensor],
        apply_colormap: bool = True,
    ):
        if type(disparity_gt) is str:
            disparity_gt = read(disparity_gt)
            disparity_gt = torch.tensor(disparity_gt)

        if type(disparity_est) is str:
            disparity_est = cv2.imread(
                disparity_est, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE
            )
            disparity_est = torch.tensor(disparity_est)

        if (
            type(disparity_est) is not torch.Tensor
            or type(disparity_gt) is not torch.Tensor
        ):
            raise ValueError(
                "disparity_gt and disparity_est must be either a path or a torch.Tensor"
            )
        diffs = torch.abs(disparity_gt - disparity_est)
        diffs = torch.log(diffs + 1)
        diffs = 255 - torch.clamp(diffs, 0, 4) / 4 * 255
        diffs = diffs.to(torch.uint8)
        if apply_colormap:
            diffs = diffs.numpy()  # OpenCV expects a NumPy array
            return cv2.applyColorMap(diffs, cv2.COLORMAP_JET)
        return diffs

    def filter_image_burn(self, image):
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