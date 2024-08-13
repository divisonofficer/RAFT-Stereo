import os
from typing import Optional, Tuple
import torch
import torch.utils.data as data
import numpy as np
from core.utils.utils import InputPadder
from PIL import Image
import json
import pfmread
from torchvision.transforms.functional import pad
import cv2


DRIVING_JSON = "flyingthings3d.json"
REAL_DATA_JSON = "real_data.json"
FLYING_JSON = "Flow3dFlyingThings3d.json"


class StereoDataset(data.Dataset):
    def __init__(
        self,
        folder: str = "/bean/depth",
        real_data_json=False,
        real_data_validate=False,
        flow3d_driving_json=False,
        flying3d_json=False,
        gt_depth=False,
        copy_of_self=False,
        validate_json=False,
        cut_resolution: Optional[Tuple[int, int]] = None,
    ):
        self.gt_depth = gt_depth
        self.cut_resolution = cut_resolution
        self.flow3d_driving_prejson = flow3d_driving_json
        if copy_of_self:
            return
        self.input_list = []
        if flow3d_driving_json:
            self.input_list += self.flow3d_driving_json(DRIVING_JSON, validate_json)
        if flying3d_json:
            self.input_list += self.flow3d_driving_json(FLYING_JSON, validate_json)
        if real_data_json:
            with open(REAL_DATA_JSON, "r") as file:
                self.input_list = json.load(file)
        if real_data_validate:
            self.input_list = self.extract_input_folder(folder)
            with open(REAL_DATA_JSON, "w") as file:
                json.dump(self.input_list, file)

    def input_resolution(self):
        image = cv2.imread(
            self.input_list[0][0][0]
            if isinstance(self.input_list[0][0], tuple)
            else self.input_list[0][0]
        )
        return image.shape[:2]

    def __to_tensor(self, filename, reduce_luminance=False):
        if filename.endswith(".pfm"):
            img = pfmread.read(filename)
        else:
            img = np.array(Image.open(filename)).astype(np.uint8)
            if reduce_luminance:
                img = self.darker_image(img)
        if self.cut_resolution is not None and img.shape[0] != self.cut_resolution[0]:
            w_f = int(img.shape[1] / 2 - self.cut_resolution[1] / 2)
            h_f = int(img.shape[0] / 2 - self.cut_resolution[0] / 2)
            w_t = int(img.shape[1] / 2 + self.cut_resolution[1] / 2)
            h_t = int(img.shape[0] / 2 + self.cut_resolution[0] / 2)
            img = img[h_f:h_t, w_f:w_t]

        tensor = torch.from_numpy(img.copy())
        if tensor.dim() == 2:
            return tensor.unsqueeze(0).float()
        return tensor.permute(2, 0, 1).float()

    def partial(self, start, end):
        copy_of_self = StereoDataset(
            "",
            copy_of_self=True,
            gt_depth=self.gt_depth,
            flow3d_driving_json=self.flow3d_driving_prejson,
            cut_resolution=self.cut_resolution,
        )
        copy_of_self.input_list = self.input_list[start:end]

        return copy_of_self

    def darker_image(self, img):
        darkened_image = img * 0.05
        darkened_image[0] = darkened_image[0] + 30
        alpha = 1.0  # Contrast control (1.0-3.0)
        beta = 0  # Brightness control (0-100)
        night_image = cv2.convertScaleAbs(darkened_image, alpha=alpha, beta=beta)
        return night_image

    def flow3d_driving_json(self, filename, validate=False):
        with open(filename, "r") as file:
            entries = json.load(file)
        self.entries = []
        validate_entries = []

        for entry in entries:

            nir = (
                entry["rgb"][0].replace("frames_cleanpass", "nir_rendered"),
                entry["rgb"][1].replace("frames_cleanpass", "nir_rendered"),
            )
            nir_ambient = (
                entry["rgb"][0].replace("frames_cleanpass", "nir_ambient"),
                entry["rgb"][1].replace("frames_cleanpass", "nir_ambient"),
            )
            if validate:
                if not os.path.exists(nir[0]) or not os.path.exists(nir[1]):
                    continue
                validate_entries.append(entry)

            self.entries.append((entry["rgb"], nir, entry["disparity"], False))
            self.entries.append((entry["rgb"], nir_ambient, entry["disparity"], False))
            self.entries.append((entry["rgb"], nir, entry["disparity"], True))
            self.entries.append((entry["rgb"], nir_ambient, entry["disparity"], True))

        if validate:
            with open(filename, "w") as file:
                json.dump(validate_entries, file)
        return self.entries

    def collate_fn(self, batch):
        max_width = max([image[1].shape[-1] for image in batch])
        max_height = max([image[1].shape[-2] for image in batch])
        padded_images = []
        input_arr = []

        for item in batch:
            input, *images = item
            input_arr.append(input)
            for idx, img in enumerate(images):
                if len(padded_images) <= idx:
                    padded_images.append([])
                if img.size(-1) == max_width and img.size(-2) == max_height:
                    padded_images[idx].append(img)
                    continue
                img = img.unsqueeze(0)
                pad_img = pad(
                    img,
                    [
                        0,
                        0,
                        max_width - img.shape[-1],
                        max_height - img.shape[-2],
                    ],
                )

                padded_images[idx].append(pad_img[0])

        padded_images = [torch.stack(padded_image) for padded_image in padded_images]

        return input_arr, *padded_images

    def __getitem__(self, index):
        """
        return: file_name_list, (img_viz_left, img_viz_right, img_nir_left, img_nir_right)
        """

        if self.gt_depth:
            (
                (img_viz_left, img_viz_right),
                (img_nir_left, img_nir_right),
                (dis_gt_left, dis_gt_right),
                viz_luminance_reduce,
            ) = self.input_list[index]

        else:
            (
                img_viz_left,
                img_viz_right,
                img_nir_left,
                img_nir_right,
                dis_viz,
                dis_nir,
            ) = self.input_list[index]
            viz_luminance_reduce = False
        disp = (
            []
            if not self.gt_depth
            else (self.__to_tensor(dis_gt_left), self.__to_tensor(dis_gt_right))
        )
        return (
            self.input_list[index],
            self.__to_tensor(img_viz_left, viz_luminance_reduce),
            self.__to_tensor(img_viz_right, viz_luminance_reduce),
            self.__to_tensor(img_nir_left),
            self.__to_tensor(img_nir_right),
            *disp,
        )

    def __len__(self):
        return len(self.input_list)

    def extract_input_folder(self, folder: str):
        files = os.listdir(folder)
        if "rgb" in files and "nir" in files:
            return self.extract_input_from_item(folder)

        sub_folders = [
            os.path.join(folder, f)
            for f in files
            if os.path.isdir(os.path.join(folder, f))
        ]
        merged_pngs = [f for f in files if f.endswith("merged.png")]

        if len(sub_folders) > 0 and len(sub_folders) == len(merged_pngs):
            # item group folder
            subfolder = [self.extract_input_folder(f) for f in sub_folders]
            return [x for x in subfolder if x is not None]

        # not support
        if len(sub_folders) < 1:
            return None

        input_list = []
        for sub_folder in sub_folders:
            inputs = self.extract_input_folder(sub_folder)
            if inputs is None or len(inputs) == 0:
                continue
            if isinstance(inputs[0], tuple):
                input_list.extend(inputs)
            else:
                input_list.append(inputs)
        return input_list

    def extract_input_from_item(self, folder: str):
        img_viz_left = os.path.join(folder, "rgb", "left.png")
        img_viz_right = os.path.join(folder, "rgb", "right.png")
        img_nir_left = os.path.join(folder, "nir", "left.png")
        img_nir_right = os.path.join(folder, "nir", "right.png")
        disparity_viz = os.path.join(folder, "rgb", "disparity.png")
        disparity_nir = os.path.join(folder, "nir", "disparity.png")
        if not os.path.exists(img_viz_left) or not os.path.exists(img_nir_right):
            return None
        return (
            img_viz_left,
            img_viz_right,
            img_nir_left,
            img_nir_right,
            disparity_viz,
            disparity_nir,
        )
