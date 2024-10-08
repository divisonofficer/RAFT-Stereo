import os
import random
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.data as data
import numpy as np

from PIL import Image
import json

import tqdm
from myutils.image_process import gamma_correction, guided_filter
import pfmread
import cv2


DRIVING_JSON = "flyingthings3d.json"
REAL_DATA_JSON = "real_data.json"
FLYING_JSON = "Flow3dFlyingThings3d.json"


class Entity:
    def get_item(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        raise NotImplementedError("You must implement get_item method")


class EntityFlying3d(Entity):

    cut_resolution = (540, 720)

    def __init__(
        self,
        images: List[str],
        disparity: List[str],
        guided_noise=None,
        gamma_noise=None,
    ):
        self.images = images
        self.disparity = disparity
        self.guided_noise = guided_noise
        self.gamma_noise = gamma_noise

    def __read_img(self, filename):
        if filename.endswith(".pfm"):
            img = pfmread.read(filename)
        else:
            img = np.array(Image.open(filename)).astype(np.uint8)
        if self.cut_resolution is not None and (
            img.shape[0] != self.cut_resolution[0]
            or img.shape[1] != self.cut_resolution[1]
        ):
            w_f = int(img.shape[1] / 2 - self.cut_resolution[1] / 2)
            h_f = int(img.shape[0] / 2 - self.cut_resolution[0] / 2)
            w_t = int(img.shape[1] / 2 + self.cut_resolution[1] / 2)
            h_t = int(img.shape[0] / 2 + self.cut_resolution[0] / 2)
            img = img[h_f:h_t, w_f:w_t]

        return img

    def __to_tensor(self, filename: Union[str, np.ndarray]):
        if isinstance(filename, np.ndarray):
            img = filename
        else:
            img = self.__read_img(filename)

        tensor = torch.from_numpy(img.copy())
        if tensor.dim() == 2:
            return tensor.unsqueeze(0).float()
        return tensor.permute(2, 0, 1).float()

    def get_item(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        images = [self.__read_img(img) for img in self.images]

        if self.guided_noise is not None:
            images[0] = guided_filter(images[2], images[0], self.guided_noise + 2, 1e-6)
            images[1] = guided_filter(images[3], images[0], self.guided_noise + 2, 1e-6)
        if self.gamma_noise is not None:
            images[0] = gamma_correction(
                images[0], self.gamma_noise + random.random() + 0.1
            )
            images[1] = gamma_correction(
                images[1], self.gamma_noise + random.random() + 0.1
            )

        images = [self.__to_tensor(img) for img in images]

        indices = torch.randperm(self.cut_resolution[1] * self.cut_resolution[0])[:5000]
        u = indices % self.cut_resolution[1]
        v = indices // self.cut_resolution[1]

        disparity = self.__to_tensor(self.disparity[0])
        disparity_sampled = disparity[:, v, u]
        disparity_points = torch.stack((u, v, disparity_sampled[0]), dim=0).T.float()
        return (images[0], images[1], images[2], images[3], disparity_points, disparity)


class EntityDataSet(data.Dataset):
    input_list: List[Entity]

    def __init__(self, input_list: List[Entity]):
        self.input_list = input_list

    def __getitem__(self, index):
        return self.input_list[index].get_item()

    def __len__(self):
        return len(self.input_list)


class Entry:
    def __init__(
        self, rgb: Tuple[str, str], nir: Tuple[str, str], disparity: Optional[str]
    ):
        self.rgb = rgb
        self.nir = nir
        self.disparity = disparity

    def __tuple__(self):
        output = (*self.rgb, *self.nir)
        if self.disparity is not None:
            output += (*self.disparity,)
        return output

    def __from_tuple__(self, t):
        self.rgb = t[:2]
        self.nir = t[2:4]
        self.disparity = t[4:5]
        return self

    def __key__(self):
        return self.nir[0]


class StereoDatasetArgs:
    def __init__(
        self,
        folder: str = "/bean/depth",
        flow3d_driving_json=False,
        flying3d_json=False,
        fast_test=False,
        synth_no_filter=False,
        synth_no_rgb=False,
        validate_json=False,
        noised_input=False,
    ):
        self.folder = folder
        self.flow3d_driving_json = flow3d_driving_json
        self.flying3d_json = flying3d_json
        self.synth_no_filter = synth_no_filter
        self.validate_json = validate_json
        self.synth_no_rgb = synth_no_rgb
        self.fast_test = fast_test
        self.noised_input = noised_input


class StereoDataset(EntityDataSet):
    input_list: List[EntityFlying3d]

    def __init__(
        self,
        args: StereoDatasetArgs,
        copy_of_self=False,
        cut_resolution: Optional[Tuple[int, int]] = None,
    ):
        self.args = args
        self.cut_resolution = cut_resolution
        if copy_of_self:
            return
        self.input_list = []
        if args.flow3d_driving_json:
            self.input_list += self.flow3d_driving_json(
                DRIVING_JSON, args.validate_json
            )
        if args.flying3d_json:
            self.input_list += self.flow3d_driving_json(FLYING_JSON, args.validate_json)

    def flow3d_driving_json(self, filename: str, validate=False):
        with open(filename, "r") as file:
            entries = json.load(file)
        self.entries: List[EntityFlying3d] = []
        validate_entries = []
        for idx, entry in enumerate(tqdm.tqdm(entries)):
            if self.args.synth_no_filter and "frame_burnt_filtered" in entry:
                continue
            if self.args.fast_test and idx > 100:
                break

            if "nir" in entry:
                nir = entry["nir"]
            else:
                nir = (
                    entry["rgb"][0].replace("frames_cleanpass", "nir_rendered"),
                    entry["rgb"][1].replace("frames_cleanpass", "nir_rendered"),
                )
                entry["nir"] = nir
            if not self.args.synth_no_rgb:
                self.entries.append(
                    EntityFlying3d([*entry["rgb"], *nir], entry["disparity"])
                )
                if self.args.noised_input:
                    for _ in range(10):
                        self.entries.append(
                            EntityFlying3d(
                                [*entry["rgb"], *nir],
                                entry["disparity"],
                                guided_noise=int((random.random() * 100) % 20),
                                gamma_noise=(random.random() * 2),
                            )
                        )

            for filter in [
                "frame_burnt_filtered",
                "frame_burnt_light_filtered",
                "frame_darken_filtered",
                "frame_darken_gain_filtered",
            ]:
                if filter in entry:
                    filtered = entry[filter]
                elif not os.path.exists(
                    entry["rgb"][0].replace("frames_cleanpass", filter)
                ) or not os.path.exists(
                    entry["rgb"][1].replace("frames_cleanpass", filter)
                ):
                    continue
                else:
                    filtered = (
                        entry["rgb"][0].replace("frames_cleanpass", filter),
                        entry["rgb"][1].replace("frames_cleanpass", filter),
                    )
                    entry[filter] = filtered

                self.entries.append(
                    EntityFlying3d([*filtered, *nir], entry["disparity"])
                )
                if self.args.noised_input:
                    for _ in range(3):
                        self.entries.append(
                            EntityFlying3d(
                                [*filtered, *nir],
                                entry["disparity"],
                                guided_noise=int((random.random() * 100) % 20),
                                gamma_noise=(random.random() * 2),
                            )
                        )

            if validate:
                validated = True
                for key, value in entry.items():
                    if (isinstance(value, str) and not os.path.exists(value)) or (
                        isinstance(value, tuple)
                        and (
                            not os.path.exists(value[0]) or not os.path.exists(value[1])
                        )
                    ):
                        validated = False
                        break
                try:
                    disparity = pfmread.readPFM(entry["disparity"][0])
                except Exception as e:
                    validated = False

                if validated:
                    validate_entries.append(entry)

        if validate:
            with open(filename, "w") as file:
                json.dump(validate_entries, file)
        return self.entries

    def __getitem__(self, index):
        """
        return: file_name_list, (img_viz_left, img_viz_right, img_nir_left, img_nir_right)
        """
        return self.input_list[index].get_item()

    def __len__(self):
        return len(self.input_list)
