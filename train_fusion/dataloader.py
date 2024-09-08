import os
from typing import List, Optional, Tuple
import torch
import torch.utils.data as data
import numpy as np

from PIL import Image
import json
import pfmread
import cv2


DRIVING_JSON = "flyingthings3d.json"
REAL_DATA_JSON = "real_data.json"
FLYING_JSON = "Flow3dFlyingThings3d.json"


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
        real_data_json=False,
        real_data_validate=False,
        flow3d_driving_json=False,
        flying3d_json=False,
        gt_depth=False,
        synth_no_filter=False,
        synth_no_rgb=False,
        validate_json=False,
        
    ):
        self.folder = folder
        self.real_data_json = real_data_json
        self.real_data_validate = real_data_validate
        self.flow3d_driving_json = flow3d_driving_json
        self.flying3d_json = flying3d_json
        self.gt_depth = gt_depth
        self.synth_no_filter = synth_no_filter
        self.validate_json = validate_json
        self.synth_no_rgb = synth_no_rgb


class StereoDataset(data.Dataset):
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
        self.input_list: List[Entry] = []
        if args.flow3d_driving_json:
            self.input_list += self.flow3d_driving_json(
                DRIVING_JSON, args.validate_json
            )
        if args.flying3d_json:
            self.input_list += self.flow3d_driving_json(FLYING_JSON, args.validate_json)
        if args.real_data_json:
            with open(REAL_DATA_JSON, "r") as file:
                self.input_list = json.load(file)
        if args.real_data_validate:
            input_list = self.extract_input_folder(args.folder)
            if not isinstance(input_list, list):
                raise ValueError("folder must contain subfolders")
            self.input_list = input_list
            with open(REAL_DATA_JSON, "w") as file:
                json.dump(self.input_list, file)
        self.input_list.sort(key=lambda x: x.__key__())

    def __to_tensor(self, filename):
        if filename.endswith(".pfm"):
            img = pfmread.read(filename)
        else:
            img = np.array(Image.open(filename)).astype(np.uint8)
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
            self.args,
            copy_of_self=True,
            cut_resolution=self.cut_resolution,
        )
        copy_of_self.input_list = self.input_list[start:end]

        return copy_of_self

    def flow3d_driving_json(self, filename: str, validate=False):
        with open(filename, "r") as file:
            entries = json.load(file)
        self.entries: List[Entry] = []
        validate_entries = []
        for entry in entries:
            if self.args.synth_no_filter and "frame_burnt_filtered" in entry:
                continue

            if "nir" in entry:
                nir = entry["nir"]
            else:
                nir = (
                    entry["rgb"][0].replace("frames_cleanpass", "nir_rendered"),
                    entry["rgb"][1].replace("frames_cleanpass", "nir_rendered"),
                )
                entry["nir"] = nir
            if not self.args.synth_no_rgb:
                self.entries.append(Entry(entry["rgb"], nir, entry["disparity"]))

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

                self.entries.append(Entry(filtered, nir, entry["disparity"]))

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

        if self.args.gt_depth:
            (
                img_viz_left,
                img_viz_right,
                img_nir_left,
                img_nir_right,
                dis_gt_left,
                dis_gt_right,
            ) = self.input_list[index].__tuple__()

        else:
            (
                img_viz_left,
                img_viz_right,
                img_nir_left,
                img_nir_right,
            ) = self.input_list[index].__tuple__()

        disp = (
            []
            if not self.args.gt_depth
            else (self.__to_tensor(dis_gt_left), self.__to_tensor(dis_gt_right))
        )
        return (
            self.input_list[index].__tuple__(),
            self.__to_tensor(img_viz_left),
            self.__to_tensor(img_viz_right),
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

        if len(sub_folders) < 1:
            return None

        input_list: List[Entry] = []
        for sub_folder in sub_folders:
            inputs = self.extract_input_folder(sub_folder)
            if inputs is None or (isinstance(inputs, list) and len(inputs)) == 0:
                continue
            if isinstance(inputs, List):
                input_list.extend(inputs)
            else:
                input_list.append(inputs)
        return input_list

    def extract_input_from_item(self, folder: str):
        img_viz_left = os.path.join(folder, "rgb", "left.png")
        img_viz_right = os.path.join(folder, "rgb", "right.png")
        img_nir_left = os.path.join(folder, "nir", "left.png")
        img_nir_right = os.path.join(folder, "nir", "right.png")
        # disparity_viz = os.path.join(folder, "rgb", "disparity.png")
        # disparity_nir = os.path.join(folder, "nir", "disparity.png")
        if not os.path.exists(img_viz_left) or not os.path.exists(img_nir_right):
            return None
        return Entry(
            (img_viz_left, img_viz_right),
            (img_nir_left, img_nir_right),
            None,
        )
