import random
from typing import List, Optional, Tuple
import cv2
import h5py
import os
import torch
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from myutils.hy5py import calibration_property, read_lidar
from myutils.matrix import rmse_loss
from myutils.points import (
    combine_disparity_by_edge,
    combine_disparity_by_lidar,
    pad_lidar_points,
    project_points_on_camera,
    refine_disparity,
    refine_disparity_with_monodepth,
    transform_point_inverse,
)


from train_fusion.dataloader import Entity, EntityDataSet


class MyH5Entity(Entity):
    def __init__(self, h5_path, frame_path):
        self.h5_path = h5_path
        self.frame_path = frame_path

    def imread(self, path: str, gray=False):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_ANYCOLOR)
        if not gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).float()
        if gray:
            tensor = tensor.unsqueeze(-1)
        tensor = tensor.permute(2, 0, 1)

        if tensor.shape[-2] < 540:
            padding = (
                (720 - tensor.shape[-1]) // 2,
                (720 - tensor.shape[-1]) // 2,
                (540 - tensor.shape[-2]) // 2,
                (540 - tensor.shape[-2]) // 2,
            )
            tensor = F.pad(tensor, padding)

        return tensor

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
        h5_path = self.h5_path
        frame_path = self.frame_path
        if type(h5_path) == bytes:
            h5_path = h5_path.decode("utf-8")
            frame_path = frame_path.decode("utf-8")
        transform_mtx = np.load("jai_transform.npy")
        resolution = cv2.imread(os.path.join(frame_path, "rgb", "left.png")).shape
        with h5py.File(h5_path, "r", swmr=True) as f:
            frame = f.require_group(f"frame/{frame_path.split('/')[-1]}")
            focal_length, baseline, cx, cy = calibration_property(
                f["calibration"].attrs
            )

            lidar_points = read_lidar(frame)
            lidar_points = transform_point_inverse(lidar_points, transform_mtx)
            lidar_projected_points = project_points_on_camera(
                lidar_points, focal_length, cx, cy, resolution[1], resolution[0]
            )
            cx = 351.19399470967926
            cx2 = 351.89804045
            lidar_projected_points[:, 2] = (
                focal_length * baseline / lidar_projected_points[:, 2] - 1
            )

            lidar_projected_points = pad_lidar_points(lidar_projected_points, 5000)

            disparity_rgb = frame["disparity/rgb"][:].squeeze()[:540, :720]
            disparity_nir = frame["disparity/nir"][:].squeeze()[:540, :720]
            monodepth_rgb = frame["depth_mono/rgb"][:].squeeze()
            monodepth_nir = frame["depth_mono/nir"][:].squeeze()
            disparity_rgb = refine_disparity_with_monodepth(
                disparity_rgb, monodepth_rgb
            )
            rgb_left = cv2.imread(
                os.path.join(frame_path, "rgb", "left.png"), cv2.IMREAD_COLOR
            )
            nir_left = cv2.imread(
                os.path.join(frame_path, "nir", "left.png"), cv2.IMREAD_GRAYSCALE
            )
            disparity_nir = refine_disparity_with_monodepth(
                disparity_nir, monodepth_nir
            )
            disparity = combine_disparity_by_edge(
                lidar_projected_points, disparity_rgb, disparity_nir, rgb_left, nir_left
            )
            disparity = torch.from_numpy(disparity).unsqueeze(-1).permute(2, 0, 1)

        lidar_projected_points = torch.from_numpy(lidar_projected_points).float()

        if resolution[0] < 540:
            lidar_projected_points[:, 1] += (540 - resolution[0]) // 2
            lidar_projected_points[:, 0] += (720 - resolution[1]) // 2
        rgb_left = self.imread(os.path.join(frame_path, "rgb", "left.png"))
        rgb_right = self.imread(os.path.join(frame_path, "rgb", "right.png"))
        nir_left = self.imread(os.path.join(frame_path, "nir", "left.png"), gray=True)
        nir_right = self.imread(os.path.join(frame_path, "nir", "right.png"), gray=True)

        return (
            rgb_left,
            rgb_right,
            nir_left,
            nir_right,
            lidar_projected_points,
            disparity,
        )


class MyH5DataSet(EntityDataSet):
    def __init__(
        self, root="/bean/depth", fast_test=False, frame_cache=False, update_cache=False
    ):
        self.transform_mtx = np.load("jai_transform.npy")
        self.frame_cache = frame_cache
        self.update_cache = update_cache
        ## find h5 files
        h5files = self.find_h5_files(root)

        ## read h5 files and get frame_id list
        frame_id_list: List[MyH5Entity] = []
        for h5file in tqdm(h5files):
            frame_id_list += list(self.read_h5_file(h5file))

            if fast_test and len(frame_id_list) > 100:
                break
        frame_id_list = random.sample(frame_id_list, len(frame_id_list))
        self.input_list = frame_id_list

    def __len__(self):
        return len(self.input_list)

    def find_h5_files(self, root):
        h5_files = []
        for folder in os.listdir(root):
            folder = os.path.join(root, folder)
            if not os.path.isdir(folder):
                continue
            for file in os.listdir(folder):
                if file.endswith(".hdf5"):
                    h5_files.append(os.path.join(folder, file))
        return h5_files

    def read_h5_file(self, h5_file):
        frame_id_ret: List[MyH5Entity] = []
        with h5py.File(h5_file, "a" if self.update_cache else "r", swmr=True) as f:
            if self.frame_cache and "frame_ids" in f:
                frame_ids = f["frame_ids"][:]
                f.close()
                return [MyH5Entity(*x) for x in frame_ids[:]]
            frame_ids = list(f["frame"].keys())
            for frame_id in frame_ids:
                frame = f.require_group(f"frame/{frame_id}")
                if "disparity" in frame:
                    if "align_error" in frame.attrs and frame.attrs["align_error"]:
                        continue
                    frame_id_ret.append(
                        MyH5Entity(
                            h5_file, os.path.join(os.path.dirname(h5_file), frame_id)
                        )
                    )
            if self.update_cache:
                if "frame_ids" in f:
                    del f["frame_ids"]
                f.create_dataset("frame_ids", data=frame_id_ret)
            f.close()
        return frame_id_ret

    def __getitem__(self, index):
        return self.input_list[index].get_item()
