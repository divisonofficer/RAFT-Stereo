from typing import List, Optional
import cv2
import h5py
import os
import torch
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from myutils.hy5py import calibration_property, read_lidar
from myutils.points import project_points_on_camera, transform_point_inverse


class MyH5DataSet(data.Dataset):

    def __init__(
        self,
        root="/bean/depth",
        fast_test=False,
        frame_cache=False,
        update_cache=False,
        id_list: Optional[List[str]] = None,
    ):
        self.transform_mtx = np.load("jai_transform.npy")
        self.frame_cache = frame_cache
        self.update_cache = update_cache
        if id_list is not None:
            self.frame_id_list = id_list
            return
        ## find h5 files
        h5files = self.find_h5_files(root)

        ## read h5 files and get frame_id list
        frame_id_list = []
        for h5file in tqdm(h5files):
            frame_id_list += list(self.read_h5_file(h5file))

            if fast_test and len(frame_id_list) > 100:
                break
        self.frame_id_list = frame_id_list

    def __len__(self):
        return len(self.frame_id_list)

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
        frame_id_ret = []
        with h5py.File(h5_file, "a" if self.update_cache else "r", swmr=True) as f:
            if self.frame_cache and "frame_ids" in f:
                frame_ids = f["frame_ids"][:]
                f.close()
                return frame_ids
            frame_ids = list(f["frame"].keys())
            for frame_id in frame_ids:
                frame = f.require_group(f"frame/{frame_id}")
                if "disparity" in frame:
                    if "align_error" in frame.attrs and frame.attrs["align_error"]:
                        continue
                    frame_id_ret.append(
                        (h5_file, os.path.join(os.path.dirname(h5_file), frame_id))
                    )
            if self.frame_cache and self.update_cache:
                if "frame_ids" in f:
                    del f["frame_ids"]
                f.create_dataset("frame_ids", data=frame_id_ret)
            f.close()
        return frame_id_ret

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

    def __getitem__(self, index):
        h5_path, frame_path = self.frame_id_list[index]
        if type(h5_path) == bytes:
            h5_path = h5_path.decode("utf-8")
            frame_path = frame_path.decode("utf-8")
        resolution = cv2.imread(os.path.join(frame_path, "rgb", "left.png")).shape
        with h5py.File(h5_path, "r", swmr=True) as f:
            frame = f.require_group(f"frame/{frame_path.split('/')[-1]}")
            focal_length, baseline, cx, cy = calibration_property(
                f["calibration"].attrs
            )

            lidar_points = read_lidar(frame)
            lidar_points = transform_point_inverse(lidar_points, self.transform_mtx)
            lidar_projected_points = project_points_on_camera(
                lidar_points, focal_length, cx, cy, resolution[1], resolution[0]
            )
            lidar_projected_points[:, 2] = (
                focal_length * baseline / lidar_projected_points[:, 2]
            )

            while len(lidar_projected_points) < 10000:
                lidar_projected_points = np.concatenate(
                    [lidar_projected_points, lidar_projected_points], axis=0
                )
            lidar_projected_points = lidar_projected_points[:10000]
        lidar_projected_points = torch.from_numpy(lidar_projected_points).float()

        if resolution[0] < 540:
            lidar_projected_points[:, 1] += (540 - resolution[0]) // 2
            lidar_projected_points[:, 0] += (720 - resolution[1]) // 2
        rgb_left = self.imread(os.path.join(frame_path, "rgb", "left.png"))
        rgb_right = self.imread(os.path.join(frame_path, "rgb", "right.png"))
        nir_left = self.imread(os.path.join(frame_path, "nir", "left.png"), gray=True)
        nir_right = self.imread(os.path.join(frame_path, "nir", "right.png"), gray=True)
        return rgb_left, rgb_right, nir_left, nir_right, lidar_projected_points
