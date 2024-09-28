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
from myutils.matrix import rmse_loss
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
            if self.update_cache:
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

    def combine_disparity(
        self, lidar_points, disparity_rgb, disparity_nir, block_width=24
    ):
        # Get image dimensions

        height, width = disparity_rgb.shape
        u, v, z = lidar_points.T
        u = u.astype(np.int32)
        v = v.astype(np.int32)

        num_blocks = (width + block_width - 1) // block_width  # Ceiling division

        # Initialize combined disparity map
        combined_disparity = np.zeros_like(disparity_rgb)

        for block_idx in range(num_blocks):
            # Define the horizontal range for the current block
            start_u = block_idx * block_width
            end_u = min((block_idx + 1) * block_width, width)

            # Identify LiDAR points within the current block
            in_block = (u >= start_u) & (u < end_u)

            if not np.any(in_block):
                # If no points in this block, default to disparity_rgb
                combined_disparity[:, start_u:end_u] = disparity_rgb[:, start_u:end_u]
                continue

            # Get the indices of points in the current block
            block_u = u[in_block]
            block_v = v[in_block]
            block_z = z[in_block]

            # Ensure u and v are within image bounds
            valid = (
                (block_u >= 0) & (block_u < width) & (block_v >= 0) & (block_v < height)
            )
            block_u = block_u[valid]
            block_v = block_v[valid]
            block_z = block_z[valid]

            if len(block_z) == 0:
                # No valid points after filtering
                combined_disparity[:, start_u:end_u] = disparity_rgb[:, start_u:end_u]
                continue

            # Sample disparity values from both maps
            sampled_rgb = disparity_rgb[block_v, block_u]
            sampled_nir = disparity_nir[block_v, block_u]

            # Compute RMSE loss for both disparity maps
            rgb_loss = rmse_loss(block_z, sampled_rgb)
            nir_loss = rmse_loss(block_z, sampled_nir)

            # Choose the disparity map with lower loss for this block
            if rgb_loss < nir_loss:
                chosen_disparity = disparity_rgb[:, start_u:end_u]
            else:
                chosen_disparity = disparity_nir[:, start_u:end_u]

            # Assign the chosen disparity to the combined map
            combined_disparity[:, start_u:end_u] = chosen_disparity
        return combined_disparity

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
            cx = 351.19399470967926
            cx2 = 351.89804045
            lidar_projected_points[:, 2] = (
                focal_length * baseline / lidar_projected_points[:, 2] - cx2 + cx
            )

            disparity_rgb = frame["disparity/rgb"][:].squeeze()
            disparity_nir = frame["disparity/nir"][:].squeeze()
            disparity = self.combine_disparity(
                lidar_projected_points, disparity_rgb, disparity_nir
            )
            disparity = torch.from_numpy(disparity).unsqueeze(-1).permute(2, 0, 1)

            while len(lidar_projected_points) < 5000:
                lidar_projected_points = np.concatenate(
                    [lidar_projected_points, lidar_projected_points], axis=0
                )
            lidar_projected_points = lidar_projected_points[:5000]
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
