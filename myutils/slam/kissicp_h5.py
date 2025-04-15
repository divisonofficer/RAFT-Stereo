from .kissicp import KissICPWrapper, Dataset
from typing import Optional
import h5py
import open3d as o3d
import numpy as np


class H5LiDARDataset(Dataset):
    def __init__(self, scene_path: str, frame_idx: int, frame_count: int, steps=1):
        self.scene_path = scene_path
        self.frame_idx = frame_idx
        self.frame_count = frame_count
        self.data_dir = ""
        self.cached_scene: Optional[h5py.File] = None
        self.cached_scene_id = -1
        self.steps = steps

    def __len__(self):
        return self.frame_count

    def downsample_points(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        """
        포인트 클라우드를 다운샘플링합니다.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        return np.asarray(downsampled_pcd.points)

    def get_frame(self, idx):
        if idx < 10:
            frame_idx = self.frame_idx + idx
        else:
            frame_idx = self.frame_idx + (idx - 10) * self.steps + 10
        scene_idx = frame_idx // 100
        if scene_idx == self.cached_scene_id:
            scene = self.cached_scene
        else:
            if self.cached_scene is not None:
                self.cached_scene.close()
            scene = h5py.File(f"{self.scene_path}/{scene_idx}.hdf5", "r")
            self.cached_scene = scene
            self.cached_scene_id = scene_idx

        frame_ids = list(scene["frame"])
        frame_id = frame_ids[frame_idx % 100]
        frame = scene["frame"][frame_id]
        return frame

    def __getitem__(self, idx):
        frame = self.get_frame(idx)
        lidar_points = frame["lidar"]["points"][:].reshape(-1, 3)  # type: ignore
        lidar_points = self.downsample_points(lidar_points, 0.1)
        return lidar_points

    def __del__(self):
        if self.cached_scene is not None:
            self.cached_scene.close()


class H5KissICP(KissICPWrapper):
    def load_h5_dataset(self, scene_path: str, frame_idx: int, stack_cnt=50, steps=1):
        self.dataset = H5LiDARDataset(scene_path, frame_idx, stack_cnt, steps=steps)
        return self.dataset

    def get_stacked_lidar(self):
        poses = self.get_h5_lidar_poses()
        merged = self.merge_points(self.dataset, poses)
        return merged

    def get_h5_lidar_poses(self):
        return self.get_sequence_extrinsics(self.dataset)
