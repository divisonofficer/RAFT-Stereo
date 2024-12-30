from attr import dataclass
import numpy as np
from tqdm import tqdm
from typing import Callable, Dict, List
from kiss_icp.pipeline import OdometryPipeline
import open3d as o3d


class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx) -> np.ndarray:
        raise NotImplementedError


@dataclass
class Metric:
    units: str
    values: List


class KissICPWrapper:

    def run_sequence(self, kiss_pipeline: Callable, results: Dict, **kwargs):
        # Create pipeline object
        pipeline: OdometryPipeline = kiss_pipeline(kwargs.pop("sequence"))

        # New entry to the results dictionary
        results.setdefault("dataset_name", pipeline.dataset_name)

        # Run pipeline
        print(f"Now evaluating sequence {pipeline.dataset_sequence}")
        seq_res = pipeline.run()
        seq_res.print()

        # Update the metrics dictionary
        for result in seq_res:
            results.setdefault("metrics", {}).setdefault(
                result.desc, Metric(result.units, [])
            ).values.append(result.value)

        # Update the trajectories results
        results.setdefault("trajectories", {}).update(
            {
                pipeline.dataset_sequence: {
                    "gt_poses": pipeline.gt_poses,
                    "poses": np.asarray(pipeline.poses).reshape(
                        len(pipeline.poses), 4, 4
                    ),
                }
            }
        )

    def merge_points(
        self, dataset: Dataset, pose_list: List
    ) -> o3d.geometry.PointCloud:
        merged_pcd = None
        for i in tqdm(range(len(dataset))):
            points = dataset[i]
            # Extract the pose for the current lidar frame
            pose = pose_list[i]
            R = pose[:3, :3]
            t = pose[:3, 3]

            # Transform the lidar points using the pose
            points = points[points[..., 1] < 0]
            points = points[np.linalg.norm(points, axis=1) > 1]

            transformed_points = (R @ points.T).T + t

            # Create an Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(transformed_points)

            # Downsample the point cloud using voxel grid
            voxel_size = (i * 1 + 1) / 100
            downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

            # Merge the downsampled point cloud into a single point cloud
            if merged_pcd is None:
                merged_pcd = downsampled_pcd
            else:
                merged_pcd += downsampled_pcd
            merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.001)
        return merged_pcd

    def get_sequence_extrinsics(self, dataset: Dataset):
        def sequence_pipe(sequence: int):
            return OdometryPipeline(
                dataset=dataset,
            )

        result = {}
        self.run_sequence(sequence_pipe, result, sequence=0)
        return result["trajectories"][""]["poses"]
