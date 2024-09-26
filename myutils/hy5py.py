from typing import Callable, Union
import h5py
import numpy as np


def read_calibration(h5path: Union[str, h5py.File]):
    """
    Read calibration attributes from an HDF5 file.
    args:
        h5path: str or h5py.File
            Path to the HDF5 file or h5py.File object.
    returns:
        dict
            Dictionary of calibration attributes.
    """
    if isinstance(h5path, h5py.File):
        return h5path["calibration"].attrs
    with h5py.File(h5path, "r") as f:
        return f["calibration"].attrs


def read_lidar(frame: h5py.Group, scale=1000):
    """
    Read LiDAR data from a frame.
    args:
        frame: h5py.Group
            HDF5 group of a frame.
    returns:
        np.ndarray
            LiDAR data.
    """
    return frame["lidar/points"][:] * scale


def calibration_property(calibration: dict):
    """
    Get calibration properties from a calibration dictionary.
    args:
        calibration: dict
            Dictionary of calibration attributes.
    returns:
        focal_length: float
        baseline: float
        cx: float
        cy: float
    """
    focal_length = calibration["mtx_left"][0, 0]
    cx = calibration["mtx_left"][0, 2]
    cy = calibration["mtx_left"][1, 2]
    baseline = np.linalg.norm(calibration["T"][:])
    return (
        focal_length,
        baseline,
        cx,
        cy,
    )


def process_frames(
    h5file: str, frame_callback: Callable[[h5py.Group], None], update=False
):
    """
    Process frames in an HDF5 file.
    args:
        h5file: str
            Path to the HDF5 file.
        frame_callback: Callable[[h5py.Group], None]
            Function to call for each frame.
    """
    with h5py.File(h5file, "a" if update else "r") as f:
        for frame_id in f["frame"]:
            frame_callback(f.require_group(f"frame/{frame_id}"))
