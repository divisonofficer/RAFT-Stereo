from collections import OrderedDict
import numpy as np
import cv2
import torch


def disparity_to_depth(disparity: np.ndarray, focal_length: float, baseline: float):
    disparity = disparity.astype(np.float32)
    depth = focal_length * baseline / disparity
    depth[depth < 0] = 0
    depth[np.isinf(depth)] = 0
    depth[np.isnan(depth)] = 0
    return depth


def disparity_color(
    disparity: np.ndarray, max_disparity=32, colormap=cv2.COLORMAP_MAGMA
):

    disparity = disparity / max_disparity
    disparity = np.clip(disparity, 0, 1)
    disparity = (disparity * 255).astype(np.uint8)
    disparity = cv2.applyColorMap(disparity, colormap)
    return disparity


from core.raft_stereo import RAFTStereo
from fusion_args import FusionArgs


def get_raft_stereo():
    args = FusionArgs()
    args.shared_backbone = True
    args.n_downsample = 3
    args.n_gru_layers = 2
    model = RAFTStereo(args)

    checkpoint = torch.load("models/raftstereo-realtime.pth")
    # 새로운 state_dict 생성
    new_state_dict = OrderedDict()

    for key, value in checkpoint.items():
        if key.startswith("module."):
            # 'module.' 접두사 제거
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    model.eval()
    return model
