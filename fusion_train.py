from core.raft_stereo_fusion import RAFTStereoFusion
from train_fusion.dataloader import StereoDataset
from tqdm.notebook import tqdm
from train_fusion.train import train, self_supervised_real_batch
from train_fusion.loss_function import self_supervised_loss, self_fm_loss

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

args = type("", (), {})()
args.hidden_dims = [128, 128, 128]
args.corr_levels = 4
args.corr_radius = 4
args.n_downsample = 3
args.context_norm = "batch"
args.n_gru_layers = 2
args.shared_backbone = True
args.mixed_precision = True
args.corr_implementation = "reg_cuda"
args.slow_fast_gru = False
args.restore_ckpt = "models/raftstereo-realtime.pth"
args.lr = 0.001
args.train_iters = 7
args.wdecay = 0.0001
args.num_steps = 100000
args.name = "StereoFusion"
args.batch_size = 4
args.fusion = "ConCat"

train_loader = DataLoader(
    StereoDataset("/bean/depth"),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
)

model = nn.DataParallel(RAFTStereoFusion(args)).cuda()

model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
model = model
model.train()


train(args, model, train_loader, tqdm, self_supervised_real_batch, self_supervised_loss)
