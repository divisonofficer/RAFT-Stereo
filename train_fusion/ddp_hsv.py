from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import torch
import os
import sys


project_root = os.path.dirname(os.path.abspath(__file__))
print(project_root)
sys.path.append(project_root + "/..")

from torch.nn.modules import Module
from torch.utils.data import DataLoader, DistributedSampler

try:
    from core.raft_stereo import RAFTStereo
except ImportError:
    import os

    os.chdir("/RAFT-Stereo")
    from core.raft_stereo import RAFTStereo
from fusion_args import FusionArgs
from train_fusion.ddp import DDPTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from hsvfusion.model import HSVNet
from train_fusion.ddp_loss import SelfLoss

from train_fusion.loss_function import (
    DynamicRangeLoss,
    loss_fn_depth_gt_box,
    reproject_disparity,
    self_supervised_loss,
    gt_loss,
    warp_reproject_loss,
    disparity_smoothness,
)
from train_fusion.my_h5_dataloader import MyH5DataSet, MyRefinedH5DataSet
from train_fusion.dataloader import EntityDataSet, StereoDataset, StereoDatasetArgs
import matplotlib.pyplot as plt
from torch.nn.parallel import DataParallel
from collections import OrderedDict
from tqdm import tqdm


class RaftTrainer(DDPTrainer):
    def __init__(self):

        args = FusionArgs()
        args.restore_ckpt = "models/raftstereo-eth3d.pth"
        # args.restore_ckpt = "checkpoints/latest_HSVFusionSynth.pth"
        args.shared_backbone = True
        args.n_gru_layers = 2
        args.n_downsample = 3
        args.batch_size = 4
        args.valid_steps = 100
        args.lr = 0.0001
        # args.corr_implementation = "reg"
        args.name = "HSVFusionSynth"
        args.shared_fusion = True
        args.mixed_precision = False
        args.freeze_backbone = ["Extractor", "Updater", "Volume", "BatchNorm"]
        self.args = args
        super().__init__(args)

    def init_models(self) -> Module:
        # raft_model = RAFTStereo(self.args).to(self.device)

        model = HSVNet(self.args).to(self.device)
        model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,
        )
        print(model.module.encoder.conv1.state_dict()["weight"][0])
        model.load_state_dict(torch.load(self.args.restore_ckpt), strict=False)
        print(model.module.encoder.conv1.state_dict()["weight"][0])
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

        # 수정된 state_dict 로드
        model.module.raft_stereo.load_state_dict(new_state_dict, strict=True)
        model.module.raft_stereo.eval()
        model.module.encoder.eval()
        for name, param in model.module.raft_stereo.named_parameters():
            param.require_grad = False
        for name, param in model.module.encoder.named_parameters():
            param.require_grad = False

        return model

    def train_mode(self):
        self.model.train()
        self.model.module.freeze_raft()

        fixed_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
            if not param.requires_grad
        }

    def init_dataloader(
        self,
    ) -> Tuple[DistributedSampler, DistributedSampler, DataLoader, DataLoader]:
        # dataset = MyH5DataSet(frame_cache=True, use_right_shift=True)
        dataset_refined = MyRefinedH5DataSet(use_right_shift=True)
        dataset = StereoDataset(
            StereoDatasetArgs(
                flying3d_json=True,
                shift_filter=True,
                noised_input=False,
                rgb_rendered=True,
            )
        )
        dataset_train = EntityDataSet(input_list=dataset.input_list)
        dataset_valid = EntityDataSet(input_list=dataset_refined.input_list[:100])
        print(len(dataset_valid))
        train_sampler = DistributedSampler(dataset_train)
        valid_sampler = DistributedSampler(dataset_valid)
        return (
            train_sampler,
            valid_sampler,
            DataLoader(
                dataset_train,
                batch_size=self.args.batch_size,
                sampler=train_sampler,
                num_workers=1,
            ),
            DataLoader(
                dataset_valid,
                batch_size=1,
                sampler=valid_sampler,
                num_workers=1,
            ),
        )

    def create_image_figure(self, image, cmap=None):
        fig, ax = plt.subplots()
        if image.ndim > 3:
            image = image[0]
        if image.shape[0] < 100:
            image = image.permute(1, 2, 0).cpu().numpy()
        if cmap is not None:
            ax.imshow(image, cmap=cmap, vmin=0, vmax=64)
        else:
            ax.imshow(image.astype(np.uint8))
        return fig

    def log_figures(self, idx: int, batch: List[torch.Tensor]):
        *inputs, _, disp_gt = batch
        rgb = torch.concat([inputs[0], inputs[1]], dim=3)
        nir = torch.concat([inputs[2], inputs[3]], dim=3)
        with torch.no_grad():
            fusion, flow = self.model(rgb, nir)
            flow_rgb = self.model.module.raft_stereo(
                inputs[0].cuda(), inputs[1].cuda(), test_mode=True
            )[1]

        self.logger.writer.add_figure(
            "disparity",
            self.create_image_figure(-flow[0, 0].cpu().numpy(), "magma"),
            idx,
        )
        self.logger.writer.add_figure(
            "disparity_rgb",
            self.create_image_figure(-flow_rgb[0, 0].cpu().numpy(), "magma"),
            idx,
        )
        self.logger.writer.add_figure("rgb", self.create_image_figure(rgb), idx)
        self.logger.writer.add_figure("nir", self.create_image_figure(nir), idx)
        self.logger.writer.add_figure("fusion", self.create_image_figure(fusion), idx)

    def init_loss_function(self) -> Callable[..., Any]:
        self.self_loss = SelfLoss()

        def loss_fn(
            fusion: torch.Tensor,
            flow: torch.Tensor,
            inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            target_gt: torch.Tensor,
            disparity_gt: torch.Tensor,
        ):
            disparity_gt = disparity_gt[..., :-8] + 16
            pd = 8
            b, c, h, w = inputs[0].shape
            fusion_left, fusion_right = torch.split(fusion, w, -1)
            concat_left = torch.cat([inputs[0], inputs[2], fusion_left], dim=1)[
                :, :, :, : w - pd
            ]
            concat_right = torch.cat([inputs[1], inputs[3], fusion_right], dim=1)[
                :, :, :, pd:
            ]
            loss_warp, metric = self.self_loss.compute_losses(
                concat_left, concat_right, [flow]
            )

            loss_expo = (torch.abs(fusion_left - fusion_right) / 255).mean()
            loss_hdr = DynamicRangeLoss()(fusion)

            loss_bright = torch.clip(96 - fusion.mean(), 0, 64) + torch.clip(
                fusion.mean() - 210, 0, 64
            )
            target_gt[..., 2] = -target_gt[..., 2]

            epe = torch.abs(disparity_gt + flow).mean(dim=1).mean()

            metric["exposure_balance"] = loss_expo
            metric["hdr_range"] = loss_hdr
            metric["brighness_center"] = loss_bright
            metric["epe"] = epe

            return loss_warp + loss_expo + loss_bright / 100 + epe, metric

        return loss_fn

    def process_batch(self, data_blob):
        inputs = [x.to(self.device) for x in data_blob]
        target_gt = inputs[-2]
        disp_gt = inputs[-1]

        rgb = torch.concat([inputs[0], inputs[1]], dim=3)
        nir = torch.concat([inputs[2], inputs[3]], dim=3)
        fusion, flow = self.model(rgb, nir)

        loss, metrics = self.loss_fn(fusion, flow, inputs[:4], target_gt, disp_gt)
        return loss, metrics

    def validate(self, model, valid_loader: DataLoader):
        model.eval()
        metrics: Dict[str, torch.Tensor] = {}
        losses = []
        with torch.no_grad():
            for i_batch, input_valid in tqdm(enumerate(valid_loader)):
                inputs = [x.to(self.device) for x in input_valid]
                target_gt = inputs[-2]
                disp_gt = inputs[-1]
                rgb = torch.concat([inputs[0], inputs[1]], dim=3)
                nir = torch.concat([inputs[2], inputs[3]], dim=3)
                fusion, flow = self.model(rgb, nir)
                loss, metric = self.loss_fn(
                    fusion, flow, inputs[:4], target_gt, disp_gt
                )
                for k, v in metric.items():
                    k = f"valid_{k}"
                    if k not in metrics:
                        metrics[k] = torch.tensor(0.0).to(self.device)
                    metrics[k] += float(v) / len(valid_loader)
                losses.append(loss.item())

        loss = sum(losses) / len(losses)

        return loss, metrics


if __name__ == "__main__":
    trainer = RaftTrainer()
    trainer.train()
