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

from train_fusion.loss_function import (
    reproject_disparity,
    self_supervised_loss,
    gt_loss,
    warp_reproject_loss,
)
from train_fusion.my_h5_dataloader import MyH5DataSet
from train_fusion.dataloader import EntityDataSet
import matplotlib.pyplot as plt
from torch.nn.parallel import DataParallel


class RaftTrainer(DDPTrainer):
    def __init__(self):

        args = FusionArgs()
        args.restore_ckpt = "models/raftstereo-eth3d.pth"
        # args.restore_ckpt = "checkpoints/200_RaftFusion.pth"

        args.n_gru_layers = 3
        args.n_downsample = 2
        args.batch_size = 4
        args.valid_steps = 2000 // args.batch_size
        args.lr = 0.0001
        args.name = "RaftFusionFreezeRaftLarge"
        args.shared_fusion = True
        args.freeze_backbone = ["Extractor", "Updater", "Volume", "BatchNorm"]
        super().__init__(args)

    def init_models(self) -> Module:
        raft_model = DataParallel(RAFTStereo(self.args)).to(self.device)

        model = HSVNet().to(self.device)
        model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,
        )
        model.load_state_dict(torch.load(self.args.restore_ckpt), strict=False)
        raft_model.load_state_dict(torch.load(self.args.restore_ckpt), strict=False)
        self.raft_mode = raft_model.module
        self.raft_mode.eval()
        return model

    def train_mode(self):
        self.model.train()

    def init_dataloader(
        self,
    ) -> Tuple[DistributedSampler, DistributedSampler, DataLoader, DataLoader]:
        dataset = MyH5DataSet(frame_cache=True)
        train_cnt = int(len(dataset) * 0.95)
        dataset_train = EntityDataSet(input_list=dataset.input_list[:train_cnt])
        dataset_valid = EntityDataSet(input_list=dataset.input_list[train_cnt:])
        train_sampler = DistributedSampler(dataset_train)
        valid_sampler = DistributedSampler(dataset_valid)
        return (
            train_sampler,
            valid_sampler,
            DataLoader(
                dataset_train,
                batch_size=self.args.batch_size,
                sampler=train_sampler,
            ),
            DataLoader(
                dataset_valid,
                batch_size=self.args.batch_size,
                sampler=valid_sampler,
            ),
        )

    def create_image_figure(self, image, cmap=None):
        fig, ax = plt.subplots()
        if image.ndim > 3:
            image = image[0]
        if image.shape[0] < 100:
            image = image.permute(1, 2, 0).cpu().numpy()
        if cmap is not None:
            ax.imshow(image, cmap=cmap, vmin=0, vmax=32)
        else:
            ax.imshow(image.astype(np.uint8))
        return fig

    def log_figures(self, idx: int, batch: List[torch.Tensor]):
        *inputs, _, disp_gt = batch
        rgb = torch.concat([inputs[0], inputs[1]], dim=3)
        nir = torch.concat([inputs[2], inputs[3]], dim=3)
        with torch.no_grad():
            fusion = self.model(rgb, nir)
            flow = self.raft_mode(fusion, fusion, test_mode=True)[1]
        self.logger.writer.add_figure(
            "disparity",
            self.create_image_figure(-flow[0, 0].cpu().numpy(), "magma"),
            idx,
        )
        self.logger.writer.add_figure("rgb", self.create_image_figure(rgb), idx)
        self.logger.writer.add_figure("nir", self.create_image_figure(nir), idx)
        self.logger.writer.add_figure("fusion", self.create_image_figure(fusion), idx)

    def init_loss_function(self) -> Callable[..., Any]:
        def loss_fn(
            fusion: torch.Tensor,
            inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            target_gt: torch.Tensor,
            disparity_gt: torch.Tensor,
        ):
            pd = 12
            b, c, h, w = fusion.shape
            w = w // 2
            fusion_left, fusion_right = torch.split(
                fusion, fusion.shape[-1] // 2, dim=3
            )
            fusion_left = fusion_left[:, :, :, : w - pd]
            fusion_right = fusion_right[:, :, :, pd:]
            concat_left = torch.cat([inputs[0], inputs[2]], dim=1)[:, :, :, : w - pd]
            concat_right = torch.cat([inputs[1], inputs[3]], dim=1)[:, :, :, pd:]
            with torch.no_grad():
                flow = self.raft_mode(fusion_left, fusion_right, test_mode=True)[1]
            loss_warp, metric = warp_reproject_loss(flow, concat_left, concat_right)
            return loss_warp, metric

        return loss_fn

    def process_batch(self, data_blob):
        inputs = [x.to(self.device) for x in data_blob]
        target_gt = inputs[-2]
        disp_gt = inputs[-1]

        rgb = torch.concat([inputs[0], inputs[1]], dim=3)
        nir = torch.concat([inputs[2], inputs[3]], dim=3)
        fusion = self.model(rgb, nir)

        loss, metrics = self.loss_fn(fusion, inputs[:4], target_gt, disp_gt)
        return loss, metrics

    def validate(self, model, valid_loader: DataLoader):
        model.eval()
        metrics: Dict[str, torch.Tensor] = {}
        losses = []
        with torch.no_grad():
            for i_batch, input_valid in enumerate(valid_loader):
                inputs = [x.to(self.device) for x in input_valid]
                target_gt = inputs[-2]
                disp_gt = inputs[-1]
                rgb = torch.concat([inputs[0], inputs[1]], dim=3)
                nir = torch.concat([inputs[2], inputs[3]], dim=3)
                fusion = self.model(rgb, nir)
                loss, metric = self.loss_fn(fusion, inputs[:4], target_gt, disp_gt)
                for k, v in metric.items():
                    k = f"valid_{k}"
                    if k not in metrics:
                        metrics[k] = torch.tensor(0.0).to(self.device)
                    metrics[k] += v / len(valid_loader)
                losses.append(loss.item())

        loss = sum(losses) / len(losses)

        return loss, metrics


if __name__ == "__main__":
    trainer = RaftTrainer()
    trainer.train()
