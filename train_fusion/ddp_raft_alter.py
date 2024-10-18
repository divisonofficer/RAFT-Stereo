import random
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import torch
import os
import sys
import torch.nn.functional as F
import tqdm


project_root = os.path.dirname(os.path.abspath(__file__))
print(project_root)
sys.path.append(project_root + "/..")

from torch.nn.modules import Module
from torch.utils.data import DataLoader, DistributedSampler

try:
    from core.raft_stereo_fusion_alter import RAFTStereoFusionAlter
except ImportError:
    import os

    os.chdir("/RAFT-Stereo")
    from core.raft_stereo_fusion_alter import RAFTStereoFusionAlter
from fusion_args import FusionArgs
from train_fusion.ddp import DDPTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from train_fusion.ddp_loss import SelfLoss
from train_fusion.dataloader import EntityDataSet, StereoDataset, StereoDatasetArgs
from train_fusion.loss_function import (
    disparity_smoothness,
    loss_fn_depth_gt_box,
    reproject_disparity,
    self_supervised_loss,
    gt_loss,
    ssim as ssim_torch,
    warp_reproject_loss,
)
from train_fusion.my_h5_dataloader import MyH5DataSet, MyRefinedH5DataSet
import matplotlib.pyplot as plt


class RaftTrainer(DDPTrainer):
    def __init__(self):
        args = FusionArgs()
        args.restore_ckpt = "models/raftstereo-eth3d.pth"
        args.restore_ckpt = "checkpoints/latest_BAFFAlterSynth.pth"
        args.n_gru_layers = 3
        args.n_downsample = 2
        args.batch_size = 6
        args.valid_steps = 100
        args.lr = 0.0005
        args.train_iters = 7
        args.valid_iters = 7
        args.logger_dir = "runs_raft"
        args.fusion = "bAFF"
        args.name = "BAFFAlterSynth"

        args.shared_fusion = True
        args.shared_backbone = False
        args.freeze_backbone = ["Extractor", "Updater", "Volume", "BatchNorm"]
        # args.freeze_backbone = ["BatchNorm"]
        super().__init__(args)

    def init_models(self) -> Module:
        model = RAFTStereoFusionAlter(self.args).to(self.device)
        if self.args.restore_ckpt.isdigit():
            self.args.restore_ckpt = (
                f"checkpoints/{self.args.restore_ckpt}_{self.args.name}.pth"
            )
        w_dict = torch.load(self.args.restore_ckpt)
        if "total_steps" in w_dict:
            self.total_steps = w_dict["total_steps"]
            w_dict = w_dict["model_state_dict"]
        is_module = "module." in list(w_dict.keys())[0]
        if not is_module:
            model.load_state_dict(w_dict, strict=False)
        model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,
        )
        if is_module:
            model.load_state_dict(w_dict, strict=False)

        return model

    def train_mode(self):
        self.model.train()
        self.model.module.freeze_raft()

    def init_dataloader(
        self,
    ) -> Tuple[DistributedSampler, DistributedSampler, DataLoader, DataLoader]:

        dataset = MyH5DataSet(frame_cache=True, use_right_shift=True)
        # dataset = MyRefinedH5DataSet(use_right_shift=True)
        dataset_flying = StereoDataset(
            StereoDatasetArgs(
                flying3d_json=True,
                noised_input=False,
                shift_filter=True,
                rgb_rendered=True,
            )
        )
        # dataset_driving = StereoDataset(
        #     StereoDatasetArgs(
        #         flow3d_driving_json=True,
        #         noised_input=True,
        #         vertical_scale=True,
        #         shift_filter=True,
        #     )
        # )
        cnt = len(dataset)
        train_cnt = cnt - 200
        valid_cnt = min(1000, len(dataset))
        dataset_valid = EntityDataSet(dataset.input_list[train_cnt:])

        dataset_train = EntityDataSet(
            # dataset.input_list[:train_cnt]
            # dataset_driving.input_list[: int(len(dataset_driving))]
            dataset_flying.input_list[: int(len(dataset_flying))]
        )
        print(len(dataset_train))
        # dataset_valid = EntityDataSet(dataset_flying.input_list[-500:])
        train_sampler = DistributedSampler(dataset_train)
        valid_sampler = DistributedSampler(dataset_valid)
        return (
            train_sampler,
            valid_sampler,
            DataLoader(
                dataset_train,
                batch_size=self.args.batch_size,
                sampler=train_sampler,
                num_workers=2,
            ),
            DataLoader(
                dataset_valid,
                batch_size=1,
                sampler=valid_sampler,
                num_workers=2,
            ),
        )

    def create_image_figure(self, image, cmap=None, vmax=32):
        fig, ax = plt.subplots()
        if image.ndim > 3:
            image = image[0]
        if image.shape[0] < 100:
            image = image.permute(1, 2, 0).cpu().numpy()
        if cmap is not None:
            ax.imshow(image, cmap=cmap, vmin=0, vmax=vmax)
        else:
            ax.imshow(image.astype(np.uint8))
        return fig

    def log_figures(self, idx: int, batch: List[torch.Tensor]):
        left_rgb, right_rgb, left_nir, right_nir, _, disp_gt = [
            x.to(self.device) for x in batch
        ]
        with torch.no_grad():
            _, flow = self.model(
                left_rgb, right_rgb, left_nir, right_nir, iters=7, test_mode=True
            )
        idx = self.total_steps
        right_rgb_warped = self.self_loss.disocc_detection(flow, left_rgb)[1]
        ssim_loss = ssim_torch(right_rgb, right_rgb_warped)
        self.logger.writer.add_figure(
            "disparity",
            self.create_image_figure(-flow[0, 0].cpu().numpy(), "magma"),
            idx,
        )
        self.logger.writer.add_figure(
            "disparity_gt",
            self.create_image_figure(disp_gt[0, 0].cpu().numpy(), "magma"),
            idx,
        )
        self.logger.writer.add_figure(
            "left_rgb", self.create_image_figure(left_rgb[0]), idx
        )
        self.logger.writer.add_figure(
            "right_rgb", self.create_image_figure(right_rgb[0]), idx
        )
        self.logger.writer.add_figure(
            "right_rgb_warped", self.create_image_figure(right_rgb_warped[0]), idx
        )

        self.logger.writer.add_figure(
            "right_warp_ssim",
            self.create_image_figure(ssim_loss[0, 0].cpu().numpy(), "OrRd", vmax=1),
            idx,
        )

    def init_loss_function(self) -> Callable[..., Any]:
        self.self_loss = SelfLoss(w_smooth=20)

        def loss_fn(
            flow: List[torch.Tensor],
            inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            target_gt: torch.Tensor,
            disparity_gt: torch.Tensor,
            disp_loss=True,
            lidar_loss=False,
            self_loss=False,
        ):
            loss_dict = {}
            total_loss = 0
            rgb_left, rgb_right, nir_left, nir_right = inputs
            flow = [
                x[:, :, : rgb_left[0].shape[-2], : rgb_left[0].shape[-1]] for x in flow
            ]
            if disp_loss:
                try:
                    disparity_loss, dis_metric = gt_loss(None, disparity_gt, flow)
                    for k, v in dis_metric.items():
                        if not isinstance(v, torch.Tensor):
                            v = torch.tensor(v, device=flow[-1].device)
                        loss_dict[k] = v

                    total_loss = torch.add(total_loss, disparity_loss)
                except AssertionError:
                    self.log_figures(
                        self.total_steps, [*inputs, target_gt, disparity_gt]
                    )

            if lidar_loss:
                target_gt[..., 2] = -target_gt[..., 2]
                depth_loss, depth_loss_last = loss_fn_depth_gt_box(
                    flow, target_gt, weight=0.8
                )
                loss_dict["depth_loss"] = depth_loss
                loss_dict["depth_loss_last"] = depth_loss_last
                total_loss += depth_loss * 0.02
            if self_loss:
                left = torch.concat([rgb_left, nir_left], dim=1)
                right = torch.concat([rgb_right, nir_right], dim=1)
                warp_loss, warp_metric = self.self_loss.compute_losses(
                    left, right, flow
                )
                for k, v in warp_metric.items():
                    if not isinstance(v, torch.Tensor):
                        v = torch.tensor(v, device=flow[-1].device)
                    loss_dict[k] = v
                total_loss = torch.add(total_loss, warp_loss)

            # total_loss = torch.clip(total_loss, 0, 100)
            return total_loss, loss_dict

        return loss_fn

    def process_batch(self, data_blob):
        inputs = [x.to(self.device).to(torch.float32) for x in data_blob]
        target_gt = inputs[-2]
        disp_gt = inputs[-1]
        total_loss = 0
        flow = self.model(*inputs[:4], iters=7)
        loss, metrics = self.loss_fn(
            flow,
            inputs[:4],
            target_gt,
            disp_gt,
            True,
            lidar_loss=False,
            self_loss=False,
        )
        total_loss += loss
        return total_loss, metrics

    @torch.no_grad()
    def validate(self, model, valid_loader: DataLoader):
        model.eval()
        metrics: Dict[str, torch.Tensor] = {}
        losses = []
        with torch.cuda.amp.autocast(enabled=True):
            for i_batch, input_valid in enumerate(tqdm.tqdm(valid_loader)):
                inputs = [x.to(self.device).to(torch.float32) for x in input_valid]
                target_gt = inputs[-2]
                disp_gt = inputs[-1]
                flow = model(*inputs[:4], iters=7)

                loss, metric = self.loss_fn(
                    flow, inputs[:4], target_gt, disp_gt, False, True, True
                )

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
