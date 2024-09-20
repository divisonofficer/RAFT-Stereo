from typing import Any, Callable, Dict, List, Tuple
import torch
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
print(project_root)
sys.path.append(project_root + "/..")

from torch.nn.modules import Module
from torch.utils.data import DataLoader, DistributedSampler
from core.raft_stereo import RAFTStereo
from fusion_args import FusionArgs
from train_fusion.ddp import DDPTrainer
from torch.nn.parallel import DistributedDataParallel as DDP

from train_fusion.loss_function import warp_reproject_loss
from train_fusion.my_h5_dataloader import MyH5DataSet


class RaftTrainer(DDPTrainer):
    def __init__(self):

        args = FusionArgs()
        args.restore_ckpt = "models/raftstereo-realtime.pth"
        args.batch_size = 4
        args.input_channel = 4
        args.valid_steps = 100
        args.name = "Raft4Channel"
        super().__init__(args)

    def init_models(self) -> Module:
        model = RAFTStereo(self.args).to(self.device)
        return DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)

    def init_dataloader(
        self,
    ) -> Tuple[DistributedSampler, DistributedSampler, DataLoader, DataLoader]:
        dataset = MyH5DataSet(frame_cache=True)
        train_cnt = int(len(dataset) * 0.95)
        dataset_train = MyH5DataSet(id_list=dataset.frame_id_list[:train_cnt])
        dataset_valid = MyH5DataSet(id_list=dataset.frame_id_list[train_cnt:])
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

    def init_loss_function(self) -> Callable[..., Any]:
        def loss_fn_detph_gt(flow: torch.Tensor, target_gt: torch.Tensor):
            gt_u = target_gt[:, :, 1].long()
            gt_v = target_gt[:, :, 0].long()
            gt_u = torch.clamp(gt_u, 0, flow.shape[-2] - 1)
            gt_v = torch.clamp(gt_v, 0, flow.shape[-1] - 1)
            B, N = gt_u.shape
            batch_indices = torch.arange(B).view(B, 1).expand(B, N).to(flow.device)
            target_pred = -flow[batch_indices, :, gt_u, gt_v].squeeze()

            target_depth = target_gt[:, :, 2]
            depth_loss = torch.sqrt(
                torch.mean((target_pred - target_depth) ** 2, dim=1)
            )

            return depth_loss

        def loss_fn(
            flow: List[torch.Tensor],
            inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            target_gt: torch.Tensor,
        ):
            rgb_left, rgb_right, nir_left, nir_right = inputs
            flow = [
                x[:, :, : rgb_left[0].shape[-2], : rgb_left[0].shape[-1]] for x in flow
            ]

            warp_loss_rgb, warp_metric_rgb = warp_reproject_loss(
                flow, rgb_left, rgb_right
            )
            warp_loss_nir, warp_metric_nir = warp_reproject_loss(
                flow, nir_left, nir_right
            )

            depth_loss = loss_fn_detph_gt(flow[-1], target_gt)
            print(depth_loss)
            depth_loss = depth_loss.mean()

            # Ensure warp metrics are tensors
            loss_dict = {}
            for k, v in warp_metric_rgb.items():
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v, device=flow[-1].device)
                loss_dict[k] = v
            for k, v in warp_metric_nir.items():
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v, device=flow[-1].device)
                loss_dict[f"{k}_nir"] = v

            # Ensure depth_loss is a tensor
            if not isinstance(depth_loss, torch.Tensor):
                depth_loss = torch.tensor(depth_loss, device=flow[-1].device)

            loss_dict["depth_loss"] = depth_loss

            total_loss = warp_loss_rgb.mean() + warp_loss_nir.mean() + depth_loss
            return total_loss, loss_dict

        return loss_fn

    def process_batch(self, data_blob):
        inputs = [x.to(self.device) for x in data_blob]
        target_gt = inputs[-1]
        input_left = torch.concat([inputs[0], inputs[2]], dim=1)
        input_right = torch.concat([inputs[1], inputs[3]], dim=1)
        flow = self.model(input_left, input_right)
        loss, metrics = self.loss_fn(flow, inputs[:4], target_gt)
        return loss, metrics

    def validate(self, model, valid_loader: DataLoader):
        model.eval()
        metrics: Dict[str, torch.Tensor] = {}
        losses = []
        with torch.no_grad():
            for i_batch, input_valid in enumerate(valid_loader):
                image1, image2, image3, image4, depth = [
                    x.to(self.device) for x in input_valid
                ]

                fused_input1 = torch.cat([image1, image3], dim=1)
                fused_input2 = torch.cat([image2, image4], dim=1)
                _, flow = model(fused_input1, fused_input2, test_mode=True)

                loss, metric = self.loss_fn(
                    [flow],
                    (image1, image2, image3, image4),
                    depth,
                )

                for k, v in metric.items():
                    if k not in metrics:
                        metrics[k] = torch.tensor(0.0).to(self.device)
                    metrics[k] += v / len(valid_loader)
                losses.append(loss.item())

        loss = sum(losses) / len(losses)

        return loss, metrics


if __name__ == "__main__":
    trainer = RaftTrainer()
    trainer.train()
