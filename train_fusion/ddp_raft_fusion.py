from typing import Any, Callable, Dict, List, Tuple
import torch
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
print(project_root)
sys.path.append(project_root + "/..")

from torch.nn.modules import Module
from torch.utils.data import DataLoader, DistributedSampler

try:
    from core.raft_stereo_fusion import RAFTStereoFusion
except ImportError:
    import os

    os.chdir("/RAFT-Stereo")
    from core.raft_stereo_fusion import RAFTStereoFusion
from fusion_args import FusionArgs
from train_fusion.ddp import DDPTrainer
from torch.nn.parallel import DistributedDataParallel as DDP

from train_fusion.loss_function import self_supervised_loss, gt_loss
from train_fusion.my_h5_dataloader import MyH5DataSet


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
        model = RAFTStereoFusion(self.args).to(self.device)
        model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,
        )
        model.load_state_dict(torch.load(self.args.restore_ckpt), strict=False)
        return model

    def train_mode(self):
        self.model.train()
        self.model.module.freeze_raft()

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
        def loss_fn_detph_gt(
            flow_preds: List[torch.Tensor], target_gt: torch.Tensor, weight=0.9
        ):
            gt_u = target_gt[:, :, 1].long()
            gt_v = target_gt[:, :, 0].long()
            gt_u = torch.clamp(gt_u, 0, flow_preds[-1].shape[-2] - 1)
            gt_v = torch.clamp(gt_v, 0, flow_preds[-1].shape[-1] - 1)
            B, N = gt_u.shape
            batch_indices = (
                torch.arange(B).view(B, 1).expand(B, N).to(flow_preds[-1].device)
            )
            depth_loss = 0.0
            flows_len = len(flow_preds)
            depth_loss_last = 0
            for idx, flow in enumerate(flow_preds):
                target_pred = -flow[batch_indices, :, gt_u, gt_v].squeeze()

                target_depth = target_gt[:, :, 2]
                depth_loss_last = torch.mean(
                    torch.abs(target_pred - target_depth), dim=1
                ) * (weight ** (flows_len - idx))
                depth_loss += depth_loss_last

            return depth_loss, depth_loss_last

        def loss_fn(
            flow: List[torch.Tensor],
            inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            target_gt: torch.Tensor,
            disparity_gt: torch.Tensor,
        ):
            loss = 0.0
            loss_dict = {}
            rgb_left, rgb_right, nir_left, nir_right = inputs
            flow = [
                x[:, :, : rgb_left[0].shape[-2], : rgb_left[0].shape[-1]] for x in flow
            ]

            # warp_loss, warp_metric = self_supervised_loss(
            #     (rgb_left, rgb_right, nir_left, nir_right), flow
            # )

            # for k, v in warp_metric.items():
            #     if not isinstance(v, torch.Tensor):
            #         v = torch.tensor(v, device=flow[-1].device)
            #     loss_dict[k] = v

            disparity_loss, dis_metric = gt_loss(None, disparity_gt, flow)
            for k, v in dis_metric.items():
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v, device=flow[-1].device)
                loss_dict[k] = v

            depth_loss, depth_loss_last = loss_fn_detph_gt(flow, target_gt, weight=0.8)
            depth_loss = depth_loss.mean()

            # Ensure depth_loss is a tensor
            if not isinstance(depth_loss, torch.Tensor):
                depth_loss = torch.tensor(depth_loss, device=flow[-1].device)

            loss_dict["depth_loss"] = depth_loss
            loss_dict["depth_loss_last"] = depth_loss_last.mean()

            total_loss = disparity_loss / 5.0 + depth_loss / 20.0

            return total_loss, loss_dict

        return loss_fn

    def process_batch(self, data_blob):
        inputs = [x.to(self.device) for x in data_blob]
        target_gt = inputs[-2]
        disp_gt = inputs[-1]
        flow = self.model(
            {
                "image_viz_left": inputs[0],
                "image_viz_right": inputs[1],
                "image_nir_left": inputs[2],
                "image_nir_right": inputs[3],
                "iters": self.args.train_iters,
                "test_mode": False,
                "flow_init": None,
                "heuristic_nir": False,
                "attention_out_mode": False,
            }
        )
        loss, metrics = self.loss_fn(flow, inputs[:4], target_gt, disp_gt)
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
                flow = model(
                    {
                        "image_viz_left": inputs[0],
                        "image_viz_right": inputs[1],
                        "image_nir_left": inputs[2],
                        "image_nir_right": inputs[3],
                        "iters": self.args.valid_iters,
                        "test_mode": False,
                        "flow_init": None,
                        "heuristic_nir": False,
                        "attention_out_mode": False,
                    }
                )

                loss, metric = self.loss_fn(flow, inputs[:4], target_gt, disp_gt)

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
