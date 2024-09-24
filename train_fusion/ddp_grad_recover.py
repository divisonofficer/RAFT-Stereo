from typing import Any, Callable, Dict, List, Tuple
import torch
import os
import sys
import torch.nn.functional as F

project_root = os.path.dirname(os.path.abspath(__file__))
print(project_root)
sys.path.append(project_root + "/..")

from torch.nn.modules import Module
from torch.utils.data import DataLoader, DistributedSampler
from core.raft_stereo import RAFTStereo
from fusion_args import FusionArgs
from train_fusion.ddp import DDPTrainer
from torch.nn.parallel import DistributedDataParallel as DDP

from train_fusion.loss_function import ssim
from train_fusion.my_h5_dataloader import MyH5DataSet
from grad2rgb import UNetResNet


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
        model = UNetResNet().cuda()
        print(model)
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
        def loss_fn(
            output: torch.Tensor,
            target: torch.Tensor,
        ):
            ssim_loss = 1 - ssim(output, target, channel=target.shape[1]).mean()
            metric = {
                "ssim_loss": ssim_loss,
            }
            # print(ssim_loss, metric)
            return ssim_loss, metric

        return loss_fn

    def pad_img(self, img):
        batch, channel, height, width = img.shape
        pad_height = (height // 32 + 1) * 32 - height
        pad_width = (width // 32 + 1) * 32 - width
        padding = torch.nn.ZeroPad2d((0, pad_width, 0, pad_height))
        padded_img = padding(img)
        return padded_img

    def compute_gradient_torch(self, batch_imgs):
        """
        Computes the x and y gradients for a batch of images using PyTorch.

        Parameters:
        - batch_imgs: torch.Tensor of shape (batch, channel, height, width)

        Returns:
        - gradients: torch.Tensor of shape (batch, channel * 2, height, width)
                    The gradients are ordered as [channel0_x, channel0_y, channel1_x, channel1_y, ...]
        """
        batch, channel, height, width = batch_imgs.shape

        # Define gradient kernels for x and y directions
        # For x-gradient: [[1, -1]]
        # For y-gradient: [[1], [-1]]
        grad_kernel_x = torch.tensor(
            [[1, -1]], dtype=batch_imgs.dtype, device=batch_imgs.device
        ).reshape(1, 1, 1, 2)
        grad_kernel_y = torch.tensor(
            [[1], [-1]], dtype=batch_imgs.dtype, device=batch_imgs.device
        ).reshape(1, 1, 2, 1)

        # Repeat the kernels for each channel
        grad_kernel_x = grad_kernel_x.repeat(
            channel, 1, 1, 1
        )  # Shape: (channel, 1, 1, 2)
        grad_kernel_y = grad_kernel_y.repeat(
            channel, 1, 1, 1
        )  # Shape: (channel, 1, 2, 1)

        # Compute gradients using convolution
        # For padding, we add one column/row to preserve the original size
        grad_x = F.conv2d(batch_imgs, grad_kernel_x, padding=(0, 1), groups=channel)
        grad_y = F.conv2d(batch_imgs, grad_kernel_y, padding=(1, 0), groups=channel)

        # Remove the extra padding introduced by the convolution
        grad_x = grad_x[:, :, :, :width]
        grad_y = grad_y[:, :, :height, :]

        # Handle the (0,0) position as in the original function
        # This sets grad_x[:, :, 0, 0] and grad_y[:, :, 0, 0] to the original pixel values
        grad_x[:, :, 0, 0] = batch_imgs[:, :, 0, 0]
        grad_y[:, :, 0, 0] = batch_imgs[:, :, 0, 0]

        # Concatenate gradients along the channel dimension
        gradients = torch.cat(
            [grad_x, grad_y], dim=1
        )  # Shape: (batch, channel * 2, height, width)

        return gradients

    def process_batch(self, data_blob):
        inputs = [x.to(self.device) for x in data_blob]
        inputs[0] = self.pad_img(inputs[0])
        inputs[1] = self.pad_img(inputs[1])
        inputs[0] /= 255.0
        inputs[1] /= 255.0
        img_concat = torch.cat([inputs[0], inputs[1]], dim=0)
        input_left = self.compute_gradient_torch(inputs[0])
        input_right = self.compute_gradient_torch(inputs[1])

        input_concat = torch.cat([input_left, input_right], dim=0)
        output = self.model(input_concat)

        loss, metric = self.loss_fn(output, img_concat)
        return loss, metric

    def validate(self, model, valid_loader: DataLoader):
        model.eval()
        metrics: Dict[str, torch.Tensor] = {}
        losses = []
        with torch.no_grad():
            for i_batch, input_valid in enumerate(valid_loader):
                loss, metric = self.process_batch(input_valid)

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
