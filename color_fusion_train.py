import os
import sys
import signal
import traceback
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 필요한 모듈 임포트
import numpy as np
from train_fusion.dataloader import StereoDataset, StereoDatasetArgs
from train_fusion.my_h5_dataloader import MyH5DataSet
from fusion_args import FusionArgs
from color_fusion_model import RGBNIRFusionNet
from rgb_thermal_fusion_net import RGBThermalFusionNet
from train_fusion.loss_function import warp_reproject_loss, reproject_disparity
from train_stereo import Logger
from tqdm import tqdm


def main():
    # 인자 파싱 (필요 시)
    args = FusionArgs()
    # 필요한 args 설정
    args.restore_ckpt = "models/raftstereo-realtime.pth"
    args.lr = 0.001
    args.train_iters = 7
    args.valid_iters = 12
    args.num_steps = 100000
    args.valid_steps = 50
    args.name = "ColorChannel4"
    args.batch_size = 10
    args.fusion = "AFF"
    args.shared_fusion = True
    args.freeze_backbone = []
    args.both_side_train = False
    args.input_channel = 4

    # DDP 초기화
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    torch.autograd.set_detect_anomaly(True)

    # 신호 처리 핸들러 설정
    def signal_handler(sig, frame):
        print(f"Process {dist.get_rank()} received signal {sig}")
        if dist.get_rank() == 0:
            print("Interrupt received. Saving model and performing validation...")
            torch.save(fusion_model.module.state_dict(), "interrupted_model.pth")
            validate_things(fusion_model.module, valid_loader)
        dist.barrier()  # 모든 프로세스가 동기화될 때까지 대기
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    def cleanup():
        dist.destroy_process_group()

    # 모델 및 데이터 설정
    # RAFTStereo 모델 로드
    try:
        from core.raft_stereo import RAFTStereo
    except ImportError:
        os.chdir("/RAFT-Stereo")
        from core.raft_stereo import RAFTStereo

    # 환경 변수 설정
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    # 디바이스 설정
    device = torch.device(f"cuda:{local_rank}")

    # 모델 초기화
    # fusion_model = RGBThermalFusionNet(hidden_dim=16).to(device)
    fusion_model = RGBNIRFusionNet().to(device)
    fusion_model = DDP(fusion_model, device_ids=[local_rank], output_device=local_rank)

    raft_model = RAFTStereo(args).to(device)
    # 사전 학습된 가중치 로드
    raft_model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
    raft_model.eval()
    raft_model.freeze_bn()
    # raft_model은 추론에만 사용되므로 DDP로 래핑하지 않음

    # 데이터셋 로드
    dataset = MyH5DataSet(frame_cache=True)
    cnt = len(dataset)
    train_cnt = int(cnt * 0.95)
    valid_cnt = cnt - train_cnt
    print(f"Total dataset size: {cnt}")

    dataset_train = MyH5DataSet(id_list=dataset.frame_id_list[:train_cnt])
    dataset_valid = MyH5DataSet(id_list=dataset.frame_id_list[train_cnt:])

    # DistributedSampler 생성
    train_sampler = DistributedSampler(dataset_train)
    valid_sampler = DistributedSampler(dataset_valid)

    # DataLoader 생성
    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        sampler=valid_sampler,
        num_workers=0,
        drop_last=True,
    )

    # 손실 함수 및 기타 유틸리티 함수 정의
    def compute_disparity(left: torch.Tensor, right: torch.Tensor):
        if left.shape[-3] == 1:
            left = left.repeat(1, 3, 1, 1)
            right = right.repeat(1, 3, 1, 1)
        _, flow = raft_model(left, right, test_mode=True)
        return flow

    def loss_fn_detph_gt(flow: torch.Tensor, target_gt: torch.Tensor):
        gt_u = target_gt[:, :, 1].long()
        gt_v = target_gt[:, :, 0].long()
        gt_u = torch.clamp(gt_u, 0, flow.shape[-2] - 1)
        gt_v = torch.clamp(gt_v, 0, flow.shape[-1] - 1)
        B, N = gt_u.shape
        batch_indices = torch.arange(B).view(B, 1).expand(B, N).to(flow.device)
        target_pred = -flow[batch_indices, :, gt_u, gt_v].squeeze()

        target_depth = target_gt[:, :, 2]
        depth_loss = torch.sqrt(torch.mean((target_pred - target_depth) ** 2, dim=1))

        return depth_loss

    def loss_fn(pred, target, target_gt):
        flow = compute_disparity(pred[0], pred[1])
        flow = flow[:, :, : pred[0].shape[-2], : pred[0].shape[-1]]

        warp_loss_rgb, warp_metric_rgb = warp_reproject_loss([flow], *target[0])
        warp_loss_nir, warp_metric_nir = warp_reproject_loss([flow], *target[1])

        depth_loss = loss_fn_detph_gt(flow, target_gt)
        print(depth_loss)
        depth_loss = depth_loss.mean()

        # Ensure warp metrics are tensors
        loss_dict = {}
        for k, v in warp_metric_rgb.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, device=flow.device)
            loss_dict[k] = v
        for k, v in warp_metric_nir.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, device=flow.device)
            loss_dict[f"{k}_nir"] = v

        # Ensure depth_loss is a tensor
        if not isinstance(depth_loss, torch.Tensor):
            depth_loss = torch.tensor(depth_loss, device=flow.device)

        loss_dict["depth_loss"] = depth_loss

        total_loss = warp_loss_rgb.mean() + warp_loss_nir.mean() + depth_loss
        return total_loss, loss_dict

    def validate_things(model, valid_loader: DataLoader):
        model.eval()
        metrics: Dict[str, torch.Tensor] = {}
        losses = []
        with torch.no_grad():
            for i_batch, input_valid in enumerate(valid_loader):
                image1, image2, image3, image4, depth = [
                    x.to(device) for x in input_valid
                ]

                fused_input1 = torch.cat([image1, image3], dim=1)
                fused_input2 = torch.cat([image2, image4], dim=1)

                image_fusion_1 = model(fused_input1)
                image_fusion_2 = model(fused_input2)

                loss, metric = loss_fn(
                    (image_fusion_1, image_fusion_2),
                    ((image1, image2), (image3, image4)),
                    depth,
                )

                for k, v in metric.items():
                    if k not in metrics:
                        metrics[k] = torch.tensor(0.0).to(device)
                    metrics[k] += v / len(valid_loader)
                losses.append(loss.item())

        loss = sum(losses) / len(losses)

        return loss, metrics

    # 옵티마이저, 스케줄러 및 로거 초기화
    optimizer = optim.AdamW(
        fusion_model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    total_steps = 0

    # rank 0에서만 로거 초기화
    if dist.get_rank() == 0:
        logger = Logger(fusion_model.module, scheduler)

    scaler = GradScaler(enabled=args.mixed_precision)

    fusion_model.train()
    raft_model.eval()

    should_keep_training = True
    global_batch_num = 0

    # raft_model 파라미터 고정
    for param in raft_model.parameters():
        param.requires_grad = False

    while should_keep_training:
        # 샘플러에 에폭 설정
        train_sampler.set_epoch(total_steps)
        for i_batch, data_blob in enumerate(
            tqdm(train_loader, disable=(dist.get_rank() != 0))
        ):

            try:
                optimizer.zero_grad()
                image1, image2, image3, image4, depth = [
                    x.to(device) for x in data_blob
                ]

                fused_input1 = torch.cat([image1, image3], dim=1)
                fused_input2 = torch.cat([image2, image4], dim=1)

                image_fusion_1 = fusion_model(fused_input1)
                image_fusion_2 = fusion_model(fused_input2)

                loss, metrics = loss_fn(
                    (image_fusion_1, image_fusion_2),
                    ((image1, image2), (image3, image4)),
                    depth,
                )

                # 손실 및 메트릭을 모든 프로세스에서 합산
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss = loss / dist.get_world_size()

                for k in metrics:
                    dist.all_reduce(metrics[k], op=dist.ReduceOp.SUM)
                    metrics[k] = metrics[k] / dist.get_world_size()

                if dist.get_rank() == 0:
                    logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
                    logger.writer.add_scalar(
                        "learning_rate",
                        optimizer.param_groups[0]["lr"],
                        global_batch_num,
                    )
                    global_batch_num += 1
                    logger.push(metrics)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                total_steps += 1

                if dist.get_rank() == 0 and total_steps % args.valid_steps == 0:
                    save_path = Path(f"checkpoints/{total_steps}_{args.name}.pth")
                    logging.info(f"Saving file {save_path.absolute()}")
                    torch.save(fusion_model.module.state_dict(), save_path)

                    val_loss, val_metrics = validate_things(
                        fusion_model.module, valid_loader
                    )

                    logger.write_dict(val_metrics)
                    logger.writer.add_scalar("valid_loss", val_loss, total_steps)
                    fusion_model.train()

                if total_steps > args.num_steps:
                    should_keep_training = False
                    break

            except Exception as e:
                print(f"Exception occurred in process {dist.get_rank()}: {e}")
                traceback.print_exc()
                should_keep_training = False
                break

    if dist.get_rank() == 0:
        print("FINISHED TRAINING")
        logger.close()
        # 최종 모델 저장
        torch.save(
            fusion_model.module.state_dict(), f"checkpoints/final_{args.name}.pth"
        )

    cleanup()


if __name__ == "__main__":
    main()
