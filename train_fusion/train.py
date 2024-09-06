from typing import Callable, Dict
from fusion_args import FusionArgs
from train_stereo import Logger
from torch.cuda.amp import GradScaler

import logging
from pathlib import Path
from torch.utils.data import DataLoader
from core.raft_stereo_fusion import RAFTStereoFusion
import torch
from torch import optim
import os
from datastructure.train_input import TrainInput


def train(
    args: FusionArgs,
    model: RAFTStereoFusion,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    tqdm: Callable,
    batch_loader_function: Callable[[FusionArgs, tuple, bool], tuple[TrainInput, list]],
    loss_function,
):
    os.makedirs("checkpoints", exist_ok=True)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8
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
    logger = Logger(model, scheduler)

    model.module.freeze_raft()  # We keep the RAFT backbone frozen
    validation_frequency = args.valid_steps

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True

    def backward(
        scaler: GradScaler,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.OneCycleLR,
        model,
        loss,
    ):
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

    while should_keep_training:

        for i_batch, input_train in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            assert model.training
            """
            Batch Loader : for each batch from the train_loader,
            create a input dict for model
            """
            batch_load, input_arr = batch_loader_function(args, input_train, False)
            flow_predictions = model(batch_load)
            assert model.training
            """
            Inputarr for loss function
            """
            loss, metric = loss_function(model.module, input_arr, flow_predictions)

            if args.both_side_train:
                input_train_right = ()
                input_train_right += (input_train[0],)
                input_train_right += (torch.flip(input_train[2].clone(), dims=[-1]),)
                input_train_right += (torch.flip(input_train[1].clone(), dims=[-1]),)
                input_train_right += (torch.flip(input_train[4].clone(), dims=[-1]),)
                input_train_right += (torch.flip(input_train[3].clone(), dims=[-1]),)

                if len(input_train) > 5:
                    input_train_right += (
                        torch.flip(input_train[6].clone(), dims=[-1]),
                    )
                    input_train_right += (
                        torch.flip(input_train[5].clone(), dims=[-1]),
                    )

                batch_load, input_arr = batch_loader_function(
                    args, input_train_right, False
                )
                flow_predictions = model(batch_load)

                loss_right, metric_right = loss_function(
                    model.module, input_arr, flow_predictions
                )
                loss += loss_right
                for k, v in metric_right.items():
                    metric[f"{k}_right"] = v

            logger.writer.add_scalar("live_loss", loss.item(), total_steps)
            logger.writer.add_scalar(
                f"learning_rate", optimizer.param_groups[0]["lr"], total_steps
            )
            logger.write_dict(metric)

            print(f"Batch {i_batch} Loss {loss}")
            backward(scaler, optimizer, scheduler, model, loss)

            if (total_steps + 1) % validation_frequency == 0:
                save_path = Path("checkpoints/%d_%s.pth" % (total_steps + 1, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)

                loss, results = validate_things(
                    model.module,
                    args,
                    logger,
                    valid_loader,
                    batch_loader_function,
                    loss_function,
                )
                logger.writer.add_scalar("valid_loss", loss, total_steps)
                logger.write_dict(results)

                model.train()
                model.module.freeze_raft()

                if total_steps > args.num_steps:
                    should_keep_training = False
                    break

            total_steps += 1
            logger.total_steps = total_steps

    print("FINISHED TRAINING")
    logger.close()
    PATH = "checkpoints/%s.pth" % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


def validate_things(
    model,
    args: FusionArgs,
    logger,
    valid_loader: DataLoader,
    batch_loader_function: Callable[[FusionArgs, tuple, bool], tuple[TrainInput, list]],
    loss_function,
):
    model.eval()
    metrics: Dict[str, torch.Tensor] = {}
    losses = []
    for i_batch, input_valid in enumerate(valid_loader):
        batch_load, input_arr = batch_loader_function(args, input_valid, True)
        flow_predictions = model(batch_load)
        loss, metric = loss_function(model, input_arr, flow_predictions)

        print(f"Batch {i_batch} Loss {loss}")
        for k, v in metric.items():
            if k not in metrics:
                metrics[k] = torch.tensor(0.0)
            metrics[k] += v / len(valid_loader)
        losses.append(loss.item())

    loss = sum(losses) / len(losses)

    return loss, metrics


def batch_input_dict(args, input_tuple, valid_mode=False):
    input = TrainInput.from_image_tuple(input_tuple[:4])
    input.iters = args.train_iters if not valid_mode else args.valid_iters
    return input.data_dict


def self_supervised_real_batch(args, input, valid_mode=False):
    """
    Batch Load function for real input data
    """
    image_list, *blob = input
    img_cuda = [img.cuda() for img in blob]
    return batch_input_dict(args, img_cuda, valid_mode), img_cuda


def flow_gt_batch(args, input, valid_mode=False):
    """
    Batch Load function for real input data
    """
    image_list, *blob = input
    img_cuda = [img.cuda() for img in blob]
    return batch_input_dict(args, img_cuda[:4], valid_mode), [img_cuda[4]]
