from train_stereo import Logger
from torch.cuda.amp import GradScaler

import logging
from pathlib import Path
from torch.utils.data import DataLoader
from core.raft_stereo_fusion import RAFTStereoFusion
from core.utils.utils import InputPadder
import torch
from torch import optim
import os
from typing import Optional


def train(
    args,
    model: RAFTStereoFusion,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    tqdm,
    batch_loader_function,
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
    global_batch_num = 0

    while should_keep_training:

        for i_batch, input_train in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            assert model.training
            """
            Batch Loader : for each batch from the train_loader,
            create a input dict for model
            """
            batch_load, input_arr = batch_loader_function(args, input_train)
            flow_predictions = model(batch_load)
            assert model.training
            """
            Inputarr for loss function
            """
            loss, metric = loss_function(model.module, input_arr, flow_predictions)
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(
                f"learning_rate", optimizer.param_groups[0]["lr"], global_batch_num
            )
            logger.write_dict(metric)
            logger.total_steps += 1
            global_batch_num += 1
            print(f"Batch {i_batch} Loss {loss}")
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

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

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        # save_path = Path(
        #     "checkpoints/%d_epoch_%s.pth.gz" % (total_steps + 1, args.name)
        # )
        # logging.info(f"Saving file {save_path}")
        # torch.save(model.state_dict(), save_path)

    print("FINISHED TRAINING")
    logger.close()
    PATH = "checkpoints/%s.pth" % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


def validate_things(
    model,
    args,
    logger,
    valid_loader,
    batch_loader_function,
    loss_function,
):
    model.eval()
    metrics = {}
    losses = []
    for i_batch, input_valid in enumerate(valid_loader):
        batch_load, input_arr = batch_loader_function(
            args, input_valid, valid_mode=True
        )
        flow_predictions = model(batch_load)
        loss, metric = loss_function(model, input_arr, flow_predictions)

        print(f"Batch {i_batch} Loss {loss}")
        for k, v in metric.items():
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)
        losses.append(loss.item())

    for k, v in metrics.items():
        metrics[k] = sum(v) / len(v)
    loss = sum(losses) / len(losses)

    return loss, metrics


def batch_input_dict(args, input_tuple, valid_mode=False):
    return {
        "image_viz_left": input_tuple[0],
        "image_viz_right": input_tuple[1],
        "image_nir_left": input_tuple[2],
        "image_nir_right": input_tuple[3],
        "iters": args.train_iters if not valid_mode else args.valid_iters,
        "test_mode": False,
        "flow_init": None,
        "heuristic_nir": False,
    }


def self_supervised_real_batch(args, input, valid_mode=False):
    """
    Batch Load function for real input data
    """
    image_list, *blob = input
    image1, image2, image3, image4 = [img.cuda() for img in blob]
    return batch_input_dict(args, (image1, image2, image3, image4), valid_mode), [
        image1,
        image2,
        image3,
        image4,
    ]


def flow_gt_batch(args, input, valid_mode=False):
    """
    Batch Load function for real input data
    """
    image_list, *blob = input
    image1, image2, image3, image4, gt, _ = [img.cuda() for img in blob]
    return batch_input_dict(args, (image1, image2, image3, image4), valid_mode), [gt]
