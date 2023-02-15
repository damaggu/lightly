# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
from einops import rearrange

from timm.data import Mixup
from timm.utils import accuracy

import fb_MAE.util.misc as misc
import fb_MAE.util.lr_sched as lr_sched
# from model import pad_cut_image


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    transform = None,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, addition_model=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args["accum_iter"]

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, (samples, targets, fnames) in enumerate(metric_logger.log_every(data_loader, 500, header)):
    # for data_iter_step, a in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if args["dataset"] == "hmdb":
            samples = [pad_cut_image(x).unsqueeze_(0) for x in samples]
        else:
            pass
        # check if images is torch.Tensor
        if not isinstance(samples, torch.Tensor):
            samples = torch.stack(samples, 0)
        # check if dimension 1 is 1
        if samples.shape[1] == 1:
            samples = torch.squeeze(samples, dim=1)
        if samples.shape[-1] == 3 and args["is_3d"]:
            samples = rearrange(samples, "b d h w c -> b d c h w")
        if args["is_3d"]:
            samples = rearrange(samples, "b d c h w -> b c d h w")

        if transform is not None:
            samples = torch.cat([transform(x).unsqueeze_(0) for x in samples])

        # if len(samples.shape) == 5:
        #     samples = rearrange(samples, "b d c h w -> b c d h w")

        # samples = samples.to(device, non_blocking=True)
        # targets = targets.to(device, non_blocking=True)
        samples = samples.to(device)
        targets = targets.to(device)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            if addition_model:
                samples = addition_model.images_to_codes(samples)
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        #TODO: double check commented line
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    # update_grad=(data_iter_step + 1) % accum_iter == 0
                    )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, transform=None, addition_model=None, verbose=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    # for batch in metric_logger.log_every(data_loader, 10, header):
    for (images, target, fnames) in metric_logger.log_every(data_loader, 500, header):
        if args["dataset"] == "hmdb":
            images = [pad_cut_image(x).unsqueeze_(0) for x in images]
        else:
            pass
        # check if images is torch.Tensor
        if not isinstance(images, torch.Tensor):
            images = torch.stack(images, 0)
        # check if dimension 1 is 1
        if images.shape[1] == 1:
            images = torch.squeeze(images, dim=1)
        if images.shape[-1] == 3 and args["is_3d"]:
            images = rearrange(images, "b d h w c -> b d c h w")
        if args["is_3d"]:
            images = rearrange(images, "b d c h w -> b c d h w")

        if transform is not None:
            images = torch.cat([transform(x).unsqueeze_(0) for x in images])

        if len(images.shape) == 5:
            images = rearrange(images, "b d c h w -> b c d h w")

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            if addition_model:
                images = addition_model.images_to_codes(images)
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}