import time
from collections import defaultdict
from typing import Iterable

import torch
import torch.distributed as dist
from utils.dist import is_dist_avail_and_initialized
from utils.device import device


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: Iterable,
    criterion,
    optimizer,
    epoch,
    logger,
    log_freq,
    need_closure,
    optimizer_argument,
    extensive_metrics_mode,
    schedule,
):
    model.train()

    _memory = MetricLogger()
    _memory.add_meter("train_loss", Metric())
    _memory.add_meter("train_acc1", Metric())
    _memory.add_meter("train_acc5", Metric())
    _memory.add_meter("images/s", Metric())
    for batch_idx, (images, targets) in enumerate(train_loader):
        batch_start = time.time()

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forard- and Backward-pass function.
        # Efficiency is mainly about how often, and which part, of this function gets called.
        def closure(computeForward, computeBackprop):
            if computeForward:
                output = model(images)
                loss = criterion(output, targets)
            if computeBackprop:
                optimizer.zero_grad()
                loss.backward()

            # computeForward=True in second_step() holds always.
            # We return output and loss at the perturbed position
            # as this is the bound on the generalization error up to a monotonic function (cf. Foret et. al.: SAM).
            if computeForward:
                return output, loss

        # sgd needs "normal" output and loss calculation
        if optimizer_argument[:3] == "sgd" and extensive_metrics_mode:
            total_sgd_norm = (
                sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
            )
            logger.wandb_log_batch(
                **{
                    "||g_{SGD}||": total_sgd_norm,
                    "global_batch_counter": epoch * len(train_loader) + batch_idx,
                }
            )

        if need_closure:
            output, loss = optimizer.step(
                closure,
                epoch=epoch,
                step=epoch * len(train_loader) + batch_idx,
                batch_idx=batch_idx,
                model=model,
                images=images,
                targets=targets,
                criterion=criterion,
                train_data=train_loader.dataset,
                logger=logger,
            )
        else:
            with torch.enable_grad():
                output, loss = closure(True, True)
            optimizer.step()

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        batch_num = images.shape[0]
        batch_t = time.time() - batch_start
        _memory.update_meter("train_loss", loss.item(), n=batch_num)
        _memory.update_meter("train_acc1", acc1.item(), n=batch_num)
        _memory.update_meter("train_acc5", acc5.item(), n=batch_num)
        _memory.update_meter("images/s", batch_num / batch_t, n=1)

        msg = " ".join(
            [
                "Epoch: {epoch}",
                "[{batch_id}/{batch_len}]",
                "lr:{lr:.6f}",
                "Train Loss:{train_loss:.4f}",
                "Train Acc1:{train_acc1:.4f}",
                "Train Acc5:{train_acc5:.4f}",
                "Time:{batch_time:.3f}s",
            ]
        )
        if batch_idx % log_freq == 0:
            logger.log(
                msg.format(
                    epoch=epoch,
                    batch_id=batch_idx,
                    batch_len=len(train_loader),
                    lr=optimizer.param_groups[0]["lr"],
                    train_loss=_memory.meters["train_loss"].global_avg,
                    train_acc1=_memory.meters["train_acc1"].global_avg,
                    train_acc5=_memory.meters["train_acc5"].global_avg,
                    batch_time=batch_t,
                )
            )
        _memory.synchronize_between_processes()
    return {name: meter.global_avg for name, meter in _memory.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: Iterable,
):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    _memory = MetricLogger()
    _memory.add_meter("test_loss", Metric())
    _memory.add_meter("test_acc1", Metric())
    _memory.add_meter("test_acc5", Metric())

    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, targets)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))

        batch_num = images.shape[0]
        _memory.update_meter("test_loss", loss.item(), n=batch_num)
        _memory.update_meter("test_acc1", acc1.item(), n=batch_num)
        _memory.update_meter("test_acc5", acc5.item(), n=batch_num)
    _memory.synchronize_between_processes()
    return {name: meter.global_avg for name, meter in _memory.meters.items()}


def accuracy(output, targets, topk=(1,)):
    # output: [b, n]
    # targets: [b]
    batch_size, n_classes = output.size()
    maxk = min(max(topk), n_classes)
    _, pred = torch.topk(output, k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # pred: [b, maxk] -> [maxk, b]
    correct = pred.eq(
        targets.reshape(1, -1).expand_as(pred)
    )  # targets: [b] -> [1, b] -> [maxk, b]; correct(bool): [maxk, b]
    return [
        correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size
        for k in topk
    ]


class Metric:
    def __init__(self) -> None:
        self.value = 0
        self.num = 0

    def update(self, value, n=1):
        self.num += n
        self.value += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.num, self.value], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.num = int(t[0])
        self.value = t[1]

    @property
    def global_avg(self):
        return self.value / self.num


class MetricLogger:
    def __init__(self) -> None:
        self.meters = defaultdict(Metric)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def update_meter(self, name, value, n):
        self.meters[name].update(value, n)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
