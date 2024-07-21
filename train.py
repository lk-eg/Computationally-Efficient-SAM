import os
import time
import datetime

import torch
import numpy as np

from models.build import build_model
from data.build import (
    build_dataset,
    build_train_dataloader,
    build_val_dataloader,
    build_full_train_dataloader,
)
from solver.build import build_optimizer, build_lr_scheduler

from utils.logger import Logger
from utils.dist import init_distributed_model, is_main_process
from utils.seed import setup_seed
from utils.engine import train_one_epoch, evaluate
from utils.optimiser_based_selection import (
    need_closure_fn,
    schedule_epoch_ranges,
    scheduling,
    optimiser_overhead_calculation,
)
from utils.global_results_collection import (
    comp_file_logging,
    training_result_save,
)

from hessian_eigenthings import compute_hessian_eigenthings


def main(args):
    # init seed
    setup_seed(args)

    # init dist
    init_distributed_model(args)

    # init log
    logger = Logger(args)
    logger.log(args)

    # build dataset and dataloader
    train_data, val_data, n_classes = build_dataset(args)
    train_loader = build_train_dataloader(train_dataset=train_data, args=args)
    val_loader = build_val_dataloader(val_dataset=val_data, args=args)
    args.n_classes = n_classes
    logger.log(f"Train Data: {len(train_data)}, Test Data: {len(val_data)}.")

    # build model
    model = build_model(args)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    logger.log(f"Model: {args.model}")

    # build loss
    criterion = torch.nn.CrossEntropyLoss()

    # build solver
    optimizer, base_optimizer = build_optimizer(
        args, model=model_without_ddp, logger=logger
    )
    use_optimizer = optimizer
    lr_scheduler = build_lr_scheduler(args, optimizer=base_optimizer)
    logger.log(f"Optimizer: {type(optimizer)}")
    logger.log(f"LR Scheduler: {type(lr_scheduler)}")

    # just for SGD
    if args.extensive_metrics_mode:
        logger.wandb_define_metrics_per_batch(["||g_{SGD}||"])

    # logger.wandb_define_runtime_metric("acc")

    # resume
    if args.resume:
        checkpoint = torch.load(args.resume_path, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        lr_scheduler.step(args.start_epoch)
        logger.log(f"Resume training from {args.resmue_path}.")

    # Re: Scheduling Optimizer
    # schedule = True if in a VaSSO epoch
    schedule = False
    if args.crt == "schedule":
        schedule = True
        sch_epoch_ranges = schedule_epoch_ranges(args.crt_s)

    need_closure = need_closure_fn(args)

    # start train:
    logger.log(f"Start training for {args.epochs} Epochs.")
    start_training = time.time()
    max_acc = 0.0
    images_per_second_list = []
    for epoch in range(args.start_epoch, args.epochs):
        start_epoch = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        if schedule:
            if scheduling(current_epoch=epoch, epoch_ranges=sch_epoch_ranges):
                use_optimizer = optimizer
                need_closure = True
            else:
                use_optimizer = base_optimizer
                need_closure = False

        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=use_optimizer,
            epoch=epoch,
            logger=logger,
            log_freq=args.log_freq,
            need_closure=need_closure,
            optimizer_argument=args.opt,
            extensive_metrics_mode=args.extensive_metrics_mode,
        )
        lr_scheduler.step(epoch)
        val_stats = evaluate(model, val_loader)

        if max_acc < val_stats["test_acc1"]:
            max_acc = val_stats["test_acc1"]
            # COMMENTED OUT because saving model checkpoints consumes too much memory
            # if is_main_process:
            #     torch.save(
            #         {
            #             "model": model_without_ddp.state_dict(),
            #             "optimizer": optimizer.state_dict(),
            #             "lr_scheduler": lr_scheduler.state_dict(),
            #             "epoch": epoch,
            #             "args": args,
            #         },
            #         os.path.join(args.output_dir, args.output_name, "checkpoint.pth"),
            #     )

        custom_metrics_per_epoch = [
            "train_loss",
            "train_acc1",
            "train_acc5",
            "images/s",
            "test_loss",
            "test_acc1",
            "test_acc5",
        ]
        logger.wandb_define_metrics_per_epoch(custom_metrics_per_epoch)

        logger.wandb_log_epoch(**train_stats, epoch=epoch)
        logger.wandb_log_epoch(**val_stats, epoch=epoch)
        msg = " ".join(
            [
                "Epoch:{epoch}",
                "Train Loss:{train_loss:.4f}",
                "Train Acc1:{train_acc1:.4f}",
                "Train Acc5:{train_acc5:.4f}",
                "Test Loss:{test_loss:.4f}",
                "Test Acc1:{test_acc1:.4f}(Max:{max_acc:.4f})",
                "Test Acc5:{test_acc5:.4f}",
                "Time:{epoch_time:.3f}s",
            ]
        )
        logger.log(
            msg.format(
                epoch=epoch,
                **train_stats,
                **val_stats,
                max_acc=max_acc,
                epoch_time=time.time() - start_epoch,
            )
        )
        train_acc1 = train_stats["train_acc1"]
        test_loss = val_stats["test_loss"]
        train_loss = train_stats["train_loss"]
        images_per_second_list.append(train_stats["images/s"])
    logger.log("Train Finish. Max Test Acc1:{:.4f}".format(max_acc))
    end_training = time.time()
    used_training = str(datetime.timedelta(seconds=end_training - start_training))
    training_duration = end_training - start_training
    training_duration_minutes = training_duration / 60
    logger.log("Training Time:{}".format(used_training))

    # taken from the last round
    overfitting_indicator = train_loss - test_loss

    optim_overhead_calc = optimiser_overhead_calculation(args)
    if optim_overhead_calc:
        logger.log(
            "Total inner gradient calculations: {}, Total iterations: {}".format(
                optimizer.inner_gradient_calculation_counter,
                optimizer.iteration_step_counter,
            )
        )
        logger.log(
            "Total inner forward passes: {}, Total iterations: {}".format(
                optimizer.inner_fwp_calculation_counter,
                optimizer.iteration_step_counter,
            )
        )
        fwp_overhead_over_sgd = 1 + optimizer.inner_fwp_calculation_counter / (
            optimizer.iteration_step_counter
        )
        bwp_overhead_over_sgd = 1 + optimizer.inner_gradient_calculation_counter / (
            optimizer.iteration_step_counter
        )
        logger.log("Overhead over SGD: {:.2f}".format(bwp_overhead_over_sgd))
    elif args.opt == "sgd" or args.opt[:4] == "adam":
        fwp_overhead_over_sgd = 1.0
        bwp_overhead_over_sgd = 1.0
    elif args.opt[:3] == "sam" or args.opt[:5] == "vasso":
        fwp_overhead_over_sgd = 2.0
        bwp_overhead_over_sgd = 2.0

    np_images_per_second = np.array(images_per_second_list)
    images_per_sec = np.mean(np_images_per_second)

    lambda_1, lambda_5 = None, None
    # Computing the Hessian spectrum of the solution
    # if not (args.model == "resnet18" and args.dataset[:7] == "CIFAR10"):
    #     lambda_1, lambda_5 = None, None
    # else:
    #     # full_traindataloader = build_full_train_dataloader(args.dataset)
    #     num_eigenthings = 10
    #     hessian_eigenthings = compute_hessian_eigenthings(
    #         model, train_loader, criterion, num_eigenthings, mode="lanczos"
    #     )
    #     hessian_spectrum = hessian_eigenthings[0]
    #     lambda_1 = round(hessian_spectrum[0], 4)
    #     lambda_5 = round(hessian_spectrum[4], 4)

    # logger.mv("{}_{:.4f}".format(logger.logger_path, max_acc))
    # comp_file_logging(
    #     args,
    #     max_acc,
    #     train_acc1,
    #     test_loss,
    #     train_loss,
    #     overhead,
    #     used_training,
    #     optim_overhead_calc,
    # )
    # saving everything into the csv is enough
    training_result_save(
        args,
        max_acc,
        overfitting_indicator,
        fwp_overhead_over_sgd,
        bwp_overhead_over_sgd,
        images_per_sec=images_per_sec,
        runtime=training_duration_minutes,
        lambda_1=lambda_1,
        lambda_5=lambda_5,
    )


if __name__ == "__main__":
    from configs.defaulf_cfg import default_parser

    cfg_file = default_parser()
    args = cfg_file.get_args()
    main(args)
