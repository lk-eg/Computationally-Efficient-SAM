import os
import time

from pathlib import Path
from utils.configurable import configurable
from utils.dist import is_main_process

try:
    import wandb

    _has_wandb = True
except Exception as e:
    print(f"No wandb found. Exception: {e}")
    _has_wandb = False


class Logger:
    @configurable
    def __init__(
        self,
        output_dir,
        output_name,
        enable_wandb,
        wandb_project,
        wandb_name,
        distributed,
        time_fmt,
        args,
    ):
        self.time_fmt = time_fmt
        self.output_dir = output_dir
        self.output_name = output_name
        self.logger_path = os.path.join(output_dir, output_name)
        Path(self.logger_path).mkdir(parents=True, exist_ok=True)

        self.enable_wandb = enable_wandb
        if enable_wandb:
            wandb_dict = {
                "project": wandb_project,
                "name": wandb_name,
            }
            if distributed:
                wandb_dict["group"] = "DDP"
            self.run = wandb.init(**wandb_dict, config=args)

            wandb.define_metric("global_batch_counter")
            wandb.define_metric("epoch")
            wandb.define_metric("training_stage_%5")
        else:
            self.run = None

    @classmethod
    def from_config(cls, args):
        return {
            "output_dir": args.output_dir,
            "output_name": args.output_name,
            "enable_wandb": args.wandb and _has_wandb,
            "wandb_project": args.wandb_project,
            "wandb_name": args.wandb_name,
            "time_fmt": "%Y-%m-%d %H:%M:%S",
            "distributed": args.distributed,
            "args": args,
        }

    # def wandb_define_runtime_metric(self, metric):
    #     if self.enable_wandb:
    #         wandb.define_metric(metric, step_metric="runtime")
    #         wandb.define_metric(metric, step_metric="Bwpass x SGD")
    #         wandb.define_metric(metric, step_metric="Fwpass x SGD")
    #         wandb.define_metric(metric, step_metric="images/s")
    #     else:
    #         return

    # called from within the constructor of the optimization algorithm class,
    # so from class VaSSO, class SAM, etc.
    def wandb_define_metrics_per_batch(self, custom_metrics):
        if self.enable_wandb:
            for metric in custom_metrics:
                wandb.define_metric(metric, step_metric="global_batch_counter")
        else:
            return

    # one `training stage` is every 5 epochs
    def wandb_define_metrics_per_training_stage(self, custom_metrics):
        if self.enable_wandb:
            for metric in custom_metrics:
                wandb.define_metric(metric, step_metric="training_stage_%5")
        else:
            return

    def wandb_define_metrics_per_epoch(self, custom_metrics):
        if self.enable_wandb:
            for metric in custom_metrics:
                wandb.define_metric(metric, step_metric="epoch")
        else:
            return

    @is_main_process
    def log(self, info, printf=True):
        header = (
            " ".join(
                [
                    time.strftime(self.time_fmt, time.localtime()),
                    self.output_name,
                ]
            )
            + ": "
        )
        # with open(os.path.join(self.logger_path, "log.info"), "a") as f:
        #     f.write(header + str(info) + "\n")

        if printf:
            print(header + str(info) + "\n")

    def wandb_log_epoch(self, **stats):
        if self.enable_wandb:
            self.run.log(stats)
        else:
            return

    def wandb_log_batch(self, **stats):
        if self.enable_wandb:
            self.run.log(stats)
        else:
            return

    @is_main_process
    def mv(self, new_name):
        os.system("mv {} {}".format(self.logger_path, new_name))
