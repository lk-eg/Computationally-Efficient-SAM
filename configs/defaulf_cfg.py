import argparse
from utils.device import dataset_directory


class default_parser:
    def __init__(self) -> None:
        pass

    def wandb_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument(
            "--wandb_project", type=str, default="VASSO", help="Project name in wandb."
        )
        parser.add_argument(
            "--wandb_name",
            type=str,
            default="Default",
            help="Experiment name in wandb.",
        )
        return parser

    def base_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--output_dir",
            type=str,
            default="logs",
            help="Name of dir where save all experiments.",
        )
        parser.add_argument(
            "--output_name",
            type=str,
            default=None,
            help="Name of dir where save the log.txt&ckpt.pth of this experiment. (None means auto-set)",
        )
        parser.add_argument(
            "--resume", action="store_true", help="resume model,opt,etc."
        )
        parser.add_argument("--resume_path", type=str, default=".")

        parser.add_argument("--seed", type=int, default=3107)
        parser.add_argument(
            "--log_freq",
            type=int,
            default=10,
            help="Frequency of recording information.",
        )

        # Three different log modes can be specified
        parser.add_argument(
            "--extensive_metrics_mode",
            action="store_true",
            help="Extensive Metrics Mode: huge logging and metric facility",
        )
        parser.add_argument(
            "--performance_scores_mode",
            action="store_true",
            help="Performance Scores Mode: for correct inner forward pass loss calc even if forward pass is not needed",
        )
        parser.add_argument(
            "--logging_mode",
            action="store_true",
            help="If we need console, output file logging, and in general a logger",
        )

        parser.add_argument("--start_epoch", type=int, default=0)
        parser.add_argument(
            "--epochs", type=int, default=200, help="Epochs of training."
        )
        parser.add_argument("--dataset_nn_combination", type=str)
        parser.add_argument(
            "--exclusive_run",
            action="store_true",
            help="Run used for runtime and memory benchmarking purposes",
        )
        return parser

    def dist_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--dist_url",
            default="env://",
            help="url used to set up distributed training",
        )
        return parser

    def data_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--dataset",
            type=str,
            default="CIFAR10_cutout",
            help="Dataset name in `DATASETS` registry.",
        )
        datasetdirectory = dataset_directory()
        parser.add_argument(
            "--datadir",
            type=str,
            default=datasetdirectory,
            help="Path to your dataset.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=128,
            help="Batch size used in training and validation.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="Number of CPU threads for dataloaders.",
        )
        parser.add_argument("--pin_memory", action="store_true", default=True)
        parser.add_argument("--drop_last", action="store_true", default=True)
        parser.add_argument(
            "--distributed_val",
            action="store_true",
            help="Enabling distributed evaluation (Only works when use multi gpus).",
        )
        return parser

    def base_opt_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--opt",
            type=str,
            default="sgd",
            help="sgd, sam-sgd, vasso-sgd, sam-adamw, vasso-adamw, vassore-sgd, vassore-adamw",
        )
        parser.add_argument("--lr", type=float, default=0.05)
        parser.add_argument("--weight_decay", type=float, default=1e-3)
        # sgd
        parser.add_argument(
            "--momentum",
            type=float,
            default=0.9,
            help="Momentum for SGD.(None means the default in optm)",
        )
        parser.add_argument("--nesterov", action="store_true")
        # adam
        parser.add_argument(
            "--betas",
            type=float,
            default=None,
            nargs="+",
            help="Betas for AdamW Optimizer.(None means the default in optm)",
        )
        parser.add_argument(
            "--eps",
            type=float,
            default=None,
            help="Epsilon for AdamW Optimizer.(None means the default in optm)",
        )
        return parser

    def sam_opt_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--rho",
            type=float,
            default=0.1,
            help="Perturbation intensity of SAM type optims.",
        )
        parser.add_argument(
            "--theta", type=float, default=0.4, help="Moving average for VASSO"
        )

        # Criteria for making SAM efficient
        parser.add_argument(
            "--crt",
            type=str,
            default="baseline",
            choices=[
                "baseline",
                "naive",
                "random",
                "gSAMsharp",
                "gSAMflat",
                "gSAMratio",
                "schedule",
                "cosSim",
                "variance",
            ],
        )
        parser.add_argument(
            "--crt_k",
            type=int,
            default=2,
            help="Re-use of eps: new calculation every k steps",
        )
        parser.add_argument(
            "--crt_p",
            type=float,
            default=0.5,
            help="Re-use of eps: new calculation with probability p in each iteration_step",
        )
        # gSAMflat and gSAMsharp
        parser.add_argument(
            "--lam", type=float, default=0.1, help="parameter for weight for tau"
        )
        parser.add_argument(
            "--crt_z",
            type=float,
            default=1.0,
            help="comparison factor for tau-decision rule. For gSAMsharp <= 1.0, for gSAMflat >= 1.0",
        )
        parser.add_argument(
            "--z_two",
            type=float,
            default=0.5,
            help="second comparison factor for combined gSAMratio decision rule.",
        )
        parser.add_argument(
            "--var_delta",
            type=float,
            default=1.0,
            help="threshold when the variance of outer gradient norms is too high so we trigger re-calculation of perturbation",
        )
        parser.add_argument(
            "--crt_s",
            type=str,
            default="[100-200]",
            help="Scheduling of VaSSO optimizer: Run VaSSO in input epochs, baseline optimizer in other epochs",
        )
        parser.add_argument(
            "--crt_c",
            type=float,
            default=0,
            help="Check if cosSim criterion >0 or <0 (False)",
        )
        return parser

    def lr_scheduler_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--warmup_epoch", type=int, default=0)
        parser.add_argument("--warmup_init_lr", type=float, default=0.0)
        parser.add_argument("--lr_scheduler", type=str, default="CosineLRscheduler")
        # CosineLRscheduler
        parser.add_argument("--eta_min", type=float, default=0)
        # MultiStepLRscheduler
        parser.add_argument(
            "--milestone",
            type=int,
            nargs="+",
            default=[60, 120, 160],
            help="Milestone for MultiStepLRscheduler.",
        )
        parser.add_argument(
            "--gamma", type=float, default=0.2, help="Gamma for MultiStepLRscheduler."
        )
        return parser

    def model_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--model", type=str, default="resnet18", help="Model in registry to use."
        )
        return parser

    def get_args(self):
        all_parser_funcs = []
        for func_or_attr in dir(self):
            if (
                callable(getattr(self, func_or_attr))
                and not func_or_attr.startswith("_")
                and func_or_attr[-len("parser") :] == "parser"
            ):
                all_parser_funcs.append(getattr(self, func_or_attr))
        all_parsers = [parser_func() for parser_func in all_parser_funcs]

        final_parser = argparse.ArgumentParser(parents=all_parsers)
        args = final_parser.parse_args()
        self.auto_set_name(args)
        return args

    def auto_set_name(self, args):
        def reuse_naming(args, output_name):
            crt = args.crt
            output_name.extend(["crt={}".format(crt)])
            if crt == "naive":
                output_name.extend(["k={}".format(args.crt_k)])
            elif crt == "random":
                output_name.extend(["p={}".format(args.crt_p)])
            elif crt == "gSAMsharp":
                output_name.extend(["crt_z={}".format(args.crt_z)])
            elif crt == "gSAMflat":
                output_name.extend(["crt_z={}".format(args.crt_z)])

        def sam_hyper_param(args):
            args_opt = args.opt.split("-")
            if len(args_opt) == 1:
                return []
            elif len(args_opt) == 2:
                sam_opt, _base_opt = args_opt[0], args_opt[1]
            # SAM, VASSO
            output_name = ["rho{}".format(args.rho)]
            if sam_opt[:5].upper() == "VASSO":
                output_name.extend(["theta={}".format(args.theta)])
            if sam_opt[:7].upper() == "VASSORE":
                reuse_naming(args, output_name)
            return output_name

        if args.output_name is None:
            args.output_name = "_".join(
                [
                    args.dataset,
                    # "bsz" + str(args.batch_size),
                    # "epoch" + str(args.epochs),
                    args.model,
                    # "lr" + str(args.lr),
                    str(args.opt),
                ]
                + sam_hyper_param(args)
                # + ["seed{}".format(args.seed)]
            )
