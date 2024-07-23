def dict_creation(
    opt: str,
    dataset: str = "CIFAR100_cutout",
    model: str = "wideresnet28x10",
    rho: float = 0.2,
    t: float = 0.4,
    w: float = 1e-3,
    crt: str = "baseline",
    crt_k: int = 2,
    crt_p: float = 0.5,
    crt_z: float = 1.0,
    crt_s: str = "[100-200]",
    epochs: int = 200,
    dataset_nn_combination: str = "cifar100_wrn28-10",
):
    d = {}
    d["dataset"] = dataset
    d["model"] = model
    d["opt"] = opt
    d["rho"] = rho
    d["theta"] = t
    d["weight_decay"] = w
    d["crt"] = crt
    d["k"] = crt_k
    d["p"] = crt_p
    d["z"] = crt_z
    d["s"] = crt_s
    d["epochs"] = epochs
    d["dataset_nn_combination"] = dataset_nn_combination
    return d


baseline_opts = ["sgd", "sam", "vasso", "adamw"]
crt_opts = ["vassore", "vassoremu"]
crts = ["naive", "random", "schedule"]
ks = [2, 5, 10, 20, 100]
ps = [0.5, 0.2, 0.1, 0.05, 0.01]
ss_endblock = ["[100-200]", "[134-200]", "[160-200]", "[180-200]", "[198-200]"]


# Filling of experiment creation commands
def filling_out_experiment_commands():
    experiments = []

    for baseline_opt in baseline_opts:
        experiments.append(dict_creation(baseline_opt))
        experiments.append(dict_creation("vasso", t=0.2))

    for crt_opt in crt_opts:
        for crt in crts:
            if crt == "naive":
                for k in ks:
                    experiments.append(dict_creation(crt_opt, crt=crt, crt_k=k))
            if crt == "random":
                for p in ps:
                    experiments.append(dict_creation(crt_opt, crt=crt, crt_p=p))
            if crt == "schedule":
                for s in ss_endblock:
                    experiments.append(dict_creation(crt_opt, crt=crt, crt_s=s))


command = """
    --dataset {dataset} \
    --model {model} \
    --opt {opt} \
    --rho {rho} \
    --theta {theta} \
    --weight_decay {weight_decay} \
    --crt {crt} \
    --crt_k {k} \
    --crt_p {p} \
    --crt_z {z} \
    --crt_s {s} \
    --epochs {epochs} \
    --dataset_nn_combination {dataset_nn_combination} \
    --exclusive_run
"""


def benchmarking_experiments():
    script_commands = []
    for experiment in filling_out_experiment_commands():
        command_content = command.format(**experiment)
        script_commands.append(command_content)
