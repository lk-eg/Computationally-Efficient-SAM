import os


def dict_creation(
    name: str,
    dir: str,
    opt: str,
    seed: int,
    mem: str = "2G",
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
    dataset_nn_combination: str = "cifar100_wrn28-10_mass",
):
    d = {}
    d["name"] = name + "_" + str(seed)
    d["output_dir"] = dir
    d["memcpu"] = mem
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
    d["seed"] = seed
    d["dataset_nn_combination"] = dataset_nn_combination
    return d


seeds = [1234, 42, 87283, 913248]
baseline_opts = ["sgd", "sam-sgd", "vasso-sgd", "adamw"]
crt_opts = ["vassore-sgd", "vassoremu-sgd"]
crts = ["naive", "random", "schedule"]
ks = [2, 3, 5, 10, 20, 100]
ps = [0.5, 0.33, 0.2, 0.1, 0.05, 0.01]
ss_endblock = [
    "[100-200]",
    "[134-200]",
    "[160-200]",
    "[180-200]",
    "[190-200]",
    "[198-200]",
]


# Filling of experiment creation commands
def filling_out_experiment_commands() -> list:
    experiments = []

    for seed in seeds:
        for baseline_opt in baseline_opts:
            name = baseline_opt + "_baseline"
            if baseline_opt == "sgd":
                experiments.append(
                    dict_creation(
                        name=name, dir="baseline", opt=baseline_opt, seed=seed, w=5e-4
                    )
                )
                continue
            experiments.append(
                dict_creation(name=name, dir="baseline", opt=baseline_opt, seed=seed)
            )
        experiments.append(
            dict_creation(
                name="vasso_baseline_t=0.2",
                dir="baseline",
                opt="vasso-sgd",
                seed=seed,
                t=0.2,
            )
        )

    for seed in seeds:
        for crt_opt in crt_opts:
            for crt in crts:
                if crt == "naive":
                    for k in ks:
                        name = crt_opt + "_" + crt + "_" + "k=" + str(k)
                        experiments.append(
                            dict_creation(
                                name=name,
                                dir=crt,
                                opt=crt_opt,
                                seed=seed,
                                crt=crt,
                                crt_k=k,
                            )
                        )
                if crt == "random":
                    for p in ps:
                        name = crt_opt + "_" + crt + "_" + "p=" + str(p)
                        experiments.append(
                            dict_creation(
                                name=name,
                                dir=crt,
                                opt=crt_opt,
                                seed=seed,
                                crt=crt,
                                crt_p=p,
                            )
                        )
        if crt == "schedule":
            for s in ss_endblock:
                name = "vasso-sgd" + "_" + crt + "_" + "s=" + s
                experiments.append(
                    dict_creation(
                        name=name, dir=crt, opt="vasso_sgd", seed=seed, crt=crt, crt_s=s
                    )
                )

    return experiments


slurm_template = """#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --job-name={name}_c100_wrn2810
#SBATCH --mem-per-cpu={memcpu}
#SBATCH --output={output_dir}/outputs/{name}.out
#SBATCH --error={output_dir}/errors/{name}.err
#SBATCH --open-mode=truncate
#SBATCH --mail-type=END

module load eth_proxy
module load stack/2024-06
module load python_cuda/3.11.6
module load py-distro/1.8.0-4tnktx7

cd ~/sam/VaSSO

python3 train.py \
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
    --seed {seed} \
    --dataset_nn_combination {dataset_nn_combination}
"""

for experiment in filling_out_experiment_commands():
    script_content = slurm_template.format(**experiment)
    output_dir = experiment["output_dir"]
    os.makedirs(os.path.join(output_dir, "scripts"), exist_ok=True)
    script_filename = os.path.join(output_dir, "scripts", f"{experiment['name']}.sh")
    with open(script_filename, "w") as file:
        file.write(script_content)
