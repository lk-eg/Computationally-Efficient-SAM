import os


def dict_creation(
    name: str,
    opt: str,
    seed: int,
    dir: str = "cosSim",
    mem: str = "2G",
    dataset: str = "CIFAR10_cutout",
    model: str = "resnet18",
    rho: float = 0.1,
    t: float = 0.4,
    w: float = 1e-3,
    crt: str = "cosSim",
    c: float = 0,
    dataset_nn_combination: str = "cifar10_rn18_cosSim",
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
    d["c"] = c
    d["seed"] = seed
    d["dataset_nn_combination"] = dataset_nn_combination
    return d


seeds = [3107, 1234, 42, 87283, 913248]
crt_opts = ["vassoremu-sgd"]
cs = [0]


def cosSim_experiments() -> list:
    experiments = []
    for crt_opt in crt_opts:
        for c in cs:
            for seed in seeds:
                name = f"cosSim={c}"
                experiments.append(
                    dict_creation(name=name, opt=crt_opt, seed=seed, c=c)
                )
    return experiments


slurm_template = """#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --job-name={name}_c10_rn18
#SBATCH --mem-per-cpu={memcpu}
#SBATCH --output={output_dir}/outputs/{name}.out
#SBATCH --error={output_dir}/errors/{name}.err
#SBATCH --open-mode=truncate

module load eth_proxy
module load stack/2024-06
module load python_cuda/3.11.6
module load py-distro/1.8.0-4tnktx7

source ~/myenv/bin/activate

cd ~/sam/VaSSO

python3 train.py \
    --dataset {dataset} \
    --model {model} \
    --opt {opt} \
    --rho {rho} \
    --theta {theta} \
    --crt {crt} \
    --crt_c {c} \
    --seed {seed} \
    --dataset_nn_combination {dataset_nn_combination} \
"""

for experiment in cosSim_experiments():
    script_content = slurm_template.format(**experiment)
    output_dir = experiment["output_dir"]
    os.makedirs(os.path.join(output_dir, "scripts"), exist_ok=True)
    script_filename = os.path.join(output_dir, "scripts", f"{experiment['name']}.sh")
    with open(script_filename, "w") as file:
        file.write(script_content)
