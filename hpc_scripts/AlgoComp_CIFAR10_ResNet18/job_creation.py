import os


def dict_creation(
    name: str,
    opt: str,
    dir: str,
    gpu: str,
    mem: str,
    t: float = 0.2,
    w: float = 1e-3,
    crt: str = "naive",
    crt_k: int = 2,
    crt_p: float = 0.5,
    zeta: float = 1.0,
):
    d = {}
    d["name"] = name
    d["opt"] = opt
    d["output_dir"] = dir
    d["gpu_model"] = gpu
    d["memcpu"] = mem
    d["theta"] = t
    d["weight_decay"] = w
    d["crt"] = crt
    d["k"] = crt_k
    d["p"] = crt_p
    d["zeta"] = zeta
    return d


opt_prefixes = ["vasso", "vassore", "vassoremu"]
crts = ["naive", "random", "gSAMNormEMA", "gSAMNormEMA"]
thetas = [0.2, 0.4]
ks = [2, 5, 10, 20]
ps = [0.5, 0.2, 0.1, 0.05]
zetas = [2.0, 1.0, 0.5, 0.1, 1e-2]

experiments = []
experiments.append(dict_creation("sam", "sam-sgd", "base", "rtx_3090", "4G"))
experiments.append(dict_creation("sgd", "sgd", "base", "rtx_3090", "4G", w=5e-4))

for opt in opt_prefixes:
    for theta in thetas:
        base_name = name = opt + "_theta=" + str(theta)
        if opt == "vasso":
            gpu = "rtx_3090"
            optm = opt + "-sgd"
            experiments.append(
                dict_creation(base_name, optm, "base", gpu, "6G", t=theta)
            )
        else:
            optm = opt + "-sgd"
            for crt in crts:
                if crt == "naive":
                    for k in ks:
                        full_name = base_name + "_k=" + str(k)
                        gpu = "rtx_4090"
                        experiments.append(
                            dict_creation(
                                full_name,
                                optm,
                                os.path.join(opt, crt),
                                gpu,
                                "4G",
                                theta,
                                crt=crt,
                                crt_k=k,
                            )
                        )
                if crt == "random":
                    for p in ps:
                        full_name = base_name + "_p=" + str(p)
                        gpu = "rtx_4090"
                        experiments.append(
                            dict_creation(
                                full_name,
                                optm,
                                os.path.join(opt, crt),
                                gpu,
                                "4G",
                                theta,
                                crt=crt,
                                crt_p=p,
                            )
                        )
                else:
                    for z in zetas:
                        full_name_normal = base_name + "_gSAMNormEMA_zeta=" + str(z)
                        full_name_inv = (
                            base_name + "_gSAMNormEMAInverted_zeta=" + str(z)
                        )
                        os_path_normal = os.path.join(opt, "gSAMNormEMA")
                        os_path_inv = os.path.join(opt, "gSAMNorEMAInverted")
                        experiments.append(
                            dict_creation(
                                full_name_normal,
                                optm,
                                os_path_normal,
                                "v100",
                                "4G",
                                theta,
                                crt="gSAMNormEMA",
                                zeta=z,
                            )
                        )
                        experiments.append(
                            dict_creation(
                                full_name_inv,
                                optm,
                                os_path_inv,
                                "v100",
                                "4G",
                                theta,
                                crt="gSAMNormEMAInverted",
                                zeta=z,
                            )
                        )


slurm_template = """#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --gpus={gpu_model}:1
#SBATCH --time=6:00:00
#SBATCH --job-name={name}_cifar10_resnet18
#SBATCH --mem-per-cpu={memcpu}
#SBATCH --output={output_dir}/outputs/{name}.out
#SBATCH --error={output_dir}/errors/{name}.err
#SBATCH --open-mode=truncate
#SBATCH --mail-type=END

module load eth_proxy
module load gcc/8.2.0
module load python_gpu/3.8.5

cd ~/sam/VaSSO

python3 train.py \
        --opt {opt} \
        --theta {theta} \
        --weight_decay {weight_decay} \
        --crt {crt} \
        --crt_k {k} \
        --crt_p {p} \
        --zeta {zeta}
"""

for experiment in experiments:
    script_content = slurm_template.format(**experiment)
    output_dir = experiment["output_dir"]
    os.makedirs(os.path.join(output_dir, "scripts"), exist_ok=True)
    script_filename = os.path.join(output_dir, "scripts", f"{experiment['name']}.sh")
    with open(script_filename, "w") as file:
        file.write(script_content)
