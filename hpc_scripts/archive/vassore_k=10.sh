#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=8:00:00
#SBATCH --job-name="vassore_k=10"
#SBATCH --mem-per-cpu=16384
#SBATCH --output="vassore_k=10.txt"
#SBATCH --error="vassore_k=10.txt"
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END

module load eth_proxy
module load gcc/8.2.0
module load python_gpu/3.8.5

python3 ../train.py --model resnet18 --dataset CIFAR10_cutout --datadir /cluster/home/laltun/datasets --opt vassore-sgd --rho 0.1 --theta 0.2 --k 10 --weight_decay 1e-3 --wandb --wandb_project "VaSSO - Next Stage" --wandb_name vassore_k=10