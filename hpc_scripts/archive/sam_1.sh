#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=8:00:00
#SBATCH --job-name="sam_1"
#SBATCH --mem-per-cpu=16384
#SBATCH --output="sam_1.txt"
#SBATCH --error="sam_1.txt"
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END

module load eth_proxy
module load gcc/8.2.0
module load python_gpu/3.8.5

python3 ../train.py --model resnet18 --dataset CIFAR10_cutout --datadir /cluster/home/laltun/datasets --opt sam-sgd --rho 0.1 --weight_decay 1e-3 --wandb --wandb_project "VaSSO - Next Stage" --wandb_name sam_1