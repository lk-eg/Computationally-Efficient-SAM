#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=6:00:00
#SBATCH --job-name=vasso_theta=1e-4_tuning
#SBATCH --mem-per-cpu=16384
#SBATCH --output=vasso_theta=1e-4_tuning.out
#SBATCH --error=vasso_theta=1e-4_tuning.err
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END

module load eth_proxy
module load gcc/8.2.0
module load python_gpu/3.8.5

python3 ../../train.py \
        --opt vasso-sgd \
        --weight_decay 1e-3 \
        --theta 1e-4 \
        --rho 0.1 \
        --model resnet18 \
        --dataset CIFAR10_cutout \
        --datadir /cluster/home/laltun/datasets \
        --wandb \
        --wandb_project "VaSSO Theta Tuning" \
        --wandb_name vasso_theta=1e-4