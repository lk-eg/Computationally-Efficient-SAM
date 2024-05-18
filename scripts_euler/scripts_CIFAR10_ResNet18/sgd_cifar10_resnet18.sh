#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=6:00:00
#SBATCH --job-name=sgd_cifar10_resnet18
#SBATCH --mem-per-cpu=4096
#SBATCH --output=sgd_cifar10_resnet18.out
#SBATCH --error=sgd_cifar10_resnet18.err
#SBATCH --open-mode=truncate
#SBATCH --mail-type=BEGIN,END

module load eth_proxy
module load gcc/8.2.0
module load python_gpu/3.8.5

python3 ../../train.py \
        --opt sgd \
        --weight_decay 5e-4 \
        --model resnet18 \
        --dataset CIFAR10_cutout \
        --datadir /cluster/home/laltun/datasets \
        --wandb \
        --wandb_project "AlgoComp - cifar10_resnet18" \
        --wandb_name sgd