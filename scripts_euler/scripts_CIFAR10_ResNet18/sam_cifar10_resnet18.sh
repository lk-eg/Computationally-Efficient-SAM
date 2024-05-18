#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=6:00:00
#SBATCH --job-name=sam_cifar10_resnet18
#SBATCH --mem-per-cpu=4096
#SBATCH --output=sam_cifar10_resnet18.out
#SBATCH --error=sam_cifar10_resnet18.err
#SBATCH --open-mode=truncate
#SBATCH --mail-type=BEGIN,END

module load eth_proxy
module load gcc/8.2.0
module load python_gpu/3.8.5

python3 ../../train.py \
        --opt sam-sgd \
        --weight_decay 1e-3 \
        --model resnet18 \
        --dataset CIFAR10_cutout \
        --datadir /cluster/home/laltun/datasets \
        --wandb \
        --wandb_project "AlgoComp - cifar10_resnet18" \
        --wandb_name sam