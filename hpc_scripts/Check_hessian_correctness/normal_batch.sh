#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1

module load eth_proxy
module load gcc/8.2.0
module load python_gpu/3.8.5

python3 ../../train.py \
        --opt sam-sgd \
        --weight_decay 1e-3 \
        --rho 0.1 \
        --model resnet18 \
        --dataset CIFAR10_cutout \
        --datadir /cluster/home/laltun/datasets \
        --epochs 1 \
        --dataset_nn_combination "cifar10_rn18_hessian_normal"
