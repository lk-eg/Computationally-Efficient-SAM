#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=8

module load eth_proxy
module load stack/2024-06
module load python_cuda/3.11.6
module load py-distro/1.8.0-4tnktx7

cd ~/sam/VaSSO

python3 ../../train.py \
        --opt sam-sgd \
        --weight_decay 1e-3 \
        --rho 0.1 \
        --model resnet18 \
        --dataset CIFAR10_cutout \
        --datadir /cluster/home/laltun/datasets \
        --epochs 1 \
        --dataset_nn_combination "cifar10_rn18_hessian_normal"
