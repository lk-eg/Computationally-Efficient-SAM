#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=10:00:00
#SBATCH --job-name=vasso_theta=0.2_cifar100_wrn-28-10
#SBATCH --mem-per-cpu=2G
#SBATCH --output=base/outputs/vasso_theta=0.2.out
#SBATCH --error=base/errors/vasso_theta=0.2.err
#SBATCH --open-mode=truncate
#SBATCH --mail-type=END

module load eth_proxy
module load gcc/8.2.0
module load python_gpu/3.8.5

cd ~/sam/VaSSO

python3 train.py         --dataset CIFAR100_cutout         --model wideresnet28x10         --rho 0.2         --opt vasso-sgd         --theta 0.2         --weight_decay 0.001         --crt naive         --crt_k 2         --crt_p 0.5         --zeta 1.0
