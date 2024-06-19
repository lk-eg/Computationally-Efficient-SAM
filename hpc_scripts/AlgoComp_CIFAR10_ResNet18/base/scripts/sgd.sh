#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=6:00:00
#SBATCH --job-name=sgd_cifar10_resnet18
#SBATCH --mem-per-cpu=4G
#SBATCH --output=base/outputs/sgd.out
#SBATCH --error=base/errors/sgd.err
#SBATCH --open-mode=truncate
#SBATCH --mail-type=END

module load eth_proxy
module load gcc/8.2.0
module load python_gpu/3.8.5

cd ~/sam/VaSSO

python3 train.py         --opt sgd         --theta 0.2         --weight_decay 0.0005         --crt naive         --crt_k 2         --crt_p 0.5         --zeta 1.0
