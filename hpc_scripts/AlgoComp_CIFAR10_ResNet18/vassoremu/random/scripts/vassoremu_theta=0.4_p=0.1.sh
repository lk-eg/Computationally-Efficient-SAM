#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=6:00:00
#SBATCH --job-name=vassoremu_theta=0.4_p=0.1_cifar10_resnet18
#SBATCH --mem-per-cpu=4G
#SBATCH --output=vassoremu/random/outputs/vassoremu_theta=0.4_p=0.1.out
#SBATCH --error=vassoremu/random/errors/vassoremu_theta=0.4_p=0.1.err
#SBATCH --open-mode=truncate
#SBATCH --mail-type=END

module load eth_proxy
module load gcc/8.2.0
module load python_gpu/3.8.5

cd ~/sam/VaSSO

python3 train.py         --opt vassoremu-sgd         --theta 0.4         --weight_decay 0.001         --crt random         --crt_k 2         --crt_p 0.1         --zeta 1.0
