#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --job-name="metrics_1"
#SBATCH --mem-per-cpu=16384
#SBATCH --output="metrics_1.txt"
#SBATCH --error="error_1.txt"
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END

module load eth_proxy
module load gcc/8.2.0
module load python_gpu/3.8.5

python3 ../train.py --model resnet18 --dataset CIFAR10_cutout --datadir /cluster/home/laltun/datasets --opt vasso-sgd --rho 0.1 --theta 0.4 --weight_decay 1e-3 --wandb --wandb_project "VaSSO Studies" --wandb_name metrics_1