#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --job-name="metrics_refined_1"
#SBATCH --mem-per-cpu=8000
#SBATCH --output="metrics_refined_1.txt"
#SBATCH --error="error_refined_1.txt"
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END

module load eth_proxy
module load gcc/8.2.0
module load python_gpu/3.8.5

python3 ../train.py --model resnet18 --dataset CIFAR10_cutout --datadir /cluster/home/laltun/datasets --opt vasso-sgd --rho 0.1 --theta 0.4 --weight_decay 1e-3 --wandb --wandb_project "VaSSO Studies Refined" --wandb_name metrics_1