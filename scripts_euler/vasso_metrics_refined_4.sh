#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --gpus=v100:1
#SBATCH --time=8:00:00
#SBATCH --job-name="metrics_refined_4"
#SBATCH --mem-per-cpu=8000
#SBATCH --output="metrics_refined_4.txt"
#SBATCH --error="error_refined_4.txt"
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END

module load eth_proxy
module load gcc/8.2.0
module load python_gpu/3.8.5

python3 ../train.py --model wideresnet28x10 --dataset CIFAR10_cutout --datadir /cluster/home/laltun/datasets --opt vasso-sgd --rho 0.1 --theta 0.2 --weight_decay 1e-3 --wandb --wandb_project "VaSSO - Next Stage" --wandb_name vasso_metrics_4