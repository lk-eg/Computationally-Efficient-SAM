#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=2:00:00

module load eth_proxy
module load stack/2024-06
module load python_cuda/3.11.6
module load py-distro/1.8.0-4tnktx7

source ~/myenv/bin/activate

cd ~/sam/VaSSO

python3 train.py \
    --dataset TinyImageNet \
    --model vit \
    --opt vasso-adamw \
    --rho 0.05 \
    --theta 0.4 \
    --weight_decay 0.3 \
    --batch_size 4096 \
    --epochs 3 \
    --betas 0.9998 \
    --dataset_nn_combination test_vit
