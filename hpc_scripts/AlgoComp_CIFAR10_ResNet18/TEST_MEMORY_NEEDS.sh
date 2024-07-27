#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=6:00:00
#SBATCH --job-name=TEST_MEMORY_NEEDS
#SBATCH --mem-per-cpu=1G
#SBATCH --output=MEMORY_NEEDS.out
#SBATCH --error=MEMORY_NEEDS.err
#SBATCH --open-mode=truncate
#SBATCH --mail-type=END

module load eth_proxy
module load gcc/8.2.0
module load python_gpu/3.8.5

cd ~/sam/VaSSO

python3 train.py         --opt vassoremu-sgd         --theta 0.4         --weight_decay 0.001         --crt random         --crt_k 2         --crt_p 0.5         --crt_z 1.0
