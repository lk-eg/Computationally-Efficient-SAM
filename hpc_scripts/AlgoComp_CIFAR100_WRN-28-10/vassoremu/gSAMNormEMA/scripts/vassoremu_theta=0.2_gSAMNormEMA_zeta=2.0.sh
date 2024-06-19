#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --gpus=quadro_rtx_6000:1
#SBATCH --time=10:00:00
#SBATCH --job-name=vassoremu_theta=0.2_gSAMNormEMA_zeta=2.0_cifar100_wrn-28-10
#SBATCH --mem-per-cpu=2G
#SBATCH --output=vassoremu/gSAMNormEMA/outputs/vassoremu_theta=0.2_gSAMNormEMA_zeta=2.0.out
#SBATCH --error=vassoremu/gSAMNormEMA/errors/vassoremu_theta=0.2_gSAMNormEMA_zeta=2.0.err
#SBATCH --open-mode=truncate
#SBATCH --mail-type=END

module load eth_proxy
module load gcc/8.2.0
module load python_gpu/3.8.5

cd ~/sam/VaSSO

python3 train.py         --dataset CIFAR100_cutout         --model wideresnet28x10         --rho 0.2         --opt vassoremu-sgd         --theta 0.2         --weight_decay 0.001         --crt gSAMNormEMA         --crt_k 2         --crt_p 0.5         --zeta 2.0
