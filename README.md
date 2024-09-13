## Computationally Efficient Sharpness-Aware Minimization


This repo implements the Bachelor thesis **Computationally Efficient Sharpnes-Aware Minimization** at the ODI group at ETH Zürich in PyTorch.
The code is built on top of [this repo](https://github.com/BingcongLi/VaSSO/tree/main).

SAM is an optimization algorithm that improves generalization ability significantly in deep neural networks. However, its runtime is roughly double that of
baseline optimizers such as SGD due to the computation of *two* gradients *per* iteration step.

This work improves SAM's runtime by employing decision rules at the granularity of iterations to decide whether to perform an entire backwardpass for the
computation of the perturbation \(\epsilon\), or not.

Runtime of VaSSORe(Mu) approaches that of SGD while generalizing significantly better.


### Getting started

- Clone this repo

```bash
git clone git@github.com:lk-eg/Computationally-Efficient-SAM.git
cd Computationally-Efficient-SAM
```


- Create conda environment

```bash
conda create -n comp-eff-sam python=3.8 -y
conda activate comp-eff-sam
```

- Install the other dependencies

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

You might have to install further dependencies.

- Dataset preparation

    CIFAR10, CIFAR100 and ImageNet are adopted.

    CIFAR datasets are built-in, and ImageNet can be found [here](http://image-net.org/).

### Useful config

The configs can be found in `configs/default_cfg.py`. Some important ones are listed below.

- `--opt`. The optimizer to use, e.g., `--sgd` for SGD, `--sam-adamw` for SAM with base optimizer AdamW, `vassoremu-sgd` for VaSSOReMu with base optimizer SGD.
- `--dataset`. The dataset for training, choices = [`CIFAR10_cutout`, `CIFAR100_cutout`, `ImageNet_base`].
- `--model`. The model to be trained, e.g., `resnet18`, `wideresnet28x10`.
- `--rho`. The perturbing radius for SAM, e.g., `--rho 0.1`.
- `--theta`. The hyperparameter of moving average for VaSSO, e.g., `--theta 0.9`.
- `--crt_c`. The threshold parameter for the cosine similarity based dynamic criterion, e.g., `--crt_c 0`.
- `--lam`. EMA hyperparameter for `gSAMflat`, `gSAMsharp`, `gSAMratio`, e.g., `--lam 0.1`.
- `--crt_z`. outer gradient norm deviation threshold, e.g., `--crt_z 1.5`.

### Training

Run training from the command line. For example, if you hope to train a Resnet18 on CIFAR10 with VaSSORe-SGD:

```bash
python train.py --model resnet18 --dataset CIFAR10_cutout --datadir [your path to dataset] --opt vassore-sgd --lr 0.05 --weight_decay 1e-3 --rho 0.1 --theta 0.4 --crt gSAMsharp --lam 0.2 --crt_z 1.5 --seed [how about 3107]
```
