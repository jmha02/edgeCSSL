# EdgeCSSL
The code in this repository is borrowed from [CaSSLe](https://github.com/DonkeyShot21/cassle)

# Installation
Updated to support PyTorch 2.x versions
```
conda create --name cassle python=3.9
conda activate cassle
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pytorch-lightning>=2.0.0 lightning-bolts wandb scikit-learn einops
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
```

# Commands
Here below you can find a few example commands for running our code. The bash scripts with full training configurations for our continual and linear evaluation experiments can be found in the `bash_files` folder. Use our `job_launcher.py` to launch continual self-supervised learning experiments.

NOTE: each experiment uses a different number of gpus (1 for CIFAR100, 2 for ImageNet100 and 4 for DomainNet). You can change this setting directly in the bash scripts.

## Fine-tuning
#### Class-incremental
E.g. running SimCLR:
```
DATA_DIR=./data CUDA_VISIBLE_DEVICES=0 python job_launcher.py --script bash_files/continual/cifar/simclr.sh
```
E.g. running DINO:
```
DATA_DIR=./data CUDA_VISIBLE_DEVICES=0 python job_launcher.py --script bash_files/continual/cifar/dino.sh
```

## CaSSLe
After running fine-tuning, you can also run CaSSLe by just loading the checkpoint of the first task. You will find all the checkpoints in your experiment directory (defaults to `"./experiments"`). Check the id of your run on WandB to make sure you are loading the correct checkpoint.
### CIFAR100
E.g. running Barlow Twins + CaSSLe:
```
PRETRAINED_PATH=/path/to/task0/checkpoint/ DATA_DIR=/path/to/data/dir/ CUDA_VISIBLE_DEVICES=0 python job_launcher.py --script bash_files/continual/cifar/barlow_distill.sh
```

## Linear Evaluation
For linear evaluation you do not need the job launcher. You can simply run the scripts from `bash_files/linear`, e.g., for VICReg:
```
PRETRAINED_PATH=/path/to/last/checkpoint/ DATA_DIR=/path/to/data/dir/ bash bash_files/linear/imagenet-100/class/vicreg_linear.sh
```

# Logging
Logging is performed with [WandB](https://wandb.ai/site). Please create an account and specify your `--entity YOUR_ENTITY` and `--project YOUR_PROJECT` in the bash scripts. For debugging, or if you do not want all the perks of WandB, you can disable logging by passing `--offline` in your bash scripts. After training you can always sync an offline run with the following command: `wandb sync your/wandb/run/folder`.
