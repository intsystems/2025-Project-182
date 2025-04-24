#!/bin/bash

export CUDA_VISIBLE_DEVICES=4  # Указываем, какие GPU использовать
# export WANDB_DISABLED="false"        # Включаем логирование в Weights & Biases
# export WANDB_PROJECT="zo-llm-ft"    # Название проекта в W&B
# export WANDB_API_KEY=$(cat ~/.wandb_api_key)
# export OPENBLAS_NUM_THREADS=1


# CUDA_VISIBLE_DEVICES=0 - Devise#2 в nvtop
# CUDA_VISIBLE_DEVICES=1 - Devise#3 в nvtop
# CUDA_VISIBLE_DEVICES=2 - Devise#4 в nvtop
# CUDA_VISIBLE_DEVICES=3 - Devise#5 в nvtop
# CUDA_VISIBLE_DEVICES=4 - Devise#0 в nvtop
# CUDA_VISIBLE_DEVICES=5 - Devise#1 в nvtop
# CUDA_VISIBLE_DEVICES=6 - Devise#6 в nvtop
# CUDA_VISIBLE_DEVICES=7 - Devise#7 в nvtop


python get_loss_values_transformer.py --config_path /home/moderntalker/opt_projects/Landscape/landscape-hessian/code/configs/configs_direct/mnist.yml \
                                      --save_path /home/moderntalker/opt_projects/Landscape/landscape-hessian/code/results/results_transformer \
                                      --num_epochs 3