#!/bin/bash

export CUDA_VISIBLE_DEVICES=3  # Указываем, какие GPU использовать
# export WANDB_DISABLED="false"        # Включаем логирование в Weights & Biases
# export WANDB_PROJECT="zo-llm-ft"    # Название проекта в W&B
# export WANDB_API_KEY=$(cat ~/.wandb_api_key)


python plot_differences_transformer.py --config_path /home/moderntalker/opt_projects/Landscape/landscape-hessian/code/configs/configs_plot/config_plot_transformer.yml