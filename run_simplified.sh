#!/bin/bash

# 简化的启动命令，使用正确的 NCCL 库路径

cd /root/paddlejob/workspace/env_run/wuhuiyue_new/qwen3_omni/PaddleFormers

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 使用正确的 NCCL 库路径（从 PaddlePaddle 环境中）
export LD_LIBRARY_PATH=/root/miniconda3/envs/why_pd/lib/python3.10/site-packages/nvidia/nccl/lib:/root/miniconda3/envs/why_pd/lib/python3.10/site-packages/paddle/../nvidia/nccl/lib:$LD_LIBRARY_PATH

python -m paddle.distributed.launch \
    --log_dir "./log-pdformers" \
    paddleformers/cli/launcher.py \
    train \
    configs/sft_vl_patch_multi_card.yaml
