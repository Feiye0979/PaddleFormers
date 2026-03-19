#!/bin/bash

# 调试模式的启动命令，添加 Python 调试标志

cd /root/paddlejob/workspace/env_run/wuhuiyue_new/qwen3_omni/PaddleFormers

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/usr/local/lib:/home/opt/nvidia_lib:/usr/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu/

# 添加 Python 调试标志
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export PYTHONVERBOSE=1

python -v -m paddle.distributed.launch \
    --log_dir "./log-pdformers" \
    paddleformers/cli/launcher.py \
    train \
    configs/sft_vl_patch_multi_card.yaml 2>&1 | tee debug_output.log
