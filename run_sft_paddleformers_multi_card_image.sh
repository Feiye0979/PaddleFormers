# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

MODAL_TYPE=image
LOG_PATH=log-pdformers-$MODAL_TYPE-stage2
rm -rf saved_tensors/npy/* $LOG_PATH/* core.* checkpoints/*
sh prepare_env.sh

# LD_LIBRARY_PATH=/usr/local/lib:/home/opt/nvidia_lib:/usr/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu/ \
# FLAGS_use_accuracy_compatible_kernel=1 \
# FLAGS_embedding_deterministic=1 \
# FLAGS_cudnn_deterministic=1 \
# FLAGS_share_tensor_for_grad_tensor_holder=1 \
# FLAGS_enable_dataset_debug=false \
# SKIP_TRAINING=0 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
NCCL_ASYNC_ERROR_HANDLING=1 \
PADDLE_AOA_SLOW_BATCH_SIZE=2048 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m paddle.distributed.launch \
    --log_dir "./$LOG_PATH" \
    /root/paddlejob/workspace/env_run/wuhuiyue_new/qwen3_omni/PaddleFormers/paddleformers/cli/launcher.py \
    train \
    /root/paddlejob/workspace/env_run/wuhuiyue_new/qwen3_omni/PaddleFormers/configs/sft_vl_patch_multi_card_$MODAL_TYPE.yaml
