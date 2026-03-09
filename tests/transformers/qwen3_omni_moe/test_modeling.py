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
from __future__ import annotations

import os

os.environ["FLAGS_use_accuracy_compatible_kernel"] = "1"
os.environ["FLAGS_embedding_deterministic"] = "1"
os.environ["FLAGS_cudnn_deterministic"] = "1"
# os.environ["FLAGS_gemm_use_half_precision_compute_type"] = "1"
# os.environ["FLAGS_share_tensor_for_grad_tensor_holder"] = "1"

import paddle

paddle.set_printoptions(precision=15)

import torch

torch.set_printoptions(
    precision=15,  # 小数位数
    sci_mode=False,  # 关闭科学计数法
    linewidth=200,  # 每行字符数，防止被截断
    threshold=10_000,  # 超过这个元素数才省略
)

mock_by_torch_logits_grad = False


import glob
import hashlib
import random
import sys
import traceback

import numpy as np

from paddleformers.transformers import (
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3VLMoeConfig,
    Qwen3VLMoeForConditionalGenerationDeprecated,
)

FILE_DIR = "/root/paddlejob/workspace/env_run/wuhuiyue_new/qwen3_omni/PaddleFormers/saved_tensors/npy/"
HACK_FILE_DIR = "/root/paddlejob/workspace/env_run/wuhuiyue_new/qwen3_omni/ms-swift/saved_tensors/npy/"


def compare_and_save(data, name: str, to_save: bool = False, print_tensor: bool = False):
    if print_tensor:
        print(name, type(data), data.shape if data is not None else None, data)
    try:
        if isinstance(data, dict):
            if to_save:
                os.makedirs(FILE_DIR, exist_ok=True)
                np.savez(os.path.join(FILE_DIR, f"{name}.npz"), **data)
        else:
            if isinstance(data, paddle.Tensor):
                data_float = data.astype("float32")
            else:
                data_float = data.float().contiguous()
            data_np = data_float.detach().cpu().numpy()
            array_bytes = data_np.tobytes()
            data_md5 = hashlib.md5(array_bytes).hexdigest()
            print(f"{name} md5: {data_md5}, shape: {data.shape if data is not None else None}, device: {data.device}")
            if to_save:
                os.makedirs(FILE_DIR, exist_ok=True)
                file = FILE_DIR + name + ".npy"
                np.save(file, data_np)
    except:
        print(name, traceback.format_exc())


def hack_with_torch_file(name: str, dtype, device):
    file = HACK_FILE_DIR + name + ".npy"
    np_data = np.load(file)
    pd_data = paddle.to_tensor(np_data).astype(dtype=dtype).to(device)
    return pd_data


MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Omni-30B-A3B-Instruct/"
# MODEL_PATH = "/root/paddlejob/workspace/env_run/wuhuiyue/qwen3_omni/models/customQwen3-Omni-30B-A3B-Instruct/"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)

global_rng = random.Random(SEED)


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return paddle.to_tensor(values, dtype="int64").cuda().view(shape).contiguous()


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return paddle.to_tensor(values, dtype="float32").cuda().view(shape).contiguous()


def get_deterministic_inputs(config, float32_switch=False):
    """生成确定性的多模态输入，用于精度对齐"""

    target_dtype = "float32" if float32_switch else "bfloat16"
    batch_size = 1
    seq_length = 512  # 使用较小的序列长度便于调试
    # vocab_size = config.get_text_config().vocab_size
    patch_size = config.vision_config.patch_size
    spatial_merge_size = config.vision_config.spatial_merge_size
    temporal_patch_size = config.vision_config.temporal_patch_size
    num_channels = 3
    num_mel_bins = 128

    # 图像参数
    image_row_size = 28  # 较小的图像便于调试
    image_col_size = 28
    num_image_tokens = (image_row_size * image_col_size) // (spatial_merge_size**2)

    # 视频参数
    video_temporal = 2
    video_row_size = 14
    video_col_size = 14
    num_video_tokens = (video_temporal * video_row_size * video_col_size) // (spatial_merge_size**2)

    # 音频参数
    feat_seq_length = 200
    input_lengths_leave = feat_seq_length % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    num_audio_tokens = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (feat_seq_length // 100) * 13

    # ============ 生成确定性输入 ============

    # 1. input_ids: 使用固定的 token 序列
    # 前面放多模态 token，后面放固定的文本 token
    input_ids = paddle.full([batch_size, seq_length], fill_value=1000, dtype="int64").cuda()

    # 设置多模态占位符
    pos = 0
    input_ids[0, pos : pos + num_image_tokens] = config.image_token_id
    pos += num_image_tokens
    input_ids[0, pos : pos + num_video_tokens] = config.video_token_id
    pos += num_video_tokens
    input_ids[0, pos : pos + num_audio_tokens] = config.audio_token_id
    pos += num_audio_tokens

    # 剩余位置用递增的 token id 填充（避免特殊 token）
    remaining_length = seq_length - pos
    input_ids[0, pos:] = paddle.arange(100, 100 + remaining_length, dtype="int64")

    # 2. attention_mask: 全 1
    attention_mask = paddle.ones([batch_size, seq_length], dtype="int64").to(input_ids.place)

    # 3. pixel_values: 使用归一化的线性序列
    image_seq_len = batch_size * (image_row_size * image_col_size)
    image_feat_dim = num_channels * (patch_size**2) * temporal_patch_size
    pixel_values = (
        paddle.linspace(0.0, 1.0, image_seq_len * image_feat_dim)
        .reshape([image_seq_len, image_feat_dim])
        .astype(target_dtype)
        .to(input_ids.place)
    )

    # 4. image_grid_thw
    pixel_grid_thw = paddle.to_tensor([[1, image_row_size, image_col_size]] * batch_size, dtype="int64").to(
        input_ids.place
    )

    # 5. pixel_values_videos: 使用归一化的线性序列
    video_seq_len = batch_size * (video_temporal * video_row_size * video_col_size)
    video_feat_dim = num_channels * (patch_size**2) * temporal_patch_size
    pixel_values_videos = (
        paddle.linspace(0.0, 1.0, video_seq_len * video_feat_dim)
        .reshape([video_seq_len, video_feat_dim])
        .astype(target_dtype)
        .to(input_ids.place)
    )

    # 6. video_grid_thw
    video_grid_thw = paddle.to_tensor(
        [[video_temporal, video_row_size, video_col_size]] * batch_size, dtype="int64"
    ).to(input_ids.place)

    # 7. input_features: 音频 mel-spectrogram，使用正弦波模拟
    # t = paddle.linspace(0, 2 * np.pi, feat_seq_length)
    # freq = paddle.linspace(1, num_mel_bins, num_mel_bins).unsqueeze(1)  # [num_mel_bins, 1]
    # input_features = paddle.sin(freq * t).unsqueeze(0).astype(target_dtype).to(input_ids.place)  # [1, num_mel_bins, feat_seq_length]

    t = np.linspace(0, 2 * np.pi, feat_seq_length).astype(np.float32)
    freq = np.linspace(1, num_mel_bins, num_mel_bins, dtype=np.float32).reshape(-1, 1)
    input_features = np.sin(freq * t)[np.newaxis, ...]  # [1, num_mel_bins, feat_seq_length]
    input_features = paddle.to_tensor(input_features).astype(target_dtype).to(input_ids.place)
    # 8. feature_attention_mask: 全 1
    feature_attention_mask = paddle.ones([batch_size, feat_seq_length], dtype="int64").to(input_ids.place)

    inputs_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": pixel_grid_thw,
        "pixel_values_videos": pixel_values_videos,
        "video_grid_thw": video_grid_thw,
        "input_features": input_features,
        "feature_attention_mask": feature_attention_mask,
        "labels": input_ids.clone(),
    }

    # 打印输入信息
    print("=" * 60)
    print("Deterministic Input Summary:")
    print("=" * 60)
    print(f"num_image_tokens: {num_image_tokens}")
    print(f"num_video_tokens: {num_video_tokens}")
    print(f"num_audio_tokens: {num_audio_tokens}")
    for key, value in inputs_dict.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    #   compare_and_save(value, key, False, False)
    print("=" * 60)

    return inputs_dict


def get_random_inputs(config):
    batch_size = 1
    seq_length = 2048
    vocab_size = config.get_text_config().vocab_size
    patch_size = config.vision_config.patch_size
    spatial_merge_size = config.vision_config.spatial_merge_size
    image_row_size = 56
    image_col_size = 56
    num_channels = 3
    temporal_patch_size = config.vision_config.temporal_patch_size
    num_mel_bins = 128
    feat_seq_length = 290

    # calculate image tokens
    num_image_tokens = (image_row_size * image_col_size) // (spatial_merge_size**2)

    # calculate image tokens
    video_temporal = 4
    video_row_size = 28
    video_col_size = 28
    num_video_tokens = (video_temporal * video_row_size * video_col_size) // (spatial_merge_size**2)

    # calculate audio tokens (the same to _get_feat_extract_output_lengths)
    input_lengths_leave = feat_seq_length % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    num_audio_tokens = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (feat_seq_length // 100) * 13

    input_ids = ids_tensor([batch_size, seq_length], vocab_size - 3) + 3
    # set the num_image_tokens position to image_token_id in order to match pixel_values
    input_ids[0, :num_image_tokens] = config.image_token_id
    # set the num_video_tokens position to video_token_id in order to match pixel_values_videos
    input_ids[0, num_image_tokens : num_image_tokens + num_video_tokens] = config.video_token_id
    # set the num_audio_tokens position to audio_token_id in order to match input_features
    input_ids[
        0, num_image_tokens + num_video_tokens : num_image_tokens + num_video_tokens + num_audio_tokens
    ] = config.audio_token_id

    print("====== multimodal tokens confirm ======")
    print(f"image_token_id: {config.image_token_id}")
    print(f"video_token_id: {config.video_token_id}")
    print(f"audio_token_id: {config.audio_token_id}")
    print(f"num_image_tokens (expected): {num_image_tokens}")
    print(f"num_video_tokens (expected): {num_video_tokens}")
    print(f"num_audio_tokens (expected): {num_audio_tokens}")
    print(f"image_tokens in input_ids: {(input_ids == config.image_token_id).sum().item()}")
    print(f"video_tokens in input_ids: {(input_ids == config.video_token_id).sum().item()}")
    print(f"audio_tokens in input_ids: {(input_ids == config.audio_token_id).sum().item()}")
    attention_mask = paddle.ones(input_ids.shape, dtype="int64").to(input_ids.place)

    # image data: pixel_values and image_grid_thw
    pixel_values = floats_tensor(
        [
            batch_size * (image_row_size * image_col_size),
            num_channels * (patch_size**2) * temporal_patch_size,
        ]
    ).to(input_ids.place)
    pixel_grid_thw = paddle.to_tensor(
        [[1, image_row_size, image_col_size]] * batch_size, dtype="int64", place=input_ids.place
    )

    # video data: pixel_values_videos and video_grid_thw
    # differ from image with temporal > 1
    pixel_values_videos = floats_tensor(
        [
            batch_size * (video_temporal * video_row_size * video_col_size),
            num_channels * (patch_size**2) * temporal_patch_size,
        ]
    ).to(input_ids.place)
    video_grid_thw = paddle.to_tensor(
        [[video_temporal, video_row_size, video_col_size]] * batch_size, dtype="int64", place=input_ids.place
    )

    # audio data: input_features and feature_attention_mask
    input_features_values = floats_tensor([batch_size, num_mel_bins, feat_seq_length]).to(input_ids.place)
    feature_attention_mask = paddle.ones([batch_size, feat_seq_length], dtype="int64").to(input_ids.place)

    inputs_dict = {
        "input_features": input_features_values,
        "feature_attention_mask": feature_attention_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_grid_thw": pixel_grid_thw,
        "pixel_values": pixel_values,
        "pixel_values_videos": pixel_values_videos,
        "video_grid_thw": video_grid_thw,
    }
    return inputs_dict


def test_thinker_model():
    if len(sys.argv) > 1:
        cmd_dtype = sys.argv[1]
    else:
        cmd_dtype = "bfloat16"

    float32_switch = cmd_dtype == "float32" or cmd_dtype == "fp32"
    print("float32_switch: ", float32_switch)

    config = Qwen3OmniMoeThinkerConfig.from_pretrained(MODEL_PATH)
    config.dtype = "float32" if float32_switch else "bfloat16"
    config.text_config.num_hidden_layers = 4
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    config.audio_config._attn_implementation = "eager"

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        config=config,
        load_checkpoint_format="flex_checkpoint",
        dtype=("float32" if float32_switch else "bfloat16"),
    )
    model.eval()

    for k, v in model.state_dict().items():
        if "conv2d" in k:
            compare_and_save(v, k, False, False)

    origin_inputs_dict = get_deterministic_inputs(config, float32_switch)

    target_input_keys = (
        "input_ids",
        "attention_mask",
        "input_features",
        "feature_attention_mask",
        "image_grid_thw",
        "pixel_values",
        "pixel_values_videos",
        "video_grid_thw",
        "labels",
    )
    inputs_dict = {k: v for k, v in origin_inputs_dict.items() if k in target_input_keys}
    for key, value in inputs_dict.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        compare_and_save(value, key, False, False)

    if mock_by_torch_logits_grad:
        # =======================================================
        # 替换 logits grad
        # =======================================================
        print("[Paddle] Running Forward...")
        # 注意：这里不需要传入 labels，因为我们要手动反向
        new_inputs_dict = {k: v for k, v in inputs_dict.items() if k != "labels"}
        output_ids = model(**new_inputs_dict, return_dict=True)

        print("output_ids: ", type(output_ids), output_ids)

        logits = output_ids.logits
        loss = output_ids.loss[0] if isinstance(output_ids.loss, tuple) else output_ids.loss

        compare_and_save(logits, "output_ids_logits_after_forward", True, True)
        compare_and_save(loss, "output_ids_loss_after_forward", True, True)

        # 【关键步骤 1】加载 Torch 算出来的 Logits 梯度
        logits_grad_pd = hack_with_torch_file(
            "output_ids_logits_grad_after_backward",
            dtype=("float32" if float32_switch else "bfloat16"),
            device="cuda",
        )

        print("[Paddle] Running Backward with INJECTED Gradients...")

        # 【关键步骤 2】手动反向传播
        # 含义：计算 logits 的梯度时，已知上游传来的梯度是 logits_grad_pd
        paddle.autograd.backward(
            tensors=[logits],  # 我们要对谁求导？(也就是谁是反向的起点) -> logits
            grad_tensors=[logits_grad_pd],  # 起点的梯度是多少？ -> 注入 Torch 的值
        )

        compare_and_save(logits, "output_ids_logits_after_backward", True, True)
        compare_and_save(loss, "output_ids_loss_after_backward", True, True)

    else:
        output_ids = model(**inputs_dict)

        print("output_ids: ", type(output_ids), output_ids)

        logits = output_ids.logits
        loss = output_ids.loss[0] if isinstance(output_ids.loss, tuple) else output_ids.loss

        compare_and_save(logits, "output_ids_logits_after_forward", True, True)
        compare_and_save(loss, "output_ids_loss_after_forward", True, True)

        print("[Paddle] Runing Backward...")
        loss.backward()

        compare_and_save(logits, "output_ids_logits_after_backward", True, True)
        compare_and_save(loss, "output_ids_loss_after_backward", True, True)

    print("[Paddle] Saving Gradients...")
    grads_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads_dict[name] = param.grad.cast("float32").numpy()

    compare_and_save(grads_dict, "output_ids_grads_after_backward", True, False)

    print("✅ Paddle Done.")


def test_thinker_with_dumped_inputs(dumped_input_path=None):
    """Test model with dumped inputs from training"""
    config = Qwen3OmniMoeThinkerConfig.from_pretrained(MODEL_PATH)
    config.text_config.num_hidden_layers = 12
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    config.audio_config._attn_implementation = "sdpa"
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_config(config)

    # Find dumped inputs
    if dumped_input_path is None:
        # Auto-discover the latest dumped input file
        dump_dir = "/root/paddlejob/workspace/env_run/wuhuiyue/MyPaddleFormers/qwen3_omni/dumped_inputs/"
        if not os.path.exists(dump_dir):
            print(f"Warning: Dump directory {dump_dir} does not exist")
            print("Please run training first to generate dumped inputs")
            return

        input_files = glob.glob(os.path.join(dump_dir, "*_inputs.npz"))
        if not input_files:
            print(f"Warning: No dumped input files found in {dump_dir}")
            return

        # Use the latest file
        dumped_input_path = max(input_files, key=os.path.getmtime)

    print(f"Loading dumped inputs from: {dumped_input_path}")

    # Load dumped inputs
    loaded_data = np.load(dumped_input_path)

    # Convert numpy arrays back to paddle tensors
    model_inputs = {}
    for key in loaded_data.files:
        model_inputs[key] = paddle.to_tensor(loaded_data[key])

    print(f"Loaded input keys: {list(model_inputs.keys())}")
    for key, value in model_inputs.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    # Run model with dumped inputs
    output_ids = model(**model_inputs)

    print("output_ids: ", type(output_ids), output_ids)

    return output_ids


def qwen3vlmoe():
    MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-VL-30B-A3B-Instruct/"

    config = Qwen3VLMoeConfig.from_pretrained(MODEL_PATH)
    config.text_config.num_hidden_layers = 12
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"

    model = Qwen3VLMoeForConditionalGenerationDeprecated.from_pretrained(
        MODEL_PATH,
        config=config,
        dtype="bfloat16",
        load_checkpoint_format="flex_checkpoint",
    )

    sorted_keys = sorted([key for key in model.state_dict().keys()])
    for key in sorted_keys:
        print(key)


if __name__ == "__main__":
    test_thinker_model()
    # test_talker_model()
    # qwen3vlmoe()

    # print("\n" + "=" * 60)
    # print("Test 2: Dumped input test")
    # print("=" * 60)

    # # Check if a specific dumped input path is provided
    # if len(sys.argv) > 1:
    #     test_thinker_with_dumped_inputs(sys.argv[1])
    # else:
    #     test_thinker_with_dumped_inputs()
