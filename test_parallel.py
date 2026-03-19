#!/usr/bin/env python
"""
简单的并行测试脚本，用于诊断 PaddleFormers 崩溃问题
"""
import sys
import os

print("=" * 60)
print("PaddleFormers Parallel Test Script")
print("=" * 60)

# 1. 检查 PaddlePaddle 版本
print("\n1. Checking PaddlePaddle version...")
import paddle
print(f"   Paddle version: {paddle.version.full_version}")
print(f"   CUDA compiled: {paddle.device.is_compiled_with_cuda()}")

# 2. 检查分布式环境
print("\n2. Checking distributed environment...")
from paddle.distributed import fleet
print(f"   Fleet available: True")

# 3. 检查当前配置
print("\n3. Current configuration:")
print(f"   World size: {os.environ.get('PADDLE_TRAINER_NUM', 'Not set')}")
print(f"   Current rank: {os.environ.get('PADDLE_TRAINER_ID', 'Not set')}")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# 4. 尝试初始化分布式环境
print("\n4. Attempting distributed initialization...")
try:
    from paddle.distributed import init_parallel_env
    init_parallel_env()
    print("   ✓ Distributed initialization successful")
except Exception as e:
    print(f"   ✗ Distributed initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 测试简单的并行操作
print("\n5. Testing simple parallel operations...")
try:
    # 创建一个简单的张量
    x = paddle.randn([10, 10])
    print(f"   ✓ Tensor creation successful: shape {x.shape}")

    # 测试 all_reduce
    if os.environ.get('PADDLE_TRAINER_NUM', '1') != '1':
        from paddle.distributed import all_reduce
        result = all_reduce(x)
        print(f"   ✓ All-reduce successful")
except Exception as e:
    print(f"   ✗ Parallel operations failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
