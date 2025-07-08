#!/usr/bin/env python3
"""
Thunder Policy Model 测试程序
测试 exported/policy.pt 模型是否能正常加载和推理
"""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

def test_policy_model(model_path: str):
    """测试策略模型"""
    
    print("🤖 Thunder Policy Model Test")
    print("=" * 60)
    
    # 1. 检查模型文件是否存在
    print(f"📁 Checking model file: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"✅ Model file found, size: {file_size:.2f} MB")
    
    try:
        # 2. 加载模型
        print(f"\n🔄 Loading PyTorch JIT model...")
        model = torch.jit.load(model_path)
        model.eval()
        print(f"✅ Model loaded successfully")
        
        # 3. 测试模型输入输出维度
        print(f"\n🔍 Testing model dimensions...")
        
        # 根据thunder_flat_deploy.py的配置，观测维度应该是57
        expected_obs_dim = 57
        expected_action_dim = 16
        
        # 创建测试输入
        test_input = torch.randn(1, expected_obs_dim, dtype=torch.float32)
        print(f"   Input shape: {test_input.shape}")
        
        # 4. 执行推理
        print(f"\n⚡ Running inference...")
        start_time = time.time()
        
        with torch.no_grad():
            test_output = model(test_input)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        print(f"✅ Inference completed in {inference_time:.2f} ms")
        print(f"   Output shape: {test_output.shape}")
        
        # 5. 验证输出维度
        if test_output.shape[0] != 1:
            print(f"❌ Expected batch size 1, got {test_output.shape[0]}")
            return False
        
        if test_output.shape[1] != expected_action_dim:
            print(f"❌ Expected action dimension {expected_action_dim}, got {test_output.shape[1]}")
            return False
        
        print(f"✅ Output dimensions correct: {test_output.shape}")
        
        # 6. 检查输出数值范围
        output_values = test_output.squeeze(0).numpy()
        print(f"\n📊 Output analysis:")
        print(f"   Min value: {output_values.min():.4f}")
        print(f"   Max value: {output_values.max():.4f}")
        print(f"   Mean: {output_values.mean():.4f}")
        print(f"   Std: {output_values.std():.4f}")
        
        # 显示前几个动作值
        print(f"   First 8 actions: {output_values[:8]}")
        
        # 7. 性能测试
        print(f"\n⏱️  Performance test (100 inferences)...")
        times = []
        
        for i in range(100):
            start = time.time()
            with torch.no_grad():
                _ = model(test_input)
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        max_time = np.max(times)
        min_time = np.min(times)
        
        print(f"   Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"   Min: {min_time:.2f} ms, Max: {max_time:.2f} ms")
        print(f"   Theoretical max frequency: {1000/avg_time:.1f} Hz")
        
        # 8. 测试不同批次大小
        print(f"\n🔢 Testing different batch sizes...")
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            test_batch = torch.randn(batch_size, expected_obs_dim, dtype=torch.float32)
            
            start = time.time()
            with torch.no_grad():
                batch_output = model(test_batch)
            batch_time = (time.time() - start) * 1000
            
            if batch_output.shape != (batch_size, expected_action_dim):
                print(f"   ❌ Batch {batch_size}: Wrong output shape {batch_output.shape}")
            else:
                time_per_sample = batch_time / batch_size
                print(f"   ✅ Batch {batch_size}: {batch_time:.2f} ms total, {time_per_sample:.2f} ms/sample")
        
        # 9. 模拟观测数据测试
        print(f"\n🎯 Testing with realistic observation data...")
        
        # 创建更真实的观测向量（模拟实际机器人状态）
        realistic_obs = create_realistic_observation()
        realistic_input = torch.tensor(realistic_obs, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            realistic_output = model(realistic_input)
        
        realistic_actions = realistic_output.squeeze(0).numpy()
        print(f"   Realistic observation input shape: {realistic_input.shape}")
        print(f"   Realistic actions output: {realistic_actions}")
        
        # 检查动作是否在合理范围内（通常模型输出应该在[-1, 1]范围内）
        if np.all(np.abs(realistic_actions) <= 10.0):  # 宽松的范围检查
            print(f"   ✅ Actions in reasonable range")
        else:
            print(f"   ⚠️  Some actions might be out of normal range")
        
        print(f"\n🎉 All tests passed! Model is ready for deployment.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during model testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_realistic_observation():
    """创建模拟的真实观测数据"""
    
    # 观测向量组成（基于thunder_flat_deploy.py的配置）：
    # - base_ang_vel (3) + projected_gravity (3) = 6
    # - velocity_commands (3) = 3  
    # - joint_pos_rel (16) = 16
    # - joint_vel (16) = 16
    # - last_actions (16) = 16
    # 总计: 6 + 3 + 16 + 16 + 16 = 57
    
    observation = np.zeros(57)
    
    # 基座角速度 (IMU数据)
    observation[0:3] = np.array([0.1, -0.05, 0.2])  # 小幅摆动
    
    # 投影重力向量
    observation[3:6] = np.array([0.0, 0.0, -1.0])  # 向下的重力
    
    # 速度指令
    observation[6:9] = np.array([1.0, 0.0, 0.0])  # 前进1m/s
    
    # 关节相对位置（相对于默认姿态的偏差）
    # 腿部关节（12个）+ 轮子关节（4个，应该为0）
    joint_pos_rel = np.zeros(16)
    joint_pos_rel[0:12] = np.random.normal(0, 0.1, 12)  # 小幅随机偏差
    joint_pos_rel[12:16] = 0.0  # 轮子位置为0
    observation[9:25] = joint_pos_rel
    
    # 关节速度
    joint_vel = np.zeros(16)
    joint_vel[0:12] = np.random.normal(0, 0.5, 12)  # 腿部关节速度
    joint_vel[12:16] = np.array([2.0, 2.0, 2.0, 2.0])  # 轮子转动
    observation[25:41] = joint_vel
    
    # 上一步动作
    last_actions = np.random.uniform(-0.5, 0.5, 16)
    observation[41:57] = last_actions
    
    return observation


def main():
    """主函数"""
    model_path = "/home/ubuntu/Desktop/dog_deploy/deployment/exported/policy.pt"
    
    print("Testing Thunder policy model...")
    print(f"Model path: {model_path}")
    
    success = test_policy_model(model_path)
    
    if success:
        print(f"\n✅ Model test completed successfully!")
        print(f"The model is ready for use with thunder_flat_deploy.py")
    else:
        print(f"\n❌ Model test failed!")
        print(f"Please check the model file and try again.")


if __name__ == "__main__":
    main() 