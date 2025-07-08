#!/usr/bin/env python3
"""
简化的IMU集成测试脚本
测试Thunder Robot Interface和IMU数据接收
"""

import os
import sys
import time
import numpy as np
import yaml
import logging
import asyncio

# 添加dep目录到路径 (使用异步版本的接口)
dep_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
sys.path.insert(0, dep_dir)

from thunder_robot_interface import ThunderRobotInterface

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'thunder_flat_config.yaml')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Failed to load config: {e}")
        # 返回默认配置
        return {
            'robot': {
                'joint_names': [
                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", 
                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                    "FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"
                ],
                'default_joint_pos': [
                    0.0, 0.7, -1.5,   # FL
                    0.0, -0.7, 1.5,   # FR
                    0.0, -0.7, 1.5,   # RL
                    0.0, 0.7, -1.5,   # RR
                    0.0, 0.0, 0.0, 0.0  # wheels
                ]
            },
            'state_receiver': {
                'motor_server_host': '192.168.66.159',
                'motor_server_port': 12345,
                'receive_frequency': 200,
                'buffer_size': 100,
                'imu': {
                    'enabled': True,
                    'serial_port': '/dev/ttyUSB0',
                    'baud_rate': 115200,
                    'update_frequency': 200,
                    'buffer_size': 100
                }
            }
        }

async def test_imu_integration():
    """测试IMU集成"""
    print("\n" + "="*60)
    print("Testing IMU Integration with Thunder Robot Interface")
    print("="*60)
    
    # 加载配置
    config = load_config()
    
    try:
        # 1. 初始化机器人接口
        print("\n1. 初始化Thunder Robot Interface...")
        robot_interface = ThunderRobotInterface(config)
        print("✅ Thunder Robot Interface初始化成功")
        
        # 2. 启动状态接收器
        print("\n2. 启动状态接收器...")
        if await robot_interface.state_receiver.start_receiving():
            print("✅ 状态接收器启动成功")
        else:
            print("❌ 状态接收器启动失败")
            return False
        
        # 3. 等待IMU连接
        print("\n3. 等待IMU连接...")
        imu_connected = False
        for i in range(10):  # 等待10秒
            await asyncio.sleep(1)
            stats = robot_interface.state_receiver.get_statistics()
            print(f"   尝试连接IMU... ({i+1}/10) - 统计: {stats}")
            
            # 检查是否有IMU数据
            observations = robot_interface.get_observations()
            if observations:
                imu_ang_vel = observations.get('base_ang_vel', np.zeros(3))
                if np.any(imu_ang_vel != 0):
                    imu_connected = True
                    print("✅ IMU连接成功，接收到数据")
                    break
        
        if not imu_connected:
            print("⚠️ IMU未连接或无数据，继续测试其他功能")
        
        # 4. 测试数据接收
        print("\n4. 测试数据接收...")
        successful_reads = 0
        imu_data_count = 0
        
        for i in range(20):  # 测试2秒
            observations = robot_interface.get_observations()
            
            if observations:
                successful_reads += 1
                
                # 检查数据 (使用get_observations()的字段名)
                joint_positions = observations.get('joint_pos', np.zeros(16))
                joint_velocities = observations.get('joint_vel', np.zeros(16))
                imu_ang_vel = observations.get('base_ang_vel', np.zeros(3))
                projected_gravity = observations.get('projected_gravity', np.zeros(3))
                
                # 打印状态
                if i % 5 == 0:  # 每0.5秒打印一次
                    print(f"\n--- Step {i} ---")
                    print(f"Joint positions: {joint_positions[:4]}")  # 只打印前4个
                    print(f"Joint velocities: {joint_velocities[:4]}")
                    print(f"IMU Angular Velocity: {imu_ang_vel}")
                    print(f"Projected Gravity: {projected_gravity}")
                    
                    if np.any(imu_ang_vel != 0):
                        imu_data_count += 1
            
            await asyncio.sleep(0.1)  # 10Hz
        
        # 5. 打印测试结果
        print(f"\n📊 测试结果:")
        print(f"   成功读取次数: {successful_reads}/20")
        print(f"   IMU有效数据次数: {imu_data_count}")
        print(f"   数据接收率: {successful_reads/20*100:.1f}%")
        
        # 6. 获取最终统计
        final_stats = robot_interface.state_receiver.get_statistics()
        print(f"\n📡 接收器统计:")
        print(f"   {final_stats}")
        
        # 7. 停止接收器
        print("\n5. 停止状态接收器...")
        await robot_interface.state_receiver.stop_receiving()
        print("✅ 状态接收器已停止")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行异步测试
    success = asyncio.run(test_imu_integration())
    
    if success:
        print("\n🎉 IMU集成测试完成!")
        print("\n接下来你可以:")
        print("1. 检查IMU连接和数据接收")
        print("2. 运行完整的部署测试")
        print("3. 验证观测数据的正确性")
    else:
        print("\n❌ IMU集成测试失败")
        print("\n请检查:")
        print("1. IMU设备是否正确连接到 /dev/ttyUSB0")
        print("2. 串口权限是否正确")
        print("3. 网络连接是否正常")
        print("4. 配置文件是否正确")

if __name__ == "__main__":
    main() 