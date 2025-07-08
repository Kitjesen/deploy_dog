#!/usr/bin/env python3
"""
简化的实时状态测试程序
"""

import os
import sys
import time
import numpy as np

# 添加部署目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thunder_robot_interface import ThunderRobotInterface

def create_config():
    """创建配置"""
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
                0.1, 0.8, -1.5,   # FL
                -0.1, 0.8, -1.5,  # FR
                0.1, 1.0, -1.5,   # RL
                -0.1, 1.0, -1.5,  # RR
                0.0, 0.0, 0.0, 0.0  # wheels
            ]
        },
        'state_receiver': {
            'receive_frequency': 200,
            'motor_server_host': '192.168.66.159',
            'motor_server_port': 12345,
            'buffer_size': 100,
            'receive_timeout': 0.1
        }
    }

def test_real_state():
    """测试实时状态获取"""
    print("Testing Real Motor State Reception")
    print("=" * 40)
    
    config = create_config()
    robot_interface = ThunderRobotInterface(config)
    
    try:
        # 启动状态接收器
        print("Starting state receiver...")
        if not robot_interface.state_receiver.start_receiving():
            print("Failed to start state receiver")
            return False
        
        print("Connected to motor server, waiting for data...")
        time.sleep(3.0)  # 等待接收数据
        
        # 获取统计信息
        stats = robot_interface.state_receiver.get_statistics()
        print(f"Received {stats['total_received']} motor states")
        print(f"Receive rate: {stats['receive_rate']:.1f} Hz")
        
        if stats['total_received'] == 0:
            print("No data received!")
            return False
        
        print("\nReal-time state data:")
        print("=" * 40)
        
        # 测试5次状态获取
        for i in range(5):
            # 直接从状态接收器获取最新状态
            latest_state = robot_interface.state_receiver.get_latest_state()
            
            if latest_state is None:
                print(f"Step {i+1}: No state available")
                continue
            
            print(f"\nStep {i+1} (timestamp: {latest_state.timestamp:.3f}):")
            
            # 显示关节位置
            joint_pos = latest_state.joint_positions
            print(f"Joint Positions (16): {joint_pos}")
            
            # 显示关节速度
            joint_vel = latest_state.joint_velocities  
            print(f"Joint Velocities (16): {joint_vel}")
            
            # 显示腿部关节具体数据
            print(f"FL leg pos: [{joint_pos[0]:6.3f}, {joint_pos[1]:6.3f}, {joint_pos[2]:6.3f}]")
            print(f"FR leg pos: [{joint_pos[3]:6.3f}, {joint_pos[4]:6.3f}, {joint_pos[5]:6.3f}]")
            print(f"RL leg pos: [{joint_pos[6]:6.3f}, {joint_pos[7]:6.3f}, {joint_pos[8]:6.3f}]")
            print(f"RR leg pos: [{joint_pos[9]:6.3f}, {joint_pos[10]:6.3f}, {joint_pos[11]:6.3f}]")
            print(f"Wheels pos: [{joint_pos[12]:6.3f}, {joint_pos[13]:6.3f}, {joint_pos[14]:6.3f}, {joint_pos[15]:6.3f}]")
            
            print(f"FL leg vel: [{joint_vel[0]:6.3f}, {joint_vel[1]:6.3f}, {joint_vel[2]:6.3f}]")
            print(f"FR leg vel: [{joint_vel[3]:6.3f}, {joint_vel[4]:6.3f}, {joint_vel[5]:6.3f}]")
            
            time.sleep(0.5)  # 等待0.5秒
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 停止状态接收器
        try:
            robot_interface.state_receiver.stop_receiving()
            print("\nState receiver stopped")
        except:
            pass

if __name__ == "__main__":
    success = test_real_state()
    if success:
        print("\nReal state test completed successfully!")
    else:
        print("\nReal state test failed!") 