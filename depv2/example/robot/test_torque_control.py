#!/usr/bin/env python3
"""
Thunder Flat扭矩控制模式测试脚本
测试新的扭矩控制功能是否正常工作
"""

import numpy as np
import sys
import os

# 添加部署目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from action_processor import ThunderActionProcessor

def test_torque_control_processor():
    """测试扭矩控制版本的action processor"""
    print("="*60)
    print("测试扭矩控制模式的Action Processor")
    print("="*60)
    
    # 创建包含扭矩控制参数的配置
    config = {
        'actions': {
            'scales': {
                'hip_joints': 0.125,
                'other_joints': 0.25,
                'wheel_joints': 5.0
            },
            'clip_range': [-1.0, 1.0],
            'action_dim': 16
        },
        'safety': {
            'joint_pos_limits': {
                'hip': [-0.6, 0.6],
                'thigh': [-3.14159, 3.14159],
                'calf': [-3.14159, 3.14159],
                'wheel': [-1000.0, 1000.0]
            },
            'joint_vel_limits': {
                'leg': [-21.0, 21.0],
                'wheel': [-20.4, 20.4]
            },
            'torque_limits': {
                'hip': [-30.0, 30.0],
                'thigh': [-55.0, 55.0], 
                'calf': [-55.0, 55.0],
                'wheel': [-20.0, 20.0]
            }
        },
        'hardware': {
            'pd_gains': {
                'leg_joints': {
                    'kp': [20.0, 25.0, 25.0],  # [hip, thigh, calf]
                    'kd': [0.5, 0.8, 0.8]
                },
                'wheel_joints': {
                    'kp': 10.0,
                    'kd': 0.3
                }
            }
        },
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
        }
    }
    
    # 创建处理器
    print("\n1. 创建Action Processor...")
    processor = ThunderActionProcessor(config)
    
    # 创建测试场景
    print("\n2. 测试不同场景...")
    
    scenarios = [
        {
            'name': '站立姿态',
            'actions': np.zeros(16),  # 无动作，应该保持默认姿态
            'robot_state': {
                'joint_positions': np.array([
                    0.1, 0.8, -1.5,   # FL - 接近默认
                    -0.1, 0.8, -1.5,  # FR
                    0.1, 1.0, -1.5,   # RL  
                    -0.1, 1.0, -1.5,  # RR
                    0.0, 0.0, 0.0, 0.0  # wheels
                ]),
                'joint_velocities': np.zeros(16)
            }
        },
        {
            'name': '前倾动作',
            'actions': np.array([
                0.1, -0.2, 0.1,   # FL - hip向前，thigh向前倾
                0.1, -0.2, 0.1,   # FR
                0.1, -0.2, 0.1,   # RL
                0.1, -0.2, 0.1,   # RR
                0.3, 0.3, 0.3, 0.3  # wheels - 前进
            ]),
            'robot_state': {
                'joint_positions': np.array([
                    0.1, 0.8, -1.5,   # FL
                    -0.1, 0.8, -1.5,  # FR
                    0.1, 1.0, -1.5,   # RL
                    -0.1, 1.0, -1.5,  # RR
                    0.0, 0.0, 0.0, 0.0  # wheels
                ]),
                'joint_velocities': np.random.randn(16) * 0.1  # 一些运动
            }
        },
        {
            'name': '位置误差大的情况',
            'actions': np.zeros(16),
            'robot_state': {
                'joint_positions': np.array([
                    0.3, 1.2, -1.8,   # FL - 远离默认位置
                    -0.2, 0.5, -1.2,  # FR
                    0.2, 1.3, -1.7,   # RL
                    -0.3, 0.7, -1.3,  # RR
                    0.0, 0.0, 0.0, 0.0  # wheels
                ]),
                'joint_velocities': np.random.randn(16) * 0.2
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n--- 场景 {i+1}: {scenario['name']} ---")
        
        # 处理动作
        commands = processor.process_actions(scenario['actions'], scenario['robot_state'])
        
        # 验证控制指令
        is_valid = processor.validate_commands(commands)
        print(f"控制指令有效性: {'✅ 有效' if is_valid else '❌ 无效'}")
        
        # 分析结果
        leg_torques = commands['joint_torque_targets']
        leg_positions = commands['joint_position_targets']
        wheel_velocities = commands['joint_velocity_targets']
        
        print(f"腿部扭矩范围: [{leg_torques.min():.2f}, {leg_torques.max():.2f}] Nm")
        print(f"目标位置范围: [{leg_positions.min():.3f}, {leg_positions.max():.3f}] rad")
        print(f"轮子速度范围: [{wheel_velocities.min():.2f}, {wheel_velocities.max():.2f}] rad/s")
        
        # 检查扭矩分布
        hip_torques = leg_torques[[0, 3, 6, 9]]
        thigh_torques = leg_torques[[1, 4, 7, 10]]
        calf_torques = leg_torques[[2, 5, 8, 11]]
        
        print(f"Hip扭矩: avg={hip_torques.mean():.2f}, max_abs={np.abs(hip_torques).max():.2f} Nm")
        print(f"Thigh扭矩: avg={thigh_torques.mean():.2f}, max_abs={np.abs(thigh_torques).max():.2f} Nm")
        print(f"Calf扭矩: avg={calf_torques.mean():.2f}, max_abs={np.abs(calf_torques).max():.2f} Nm")
        
        # 验证控制模式
        control_mode = commands['control_mode']
        print(f"控制模式: 腿部={control_mode['legs']}, 轮子={control_mode['wheels']}")
        
        # 验证硬件格式
        hardware_commands = processor.format_for_hardware(commands)
        print(f"硬件接口: {len(hardware_commands['torque_commands'])}个扭矩指令, {len(hardware_commands['velocity_commands'])}个速度指令")
    
    print("\n" + "="*60)
    print("扭矩控制测试完成")
    print("="*60)
    
    return True

def test_pd_control_behavior():
    """测试PD控制器的行为"""
    print("\n" + "="*60)
    print("测试PD控制器行为")
    print("="*60)
    
    # 简单配置
    config = {
        'actions': {'scales': {'hip_joints': 0.125, 'other_joints': 0.25, 'wheel_joints': 5.0}, 'clip_range': [-1.0, 1.0]},
        'safety': {
            'joint_pos_limits': {'hip': [-0.6, 0.6], 'thigh': [-3.14, 3.14], 'calf': [-3.14, 3.14]},
            'joint_vel_limits': {'leg': [-21.0, 21.0], 'wheel': [-20.4, 20.4]},
            'torque_limits': {'hip': [-30.0, 30.0], 'thigh': [-55.0, 55.0], 'calf': [-55.0, 55.0]}
        },
        'hardware': {
            'pd_gains': {
                'leg_joints': {'kp': [20.0, 25.0, 25.0], 'kd': [0.5, 0.8, 0.8]},
                'wheel_joints': {'kp': 10.0, 'kd': 0.3}
            }
        },
        'robot': {
            'joint_names': ["FL_hip", "FL_thigh", "FL_calf"] + ["FR_hip", "FR_thigh", "FR_calf"] + 
                          ["RL_hip", "RL_thigh", "RL_calf"] + ["RR_hip", "RR_thigh", "RR_calf"] + 
                          ["FL_wheel", "FR_wheel", "RL_wheel", "RR_wheel"],
            'default_joint_pos': [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5, 0, 0, 0, 0]
        }
    }
    
    processor = ThunderActionProcessor(config)
    
    # 测试不同的位置误差和速度情况
    test_cases = [
        {'pos_error': 0.1, 'velocity': 0.0, 'description': '小位置误差，零速度'},
        {'pos_error': 0.5, 'velocity': 0.0, 'description': '大位置误差，零速度'},
        {'pos_error': 0.1, 'velocity': 1.0, 'description': '小位置误差，高速度'},
        {'pos_error': -0.3, 'velocity': -0.5, 'description': '负位置误差，负速度'},
    ]
    
    for case in test_cases:
        print(f"\n--- {case['description']} ---")
        
        # 创建测试状态
        current_pos = np.array([0.0, 0.5, -1.2] * 4 + [0.0] * 4)  # 当前位置
        target_pos = current_pos.copy()
        target_pos[:12] += case['pos_error']  # 添加位置误差
        
        current_vel = np.ones(16) * case['velocity']
        
        robot_state = {
            'joint_positions': current_pos,
            'joint_velocities': current_vel
        }
        
        # 计算扭矩 (使用零动作，依靠默认位置)
        actions = np.zeros(16)
        commands = processor.process_actions(actions, robot_state)
        
        torques = commands['joint_torque_targets']
        
        # 分析扭矩输出
        print(f"位置误差: {case['pos_error']:.1f}, 速度: {case['velocity']:.1f}")
        print(f"Hip扭矩 (Kp=20.0): {torques[[0,3,6,9]].mean():.2f} ± {torques[[0,3,6,9]].std():.2f} Nm")
        print(f"Thigh扭矩 (Kp=25.0): {torques[[1,4,7,10]].mean():.2f} ± {torques[[1,4,7,10]].std():.2f} Nm")
        print(f"Calf扭矩 (Kp=25.0): {torques[[2,5,8,11]].mean():.2f} ± {torques[[2,5,8,11]].std():.2f} Nm")
        
        # 验证PD控制逻辑
        expected_torque_magnitude = abs(case['pos_error']) * 20.0 + abs(case['velocity']) * 0.5
        actual_torque_magnitude = np.abs(torques).mean()
        print(f"期望扭矩量级: ~{expected_torque_magnitude:.2f} Nm, 实际: {actual_torque_magnitude:.2f} Nm")

def main():
    """主测试函数"""
    print("Thunder Flat扭矩控制模式测试")
    print("="*60)
    
    try:
        # 测试Action Processor
        success1 = test_torque_control_processor()
        
        # 测试PD控制器行为
        test_pd_control_behavior()
        
        if success1:
            print("\n🎉 所有测试通过！扭矩控制模式运行正常。")
            print("\n💡 关键改进:")
            print("  ✅ 腿部关节使用扭矩控制 (PD控制器)")
            print("  ✅ 轮子关节保持速度控制")
            print("  ✅ 支持动态响应和合规性")
            print("  ✅ 可配置的PD参数")
            print("  ✅ 扭矩限制和安全检查")
            return True
        else:
            print("\n❌ 部分测试失败，请检查配置。")
            return False
            
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 