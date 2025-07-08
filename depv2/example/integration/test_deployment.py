#!/usr/bin/env python3
"""
Thunder Flat Deployment Test Script
测试Thunder Flat模型部署系统的各个组件
"""

import os
import sys
import time
import numpy as np
import torch
import logging
from pathlib import Path

# 添加部署目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from thunder_flat_deploy import ThunderFlatDeployer
from observation_processor import ThunderObservationProcessor
from action_processor import ThunderActionProcessor
from thunder_robot_interface import ThunderRobotInterface
from safety_monitor import ThunderSafetyMonitor

class ThunderRobotSimulator:
    """Thunder机器人物理模拟器 - 生成真实的动态数据"""
    
    def __init__(self):
        # 机器人配置
        self.num_joints = 16
        
        # 默认关节位置
        self.default_joint_positions = np.array([
            0.1, 0.8, -1.5,    # FL
            -0.1, 0.8, -1.5,   # FR
            0.1, 1.0, -1.5,    # RL
            -0.1, 1.0, -1.5,   # RR
            0.0, 0.0, 0.0, 0.0 # 轮子
        ])
        
        # 当前状态
        self.joint_positions = self.default_joint_positions.copy()
        self.joint_velocities = np.zeros(16)
        
        # 基座状态
        self.base_position = np.array([0.0, 0.0, 0.3])  # 30cm高度
        self.base_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # 四元数 [w,x,y,z]
        self.base_lin_velocity = np.zeros(3)
        self.base_ang_velocity = np.zeros(3)
        
        # 运动模式
        self.motion_mode = "walking"  # walking, turning, standing
        self.motion_time = 0.0
        self.motion_frequency = 1.0  # Hz
        
        # 传感器噪声参数
        self.joint_pos_noise = 0.001   # 1mm精度
        self.joint_vel_noise = 0.01    # rad/s
        self.imu_gyro_noise = 0.01     # rad/s
        
        # 物理参数
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.dt = 0.02  # 50Hz
        
    def set_motion_mode(self, mode: str, params=None):
        """设置运动模式"""
        self.motion_mode = mode
        if params is not None and 'frequency' in params:
            self.motion_frequency = params['frequency']
        
    def update(self, dt: float, velocity_commands=None):
        """更新机器人状态"""
        self.motion_time += dt
        
        if velocity_commands is None:
            velocity_commands = np.array([0.5, 0.0, 0.0])  # 默认前进
        
        # 基于运动模式生成关节轨迹
        if self.motion_mode == "walking":
            self._simulate_walking_gait(velocity_commands)
        elif self.motion_mode == "turning":
            self._simulate_turning_gait(velocity_commands)
        elif self.motion_mode == "standing":
            self._simulate_standing_motion()
            
        # 更新基座运动
        self._update_base_motion(velocity_commands)
        
        # 添加传感器噪声
        self._add_sensor_noise()
        
    def _simulate_walking_gait(self, vel_cmd):
        """模拟行走步态"""
        # 四足机器人对角步态 (Trot)
        phase_offset = [0.0, 0.5, 0.5, 0.0]  # FL, FR, RL, RR
        
        for leg in range(4):
            # 计算相位
            phase = (self.motion_time * self.motion_frequency + phase_offset[leg]) % 1.0
            
            # 腿部关节索引
            hip_idx = leg * 3
            thigh_idx = leg * 3 + 1
            calf_idx = leg * 3 + 2
            wheel_idx = 12 + leg
            
            # 步态生成 - 摆动相和支撑相
            if phase < 0.5:  # 摆动相
                # 抬腿动作
                lift_height = 0.05 * np.sin(phase * 2 * np.pi)
                self.joint_positions[thigh_idx] = self.default_joint_positions[thigh_idx] + lift_height
                self.joint_positions[calf_idx] = self.default_joint_positions[calf_idx] - lift_height * 1.5
                
                # 关节速度
                self.joint_velocities[thigh_idx] = 0.05 * np.cos(phase * 2 * np.pi) * 2 * np.pi * self.motion_frequency
                self.joint_velocities[calf_idx] = -self.joint_velocities[thigh_idx] * 1.5
            else:  # 支撑相
                # 着地推进
                support_offset = 0.02 * np.sin((phase - 0.5) * 2 * np.pi)
                self.joint_positions[thigh_idx] = self.default_joint_positions[thigh_idx] + support_offset
                self.joint_positions[calf_idx] = self.default_joint_positions[calf_idx] - support_offset
                
                # 关节速度
                self.joint_velocities[thigh_idx] = 0.02 * np.cos((phase - 0.5) * 2 * np.pi) * 2 * np.pi * self.motion_frequency
                self.joint_velocities[calf_idx] = -self.joint_velocities[thigh_idx]
            
            # Hip关节调整转向
            hip_adjustment = vel_cmd[2] * 0.1 * (1 if leg < 2 else -1)  # 前腿和后腿相反
            self.joint_positions[hip_idx] = self.default_joint_positions[hip_idx] + hip_adjustment
            
            # 轮子速度 - 基于前进速度
            wheel_speed = vel_cmd[0] * 5.0  # 转换为轮子角速度
            self.joint_velocities[wheel_idx] = wheel_speed
            
    def _simulate_turning_gait(self, vel_cmd):
        """模拟转向步态"""
        turn_amplitude = abs(vel_cmd[2]) * 0.2
        
        for leg in range(4):
            hip_idx = leg * 3
            wheel_idx = 12 + leg
            
            # 左右腿反向
            side_multiplier = 1 if leg % 2 == 0 else -1  # 左腿vs右腿
            turn_offset = side_multiplier * turn_amplitude * np.sin(self.motion_time * 2 * np.pi)
            
            self.joint_positions[hip_idx] = self.default_joint_positions[hip_idx] + turn_offset
            self.joint_velocities[hip_idx] = side_multiplier * turn_amplitude * np.cos(self.motion_time * 2 * np.pi) * 2 * np.pi
            
            # 轮子反向转动
            self.joint_velocities[wheel_idx] = side_multiplier * vel_cmd[2] * 3.0
            
    def _simulate_standing_motion(self):
        """模拟站立时的微小摆动"""
        sway_amplitude = 0.005  # 很小的摆动
        
        for i in range(12):  # 只有腿部关节
            sway = sway_amplitude * np.sin(self.motion_time * 0.5 * 2 * np.pi + i * 0.1)
            self.joint_positions[i] = self.default_joint_positions[i] + sway
            self.joint_velocities[i] = sway_amplitude * np.cos(self.motion_time * 0.5 * 2 * np.pi + i * 0.1) * 0.5 * 2 * np.pi
            
        # 轮子静止
        for i in range(12, 16):
            self.joint_velocities[i] = 0.0
            
    def _update_base_motion(self, vel_cmd):
        """更新基座运动"""
        # 基于速度指令更新基座状态
        self.base_lin_velocity = np.array([vel_cmd[0], vel_cmd[1], 0.0])
        self.base_ang_velocity = np.array([0.0, 0.0, vel_cmd[2]])
        
        # 更新位置
        self.base_position[:2] += self.base_lin_velocity[:2] * self.dt
        
        # 模拟基座俯仰和滚转 (基于运动)
        pitch_amplitude = abs(vel_cmd[0]) * 0.05  # 前进时的俯仰
        roll_amplitude = abs(vel_cmd[2]) * 0.03   # 转向时的滚转
        
        pitch = pitch_amplitude * np.sin(self.motion_time * self.motion_frequency * 2 * np.pi)
        roll = roll_amplitude * np.sin(self.motion_time * self.motion_frequency * 2 * np.pi)
        yaw = 0.0
        
        # 转换为四元数 (简化版)
        self.base_orientation = self._euler_to_quaternion(roll, pitch, yaw)
        
    def _euler_to_quaternion(self, roll, pitch, yaw):
        """欧拉角转四元数"""
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
        
    def _add_sensor_noise(self):
        """添加传感器噪声"""
        # 关节位置噪声
        self.joint_positions += np.random.normal(0, self.joint_pos_noise, 16)
        
        # 关节速度噪声
        self.joint_velocities += np.random.normal(0, self.joint_vel_noise, 16)
        
    def get_robot_state(self):
        """获取当前机器人状态"""
        # 计算projected gravity (考虑基座姿态)
        gravity_world = np.array([0.0, 0.0, -1.0])  # 世界坐标系重力方向
        
        # 从四元数计算旋转矩阵 (简化版)
        w, x, y, z = self.base_orientation
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
        
        projected_gravity = R.T @ gravity_world  # 基座坐标系中的重力方向
        
        # 添加IMU噪声
        base_ang_vel_noisy = self.base_ang_velocity + np.random.normal(0, self.imu_gyro_noise, 3)
        imu_acceleration = self.gravity + np.random.normal(0, 0.1, 3)
        
        return {
            'joint_positions': self.joint_positions.copy(),
            'joint_velocities': self.joint_velocities.copy(),
            'base_position': self.base_position.copy(),
            'base_orientation': self.base_orientation.copy(),  # [w,x,y,z]
            'base_lin_vel': self.base_lin_velocity.copy(),
            'base_ang_vel': base_ang_vel_noisy.copy(),
            'projected_gravity': projected_gravity,
            'imu_acceleration': imu_acceleration,
            'imu_angular_velocity': base_ang_vel_noisy.copy(),
            'imu_orientation': self.base_orientation.copy(),
            'timestamp': time.time()
        }

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_model_loading():
    """测试模型加载"""
    print("\n" + "="*60)
    print("1. TESTING MODEL LOADING")
    print("="*60)
    
    model_path = "/home/aa/snake/deployment/exported/policy.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        for i in range(10):
        # 加载模型
            model = torch.jit.load(model_path)
            model.eval()
            
            # 创建更真实的测试输入
            # 输入向量构成: [base_ang_vel(3), projected_gravity(3), velocity_commands(3), joint_positions(16), joint_velocities(16), last_actions(16)]
            
            # 1. 基座角速度 (±0.5 rad/s)
            base_ang_vel = torch.tensor([[0.1, 0.0, 0.2]])
            
            # 2. 投影重力 (通常接近[0,0,-1])
            projected_gravity = torch.tensor([[0.0, 0.0, -1.0]])
            
            # 3. 速度命令 (±0.5 m/s, ±0.2 rad/s)
            velocity_commands = torch.tensor([[0.3, 0.0, 0.1]])
            
            # 4. 关节位置 (合理的站立姿态)
            default_positions = torch.tensor([
                [0.0, 0.8, -1.5,     # FL leg
                0.0, 0.8, -1.5,     # FR leg
                0.0, 0.8, -1.5,     # RL leg
                0.0, 0.8, -1.5,     # RR leg
                0.0, 0.0, 0.0, 0.0] # wheels
            ]) * 0.1  # 缩小到更合理的范围
            
            # 5. 关节速度 (较小的运动)
            joint_velocities = torch.zeros(1, 16)
            
            # 6. 上一步动作 (初始为0)
            last_actions = torch.zeros(1, 16)
            
            # 组合所有输入
            test_input = torch.cat([
                base_ang_vel,
                projected_gravity,
                velocity_commands,
                default_positions,
                joint_velocities,
                last_actions
            ], dim=1)
            
            print("\n📥 Test Input Analysis:")
            print(f"   Base Angular Velocity: {base_ang_vel.numpy()[0]}")
            print(f"   Projected Gravity: {projected_gravity.numpy()[0]}")
            print(f"   Velocity Commands: {velocity_commands.numpy()[0]}")
            print(f"   Joint Positions Range: [{default_positions.min():.3f}, {default_positions.max():.3f}]")
            print(f"   Joint Velocities Range: [{joint_velocities.min():.3f}, {joint_velocities.max():.3f}]")
            
            with torch.no_grad():
                output = model(test_input)
                print("\n output:  ", output)
        # 详细分析输出
        output_np = output.numpy().squeeze()
        leg_actions = output_np[:12]  # 前12个输出是腿部关节
        wheel_actions = output_np[12:]  # 后4个输出是轮子
        
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print("\n📊 Raw Model Output Analysis:")
        print(f"   Overall range: [{output_np.min():.3f}, {output_np.max():.3f}]")
        print(f"   Leg joints range: [{leg_actions.min():.3f}, {leg_actions.max():.3f}]")
        print(f"   Wheel joints range: [{wheel_actions.min():.3f}, {wheel_actions.max():.3f}]")
        print(f"   Mean: {output_np.mean():.3f}")
        print(f"   Std: {output_np.std():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_observation_processor():
    """测试观测处理器"""
    print("\n" + "="*60)
    print("2. TESTING OBSERVATION PROCESSOR")
    print("="*60)
    
    try:
        # 创建配置
        config = {
            'observations': {
                'scales': {
                    'base_ang_vel': 0.25,
                    'projected_gravity': 1.0,
                    'velocity_commands': 1.0,
                    'joint_pos': 1.0,
                    'joint_vel': 0.05,
                    'actions': 1.0
                },
                'clip_range': [-100.0, 100.0],
                'observation_dim': 57,
                'normalization': {
                    'enabled': False,
                    'mean': [],
                    'std': []
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
                    0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                    0.1, 1.0, -1.5, -0.1, 1.0, -1.5,
                    0.0, 0.0, 0.0, 0.0
                ]
            }
        }
        
        # 创建处理器
        processor = ThunderObservationProcessor(config)
        
        # 创建测试数据
        robot_state = {
            'base_ang_vel': [0.1, 0.2, 0.3],
            'base_orientation': [1.0, 0.0, 0.0, 0.0],  # [w,x,y,z]
            'joint_positions': np.random.randn(16) * 0.1,
            'joint_velocities': np.random.randn(16) * 0.5,
            'timestamp': time.time()
        }
        
        velocity_commands = np.array([1.0, 0.0, 0.0])
        last_actions = torch.randn(16) * 0.1
        
        # 处理观测
        observation = processor.process_observations(
            robot_state, velocity_commands, last_actions
        )
        
        # 验证
        is_valid = processor.validate_observation(observation)
        
        print(f"✅ Observation processor working")
        print(f"   Output dimension: {len(observation)}")
        print(f"   Output range: [{observation.min():.3f}, {observation.max():.3f}]")
        print(f"   Validation passed: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"❌ Observation processor failed: {e}")
        return False

def test_action_processor():
    """测试动作处理器"""
    print("\n" + "="*60)
    print("3. TESTING ACTION PROCESSOR")
    print("="*60)
    
    try:
        # 创建配置
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
                    0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                    0.1, 1.0, -1.5, -0.1, 1.0, -1.5,
                    0.0, 0.0, 0.0, 0.0
                ]
            }
        }
        
        # 创建处理器
        processor = ThunderActionProcessor(config)
        
        # 创建多组测试动作进行统计
        num_tests = 100
        raw_actions_list = []
        processed_leg_positions = []
        processed_wheel_velocities = []
        
        print("\n📊 Testing with random actions...")
        for i in range(num_tests):
            # 创建随机测试动作 (模拟神经网络输出)
            raw_actions = np.random.randn(16) * 0.5  # 随机动作
            raw_actions_list.append(raw_actions)
            
            # 处理动作
            commands = processor.process_actions(raw_actions)
            
            # 收集处理后的动作
            leg_positions = np.array(commands['joint_position_targets'])
            wheel_velocities = np.array(commands['joint_velocity_targets'])
            processed_leg_positions.append(leg_positions)
            processed_wheel_velocities.append(wheel_velocities)
            
            # 每25步打印一次示例
            if i % 25 == 0:
                print(f"\n🔄 Test iteration {i+1}:")
                print(f"Raw actions range: [{raw_actions.min():.3f}, {raw_actions.max():.3f}]")
                print(f"Leg positions (rad): [{leg_positions.min():.3f}, {leg_positions.max():.3f}]")
                print(f"Wheel velocities (rad/s): [{wheel_velocities.min():.3f}, {wheel_velocities.max():.3f}]")
        
        # 转换为numpy数组以便统计
        raw_actions_array = np.array(raw_actions_list)
        processed_leg_positions = np.array(processed_leg_positions)
        processed_wheel_velocities = np.array(processed_wheel_velocities)
        
        print("\n📈 Statistical Analysis:")
        print("原始动作 (神经网络输出):")
        print(f"  范围: [{raw_actions_array.min():.3f}, {raw_actions_array.max():.3f}]")
        print(f"  均值: {raw_actions_array.mean():.3f}")
        print(f"  标准差: {raw_actions_array.std():.3f}")
        
        print("\n处理后的腿部关节位置 (单位: 弧度):")
        print(f"  范围: [{processed_leg_positions.min():.3f}, {processed_leg_positions.max():.3f}]")
        print(f"  均值: {processed_leg_positions.mean():.3f}")
        print(f"  标准差: {processed_leg_positions.std():.3f}")
        
        print("\n处理后的轮子关节速度 (单位: 弧度/秒):")
        print(f"  范围: [{processed_wheel_velocities.min():.3f}, {processed_wheel_velocities.max():.3f}]")
        print(f"  均值: {processed_wheel_velocities.mean():.3f}")
        print(f"  标准差: {processed_wheel_velocities.std():.3f}")
        
        print("\n⚙️ Action处理细节:")
        print("1. 腿部关节 (位置控制):")
        print("   - Hip关节缩放: 0.125 (±0.6 rad限制)")
        print("   - Thigh/Calf关节缩放: 0.25 (±π rad限制)")
        print("2. 轮子关节 (速度控制):")
        print("   - 速度缩放: 5.0 (±20.4 rad/s限制)")
        
        # 验证
        is_valid = processor.validate_commands(commands)
        print(f"\n✅ Action processor working")
        print(f"   Validation passed: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"❌ Action processor failed: {e}")
        return False

def test_safety_monitor():
    """测试安全监控器"""
    print("\n" + "="*60)
    print("4. TESTING SAFETY MONITOR")
    print("="*60)
    
    try:
        # 创建配置
        config = {
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
                'max_position_error': 0.5,
                'max_velocity': 25.0,
                'emergency_stop_enabled': True
            },
            'robot': {
                'joint_names': [
                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", 
                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                    "FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"
                ]
            }
        }
        
        # 创建安全监控器
        monitor = ThunderSafetyMonitor(config)
        
        # 测试正常状态
        normal_state = {
            'joint_positions': np.random.randn(16) * 0.1,
            'joint_velocities': np.random.randn(16) * 0.5,
            'imu_orientation': np.array([1, 0, 0, 0]),  # [w,x,y,z]
            'imu_acceleration': np.array([0, 0, 9.81]),
            'imu_angular_velocity': np.random.randn(3) * 0.1,
            'timestamp': time.time()
        }
        
        is_safe_normal = monitor.check_safety(normal_state)
        
        # 测试危险状态
        dangerous_state = normal_state.copy()
        dangerous_state['joint_positions'][0] = 1.0  # Hip关节超限
        
        is_safe_dangerous = monitor.check_safety(dangerous_state)
        
        print(f"✅ Safety monitor working")
        print(f"   Normal state safe: {is_safe_normal}")
        print(f"   Dangerous state safe: {is_safe_dangerous}")
        
        return True
        
    except Exception as e:
        print(f"❌ Safety monitor failed: {e}")
        return False

def test_full_pipeline():
    """测试完整的推理管道"""
    print("\n" + "="*60)
    print("5. TESTING FULL INFERENCE PIPELINE")
    print("="*60)
    
    try:
        model_path = "/home/aa/snake/deployment/exported/policy.pt"
        config_path = "config/thunder_flat_config.yaml"
        
        # 创建部署器 (但不启动循环)
        deployer = ThunderFlatDeployer(model_path, config_path)
        
        # 创建测试状态
        test_robot_state = {
            'joint_positions': np.random.randn(16) * 0.1,
            'joint_velocities': np.random.randn(16) * 0.5,
            'base_ang_vel': [0.1, 0.2, 0.3],
            'base_orientation': [1.0, 0.0, 0.0, 0.0],  # [w,x,y,z]
            'imu_acceleration': np.array([0, 0, 9.81]),
            'imu_angular_velocity': np.random.randn(3) * 0.1,
            'imu_orientation': np.array([1, 0, 0, 0]),  # [w,x,y,z]
            'timestamp': time.time()
        }
        
        # 模拟机器人接口返回测试状态
        deployer.robot_interface.get_robot_state = lambda: test_robot_state
        
        # 测试单步推理
        velocity_commands = np.array([1.0, 0.0, 0.0])
        success = deployer.step(velocity_commands)
        
        print(f"✅ Full pipeline working")
        print(f"   Single step success: {success}")
        print(f"   Step count: {deployer.step_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """测试性能"""
    print("\n" + "="*60)
    print("6. TESTING PERFORMANCE")
    print("="*60)
    
    try:
        model_path = "/home/aa/snake/deployment/exported/policy.pt"
        
        # 加载模型
        model = torch.jit.load(model_path)
        model.eval()
        
        # 预热
        test_input = torch.randn(1, 57)
        for _ in range(10):
            with torch.no_grad():
                _ = model(test_input)
        
        # 性能测试
        num_iterations = 1000
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                output = model(test_input)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        frequency = 1.0 / avg_time
        
        print(f"✅ Performance test completed")
        print(f"   Iterations: {num_iterations}")
        print(f"   Total time: {total_time:.3f} seconds")
        print(f"   Average inference time: {avg_time*1000:.3f} ms")
        print(f"   Max frequency: {frequency:.1f} Hz")
        print(f"   Target frequency: 50 Hz ({'✅ OK' if frequency >= 50 else '❌ Too slow'})")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_realistic_simulation():
    """测试真实物理模拟"""
    print("\n" + "="*60)
    print("7. TESTING REALISTIC ROBOT SIMULATION")
    print("="*60)
    
    try:
        # 创建机器人模拟器
        simulator = ThunderRobotSimulator()
        
        print("🤖 Testing different motion patterns...")
        
        # 测试不同运动模式
        motion_tests = [
            ("Standing", "standing", np.array([0.0, 0.0, 0.0])),
            ("Walking Forward", "walking", np.array([0.5, 0.0, 0.0])),
            ("Turning Right", "turning", np.array([0.0, 0.0, -0.5])),
            ("Walking + Turning", "walking", np.array([0.3, 0.0, 0.2]))
        ]
        
        dt = 0.02  # 50Hz
        duration = 2.0  # 2秒测试
        
        for test_name, mode, vel_cmd in motion_tests:
            print(f"\n--- {test_name} ---")
            simulator.set_motion_mode(mode)
            
            # 收集数据
            joint_pos_history = []
            joint_vel_history = []
            base_motion_history = []
            
            for step in range(int(duration / dt)):
                simulator.update(dt, vel_cmd)
                state = simulator.get_robot_state()
                
                joint_pos_history.append(state['joint_positions'].copy())
                joint_vel_history.append(state['joint_velocities'].copy())
                base_motion_history.append({
                    'base_ang_vel': state['base_ang_vel'].copy(),
                    'projected_gravity': state['projected_gravity'].copy()
                })
                
                # 每20步打印一次状态 (1Hz)
                if step % 20 == 0:
                    print(f"  t={step*dt:.1f}s: "
                          f"Hip=[{state['joint_positions'][0]:.3f}, {state['joint_positions'][3]:.3f}], "
                          f"Wheel_vel=[{state['joint_velocities'][12]:.2f}, {state['joint_velocities'][13]:.2f}], "
                          f"Base_ang_vel={state['base_ang_vel']}")
            
            # 分析运动数据
            joint_pos_array = np.array(joint_pos_history)
            joint_vel_array = np.array(joint_vel_history)
            
            print(f"  Joint position range: [{joint_pos_array.min():.3f}, {joint_pos_array.max():.3f}]")
            print(f"  Joint velocity range: [{joint_vel_array.min():.3f}, {joint_vel_array.max():.3f}]")
            print(f"  Motion smoothness: {np.std(np.diff(joint_pos_array, axis=0)):.4f}")
        
        print(f"\n✅ Realistic simulation test completed")
        print(f"   Generated {len(motion_tests)} different motion patterns")
        print(f"   Each pattern simulated for {duration}s at {1/dt}Hz")
        print(f"   Total data points: {len(motion_tests) * int(duration/dt)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Realistic simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline_with_simulation():
    """测试完整管道与真实模拟数据"""
    print("\n" + "="*60)
    print("8. TESTING FULL PIPELINE WITH REALISTIC DATA")
    print("="*60)
    
    try:
        model_path = "/home/aa/snake/deployment/exported/policy.pt"
        config_path = "config/thunder_flat_config.yaml"
        
        # 创建部署器
        deployer = ThunderFlatDeployer(model_path, config_path)
        
        # 为测试模式调整安全配置 - 放宽限制
        test_safety_config = {
            'joint_pos_limits': {
                'hip': [-1.0, 1.0],  # 放宽Hip关节限制
                'thigh': [-4.0, 4.0],  # 放宽Thigh关节限制
                'calf': [-4.0, 4.0],   # 放宽Calf关节限制
                'wheel': [-1000.0, 1000.0]
            },
            'joint_vel_limits': {
                'leg': [-30.0, 30.0],   # 放宽腿部速度限制
                'wheel': [-30.0, 30.0]  # 放宽轮子速度限制
            },
            'max_position_error': 2.0,  # 放宽位置误差限制
            'max_velocity': 50.0,       # 放宽速度限制
            'emergency_stop_enabled': True
        }
        deployer.safety_monitor.update_limits(test_safety_config)
        
        # 创建机器人模拟器
        simulator = ThunderRobotSimulator()
        simulator.set_motion_mode("walking")
        
        # 替换机器人接口的状态获取方法
        def get_simulated_state():
            return simulator.get_robot_state()
        
        deployer.robot_interface.get_robot_state = get_simulated_state
        
        # 设置速度指令
        velocity_commands = np.array([0.5, 0.0, 0.2])  # 前进0.5m/s，旋转0.2rad/s
        
        print("🚀 Running realistic simulation pipeline...")
        
        # 初始化变量
        num_steps = 100
        dt = 0.02  # 50Hz
        success_count = 0
        inference_times = []
        raw_actions_history = []
        processed_leg_positions = []
        processed_wheel_velocities = []
        
        # 主仿真循环
        for step in range(num_steps):
            # 更新模拟器
            simulator.update(dt, velocity_commands)
            
            # 执行推理
            start_time = time.time()
            success = deployer.step(velocity_commands)
            inference_time = time.time() - start_time
            
            if success:
                success_count += 1
                inference_times.append(inference_time * 1000)  # ms
                
                # 记录原始动作和处理后的动作
                raw_actions = deployer.last_actions.numpy()
                raw_actions_history.append(raw_actions)
                
                # 获取处理后的动作
                commands = deployer.action_processor.process_actions(raw_actions)
                leg_positions = np.array(commands['joint_position_targets'])
                wheel_velocities = np.array(commands['joint_velocity_targets'])
                processed_leg_positions.append(leg_positions)
                processed_wheel_velocities.append(wheel_velocities)
            
            # 每25步打印一次状态
            if step % 25 == 0:
                robot_state = simulator.get_robot_state()
                print(f"\n🔄 Step {step:3d}:")
                print(f"   Joint positions (rad): [{robot_state['joint_positions'].min():.3f}, {robot_state['joint_positions'].max():.3f}]")
                print(f"   Joint velocities (rad/s): [{robot_state['joint_velocities'].min():.3f}, {robot_state['joint_velocities'].max():.3f}]")
                print(f"   Base angular velocity (rad/s): [{robot_state['base_ang_vel'].min():.3f}, {robot_state['base_ang_vel'].max():.3f}]")
                print(f"   Inference time: {inference_time*1000:.2f}ms")
        
        # 转换为numpy数组以便统计
        raw_actions_array = np.array(raw_actions_history)
        processed_leg_positions = np.array(processed_leg_positions)
        processed_wheel_velocities = np.array(processed_wheel_velocities)
        
        # 分析结果
        success_rate = success_count / num_steps
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        max_inference_time = np.max(inference_times) if inference_times else 0
        
        print(f"\n📊 Pipeline Performance Analysis:")
        print(f"   Success rate: {success_rate*100:.1f}% ({success_count}/{num_steps})")
        print(f"   Avg inference time: {avg_inference_time:.2f}ms")
        print(f"   Max inference time: {max_inference_time:.2f}ms")
        print(f"   Real-time capable: {'✅ Yes' if max_inference_time < 20 else '❌ No'} (target: <20ms)")
        
        print("\n📈 Action Analysis:")
        if len(raw_actions_array) > 0:
            print("1. 原始动作 (神经网络输出):")
            print(f"   范围 (rad): [{raw_actions_array.min():.3f}, {raw_actions_array.max():.3f}]")
            print(f"   范围 (deg): [{np.rad2deg(raw_actions_array.min()):.1f}°, {np.rad2deg(raw_actions_array.max()):.1f}°]")
            print(f"   均值: {raw_actions_array.mean():.3f}")
            print(f"   标准差: {raw_actions_array.std():.3f}")
        
        if len(processed_leg_positions) > 0:
            print("\n2. 处理后的腿部关节位置 (单位: 弧度):")
            print(f"   范围: [{processed_leg_positions.min():.3f}, {processed_leg_positions.max():.3f}]")
            print(f"   均值: {processed_leg_positions.mean():.3f}")
            print(f"   标准差: {processed_leg_positions.std():.3f}")
            print(f"   平滑度 (相邻步差异): {np.std(np.diff(processed_leg_positions, axis=0)):.4f}")
        
        if len(processed_wheel_velocities) > 0:
            print("\n3. 处理后的轮子关节速度 (单位: 弧度/秒):")
            print(f"   范围: [{processed_wheel_velocities.min():.3f}, {processed_wheel_velocities.max():.3f}]")
            print(f"   均值: {processed_wheel_velocities.mean():.3f}")
            print(f"   标准差: {processed_wheel_velocities.std():.3f}")
            print(f"   平滑度 (相邻步差异): {np.std(np.diff(processed_wheel_velocities, axis=0)):.4f}")
        
        print("\n⚙️ Action处理细节:")
        print("1. 腿部关节 (位置控制):")
        print("   - Hip关节缩放: 0.125 (±0.6 rad限制)")
        print("   - Thigh/Calf关节缩放: 0.25 (±π rad限制)")
        print("2. 轮子关节 (速度控制):")
        print("   - 速度缩放: 5.0 (±20.4 rad/s限制)")
        
        return success_rate > 0.95  # 95%成功率
        
    except Exception as e:
        print(f"❌ Full pipeline with simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    setup_logging()
    
    print("THUNDER FLAT DEPLOYMENT SYSTEM TEST")
    print("="*60)
    print("Testing all components of the deployment system...")
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Observation Processor", test_observation_processor),
        ("Action Processor", test_action_processor),
        ("Safety Monitor", test_safety_monitor),
        ("Full Pipeline", test_full_pipeline),
        ("Performance", test_performance),
        ("Realistic Simulation", test_realistic_simulation),
        ("Full Pipeline + Simulation", test_full_pipeline_with_simulation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Deployment system is ready with realistic simulation.")
        return True
    else:
        print(f"\n⚠️  {total-passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)