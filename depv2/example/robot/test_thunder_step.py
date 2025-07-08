#!/usr/bin/env python3
"""
Thunder Flat Deploy Step 测试程序
测试观测获取和模型推理功能，不发送控制指令
"""

import os
import sys
import time
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, Optional
import logging

# 添加部署目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thunder_robot_interface import ThunderRobotInterface
from observation_processor import ThunderObservationProcessor
from action_processor import ThunderActionProcessor
from safety_monitor import ThunderSafetyMonitor

class ThunderStepTester:
    """Thunder Step 测试器（仅观测和推理）"""
    
    def __init__(self, model_path: str):
        """初始化测试器"""
        self.logger = self._setup_logger()
        self.model_path = model_path
        
        # 创建测试配置
        self.config = self._create_test_config()
        
        # 初始化组件
        self.model = self._load_model()
        self.robot_interface = ThunderRobotInterface(self.config)
        self.obs_processor = ThunderObservationProcessor(self.config)
        self.action_processor = ThunderActionProcessor(self.config)
        self.safety_monitor = ThunderSafetyMonitor(self.config)
        
        # 状态变量
        self.last_actions = torch.zeros(16, dtype=torch.float32)
        self.step_count = 0
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('ThunderStepTester')
    
    def _create_test_config(self) -> Dict:
        """创建测试配置"""
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
                ],
                'control_frequency': 50,
                'policy_frequency': 50.0
            },
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
                'observation_dim': 57
            },
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
                    'hip': [-120.0, 120.0],      # 使用实际电机能力
                    'thigh': [-120.0, 120.0],
                    'calf': [-120.0, 120.0],
                    'wheel': [-60.0, 60.0]
                },
                'max_position_error': 0.5,
                'max_velocity': 25.0,
                'emergency_stop_enabled': True
            },
            'commands': {
                'default_lin_vel_x': 0.0,
                'default_lin_vel_y': 0.0,
                'default_ang_vel_z': 0.0,
                'max_lin_vel_x': 1.5,
                'max_lin_vel_y': 0.8,
                'max_ang_vel_z': 1.5
            },
            'state_receiver': {
                'receive_frequency': 200,
                'motor_server_host': '192.168.66.159',
                'motor_server_port': 12345,
                'buffer_size': 100,
                'receive_timeout': 0.1
            },
            'hardware': {
                'pd_gains': {
                    'leg_joints': {
                        'kp': [80.0, 80.0, 80.0],  # [hip, thigh, calf]
                        'kd': [4.0, 4.0, 4.0]
                    },
                    'wheel_joints': {
                        'kp': 30.0,
                        'kd': 1.0
                    }
                }
            },
            'control': {
                'pd_gains': {
                    'kp': [80, 80, 80],
                    'kd': [4, 4, 4],
                    'wheel_kp': 30.0,
                    'wheel_kd': 1.0
                }
            }
        }
    
    def _load_model(self) -> torch.jit.ScriptModule:
        """加载PyTorch JIT模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            model = torch.jit.load(self.model_path)
            model.eval()
            
            # 测试模型维度
            test_input = torch.randn(1, self.config['observations']['observation_dim'])
            with torch.no_grad():
                test_output = model(test_input)
            
            expected_output_dim = self.config['actions']['action_dim']
            if test_output.shape[1] != expected_output_dim:
                raise ValueError(f"Model output dimension {test_output.shape[1]} != expected {expected_output_dim}")
            
            self.logger.info(f"Model loaded: {self.model_path}")
            self.logger.info(f"Input dim: {test_input.shape[1]}, Output dim: {test_output.shape[1]}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def test_step(self, velocity_commands: np.ndarray) -> bool:
        """
        测试一步观测获取和模型推理（不发送控制指令）
        
        Args:
            velocity_commands: 速度指令 [vx, vy, wz]
            
        Returns:
            bool: 是否成功执行
        """
        try:
            self.step_count += 1
            
            # 1. 获取真实机器人状态（直接从状态接收器获取）
            latest_robot_state = self.robot_interface.state_receiver.get_latest_state()
            if latest_robot_state is None:
                print(f"No robot state received yet in step {self.step_count}")
                return False
            
            # 转换为robot_interface格式
            robot_state = {
                'joint_positions': latest_robot_state.joint_positions,
                'joint_velocities': latest_robot_state.joint_velocities,
                'joint_torques': latest_robot_state.joint_torques,
                'base_position': latest_robot_state.base_position,
                'base_orientation': latest_robot_state.base_orientation,
                'base_lin_vel': latest_robot_state.base_linear_velocity,
                'base_ang_vel': latest_robot_state.base_angular_velocity,
                'imu_acceleration': latest_robot_state.imu_acceleration,
                'imu_angular_velocity': latest_robot_state.imu_angular_velocity,
                'imu_orientation': latest_robot_state.imu_orientation,
                'timestamp': latest_robot_state.timestamp
            }
            
            # 2. 安全检查
            if not self.safety_monitor.check_safety(robot_state):
                print(f"⚠️ Safety check failed in step {self.step_count}")
                return False
            
            # 3. 限制速度指令在安全范围内
            velocity_commands = np.clip(velocity_commands, 
                [-self.config['commands']['max_lin_vel_x'], 
                 -self.config['commands']['max_lin_vel_y'], 
                 -self.config['commands']['max_ang_vel_z']], 
                [self.config['commands']['max_lin_vel_x'], 
                 self.config['commands']['max_lin_vel_y'], 
                 self.config['commands']['max_ang_vel_z']])
            
            # 4. 构建观测向量
            observation = self.obs_processor.process_observations(
                robot_state, velocity_commands, self.last_actions.numpy()
            )
            
            # 验证观测数据
            if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
                print(f"⚠️ Invalid observation data in step {self.step_count}")
                return False
            
            # 5. 模型推理
            with torch.no_grad():
                observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                raw_actions = self.model(observation_tensor).squeeze(0)
            
            # 验证模型输出
            if torch.any(torch.isnan(raw_actions)) or torch.any(torch.isinf(raw_actions)):
                print(f"⚠️ Invalid model output in step {self.step_count}")
                return False
            
            # 6. 处理动作输出（但不发送）
            processed_actions = self.action_processor.process_actions(raw_actions.numpy(), robot_state)
            
            # ===== 输出关键信息 =====
            print(f"\n=== Step {self.step_count} ===")
            print(f"Velocity Commands: [{velocity_commands[0]:.2f}, {velocity_commands[1]:.2f}, {velocity_commands[2]:.2f}]")
            
            # 输出观测状态
            print(f"\n📊 OBSERVATION (shape: {observation.shape}):")
            print(f"  Base ang vel:      {observation[0:3]}")
            print(f"  Projected gravity: {observation[3:6]}")
            print(f"  Velocity commands: {observation[6:9]}")
            print(f"  Joint pos rel:     {observation[9:25]}")
            print(f"  Joint velocities:  {observation[25:41]}")
            print(f"  Last actions:      {observation[41:57]}")
            
            # 输出动作
            print(f"\n🎬 RAW ACTIONS (shape: {raw_actions.shape}):")
            print(f"  {raw_actions.numpy()}")
            
            # 输出处理后的动作
            if 'joint_torque_targets' in processed_actions:
                torque_targets = processed_actions['joint_torque_targets']
                print(f"\n🔧 TORQUE TARGETS (shape: {torque_targets.shape}):")
                print(f"  FL leg:  [{torque_targets[0]:6.3f}, {torque_targets[1]:6.3f}, {torque_targets[2]:6.3f}]")
                print(f"  FR leg:  [{torque_targets[3]:6.3f}, {torque_targets[4]:6.3f}, {torque_targets[5]:6.3f}]")
                print(f"  RL leg:  [{torque_targets[6]:6.3f}, {torque_targets[7]:6.3f}, {torque_targets[8]:6.3f}]")
                print(f"  RR leg:  [{torque_targets[9]:6.3f}, {torque_targets[10]:6.3f}, {torque_targets[11]:6.3f}]")
            
            if 'joint_velocity_targets' in processed_actions:
                vel_targets = processed_actions['joint_velocity_targets']
                print(f"  Wheels:  {vel_targets}")
            
            # 更新状态
            self.last_actions = raw_actions
            return True
            
        except Exception as e:
            print(f"❌ Error in step {self.step_count}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def initialize_state_receiver(self) -> bool:
        """仅初始化状态接收器（不启动控制）"""
        try:
            self.logger.info("🔄 Initializing state receiver only...")
            
            # 仅启动状态接收器
            if not self.robot_interface.state_receiver.start_receiving():
                self.logger.error("❌ Failed to start state receiver")
                return False
            
            # 等待接收到一些状态数据
            self.logger.info("⏳ Waiting for state data...")
            time.sleep(3.0)
            
            # 检查是否接收到状态
            stats = self.robot_interface.state_receiver.get_statistics()
            if stats['total_received'] > 0:
                self.logger.info(f"✅ State receiver initialized, received {stats['total_received']} states")
                return True
            else:
                self.logger.warning("⚠️  No state data received")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize state receiver: {e}")
            return False
    
    def run_test(self, num_steps: int = 10, velocity_commands: Optional[np.ndarray] = None):
        """运行测试"""
        print("Thunder Step Tester - Real Motor State & Inference")
        print("=" * 50)
        
        try:
            # 初始化状态接收器
            if not self.initialize_state_receiver():
                print("Failed to initialize state receiver")
                return False
            
            print("Connected to motor server, receiving real state data")
            
            # 设置速度指令
            if velocity_commands is None:
                velocity_commands = np.array([1.0, 0.0, 0.0])  # 前进1m/s
            
            print(f"Testing {num_steps} steps with velocity commands: {velocity_commands}")
            print("=" * 50)
            
            successful_steps = 0
            
            for i in range(num_steps):
                success = self.test_step(velocity_commands)
                
                if success:
                    successful_steps += 1
                else:
                    print(f"Step {i+1} failed")
                    break
                
                # 控制频率：50Hz = 20ms间隔
                time.sleep(0.02)
            
            # 显示测试结果
            print("\n" + "=" * 50)
            print("Test Results:")
            print(f"  Successful steps: {successful_steps}/{num_steps}")
            print(f"  Success rate: {successful_steps/num_steps*100:.1f}%")
            
            return successful_steps == num_steps
                
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
            return False
        except Exception as e:
            print(f"Test error: {e}")
            return False
        finally:
            # 停止状态接收器
            try:
                self.robot_interface.state_receiver.stop_receiving()
            except:
                pass


def main():
    """主函数"""
    model_path = "/home/ubuntu/Desktop/dog_deploy/deployment/exported/policy.pt"
    
    try:
        # 创建测试器
        tester = ThunderStepTester(model_path)
        
        # 运行测试（自动切换到模拟模式如果连接失败）
        success = tester.run_test(
            num_steps=5,  # 测试5步
            velocity_commands=np.array([1.0, 0.0, 0.0])  # 前进指令
        )
        
        if success:
            print("\nAll tests completed successfully!")
        else:
            print("\nSome tests failed!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 