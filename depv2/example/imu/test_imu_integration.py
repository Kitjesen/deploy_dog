#!/usr/bin/env python3
"""
测试IMU集成到Thunder Robot Interface
验证IMU数据是否正确集成到观测处理器中
"""

import os
import sys
import time
import numpy as np
import yaml
import logging
from pathlib import Path

# 添加部署目录到路径
deployment_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', 'deployment')
sys.path.insert(0, deployment_dir)

from thunder_robot_interface import ThunderRobotInterface
from observation_processor import ThunderObservationProcessor

class IMUIntegrationTester:
    """IMU集成测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.logger = self._setup_logger()
        self.config = self._load_config()
        
        # 初始化组件
        self.robot_interface = None
        self.obs_processor = None
    
    def _setup_logger(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('IMUIntegrationTester')
    
    def _load_config(self):
        """加载配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'thunder_flat_config.yaml')
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            # 返回默认配置
            return self._get_default_config()
    
    def _get_default_config(self):
        """获取默认配置"""
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
            }
        }
    
    def test_imu_integration(self) -> bool:
        """测试IMU集成"""
        print("\n" + "="*60)
        print("Testing IMU Integration with Thunder Robot Interface")
        print("="*60)
        
        try:
            # 1. 初始化机器人接口
            print("\n1. 初始化Thunder Robot Interface...")
            self.robot_interface = ThunderRobotInterface(self.config)
            
            # 2. 只启动状态接收器（不需要完整初始化）
            print("\n2. 启动状态接收器...")
            if not self.robot_interface.state_receiver.start_receiving():
                print("❌ 状态接收器启动失败")
                return False
            
            print("✅ 状态接收器启动成功")
            
            # 3. 初始化观测处理器
            print("\n3. 初始化观测处理器...")
            self.obs_processor = ThunderObservationProcessor(self.config)
            print("✅ 观测处理器初始化成功")
            
            # 4. 等待IMU连接
            print("\n4. 等待IMU连接...")
            imu_connected = False
            for i in range(10):  # 等待10秒
                stats = self.robot_interface.state_receiver.get_statistics()
                if 'imu' in stats and stats['imu'].get('connected', False):
                    imu_connected = True
                    print("✅ IMU连接成功")
                    break
                print(f"   等待IMU连接... ({i+1}/10)")
                time.sleep(1)
            
            if not imu_connected:
                print("⚠️ IMU未连接，将使用模拟数据继续测试")
            
            # 5. 测试观测数据处理
            print("\n5. 测试观测数据处理...")
            return self._test_observation_processing()
            
        except Exception as e:
            self.logger.error(f"IMU integration test failed: {e}")
            print(f"❌ IMU集成测试失败: {e}")
            return False
    
    def _test_observation_processing(self) -> bool:
        """测试观测数据处理"""
        print("\n测试观测数据处理流程:")
        
        # 创建测试速度指令和动作
        velocity_commands = np.array([1.0, 0.0, 0.0])  # 前进
        last_actions = np.zeros(16)
        
        observation_count = 0
        imu_data_received = 0
        
        try:
            for i in range(50):  # 测试5秒，10Hz
                # 获取机器人状态
                robot_state = self.robot_interface.get_robot_state()
                
                if robot_state:
                    observation_count += 1
                    
                    # 检查IMU数据
                    imu_ang_vel = robot_state.get('imu_angular_velocity', np.zeros(3))
                    imu_orientation = robot_state.get('imu_orientation', np.array([1,0,0,0]))
                    
                    # 检查是否为非零IMU数据（表示真实数据）
                    if np.any(imu_ang_vel != 0) or not np.allclose(imu_orientation, [1,0,0,0]):
                        imu_data_received += 1
                    
                    # 处理观测
                    observation = self.obs_processor.process_observations(
                        robot_state, velocity_commands, last_actions
                    )
                    
                    # 验证观测有效性
                    if not self.obs_processor.validate_observation(observation):
                        print(f"❌ 观测验证失败 at step {i}")
                        return False
                    
                    # 每10步打印一次状态
                    if i % 10 == 0:
                        print(f"\n--- Step {i} ---")
                        print(f"IMU Angular Velocity: {imu_ang_vel}")
                        print(f"IMU Orientation: {imu_orientation}")
                        print(f"Observation shape: {observation.shape}")
                        print(f"Observation range: [{observation.min():.3f}, {observation.max():.3f}]")
                        
                        # 打印obs breakdown
                        print("Observation breakdown:")
                        print(f"  base_ang_vel: {observation[0:3]}")
                        print(f"  projected_gravity: {observation[3:6]}")
                        print(f"  velocity_commands: {observation[6:9]}")
                
                time.sleep(0.1)  # 10Hz
            
            # 打印统计结果
            print(f"\n📊 测试统计:")
            print(f"   观测数据处理次数: {observation_count}")
            print(f"   IMU真实数据接收次数: {imu_data_received}")
            print(f"   IMU数据接收率: {imu_data_received/observation_count*100:.1f}%")
            
            # 打印最终统计信息
            stats = self.robot_interface.state_receiver.get_statistics()
            if 'imu' in stats:
                imu_stats = stats['imu']
                print(f"\n📡 IMU统计信息:")
                print(f"   连接状态: {imu_stats.get('connected', False)}")
                print(f"   数据更新率: {imu_stats.get('data_rate', 0):.1f} Hz")
                print(f"   接收包数: {imu_stats.get('packets_received', 0)}")
                print(f"   解析包数: {imu_stats.get('packets_parsed', 0)}")
                print(f"   解析错误: {imu_stats.get('parse_errors', 0)}")
                print(f"   数据延迟: {imu_stats.get('data_age', float('inf')):.3f} s")
            
            print("\n✅ 观测数据处理测试完成")
            return True
            
        except Exception as e:
            print(f"❌ 观测数据处理测试失败: {e}")
            return False
    
    def cleanup(self):
        """清理资源"""
        if self.robot_interface:
            try:
                self.robot_interface.state_receiver.stop_receiving()
                print("✅ 状态接收器已停止")
            except Exception as e:
                print(f"⚠️ 停止状态接收器时出错: {e}")

def main():
    """主函数"""
    tester = IMUIntegrationTester()
    
    try:
        # 运行测试
        success = tester.test_imu_integration()
        
        if success:
            print("\n🎉 IMU集成测试成功!")
            print("\n接下来你可以:")
            print("1. 运行 test_thunder_step.py 测试完整的步骤流程")
            print("2. 运行 thunder_flat_deploy.py 启动完整部署")
            print("3. 检查IMU数据是否准确反映机器人姿态")
        else:
            print("\n❌ IMU集成测试失败")
            print("\n请检查:")
            print("1. IMU设备是否正确连接到 /dev/ttyUSB0")
            print("2. 串口权限是否正确")
            print("3. hipnuc_imu_interface.py 中的协议解析是否正确")
            print("4. 网络连接是否正常")
            
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main() 