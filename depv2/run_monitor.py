#!/usr/bin/env python3

import time
import asyncio
import logging
import numpy as np
from typing import Dict
import signal
import sys
import torch

from lib.hardware.thunder_robot_interface import ThunderRobotInterface
from lib.core.observation_processor import ThunderObservationProcessor
from lib.core.action_processor import ThunderActionProcessor
# from lib.safety.safety_monitor import ThunderSafetyMonitor  # 禁用安全监控
from lib.state.robot_state import quat_to_projected_gravity

class ThunderHardwareMonitor:
    """Thunder硬件监控器"""
    
    def __init__(self):
        """初始化硬件监控器"""
        self.logger = self._setup_logger()
        
        # 配置
        self.config = self._create_hardware_config()
        
        # 组件
        self.robot_interface = ThunderRobotInterface(self.config)
        self.obs_processor = ThunderObservationProcessor(self.config)
        self.action_processor = ThunderActionProcessor(self.config)
        # self.safety_monitor = ThunderSafetyMonitor(self.config)  # 禁用安全监控
        
        # 模型与动作
        self.model = self._load_model()
        self.last_actions = torch.zeros(self.config['robot']['num_actions'])
        self.raw_model_output = None
        self.processed_actions = None
        
        # 监控状态
        self.is_running = False
        self.step_count = 0
        self.start_time = 0.0
        
        # IMU数据统计
        self.imu_history = []
        self.gravity_history = []
        
        # Step速率统计
        self._last_display_time = None
        self._last_display_step = 0
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Thunder Hardware Monitor initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器 - 关闭安全监控的冗余输出"""
        # 设置根日志级别
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 完全关闭SafetyMonitor的所有日志输出
        safety_logger = logging.getLogger('SafetyMonitor')
        safety_logger.disabled = True  # 完全禁用SafetyMonitor日志
        
        # 关闭其他组件的冗余日志
        logging.getLogger('HipnucIMUInterface').setLevel(logging.WARNING)
        logging.getLogger('MotorController').setLevel(logging.WARNING)
        logging.getLogger('HardwareStateReceiver').setLevel(logging.WARNING)
        
        return logging.getLogger('HardwareMonitor')
    
    def _create_hardware_config(self) -> Dict:
        """创建硬件配置"""
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
                'num_actions': 16, # 12 leg joints + 4 wheels
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
            'control': {
                'motor_server_host': '192.168.66.159',
                'motor_server_port': 12345,
                'pd_gains': {
                    'kp': [80, 80, 80],
                    'kd': [4, 4, 4],
                    'wheel_kp': 30.0,
                    'wheel_kd': 1.0
                }
            },
            'imu': {
                'enabled': True,
                'serial_port': '/dev/ttyUSB0',
                'baud_rate': 115200,
                'update_frequency': 200,
                'buffer_size': 100,
                'data_timeout': 0.5,
                'validate_data': True
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
                'max_position_error': 0.5,
                'max_velocity': 25.0,
                'emergency_stop_enabled': False  # 关闭紧急停止以减少日志
            }
        }
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False
    
    def _load_model(self):
        """加载JIT编译的PyTorch模型"""
        # TODO: 请将此路径替换为您的实际模型文件路径
        model_path = "/home/ubuntu/Desktop/dog_deploy/depv2/exported/policy.pt" # 示例路径, 请确保此文件存在
        self.logger.info(f"Attempting to load model from: {model_path}")
        
        try:
            model = torch.jit.load(model_path)
            model.eval()  # 设置为评估模式
            self.logger.info("✅ Model loaded successfully.")
            return model
        except Exception as e:
            self.logger.error(f"❌ Failed to load model from {model_path}: {e}")
            self.logger.error("Running in monitor-only mode without inference.")
            return None
    
    async def start_monitoring(self):
        """开始硬件监控"""
        print("\n" + "="*80, flush=True)
        print("🚀 THUNDER HARDWARE MONITOR - Real-time Status Display", flush=True)
        print("="*80, flush=True)
        print("📡 Connecting to hardware...", flush=True)
        
        # 连接硬件
        if not await self.robot_interface.connect():
            print("❌ Failed to connect to hardware")
            return False
        
        print("✅ Hardware connected successfully!", flush=True)
        print("📊 Starting real-time monitoring...", flush=True)
        print("   Press Ctrl+C to stop monitoring", flush=True)
        print()
        
        self.is_running = True
        self.start_time = time.time()
        
        # 监控循环
        await self._monitoring_loop()
        
        # 清理
        await self.robot_interface.disconnect()
        return True
    
    async def _monitoring_loop(self):
        """监控主循环"""
        loop_freq = self.config['robot']['control_frequency']
        loop_interval = 1.0 / loop_freq
        
        # 在监控模式下,我们使用零速度指令
        velocity_commands = np.zeros(3)

        while self.is_running:
            loop_start = time.time()
            
            try:
                # 1. 获取观测数据
                raw_obs = self.robot_interface.get_observations()

                # 2. 构建观测向量
                self.obs_processor.update_last_actions(self.last_actions.numpy())
                obs = self.obs_processor.process_observations(
                    raw_obs, velocity_commands
                )

                # 3. 模型推理 (如果模型已加载)
                if self.model is not None:
                    with torch.no_grad():
                        raw_actions = self.model(obs.unsqueeze(0)).squeeze(0)
                    self.raw_model_output = raw_actions
                    
                    # 4. 处理动作
                    self.processed_actions = self.action_processor.process_actions(raw_actions.numpy())
                    self.last_actions = raw_actions
                else:
                    self.processed_actions = np.zeros(self.config['robot']['num_actions'])

                # IMU数据分析
                self._update_imu_analysis(raw_obs)
                
                # 每5步(0.1秒)显示详细状态 - 更实时的更新
                if self.step_count % 5 == 0:
                    self._display_hardware_status(raw_obs, obs, self.processed_actions)
                
                self.step_count += 1
                
                # 控制循环频率
                loop_time = time.time() - loop_start
                if loop_time < loop_interval:
                    await asyncio.sleep(loop_interval - loop_time)
                
                # 输出本次step的耗时（ms）
                step_duration = (time.time() - loop_start) * 1000
                # print(f"[Step Interval] {step_duration:.2f} ms")
            
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(0.1)
    
    def _update_imu_analysis(self, raw_obs: Dict):
        """更新IMU数据分析 - 集成test_imu_optimized.py的功能"""
        try:
            # 获取IMU数据
            ang_vel = raw_obs.get('base_ang_vel', np.zeros(3))
            proj_gravity = raw_obs.get('projected_gravity', np.zeros(3))
            
            # 计算重力向量模长
            gravity_magnitude = np.linalg.norm(proj_gravity)
            
            # 更新历史数据（保留最近50个数据点）
            self.imu_history.append({
                'timestamp': time.time(),
                'angular_velocity': ang_vel.copy(),
                'projected_gravity': proj_gravity.copy(),
                'gravity_magnitude': gravity_magnitude
            })
            
            if len(self.imu_history) > 50:
                self.imu_history.pop(0)
                
        except Exception as e:
            pass  # 静默处理IMU数据错误
    
    def _display_hardware_status(self, raw_obs: Dict, processed_obs, processed_actions: np.ndarray):
        """显示硬件状态 - 改进的显示格式"""
        current_time = time.time()
        runtime = current_time - self.start_time
        
        # Step速率测算
        if self._last_display_time is not None:
            step_diff = self.step_count - self._last_display_step
            time_diff = current_time - self._last_display_time
            if time_diff > 0 and step_diff > 0:
                step_rate = step_diff / time_diff
                step_interval_ms = (time_diff / step_diff) * 1000
                print(f"[Step Rate] {step_rate:.2f} Hz, Interval: {step_interval_ms:.2f} ms", flush=True)
        self._last_display_time = current_time
        self._last_display_step = self.step_count
        
        # 清屏并显示标题
        print("\033[2J\033[H", end="", flush=True)  # 清屏
        
        # 实时更新指示器
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        spinner = spinner_chars[self.step_count % len(spinner_chars)]
        
        print("="*80, flush=True)
        print(f"🤖 THUNDER ROBOT STATUS {spinner} | Step: {self.step_count:6d} | Runtime: {runtime:8.1f}s | Update: 10Hz", flush=True)
        print("="*80, flush=True)
        
        # 系统连接状态
        system_status = self.robot_interface.get_system_status()
        self._display_connection_status(system_status)
        
        # IMU状态和数据
        self._display_imu_status(raw_obs, system_status)
        
        # 电机状态 - 显示全部16个
        self._display_motor_status(raw_obs)
        
        # 观测向量状态
        self._display_observation_status(processed_obs)

        # 模型动作输出
        self._display_action_status(processed_actions)
        
        print("="*80, flush=True)
        print("💡 Press Ctrl+C to stop monitoring")
    
    def _display_connection_status(self, system_status: Dict):
        """显示连接状态"""
        print("🔗 CONNECTION STATUS", flush=True)
        print("-"*40, flush=True)
        
        # 机器人接口
        robot_connected = system_status['connected']
        print(f"  Robot Interface:     {'🟢 ONLINE ' if robot_connected else '🔴 OFFLINE'}", flush=True)
        
        # 电机控制器
        motor_status = system_status.get('motor_controller', {})
        motor_connected = motor_status.get('connected', False)
        emergency_stop = motor_status.get('emergency_stop', False)
        print(f"  Motor Controller:    {'🟢 ONLINE ' if motor_connected else '🔴 OFFLINE'}", flush=True)
        if motor_connected and emergency_stop:
            print(f"                       🚨 EMERGENCY STOP ACTIVE", flush=True)
        
        # IMU
        imu_status = system_status.get('state_receiver', {}).get('imu', {})
        imu_connected = imu_status.get('connected', False)
        print(f"  IMU Sensor:          {'🟢 ONLINE ' if imu_connected else '🔴 OFFLINE'}", flush=True)
        
        print()
    
    def _display_imu_status(self, raw_obs: Dict, system_status: Dict):
        """显示IMU状态和数据"""
        print("📡 IMU SENSOR DATA", flush=True)
        print("-"*40, flush=True)
        
        imu_status = system_status.get('state_receiver', {}).get('imu', {})
        
        if imu_status.get('connected', False):
            # 数据统计
            data_rate = imu_status.get('data_rate', 0)
            packets = imu_status.get('packets_received', 0)
            parsed = imu_status.get('packets_parsed', 0)
            parse_rate = (parsed / max(1, packets)) * 100
            
            # 最后更新时间显示
            current_timestamp = time.strftime('%H:%M:%S')
            
            print(f"  Data Rate:           {data_rate:8.1f} Hz", flush=True)
            print(f"  Packets Received:    {packets:8d}", flush=True)
            print(f"  Parse Success:       {parse_rate:8.1f}%", flush=True)
            print(f"  Last Update:         {current_timestamp} (realtime)", flush=True)
            
            # 角速度数据
            ang_vel = raw_obs.get('base_ang_vel', np.zeros(3))
            print(f"  Angular Velocity:    [{ang_vel[0]:8.4f}, {ang_vel[1]:8.4f}, {ang_vel[2]:8.4f}] rad/s", flush=True)
            
            # 重力投影数据
            proj_gravity = raw_obs.get('projected_gravity', np.zeros(3))
            gravity_mag = np.linalg.norm(proj_gravity)
            print(f"  Projected Gravity:   [{proj_gravity[0]:8.4f}, {proj_gravity[1]:8.4f}, {proj_gravity[2]:8.4f}]", flush=True)
            print(f"  Gravity Magnitude:   {gravity_mag:8.4f} (Expected: ~1.0)", flush=True)
            
            # 质量评估
            if abs(gravity_mag - 1.0) < 0.1:
                quality = "🟢 EXCELLENT"
            elif abs(gravity_mag - 1.0) < 0.3:
                quality = "🟡 GOOD"
            else:
                quality = "🔴 POOR"
            print(f"  Data Quality:        {quality}", flush=True)
            
        else:
            print("  Status:              🔴 DISCONNECTED", flush=True)
            print("  Angular Velocity:    [  0.0000,   0.0000,   0.0000] rad/s", flush=True)
            print("  Projected Gravity:   [  0.0000,   0.0000,   0.0000]", flush=True)
        
        print()
    
    def _display_motor_status(self, raw_obs: Dict):
        """显示所有16个电机状态"""
        print("🦾 MOTOR STATUS (16 Joints) - Live Data", flush=True)
        print("-"*80, flush=True)
        print(f"  Last Update:         {raw_obs} (joint_pos)", flush=True)
        joint_pos = raw_obs.get('joint_pos', np.zeros(16))
        joint_vel = raw_obs.get('joint_vel', np.zeros(16))
        joint_names = self.config['robot']['joint_names']
        
        # 按腿部分组显示
        legs = ['FL', 'FR', 'RL', 'RR']
        joint_types = ['hip', 'thigh', 'calf']
        
        for i, leg in enumerate(legs):
            leg_start = i * 3
            print(f"  {leg} Leg:  ", end="", flush=True)
            for j, joint_type in enumerate(joint_types):
                idx = leg_start + j
                if idx < len(joint_pos):
                    pos = joint_pos[idx]
                    vel = joint_vel[idx]
                    print(f"{joint_type}({pos:6.2f},{vel:5.1f}) ", end="", flush=True)
            print(flush=True)
        
        # 轮子关节
        print("  Wheels: ", end="", flush=True)
        wheel_indices = [12, 13, 14, 15]
        for i, idx in enumerate(wheel_indices):
            if idx < len(joint_pos):
                pos = joint_pos[idx]
                vel = joint_vel[idx]
                leg_name = legs[i] if i < len(legs) else f"W{i}"
                print(f"{leg_name}({pos:6.2f},{vel:5.1f}) ", end="", flush=True)
        print(flush=True)
        print()
    
    def _display_observation_status(self, processed_obs):
        """显示观测向量状态"""
        print("📊 OBSERVATION VECTOR", flush=True)
        print("-"*40, flush=True)
        
        if processed_obs is not None:
            obs_np = processed_obs.numpy()
            obs_dim = len(obs_np)
            obs_min = obs_np.min()
            obs_max = obs_np.max()
            obs_mean = obs_np.mean()
            obs_std = obs_np.std()
            
            print(f"  Dimension:           {obs_dim:8d}", flush=True)
            print(f"  Value Range:         [{obs_min:8.3f}, {obs_max:8.3f}]", flush=True)
            print(f"  Mean ± Std:          {obs_mean:8.3f} ± {obs_std:6.3f}", flush=True)
            
            # 检查异常值
            nan_count = np.sum(np.isnan(obs_np))
            inf_count = np.sum(np.isinf(obs_np))
            if nan_count > 0 or inf_count > 0:
                print(f"  ⚠️  Anomalies:        NaN:{nan_count}, Inf:{inf_count}", flush=True)
            else:
                print(f"  Data Integrity:      🟢 HEALTHY", flush=True)
        else:
            print("  Status:              🔴 NO DATA", flush=True)
        
        print()

    def _display_action_status(self, processed_actions: np.ndarray):
        """显示模型动作输出"""
        print("🚀 MODEL ACTIONS (Processed)", flush=True)
        print("-"*40, flush=True)
        
        if self.model is None:
            print("  Status:              ⚪️ MODEL NOT LOADED", flush=True)
            return
            
        if processed_actions is not None:
            action_dim = len(processed_actions)
            action_min = processed_actions.min()
            action_max = processed_actions.max()
            action_mean = processed_actions.mean()
            
            print(f"  Dimension:           {action_dim:8d}", flush=True)
            print(f"  Value Range:         [{action_min:8.3f}, {action_max:8.3f}]", flush=True)
            print(f"  Mean:                {action_mean:8.3f}", flush=True)
            
            # 以紧凑格式显示动作值
            action_str = "  " + " ".join([f"{a:6.2f}" for a in processed_actions])
            print(f"  Values:              {action_str}", flush=True)
        else:
            print("  Status:              🔴 NO ACTION DATA", flush=True)

def main():
    """主函数"""
    monitor = ThunderHardwareMonitor()
    
    try:
        # 运行监控
        success = asyncio.run(monitor.start_monitoring())
        
        if success:
            print("\n✅ Hardware monitoring completed successfully", flush=True)
        else:
            print("\n❌ Hardware monitoring failed to start", flush=True)
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user", flush=True)
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}", flush=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 