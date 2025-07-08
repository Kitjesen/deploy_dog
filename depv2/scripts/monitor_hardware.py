#!/usr/bin/env python3
"""
Thunder Hardware Monitor
Real hardware monitoring program - connects and monitors actual Thunder robot hardware status
Integrated IMU testing and full 16 motor status display
"""

import time
import asyncio
import logging
import numpy as np
from typing import Dict
import signal
import sys
import torch
import os

from thunder_robot_interface import ThunderRobotInterface
from observation_processor import ThunderObservationProcessor
from action_processor import ThunderActionProcessor
# from safety_monitor import ThunderSafetyMonitor  # Disable safety monitoring
from robot_state import quat_to_projected_gravity
from config_manager import ConfigurationManager

class ThunderHardwareMonitor:
    """Thunder Hardware Monitor"""
    
    def __init__(self, config_path: str = "config/thunder_flat_config.yaml"):
        """Initialize hardware monitor"""
        self.logger = self._setup_logger()
        
        # Configuration - use ConfigurationManager
        self.config_manager = ConfigurationManager(config_path)
        self.config = self._create_hardware_config()  # Keep for backward compatibility
        
        # Components
        self.robot_interface = ThunderRobotInterface(self.config)
        self.obs_processor = ThunderObservationProcessor(self.config)
        self.action_processor = ThunderActionProcessor(self.config)
        # self.safety_monitor = ThunderSafetyMonitor(self.config)  # Disable safety monitoring
        
        # Model and actions
        self.model = self._load_model()
        self.last_actions = torch.zeros(self.config['robot']['num_actions'])
        self.raw_model_output = None
        self.processed_actions = None
        
        # Monitoring state
        self.is_running = False
        self.step_count = 0
        self.start_time = 0.0
        
        # IMU data statistics
        self.imu_history = []
        self.gravity_history = []
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Thunder Hardware Monitor initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger - disable redundant safety monitoring output"""
        # Set root log level
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Completely disable all SafetyMonitor log output
        safety_logger = logging.getLogger('SafetyMonitor')
        safety_logger.disabled = True  # Completely disable SafetyMonitor logs
        
        # Disable redundant logs from other components
        logging.getLogger('HipnucIMUInterface').setLevel(logging.WARNING)
        logging.getLogger('MotorController').setLevel(logging.WARNING)
        logging.getLogger('HardwareStateReceiver').setLevel(logging.WARNING)
        
        return logging.getLogger('HardwareMonitor')
    
    def _create_hardware_config(self) -> Dict:
        """Create hardware configuration"""
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
                'emergency_stop_enabled': False  # Disable emergency stop to reduce logs
            }
        }
    
    def _signal_handler(self, signum, frame):
        """Signal handler"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False
    
    def _load_model(self):
        """Load JIT compiled PyTorch model"""
        # Get model path from configuration
        model_config = self.config_manager.get_model_config()
        model_path = model_config.model_path
        self.logger.info(f"Attempting to load model from: {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            self.logger.warning(f"Model file not found: {model_path}")
            self.logger.warning("Running in monitor-only mode without inference.")
            return None
        
        try:
            model = torch.jit.load(model_path)
            model.eval()  # Set to evaluation mode
            self.logger.info("✅ Model loaded successfully.")
            return model
        except Exception as e:
            self.logger.error(f"❌ Failed to load model from {model_path}: {e}")
            self.logger.error("Running in monitor-only mode without inference.")
            return None
    
    async def start_monitoring(self):
        """Start hardware monitoring"""
        print("\n" + "="*80)
        print("🚀 THUNDER HARDWARE MONITOR - Real-time Status Display")
        print("="*80)
        print("📡 Connecting to hardware...")
        
        # Connect hardware
        if not await self.robot_interface.connect():
            print("❌ Failed to connect to hardware")
            return False
        
        print("✅ Hardware connected successfully!")
        print("📊 Starting real-time monitoring...")
        print("   Press Ctrl+C to stop monitoring")
        print()
        
        self.is_running = True
        self.start_time = time.time()
        
        # Monitoring loop
        await self._monitoring_loop()
        
        # Cleanup
        await self.robot_interface.disconnect()
        return True
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        loop_freq = self.config['robot']['control_frequency']
        loop_interval = 1.0 / loop_freq
        
        # In monitoring mode, we use zero velocity commands
        velocity_commands = np.zeros(3)

        while self.is_running:
            loop_start = time.time()
            
            try:
                # 1. Get observation data
                raw_obs = self.robot_interface.get_observations()

                # 2. Build observation vector
                self.obs_processor.update_last_actions(self.last_actions.numpy())
                obs = self.obs_processor.process_observations(
                    raw_obs, velocity_commands
                )

                # 3. Model inference (if model is loaded)
                if self.model is not None:
                    with torch.no_grad():
                        raw_actions = self.model(obs.unsqueeze(0)).squeeze(0)
                    self.raw_model_output = raw_actions
                    
                    # 4. Process actions
                    self.processed_actions = self.action_processor.process_actions(raw_actions.numpy())
                    self.last_actions = raw_actions
                else:
                    self.processed_actions = np.zeros(self.config['robot']['num_actions'])

                # IMU data analysis
                self._update_imu_analysis(raw_obs)
                
                # Display detailed status every 5 steps (0.1 seconds) - more real-time updates
                if self.step_count % 5 == 0:
                    self._display_hardware_status(raw_obs, obs, self.processed_actions)
                
                self.step_count += 1
                
                # Control loop frequency
                loop_time = time.time() - loop_start
                if loop_time < loop_interval:
                    await asyncio.sleep(loop_interval - loop_time)
                
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(0.1)
    
    def _update_imu_analysis(self, raw_obs: Dict):
        """Update IMU data analysis - integrate test_imu_optimized.py functionality"""
        try:
            # Get IMU data
            ang_vel = raw_obs.get('base_ang_vel', np.zeros(3))
            proj_gravity = raw_obs.get('projected_gravity', np.zeros(3))
            
            # Calculate gravity vector magnitude
            gravity_magnitude = np.linalg.norm(proj_gravity)
            
            # Update history data (keep last 50 data points)
            self.imu_history.append({
                'timestamp': time.time(),
                'angular_velocity': ang_vel.copy(),
                'projected_gravity': proj_gravity.copy(),
                'gravity_magnitude': gravity_magnitude
            })
            
            if len(self.imu_history) > 50:
                self.imu_history.pop(0)
                
        except Exception as e:
            pass  # Silently handle IMU data errors
    
    def _display_hardware_status(self, raw_obs: Dict, processed_obs, processed_actions: np.ndarray):
        """Display hardware status - improved display format"""
        current_time = time.time()
        runtime = current_time - self.start_time
        
        # Clear screen and display title
        print("\033[2J\033[H", end="")  # Clear screen
        
        # Real-time update indicator
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        spinner = spinner_chars[self.step_count % len(spinner_chars)]
        
        print("="*80)
        print(f"🤖 THUNDER ROBOT STATUS {spinner} | Step: {self.step_count:6d} | Runtime: {runtime:8.1f}s | Update: 10Hz")
        print("="*80)
        
        # System connection status
        system_status = self.robot_interface.get_system_status()
        self._display_connection_status(system_status)
        
        # IMU status and data
        self._display_imu_status(raw_obs, system_status)
        
        # Motor status - display all 16
        self._display_motor_status(raw_obs)
        
        # Observation vector status
        self._display_observation_status(processed_obs)

        # Model action output
        self._display_action_status(processed_actions)
        
        print("="*80)
        print("💡 Press Ctrl+C to stop monitoring")
    
    def _display_connection_status(self, system_status: Dict):
        """Display connection status"""
        print("🔗 CONNECTION STATUS")
        print("-"*40)
        
        # Robot interface
        robot_connected = system_status['connected']
        print(f"  Robot Interface:     {'🟢 ONLINE ' if robot_connected else '🔴 OFFLINE'}")
        
        # Motor controller
        motor_status = system_status.get('motor_controller', {})
        motor_connected = motor_status.get('connected', False)
        emergency_stop = motor_status.get('emergency_stop', False)
        print(f"  Motor Controller:    {'🟢 ONLINE ' if motor_connected else '🔴 OFFLINE'}")
        if motor_connected and emergency_stop:
            print(f"                       🚨 EMERGENCY STOP ACTIVE")
        
        # IMU
        imu_status = system_status.get('state_receiver', {}).get('imu', {})
        imu_connected = imu_status.get('connected', False)
        print(f"  IMU Sensor:          {'🟢 ONLINE ' if imu_connected else '🔴 OFFLINE'}")
        
        print()
    
    def _display_imu_status(self, raw_obs: Dict, system_status: Dict):
        """Display IMU status and data"""
        print("📡 IMU SENSOR DATA")
        print("-"*40)
        
        imu_status = system_status.get('state_receiver', {}).get('imu', {})
        
        if imu_status.get('connected', False):
            # Data statistics
            data_rate = imu_status.get('data_rate', 0)
            packets = imu_status.get('packets_received', 0)
            parsed = imu_status.get('packets_parsed', 0)
            parse_rate = (parsed / max(1, packets)) * 100
            
            # Last update time display
            current_timestamp = time.strftime('%H:%M:%S')
            
            print(f"  Data Rate:           {data_rate:8.1f} Hz")
            print(f"  Packets Received:    {packets:8d}")
            print(f"  Parse Success:       {parse_rate:8.1f}%")
            print(f"  Last Update:         {current_timestamp} (realtime)")
            
            # Angular velocity data
            ang_vel = raw_obs.get('base_ang_vel', np.zeros(3))
            print(f"  Angular Velocity:    [{ang_vel[0]:8.4f}, {ang_vel[1]:8.4f}, {ang_vel[2]:8.4f}] rad/s")
            
            # Projected gravity data
            proj_gravity = raw_obs.get('projected_gravity', np.zeros(3))
            gravity_mag = np.linalg.norm(proj_gravity)
            print(f"  Projected Gravity:   [{proj_gravity[0]:8.4f}, {proj_gravity[1]:8.4f}, {proj_gravity[2]:8.4f}]")
            print(f"  Gravity Magnitude:   {gravity_mag:8.4f} (Expected: ~1.0)")
            
            # Quality assessment
            if abs(gravity_mag - 1.0) < 0.1:
                quality = "🟢 EXCELLENT"
            elif abs(gravity_mag - 1.0) < 0.3:
                quality = "🟡 GOOD"
            else:
                quality = "🔴 POOR"
            print(f"  Data Quality:        {quality}")
            
        else:
            print("  Status:              🔴 DISCONNECTED")
            print("  Angular Velocity:    [  0.0000,   0.0000,   0.0000] rad/s")
            print("  Projected Gravity:   [  0.0000,   0.0000,   0.0000]")
        
        print()
    
    def _display_motor_status(self, raw_obs: Dict):
        """Display all 16 motor status"""
        print("🦾 MOTOR STATUS (16 Joints) - Live Data")
        print("-"*80)
        print(f"  Last Update:         {raw_obs} (joint_pos)")
        joint_pos = raw_obs.get('joint_pos', np.zeros(16))
        joint_vel = raw_obs.get('joint_vel', np.zeros(16))
        joint_names = self.config['robot']['joint_names']
        
        # Display grouped by legs
        legs = ['FL', 'FR', 'RL', 'RR']
        joint_types = ['hip', 'thigh', 'calf']
        
        for i, leg in enumerate(legs):
            leg_start = i * 3
            print(f"  {leg} Leg:  ", end="")
            for j, joint_type in enumerate(joint_types):
                idx = leg_start + j
                if idx < len(joint_pos):
                    pos = joint_pos[idx]
                    vel = joint_vel[idx]
                    print(f"{joint_type}({pos:6.2f},{vel:5.1f}) ", end="")
            print()
        
        # Wheel joints
        print("  Wheels: ", end="")
        wheel_indices = [12, 13, 14, 15]
        for i, idx in enumerate(wheel_indices):
            if idx < len(joint_pos):
                pos = joint_pos[idx]
                vel = joint_vel[idx]
                leg_name = legs[i] if i < len(legs) else f"W{i}"
                print(f"{leg_name}({pos:6.2f},{vel:5.1f}) ", end="")
        print()
        print()
    
    def _display_observation_status(self, processed_obs):
        """Display observation vector status"""
        print("📊 OBSERVATION VECTOR")
        print("-"*40)
        
        if processed_obs is not None:
            obs_np = processed_obs.numpy()
            obs_dim = len(obs_np)
            obs_min = obs_np.min()
            obs_max = obs_np.max()
            obs_mean = obs_np.mean()
            obs_std = obs_np.std()
            
            print(f"  Dimension:           {obs_dim:8d}")
            print(f"  Value Range:         [{obs_min:8.3f}, {obs_max:8.3f}]")
            print(f"  Mean ± Std:          {obs_mean:8.3f} ± {obs_std:6.3f}")
            
            # Check for anomalies
            nan_count = np.sum(np.isnan(obs_np))
            inf_count = np.sum(np.isinf(obs_np))
            if nan_count > 0 or inf_count > 0:
                print(f"  ⚠️  Anomalies:        NaN:{nan_count}, Inf:{inf_count}")
            else:
                print(f"  Data Integrity:      🟢 HEALTHY")
        else:
            print("  Status:              🔴 NO DATA")
        
        print()

    def _display_action_status(self, processed_actions: np.ndarray):
        """Display model action output"""
        print("🚀 MODEL ACTIONS (Processed)")
        print("-"*40)
        
        if self.model is None:
            print("  Status:              ⚪️ MODEL NOT LOADED")
            return
            
        if processed_actions is not None:
            action_dim = len(processed_actions)
            action_min = processed_actions.min()
            action_max = processed_actions.max()
            action_mean = processed_actions.mean()
            
            print(f"  Dimension:           {action_dim:8d}")
            print(f"  Value Range:         [{action_min:8.3f}, {action_max:8.3f}]")
            print(f"  Mean:                {action_mean:8.3f}")
            
            # Display action values in compact format
            action_str = "  " + " ".join([f"{a:6.2f}" for a in processed_actions])
            print(f"  Values:              {action_str}")
        else:
            print("  Status:              🔴 NO ACTION DATA")

def main():
    """Main function"""
    monitor = ThunderHardwareMonitor()
    
    try:
        # Run monitoring
        success = asyncio.run(monitor.start_monitoring())
        
        if success:
            print("\n✅ Hardware monitoring completed successfully")
        else:
            print("\n❌ Hardware monitoring failed to start")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 