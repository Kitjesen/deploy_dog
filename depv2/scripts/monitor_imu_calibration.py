#!/usr/bin/env python3
"""
IMU Calibration Monitor
IMU calibration monitoring program - real-time display of IMU observation data for calibration
"""

import time
import numpy as np
import logging
from typing import Dict
import signal
import sys
import os
from datetime import datetime

# Add parent directory to sys.path to import lib modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.hardware.hipnuc_imu_interface import HipnucIMUInterface
from lib.state.robot_state import quat_to_projected_gravity
from lib.config.config_manager import get_default_config

class IMUCalibrationMonitor:
    """IMU Calibration Monitor"""
    
    def __init__(self):
        """Initialize monitor"""
        self.config = get_default_config()
        self.imu_interface = HipnucIMUInterface(self.config['imu'])
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('IMUCalibrationMonitor')
        
        # Runtime control
        self.is_running = False
        
        # Data statistics
        self.sample_count = 0
        self.start_time = None
        
        # Store recent data for analysis
        self.recent_ang_vel = []
        self.recent_proj_grav = []
        self.max_history = 100
        
        # Setup signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        
        print("=" * 80)
        print("🎯 IMU Calibration Monitor")
        print("=" * 80)
        print("This program will display the following key IMU observation data:")
        print("  📐 base_ang_vel: Base angular velocity (robot coordinate system)")
        print("  🌍 projected_gravity: Projected gravity vector")
        print("=" * 80)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C signal"""
        print("\n\n🛑 Stop signal received, stopping monitor...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start monitoring"""
        print("🚀 Starting IMU interface...")
        
        if not self.imu_interface.start():
            print("❌ Failed to start IMU interface!")
            return False
        
        print("✅ IMU interface started successfully")
        print("⏱️  Waiting for IMU data to stabilize...")
        time.sleep(2)  # Wait for data to stabilize
        
        print("\n📊 Starting real-time monitoring (Press Ctrl+C to stop):")
        print("-" * 80)
        
        self.is_running = True
        self.start_time = time.time()
        
        try:
            self._monitor_loop()
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            self.logger.error(f"Error during monitoring: {e}")
            self.stop()
    
    def stop(self):
        """Stop monitoring"""
        if self.is_running:
            self.is_running = False
            print("\n🔄 Stopping IMU interface...")
            self.imu_interface.stop()
            print("✅ Monitoring stopped")
            
            # Show statistics
            self._show_statistics()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        last_display_time = 0
        display_interval = 0.2  # Display every 200ms
        
        while self.is_running:
            current_time = time.time()
            
            # Get latest IMU data
            imu_data = self.imu_interface.get_latest_data()
            
            if imu_data is not None:
                # Calculate observation data
                base_ang_vel = imu_data.angular_velocity.copy()
                projected_gravity = quat_to_projected_gravity(imu_data.orientation)
                
                # Store data for statistics
                self.recent_ang_vel.append(base_ang_vel)
                self.recent_proj_grav.append(projected_gravity)
                
                # Limit history data length
                if len(self.recent_ang_vel) > self.max_history:
                    self.recent_ang_vel.pop(0)
                    self.recent_proj_grav.pop(0)
                
                self.sample_count += 1
                
                # Display data at intervals
                if current_time - last_display_time >= display_interval:
                    self._display_data(base_ang_vel, projected_gravity, current_time)
                    last_display_time = current_time
            
            time.sleep(0.01)  # 100Hz check frequency
    
    def _display_data(self, base_ang_vel: np.ndarray, projected_gravity: np.ndarray, timestamp: float):
        """Display real-time data"""
        if self.start_time is not None:
            runtime = timestamp - self.start_time
        else:
            runtime = 0
        
        # Clear screen and display data
        print("\033[2J\033[H", end="")  # ANSI clear screen command
        
        print("=" * 80)
        print(f"🎯 IMU Calibration Monitor | Runtime: {runtime:.1f}s | Samples: {self.sample_count}")
        print("=" * 80)
        
        # Display current time
        current_time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"⏰ Time: {current_time_str}")
        print()
        
        # Display base angular velocity
        print("📐 Base Angular Velocity (base_ang_vel) [rad/s]:")
        print(f"   X-axis: {base_ang_vel[0]:+8.4f}")
        print(f"   Y-axis: {base_ang_vel[1]:+8.4f}")
        print(f"   Z-axis: {base_ang_vel[2]:+8.4f}")
        print(f"   Magnitude: {np.linalg.norm(base_ang_vel):8.4f}")
        print()
        
        # Display projected gravity
        print("🌍 Projected Gravity Vector (projected_gravity):")
        print(f"   X-axis: {projected_gravity[0]:+8.4f}")
        print(f"   Y-axis: {projected_gravity[1]:+8.4f}")
        print(f"   Z-axis: {projected_gravity[2]:+8.4f}")
        print(f"   Magnitude: {np.linalg.norm(projected_gravity):8.4f}")
        print()
        
        # Display raw quaternion
        imu_data = self.imu_interface.get_latest_data()
        if imu_data is not None:
            quat = imu_data.orientation
            print("🔄 Quaternion (w, x, y, z):")
            print(f"   w: {quat[0]:+8.4f}")
            print(f"   x: {quat[1]:+8.4f}")
            print(f"   y: {quat[2]:+8.4f}")
            print(f"   z: {quat[3]:+8.4f}")
            print()
        
        # Display statistics
        if len(self.recent_ang_vel) > 10:
            ang_vel_std = np.std(self.recent_ang_vel, axis=0)
            proj_grav_std = np.std(self.recent_proj_grav, axis=0)
            
            print("📊 Recent Data Stability (Standard Deviation):")
            print(f"   Angular velocity stability: X:{ang_vel_std[0]:.4f} Y:{ang_vel_std[1]:.4f} Z:{ang_vel_std[2]:.4f}")
            print(f"   Gravity vector stability: X:{proj_grav_std[0]:.4f} Y:{proj_grav_std[1]:.4f} Z:{proj_grav_std[2]:.4f}")
            print()
        
        # Display IMU connection status
        imu_stats = self.imu_interface.get_statistics()
        print("🔗 IMU Status:")
        print(f"   Data rate: {imu_stats.get('data_rate', 0):.1f} Hz")
        print(f"   Packets received: {imu_stats.get('packets_received', 0)}")
        print(f"   Packets parsed: {imu_stats.get('packets_parsed', 0)}")
        print(f"   Parse errors: {imu_stats.get('parse_errors', 0)}")
        print()
        
        print("=" * 80)
        print("💡 Calibration Tips:")
        print("  - When robot is stationary, angular velocity should be close to [0, 0, 0]")
        print("  - When robot is placed horizontally, projected gravity should be close to [0, 0, -1]")
        print("  - Observe data stability, smaller standard deviation is better")
        print("  - Press Ctrl+C to stop monitoring")
        print("=" * 80)
    
    def _show_statistics(self):
        """Show final statistics"""
        if not self.recent_ang_vel:
            return
        
        print("\n" + "=" * 80)
        print("📈 Calibration Session Statistics")
        print("=" * 80)
        
        if self.start_time is not None:
            runtime = time.time() - self.start_time
        else:
            runtime = 0
        print(f"Total runtime: {runtime:.1f} seconds")
        print(f"Total samples: {self.sample_count}")
        print(f"Average sampling rate: {self.sample_count / runtime:.1f} Hz" if runtime > 0 else "Average sampling rate: N/A")
        print()
        
        # Calculate statistics
        ang_vel_array = np.array(self.recent_ang_vel)
        proj_grav_array = np.array(self.recent_proj_grav)
        
        print("📐 Angular Velocity Statistics [rad/s]:")
        ang_vel_mean = np.mean(ang_vel_array, axis=0)
        ang_vel_std = np.std(ang_vel_array, axis=0)
        print(f"   Mean: X:{ang_vel_mean[0]:+8.4f} Y:{ang_vel_mean[1]:+8.4f} Z:{ang_vel_mean[2]:+8.4f}")
        print(f"   Std Dev: X:{ang_vel_std[0]:8.4f} Y:{ang_vel_std[1]:8.4f} Z:{ang_vel_std[2]:8.4f}")
        print()
        
        print("🌍 Projected Gravity Statistics:")
        proj_grav_mean = np.mean(proj_grav_array, axis=0)
        proj_grav_std = np.std(proj_grav_array, axis=0)
        print(f"   Mean: X:{proj_grav_mean[0]:+8.4f} Y:{proj_grav_mean[1]:+8.4f} Z:{proj_grav_mean[2]:+8.4f}")
        print(f"   Std Dev: X:{proj_grav_std[0]:8.4f} Y:{proj_grav_std[1]:8.4f} Z:{proj_grav_std[2]:8.4f}")
        print()
        
        # Calibration quality assessment
        print("🎯 Calibration Quality Assessment:")
        ang_vel_noise = np.mean(ang_vel_std)
        proj_grav_noise = np.mean(proj_grav_std)
        
        if ang_vel_noise < 0.01:
            print("   ✅ Angular velocity noise level: Excellent")
        elif ang_vel_noise < 0.05:
            print("   ⚠️ Angular velocity noise level: Good")
        else:
            print("   ❌ Angular velocity noise level: Needs improvement")
        
        if proj_grav_noise < 0.01:
            print("   ✅ Gravity vector stability: Excellent")
        elif proj_grav_noise < 0.05:
            print("   ⚠️ Gravity vector stability: Good")
        else:
            print("   ❌ Gravity vector stability: Needs improvement")
        
        print("=" * 80)


def main():
    """Main function"""
    monitor = IMUCalibrationMonitor()
    monitor.start()


if __name__ == "__main__":
    main() 