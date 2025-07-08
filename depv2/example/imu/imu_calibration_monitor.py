#!/usr/bin/env python3
"""
IMU Calibration Monitor
IMU标定监控程序 - 实时显示IMU观测数据，方便标定
"""

import time
import numpy as np
import logging
from typing import Dict
import signal
import sys
from datetime import datetime

from hipnuc_imu_interface import HipnucIMUInterface
from robot_state import quat_to_projected_gravity
from config_manager import get_default_config

class IMUCalibrationMonitor:
    """IMU标定监控器"""
    
    def __init__(self):
        """初始化监控器"""
        self.config = get_default_config()
        self.imu_interface = HipnucIMUInterface(self.config['imu'])
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('IMUCalibrationMonitor')
        
        # 运行控制
        self.is_running = False
        
        # 数据统计
        self.sample_count = 0
        self.start_time = None
        
        # 存储最近的数据用于分析
        self.recent_ang_vel = []
        self.recent_proj_grav = []
        self.max_history = 100
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        
        print("=" * 80)
        print("🎯 IMU Calibration Monitor")
        print("=" * 80)
        print("这个程序将显示以下关键的IMU观测数据:")
        print("  📐 base_ang_vel: 基础角速度 (机器人坐标系)")
        print("  🌍 projected_gravity: 投影重力向量")
        print("=" * 80)
    
    def _signal_handler(self, signum, frame):
        """处理Ctrl+C信号"""
        print("\n\n🛑 收到停止信号，正在停止监控...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """启动监控"""
        print("🚀 启动IMU接口...")
        
        if not self.imu_interface.start():
            print("❌ 启动IMU接口失败!")
            return False
        
        print("✅ IMU接口启动成功")
        print("⏱️  等待IMU数据稳定...")
        time.sleep(2)  # 等待数据稳定
        
        print("\n📊 开始实时监控 (按Ctrl+C停止):")
        print("-" * 80)
        
        self.is_running = True
        self.start_time = time.time()
        
        try:
            self._monitor_loop()
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            self.logger.error(f"监控过程出错: {e}")
            self.stop()
    
    def stop(self):
        """停止监控"""
        if self.is_running:
            self.is_running = False
            print("\n🔄 正在停止IMU接口...")
            self.imu_interface.stop()
            print("✅ 监控已停止")
            
            # 显示统计信息
            self._show_statistics()
    
    def _monitor_loop(self):
        """主监控循环"""
        last_display_time = 0
        display_interval = 0.2  # 每200ms显示一次
        
        while self.is_running:
            current_time = time.time()
            
            # 获取最新IMU数据
            imu_data = self.imu_interface.get_latest_data()
            
            if imu_data is not None:
                # 计算观测数据
                base_ang_vel = imu_data.angular_velocity.copy()
                projected_gravity = quat_to_projected_gravity(imu_data.orientation)
                
                # 存储数据用于统计
                self.recent_ang_vel.append(base_ang_vel)
                self.recent_proj_grav.append(projected_gravity)
                
                # 限制历史数据长度
                if len(self.recent_ang_vel) > self.max_history:
                    self.recent_ang_vel.pop(0)
                    self.recent_proj_grav.pop(0)
                
                self.sample_count += 1
                
                # 按间隔显示数据
                if current_time - last_display_time >= display_interval:
                    self._display_data(base_ang_vel, projected_gravity, current_time)
                    last_display_time = current_time
            
            time.sleep(0.01)  # 100Hz检查频率
    
    def _display_data(self, base_ang_vel: np.ndarray, projected_gravity: np.ndarray, timestamp: float):
        """显示实时数据"""
        if self.start_time is not None:
            runtime = timestamp - self.start_time
        else:
            runtime = 0
        
        # 清屏并显示数据
        print("\033[2J\033[H", end="")  # ANSI清屏命令
        
        print("=" * 80)
        print(f"🎯 IMU Calibration Monitor | 运行时间: {runtime:.1f}s | 样本: {self.sample_count}")
        print("=" * 80)
        
        # 显示当前时间
        current_time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"⏰ 时间: {current_time_str}")
        print()
        
        # 显示基础角速度
        print("📐 基础角速度 (base_ang_vel) [rad/s]:")
        print(f"   X轴: {base_ang_vel[0]:+8.4f}")
        print(f"   Y轴: {base_ang_vel[1]:+8.4f}")
        print(f"   Z轴: {base_ang_vel[2]:+8.4f}")
        print(f"   模长: {np.linalg.norm(base_ang_vel):8.4f}")
        print()
        
        # 显示投影重力
        print("🌍 投影重力向量 (projected_gravity):")
        print(f"   X轴: {projected_gravity[0]:+8.4f}")
        print(f"   Y轴: {projected_gravity[1]:+8.4f}")
        print(f"   Z轴: {projected_gravity[2]:+8.4f}")
        print(f"   模长: {np.linalg.norm(projected_gravity):8.4f}")
        print()
        
        # 显示原始四元数
        imu_data = self.imu_interface.get_latest_data()
        if imu_data is not None:
            quat = imu_data.orientation
            print("🔄 四元数 (w, x, y, z):")
            print(f"   w: {quat[0]:+8.4f}")
            print(f"   x: {quat[1]:+8.4f}")
            print(f"   y: {quat[2]:+8.4f}")
            print(f"   z: {quat[3]:+8.4f}")
            print()
        
        # 显示统计信息
        if len(self.recent_ang_vel) > 10:
            ang_vel_std = np.std(self.recent_ang_vel, axis=0)
            proj_grav_std = np.std(self.recent_proj_grav, axis=0)
            
            print("📊 近期数据稳定性 (标准差):")
            print(f"   角速度稳定性: X:{ang_vel_std[0]:.4f} Y:{ang_vel_std[1]:.4f} Z:{ang_vel_std[2]:.4f}")
            print(f"   重力向量稳定性: X:{proj_grav_std[0]:.4f} Y:{proj_grav_std[1]:.4f} Z:{proj_grav_std[2]:.4f}")
            print()
        
        # 显示IMU连接状态
        imu_stats = self.imu_interface.get_statistics()
        print("🔗 IMU状态:")
        print(f"   数据频率: {imu_stats.get('data_rate', 0):.1f} Hz")
        print(f"   接收包数: {imu_stats.get('packets_received', 0)}")
        print(f"   解析包数: {imu_stats.get('packets_parsed', 0)}")
        print(f"   解析错误: {imu_stats.get('parse_errors', 0)}")
        print()
        
        print("=" * 80)
        print("💡 标定提示:")
        print("  - 保持机器人静止时，角速度应接近[0, 0, 0]")
        print("  - 机器人水平放置时，投影重力应接近[0, 0, -1]")
        print("  - 观察数据稳定性，标准差越小越好")
        print("  - 按 Ctrl+C 停止监控")
        print("=" * 80)
    
    def _show_statistics(self):
        """显示最终统计信息"""
        if not self.recent_ang_vel:
            return
        
        print("\n" + "=" * 80)
        print("📈 标定会话统计")
        print("=" * 80)
        
        if self.start_time is not None:
            runtime = time.time() - self.start_time
        else:
            runtime = 0
        print(f"总运行时间: {runtime:.1f} 秒")
        print(f"总样本数: {self.sample_count}")
        print(f"平均采样率: {self.sample_count / runtime:.1f} Hz" if runtime > 0 else "平均采样率: N/A")
        print()
        
        # 计算统计数据
        ang_vel_array = np.array(self.recent_ang_vel)
        proj_grav_array = np.array(self.recent_proj_grav)
        
        print("📐 角速度统计 [rad/s]:")
        ang_vel_mean = np.mean(ang_vel_array, axis=0)
        ang_vel_std = np.std(ang_vel_array, axis=0)
        print(f"   均值: X:{ang_vel_mean[0]:+8.4f} Y:{ang_vel_mean[1]:+8.4f} Z:{ang_vel_mean[2]:+8.4f}")
        print(f"   标准差: X:{ang_vel_std[0]:8.4f} Y:{ang_vel_std[1]:8.4f} Z:{ang_vel_std[2]:8.4f}")
        print()
        
        print("🌍 投影重力统计:")
        proj_grav_mean = np.mean(proj_grav_array, axis=0)
        proj_grav_std = np.std(proj_grav_array, axis=0)
        print(f"   均值: X:{proj_grav_mean[0]:+8.4f} Y:{proj_grav_mean[1]:+8.4f} Z:{proj_grav_mean[2]:+8.4f}")
        print(f"   标准差: X:{proj_grav_std[0]:8.4f} Y:{proj_grav_std[1]:8.4f} Z:{proj_grav_std[2]:8.4f}")
        print()
        
        # 标定质量评估
        print("🎯 标定质量评估:")
        ang_vel_noise = np.mean(ang_vel_std)
        proj_grav_noise = np.mean(proj_grav_std)
        
        if ang_vel_noise < 0.01:
            print("   ✅ 角速度噪声水平: 优秀")
        elif ang_vel_noise < 0.05:
            print("   ⚠️ 角速度噪声水平: 良好")
        else:
            print("   ❌ 角速度噪声水平: 需要改善")
        
        if proj_grav_noise < 0.01:
            print("   ✅ 重力向量稳定性: 优秀")
        elif proj_grav_noise < 0.05:
            print("   ⚠️ 重力向量稳定性: 良好")
        else:
            print("   ❌ 重力向量稳定性: 需要改善")
        
        print("=" * 80)


def main():
    """主函数"""
    monitor = IMUCalibrationMonitor()
    monitor.start()


if __name__ == "__main__":
    main() 