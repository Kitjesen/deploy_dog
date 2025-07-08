#!/usr/bin/env python3
"""
IMU极限速率测试 V2
移除去重逻辑，测试真正的极限输出速率
"""

import time
import numpy as np
from hipnuc_imu_interface import HipnucIMUInterface

def test_max_rate_v2():
    """测试IMU真正的极限输出速率"""
    
    print("🚀 IMU极限速率测试 V2 (移除限制)")
    print("=" * 60)
    
    # 配置IMU (最高性能设置)
    config = {
        'serial_port': '/dev/ttyUSB0',
        'baud_rate': 115200,
        'update_frequency': 1000,  # 设置很高的频率
        'buffer_size': 200         # 进一步增大缓冲区
    }
    
    print(f"📡 配置: 串口={config['serial_port']}, 波特率={config['baud_rate']}")
    print(f"🔧 目标频率: {config['update_frequency']}Hz, 缓冲区: {config['buffer_size']}")
    
    # 创建IMU接口
    imu = HipnucIMUInterface(config)
    
    if not imu.start():
        print("❌ IMU启动失败")
        return False
    
    print("✅ IMU启动成功")
    
    try:
        # 短暂等待连接稳定
        time.sleep(0.5)
        print("📊 开始极限速率测试 (无去重逻辑)...")
        print("格式: [时间] 角速度=[wx, wy, wz] 四元数=[w, x, y, z]")
        print("-" * 80)
        
        start_time = time.time()
        sample_count = 0
        
        # 方法1: 完全无限制输出 (可能有重复)
        test_duration = 5.0  # 先测试5秒
        last_print_time = start_time
        print_interval = 0.01  # 每0.2秒打印一次状态
        
        while time.time() - start_time < test_duration:
            # 获取最新数据 (无任何限制)
            current_data = imu.get_latest_data()
            
            if current_data:
                elapsed = time.time() - start_time
                ang_vel = current_data.angular_velocity
                quat = current_data.orientation
                
                # 高频输出 (可能有重复数据)
                print(f"[{elapsed:7.3f}s] "
                      f"角速度=[{ang_vel[0]:8.5f}, {ang_vel[1]:8.5f}, {ang_vel[2]:8.5f}] "
                      f"四元数=[{quat[0]:7.4f}, {quat[1]:7.4f}, {quat[2]:7.4f}, {quat[3]:7.4f}]")
                
                sample_count += 1
                
                # 每0.2秒显示中间统计
                if time.time() - last_print_time > print_interval:
                    current_rate = sample_count / elapsed
                    stats = imu.get_statistics()
                    print(f">>> 中间统计: {elapsed:.1f}s, 输出={sample_count}, 当前频率={current_rate:.1f}Hz, IMU频率={stats['data_rate']:.1f}Hz")
                    last_print_time = time.time()
        
        total_time = time.time() - start_time
        
        print("-" * 80)
        print(f"🏁 方法1完成 (无限制输出)!")
        print(f"📈 性能统计:")
        print(f"   测试时长: {total_time:.3f} 秒")
        print(f"   输出样本: {sample_count} 个")
        print(f"   ⭐ 实际输出频率: {sample_count / total_time:.2f} Hz")
        
        # 获取IMU内部统计
        stats = imu.get_statistics()
        print(f"📊 IMU内部统计:")
        print(f"   接收包数: {stats['packets_received']}")
        print(f"   解析成功: {stats['packets_parsed']}")
        print(f"   内部频率: {stats['data_rate']:.2f} Hz")
        print(f"   解析成功率: {stats['packets_parsed']/max(1,stats['packets_received'])*100:.1f}%")
        
        print("\n" + "="*60)
        print("🔍 方法2: 测试纯数据获取速度 (不输出)")
        
        # 方法2: 纯数据获取测试，不输出到屏幕
        start_time2 = time.time()
        get_count = 0
        unique_timestamps = set()
        
        while time.time() - start_time2 < 3.0:  # 3秒纯速度测试
            current_data = imu.get_latest_data()
            if current_data:
                get_count += 1
                unique_timestamps.add(current_data.timestamp)
        
        total_time2 = time.time() - start_time2
        
        print(f"📈 纯数据获取统计:")
        print(f"   测试时长: {total_time2:.3f} 秒")
        print(f"   数据获取: {get_count} 次")
        print(f"   获取频率: {get_count / total_time2:.2f} Hz")
        print(f"   唯一时间戳: {len(unique_timestamps)} 个")
        print(f"   ⭐ 真实数据更新频率: {len(unique_timestamps) / total_time2:.2f} Hz")
        
        # 分析瓶颈
        print(f"\n🔍 瓶颈分析:")
        imu_rate = stats['data_rate']
        output_rate1 = sample_count / total_time
        get_rate = get_count / total_time2
        real_update_rate = len(unique_timestamps) / total_time2
        
        print(f"   IMU硬件频率: {imu_rate:.1f} Hz")
        print(f"   真实更新频率: {real_update_rate:.1f} Hz")
        print(f"   数据获取频率: {get_rate:.1f} Hz")
        print(f"   屏幕输出频率: {output_rate1:.1f} Hz")
        
        if real_update_rate < 80:
            print("   🔍 瓶颈: IMU硬件数据更新频率限制")
        elif get_rate < real_update_rate * 0.8:
            print("   🔍 瓶颈: Python数据获取处理速度")
        elif output_rate1 < get_rate * 0.8:
            print("   🔍 瓶颈: 屏幕输出I/O限制")
        else:
            print("   ✅ 系统运行在接近理论极限")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断测试")
        return True
        
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        imu.stop()
        print("🔄 IMU接口已关闭")

if __name__ == "__main__":
    test_max_rate_v2() 