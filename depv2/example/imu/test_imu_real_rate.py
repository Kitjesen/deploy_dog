#!/usr/bin/env python3
"""
IMU真实速率测试 - 无去重版本
显示所有解析成功的数据包，不基于时间戳去重
"""

import time
import numpy as np
from hipnuc_imu_interface import HipnucIMUInterface

def test_real_rate():
    """测试IMU真实数据流速率"""
    
    print("🔍 IMU真实数据流测试 (无去重)")
    print("=" * 60)
    
    # 配置IMU
    config = {
        'serial_port': '/dev/ttyUSB0',
        'baud_rate': 115200,
        'update_frequency': 1000,
        'buffer_size': 100
    }
    
    print(f"📡 配置: 串口={config['serial_port']}, 波特率={config['baud_rate']}")
    
    # 创建IMU接口
    imu = HipnucIMUInterface(config)
    
    if not imu.start():
        print("❌ IMU启动失败")
        return False
    
    print("✅ IMU启动成功")
    
    try:
        # 等待连接稳定
        time.sleep(0.5)
        print("📊 开始真实速率测试...")
        print("格式: [计数] [时间] 角速度=[wx, wy, wz] 时间戳差值")
        print("-" * 80)
        
        start_time = time.time()
        call_count = 0           # get_latest_data调用次数
        data_count = 0           # 获取到数据的次数  
        unique_data_count = 0    # 唯一数据计数
        last_data = None
        last_timestamp = None
        
        test_duration = 10.0  # 测试10秒
        
        # 统计时间戳分布
        timestamp_intervals = []
        
        while time.time() - start_time < test_duration:
            call_count += 1
            
            # 获取最新数据 (每次调用都计数)
            current_data = imu.get_latest_data()
            
            if current_data:
                data_count += 1
                elapsed = time.time() - start_time
                
                # 计算时间戳差值
                timestamp_diff = 0.0
                if last_timestamp:
                    timestamp_diff = current_data.timestamp - last_timestamp
                    timestamp_intervals.append(timestamp_diff)
                
                # 检查是否为新数据 (使用时间戳判断)
                is_new_data = (last_data is None or 
                              current_data.timestamp != last_data.timestamp)
                
                if is_new_data:
                    unique_data_count += 1
                    ang_vel = current_data.angular_velocity
                    
                    # 详细输出
                    print(f"[{unique_data_count:3d}] [{elapsed:7.3f}s] "
                          f"角速度=[{ang_vel[0]:8.5f}, {ang_vel[1]:8.5f}, {ang_vel[2]:8.5f}] "
                          f"Δt={timestamp_diff*1000:6.2f}ms")
                    
                    last_data = current_data
                    last_timestamp = current_data.timestamp
                
                # 每100次调用显示一次统计
                if call_count % 100 == 0:
                    print(f"    [统计] 调用:{call_count}, 有数据:{data_count}, 唯一:{unique_data_count}")
            
            # 无延迟，最大频率调用
        
        total_time = time.time() - start_time
        
        print("-" * 80)
        print(f"🏁 测试完成!")
        print(f"📈 详细统计:")
        print(f"   测试时长: {total_time:.3f} 秒")
        print(f"   🔄 总调用次数: {call_count}")
        print(f"   📦 获取数据次数: {data_count}")
        print(f"   ⭐ 唯一数据包: {unique_data_count}")
        print(f"   📞 调用频率: {call_count / total_time:.2f} Hz")
        print(f"   📊 数据获取频率: {data_count / total_time:.2f} Hz")
        print(f"   🎯 真实数据更新频率: {unique_data_count / total_time:.2f} Hz")
        
        # 分析时间戳间隔
        if timestamp_intervals:
            intervals = np.array(timestamp_intervals) * 1000  # 转换为毫秒
            print(f"📏 时间戳间隔分析:")
            print(f"   平均间隔: {np.mean(intervals):.2f} ms")
            print(f"   最小间隔: {np.min(intervals):.2f} ms")
            print(f"   最大间隔: {np.max(intervals):.2f} ms")
            print(f"   标准差: {np.std(intervals):.2f} ms")
        
        # 获取IMU内部统计
        stats = imu.get_statistics()
        print(f"📊 IMU内部统计:")
        print(f"   接收包数: {stats['packets_received']}")
        print(f"   解析成功: {stats['packets_parsed']}")
        print(f"   解析频率: {stats['data_rate']:.2f} Hz")
        print(f"   解析成功率: {stats['packets_parsed']/max(1,stats['packets_received'])*100:.1f}%")
        
        # 分析数据重复率
        if call_count > 0:
            data_hit_rate = data_count / call_count * 100
            uniqueness_rate = unique_data_count / max(1, data_count) * 100
            
            print(f"🎯 数据流分析:")
            print(f"   数据命中率: {data_hit_rate:.1f}% ({data_count}/{call_count})")
            print(f"   数据唯一率: {uniqueness_rate:.1f}% ({unique_data_count}/{data_count})")
            print(f"   重复数据: {data_count - unique_data_count} 个")
        
        # 性能诊断
        print(f"💡 性能诊断:")
        if unique_data_count / total_time < 30:
            print("   ⚠️ 数据更新频率较低，可能IMU硬件输出频率限制")
        if data_count > unique_data_count * 2:
            print("   ⚠️ 检测到大量重复数据，可能存在缓存或时间戳问题")
        if call_count / total_time > 1000:
            print("   ✅ 调用频率很高，软件性能良好")
        
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
    test_real_rate() 