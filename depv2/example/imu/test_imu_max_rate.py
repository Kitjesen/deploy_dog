#!/usr/bin/env python3
"""
IMU极限速率测试
以最高频率输出角速度和四元数，测试系统极限性能
"""

import time
import numpy as np
from hipnuc_imu_interface import HipnucIMUInterface

def test_max_rate():
    """测试IMU极限输出速率"""
    
    print("🚀 IMU极限速率测试")
    print("=" * 60)
    
    # 配置IMU (最高性能设置)
    config = {
        'serial_port': '/dev/ttyUSB0',
        'baud_rate': 115200,
        'update_frequency': 1000,  # 设置很高的频率
        'buffer_size': 100         # 增大缓冲区
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
        print("📊 开始极限速率测试...")
        print("格式: [时间] 角速度=[wx, wy, wz] 四元数=[w, x, y, z]")
        print("-" * 80)
        
        start_time = time.time()
        sample_count = 0
        last_data = None
        
        # 极限速率运行 (无延迟)
        test_duration = 10.0  # 测试10秒
        
        while time.time() - start_time < test_duration:
            # 获取最新数据 (无延迟)
            current_data = imu.get_latest_data()
            
            # 只有数据更新时才输出 (避免重复)
            if current_data and (last_data is None or 
                                current_data.timestamp != last_data.timestamp):
                
                elapsed = time.time() - start_time
                ang_vel = current_data.angular_velocity
                quat = current_data.orientation
                
                # 格式化输出
                print(f"[{elapsed:7.3f}s] "
                      f"角速度=[{ang_vel[0]:8.5f}, {ang_vel[1]:8.5f}, {ang_vel[2]:8.5f}] "
                      f"四元数=[{quat[0]:7.4f}, {quat[1]:7.4f}, {quat[2]:7.4f}, {quat[3]:7.4f}]")
                
                sample_count += 1
                last_data = current_data
        
        total_time = time.time() - start_time
        
        print("-" * 80)
        print(f"🏁 测试完成!")
        print(f"📈 性能统计:")
        print(f"   测试时长: {total_time:.3f} 秒")
        print(f"   输出样本: {sample_count} 个")
        print(f"   ⭐ 实际输出频率: {sample_count / total_time:.2f} Hz")
        
        # 获取IMU内部统计
        stats = imu.get_statistics()
        print(f"📊 IMU内部统计:")
        print(f"   接收包数: {stats['packets_received']}")
        print(f"   解析成功: {stats['packets_parsed']}")
        print(f"   解析频率: {stats['data_rate']:.2f} Hz")
        print(f"   解析成功率: {stats['packets_parsed']/max(1,stats['packets_received'])*100:.1f}%")
        print(f"   数据新鲜度: {stats['data_age']:.3f} 秒")
        
        # 性能评估
        theoretical_max = min(stats['data_rate'], 1000)  # 理论最大值
        efficiency = (sample_count / total_time) / theoretical_max * 100
        
        print(f"⚡ 性能评估:")
        print(f"   理论最大频率: {theoretical_max:.2f} Hz")
        print(f"   输出效率: {efficiency:.1f}%")
        
        if efficiency > 80:
            print("   🎉 性能优秀! 系统运行在高效率状态")
        elif efficiency > 60:
            print("   ✅ 性能良好! 满足大多数应用需求")
        else:
            print("   ⚠️ 性能一般，可能存在瓶颈")
        
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
    test_max_rate() 