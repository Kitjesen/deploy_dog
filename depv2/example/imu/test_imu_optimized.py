#!/usr/bin/env python3
"""
优化后的IMU测试程序
测试改进的时间戳机制，验证是否能正确输出94Hz的数据
"""

import time
import numpy as np
from hipnuc_imu_interface import HipnucIMUInterface
from robot_state import quat_to_projected_gravity

def test_optimized_imu():
    """测试优化后的IMU接口"""
    
    print("🚀 优化后的IMU测试 - 目标94Hz输出")
    print("=" * 60)
    
    # 配置IMU
    config = {
        'serial_port': '/dev/ttyUSB0',
        'baud_rate': 115200,
        'update_frequency': 200,
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
        print("📊 开始优化测试...")
        print("格式: [序列] [时间] 角速度=[wx, wy, wz] 重力=[gx, gy, gz] 间隔(ms)")
        print("-" * 120)
        
        start_time = time.time()
        sample_count = 0
        last_data = None
        last_timestamp = None
        
        # 记录时间间隔和重力数据
        intervals = []
        gravity_vectors = []
        quaternion_norms = []
        
        test_duration = 10.0  # 测试10秒
        
        while time.time() - start_time < test_duration:
            # 获取最新数据
            current_data = imu.get_latest_data()
            
            # 检查是否为新数据（基于序列号）
            if current_data and (last_data is None or 
                                current_data.sequence != last_data.sequence):
                
                elapsed = time.time() - start_time
                ang_vel = current_data.angular_velocity
                quat = current_data.orientation
                
                # 计算投影重力
                quat_norm = np.linalg.norm(quat)
                quaternion_norms.append(quat_norm)
                
                if quat_norm > 0.1:  # 如果四元数有效
                    normalized_quat = quat / quat_norm
                    proj_gravity = quat_to_projected_gravity(normalized_quat)
                else:
                    proj_gravity = np.array([0.0, 0.0, 0.0])  # 无效四元数
                
                gravity_vectors.append(proj_gravity.copy())
                
                # 计算时间间隔
                interval_ms = 0.0
                if last_timestamp is not None:
                    interval_ms = (current_data.timestamp - last_timestamp) * 1000
                    intervals.append(interval_ms)
                

                print(f"[{current_data.sequence:4d}] [{elapsed:7.3f}s] "
                        f"四元数=[{quat[0]:8.5f}, {quat[1]:8.5f}, {quat[2]:8.5f}, {quat[3]:8.5f}] "
                        f"角速度=[{ang_vel[0]:8.5f}, {ang_vel[1]:8.5f}, {ang_vel[2]:8.5f}] "
                        f"重力=[{proj_gravity[0]:8.5f}, {proj_gravity[1]:8.5f}, {proj_gravity[2]:8.5f}] "
                        f"间隔={interval_ms:6.2f}ms")
                
                sample_count += 1
                last_data = current_data
                last_timestamp = current_data.timestamp
            
            # 小延迟，避免过度占用CPU
            time.sleep(0.001)  # 1ms
        
        total_time = time.time() - start_time
        
        print("-" * 120)
        print(f"🏁 测试完成!")
        print(f"📈 性能统计:")
        print(f"   测试时长: {total_time:.3f} 秒")
        print(f"   输出样本: {sample_count} 个")
        print(f"   ⭐ 实际输出频率: {sample_count / total_time:.2f} Hz")
        
        # 时间间隔分析
        if intervals:
            intervals = np.array(intervals)
            print(f"📏 时间间隔分析:")
            print(f"   平均间隔: {np.mean(intervals):.2f} ms (期望: {1000/94:.2f} ms)")
            print(f"   最小间隔: {np.min(intervals):.2f} ms")
            print(f"   最大间隔: {np.max(intervals):.2f} ms")
            print(f"   标准差: {np.std(intervals):.2f} ms")
            
            # 频率稳定性分析
            expected_interval = 1000.0 / 94.0  # ~10.64ms
            deviations = np.abs(intervals - expected_interval)
            stable_count = np.sum(deviations < 2.0)  # 2ms内的变化算稳定
            stability = stable_count / len(intervals) * 100
            
            print(f"📊 频率稳定性:")
            print(f"   稳定性: {stability:.1f}% (±2ms内)")
            print(f"   最大偏差: {np.max(deviations):.2f} ms")
        
        # 四元数和重力向量分析
        if quaternion_norms and gravity_vectors:
            quat_norms = np.array(quaternion_norms)
            gravity_array = np.array(gravity_vectors)
            
            print(f"🔄 四元数分析:")
            print(f"   平均模长: {np.mean(quat_norms):.4f} (期望: 1.0000)")
            print(f"   最小模长: {np.min(quat_norms):.4f}")
            print(f"   最大模长: {np.max(quat_norms):.4f}")
            valid_quats = np.sum(quat_norms > 0.1)
            print(f"   有效四元数: {valid_quats}/{len(quat_norms)} ({valid_quats/len(quat_norms)*100:.1f}%)")
            
            print(f"🌍 重力向量分析:")
            gravity_mean = np.mean(gravity_array, axis=0)
            gravity_std = np.std(gravity_array, axis=0)
            gravity_magnitude = np.linalg.norm(gravity_mean)
            print(f"   平均重力: [{gravity_mean[0]:8.4f}, {gravity_mean[1]:8.4f}, {gravity_mean[2]:8.4f}]")
            print(f"   标准差: [{gravity_std[0]:8.4f}, {gravity_std[1]:8.4f}, {gravity_std[2]:8.4f}]")
            print(f"   模长: {gravity_magnitude:.4f} (期望: ~1.0)")
            
            # 评估重力向量质量
            if abs(gravity_magnitude - 1.0) < 0.1:
                print("   ✅ 重力向量质量: 优秀")
            elif abs(gravity_magnitude - 1.0) < 0.3:
                print("   ⚠️ 重力向量质量: 良好")
            else:
                print("   ❌ 重力向量质量: 需要改善")
        
        # 获取IMU内部统计
        stats = imu.get_statistics()
        print(f"📊 IMU内部统计:")
        print(f"   接收包数: {stats['packets_received']}")
        print(f"   解析成功: {stats['packets_parsed']}")
        print(f"   序列号: {stats['sequence_number']}")
        print(f"   解析频率: {stats['data_rate']:.2f} Hz")
        print(f"   解析成功率: {stats['packets_parsed']/max(1,stats['packets_received'])*100:.1f}%")
        
        # 性能评估
        target_freq = 94.0
        actual_freq = sample_count / total_time
        freq_accuracy = (1 - abs(actual_freq - target_freq) / target_freq) * 100
        
        print(f"🎯 性能评估:")
        print(f"   目标频率: {target_freq:.1f} Hz")
        print(f"   实际频率: {actual_freq:.2f} Hz")
        print(f"   频率准确度: {freq_accuracy:.1f}%")
        
        if freq_accuracy > 90:
            print("   🎉 优化成功! 输出频率非常接近目标94Hz")
        elif freq_accuracy > 80:
            print("   ✅ 优化有效! 输出频率较为准确")
        else:
            print("   ⚠️ 需要进一步优化")
        
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
    test_optimized_imu() 