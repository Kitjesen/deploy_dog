#!/usr/bin/env python3
"""
IMU Setup Script
设置IMU相关的依赖和权限
"""

import os
import sys
import subprocess
import logging

def setup_logger():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('IMUSetup')

def check_python_dependencies():
    """检查Python依赖"""
    logger = setup_logger()
    logger.info("检查Python依赖...")
    
    required_packages = [
        'pyserial',
        'numpy',
        'scipy',
        'pyyaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"❌ {package} 未安装")
    
    if missing_packages:
        logger.info(f"尝试安装缺失的包: {missing_packages}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            logger.info("✅ 依赖包安装完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 依赖包安装失败: {e}")
            return False
    
    return True

def check_serial_permissions():
    """检查串口权限"""
    logger = setup_logger()
    logger.info("检查串口权限...")
    
    # 检查常见的串口设备
    serial_devices = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']
    
    accessible_devices = []
    for device in serial_devices:
        if os.path.exists(device):
            if os.access(device, os.R_OK | os.W_OK):
                accessible_devices.append(device)
                logger.info(f"✅ {device} 可访问")
            else:
                logger.warning(f"⚠️ {device} 存在但无权限访问")
        else:
            logger.info(f"ℹ️ {device} 不存在")
    
    if not accessible_devices:
        logger.warning("❌ 没有找到可访问的串口设备")
        logger.info("请尝试以下方法:")
        logger.info("1. 将用户添加到dialout组: sudo usermod -a -G dialout $USER")
        logger.info("2. 重新登录或重启")
        logger.info("3. 检查IMU设备是否正确连接")
        return False
    else:
        logger.info(f"✅ 找到可访问的串口设备: {accessible_devices}")
        return True

def create_test_script():
    """创建简单的IMU测试脚本"""
    logger = setup_logger()
    logger.info("创建IMU测试脚本...")
    
    test_script = """#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(__file__))

from hipnuc_imu_interface import HipnucIMUInterface
import time

# 测试配置
config = {
    'serial_port': '/dev/ttyUSB0',
    'baud_rate': 115200,
    'update_frequency': 100,
    'buffer_size': 50
}

# 创建IMU接口
imu = HipnucIMUInterface(config)

print("启动IMU接口...")
if imu.start():
    print("IMU接口启动成功，运行10秒测试...")
    
    for i in range(100):  # 10秒，10Hz
        time.sleep(0.1)
        
        # 获取最新数据
        data = imu.get_latest_data()
        stats = imu.get_statistics()
        
        if i % 10 == 0:  # 每秒打印一次
            print(f"\\n=== 第 {i//10 + 1} 秒 ===")
            if data:
                print(f"角速度: {data.angular_velocity}")
                print(f"加速度: {data.linear_acceleration}")
                print(f"四元数: {data.orientation}")
            else:
                print("暂无IMU数据")
            
            print(f"统计: 接收={stats['packets_received']}, "
                  f"解析={stats['packets_parsed']}, "
                  f"错误={stats['parse_errors']}, "
                  f"频率={stats['data_rate']:.1f}Hz")
    
    imu.stop()
    print("\\n测试完成!")
else:
    print("IMU接口启动失败!")
"""
    
    try:
        with open('test_imu_simple.py', 'w') as f:
            f.write(test_script)
        
        # 设置可执行权限
        os.chmod('test_imu_simple.py', 0o755)
        
        logger.info("✅ 创建测试脚本: test_imu_simple.py")
        return True
    except Exception as e:
        logger.error(f"❌ 创建测试脚本失败: {e}")
        return False

def check_hipnuc_protocol():
    """检查hipnuc协议解析器"""
    logger = setup_logger()
    logger.info("检查hipnuc协议...")
    
    try:
        from hipnuc_imu_interface import HipnucIMUInterface
        logger.info("✅ hipnuc_imu_interface 模块导入成功")
        
        # 测试实例化
        test_config = {
            'serial_port': '/dev/ttyUSB0',
            'baud_rate': 115200,
            'update_frequency': 100,
            'buffer_size': 50
        }
        
        imu = HipnucIMUInterface(test_config)
        logger.info("✅ HipnucIMUInterface 实例化成功")
        return True
        
    except ImportError as e:
        logger.error(f"❌ 导入hipnuc_imu_interface失败: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 实例化HipnucIMUInterface失败: {e}")
        return False

def main():
    """主函数"""
    logger = setup_logger()
    
    print("="*60)
    print("Thunder Robot IMU Setup")
    print("="*60)
    
    success = True
    
    # 1. 检查Python依赖
    if not check_python_dependencies():
        success = False
    
    # 2. 检查串口权限
    if not check_serial_permissions():
        success = False
    
    # 3. 检查协议解析器
    if not check_hipnuc_protocol():
        success = False
    
    # 4. 创建测试脚本
    if not create_test_script():
        success = False
    
    print("\n" + "="*60)
    if success:
        print("🎉 IMU设置完成!")
        print("\n接下来你可以:")
        print("1. 运行 python test_imu_simple.py 测试IMU连接")
        print("2. 运行 python test_imu_integration.py 测试IMU集成")
        print("3. 运行 python hipnuc_imu_interface.py 测试原始接口")
    else:
        print("❌ IMU设置过程中遇到问题")
        print("\n请解决上述问题后重新运行此脚本")
    
    print("="*60)

if __name__ == "__main__":
    main() 