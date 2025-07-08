#!/usr/bin/env python3
"""
Thunder Robot Interface 测试客户端
类似于 test_client.py，但专门用于测试电机状态接收和关节映射
"""

import asyncio
import sys
import os
from collections import deque

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.state import parse_motor_states, MotorFeedback
from state.event import MotorRequestClosed

# 关节名称定义（与 thunder_robot_interface.py 保持一致）
JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",      # 0,1,2
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",      # 3,4,5
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",      # 6,7,8
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",      # 9,10,11
    "FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"  # 12,13,14,15
]

# 电机映射（与 thunder_robot_interface.py 保持一致）
MOTOR_MAPPING = {
    # FL leg (port_id=1, motor_id=1,2,3) -> joint_index 0,1,2
    (1, 1): 0,  # FL_hip_joint
    (1, 2): 1,  # FL_thigh_joint  
    (1, 3): 2,  # FL_calf_joint
    # FR leg (port_id=2, motor_id=1,2,3) -> joint_index 3,4,5
    (2, 1): 3,  # FR_hip_joint
    (2, 2): 4,  # FR_thigh_joint
    (2, 3): 5,  # FR_calf_joint
    # RL leg (port_id=3, motor_id=1,2,3) -> joint_index 6,7,8
    (3, 1): 6,  # RL_hip_joint
    (3, 2): 7,  # RL_thigh_joint
    (3, 3): 8,  # RL_calf_joint
    # RR leg (port_id=4, motor_id=1,2,3) -> joint_index 9,10,11
    (4, 1): 9,  # RR_hip_joint
    (4, 2): 10, # RR_thigh_joint
    (4, 3): 11, # RR_calf_joint
    # Wheels (port_id=1,2,3,4, motor_id=4) -> joint_index 12,13,14,15
    (1, 4): 12, # FL_foot_joint (wheel)
    (2, 4): 13, # FR_foot_joint (wheel)
    (3, 4): 14, # RL_foot_joint (wheel)
    (4, 4): 15, # RR_foot_joint (wheel)
}


async def main():
    print('🤖 Thunder Robot State Reading Test Client')
    print('Similar to test_client.py but for robot state reading')
    print('=' * 60)
    
    try:
        # 创建 TCP 连接
        reader, writer = await asyncio.open_connection('192.168.66.159', 12345)
        print('✅ Connected to motor server')
        
        buffer_queue = deque()  # 类似 FIFO 队列
        motor_states = {}  # 存储最新的电机状态
        received_count = 0
        
        # 打印电机映射关系
        print(f'\n📋 Motor Mapping ({len(MOTOR_MAPPING)} motors):')
        for (port_id, motor_id), joint_index in sorted(MOTOR_MAPPING.items()):
            joint_name = JOINT_NAMES[joint_index] if joint_index < len(JOINT_NAMES) else "UNKNOWN"
            print(f'   Port {port_id}, Motor {motor_id} -> Joint {joint_index:2d} ({joint_name})')
        
        # 创建接收数据的任务
        async def receive_data():
            nonlocal received_count
            try:
                while True:
                    data = await reader.read(1024)
                    if not data:
                        break
                    
                    # 将接收到的字节加入队列
                    buffer_queue.extend(data)
                    
                    # 解析电机状态
                    for motor_state in parse_motor_states(buffer_queue):
                        if isinstance(motor_state, MotorFeedback):
                            received_count += 1
                            motor_key = (motor_state.port_id, motor_state.motor_id)
                            motor_states[motor_key] = motor_state
                            
                            # 获取关节信息
                            joint_index = MOTOR_MAPPING.get(motor_key, -1)
                            joint_name = JOINT_NAMES[joint_index] if 0 <= joint_index < len(JOINT_NAMES) else "UNKNOWN"
                            
                            print(f'📡 Motor State #{received_count}: Port {motor_state.port_id}, '
                                  f'Motor {motor_state.motor_id} -> {joint_name} | '
                                  f'Angle: {motor_state.angle:6.3f}, '
                                  f'Vel: {motor_state.velocity:6.3f}, '
                                  f'Torque: {motor_state.torque:6.3f}')
                            
                            # 每收到100个状态，显示当前完整的机器人状态
                            if received_count % 100 == 0:
                                print(f'\n🔄 Received {received_count} motor states. Current robot state:')
                                display_robot_state(motor_states)
                        
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f'❌ Error receiving data: {e}')
        
        # 启动接收任务
        receive_task = asyncio.create_task(receive_data())
        
        # 等待 5 秒接收数据
        print(f'\n⏱️  Receiving motor states for 5 seconds...\n')
        await asyncio.sleep(5)
        
        # 发送关闭请求
        print(f'\n📤 Sending close request...')
        close_command = MotorRequestClosed()
        writer.write(close_command.to_bytes())
        await writer.drain()
        
        # 取消接收任务并关闭连接
        receive_task.cancel()
        writer.close()
        await writer.wait_closed()
        print('❌ Connection closed')
        
        # 显示最终统计
        print(f'\n📊 Final Statistics:')
        print(f'   Total motor states received: {received_count}')
        print(f'   Unique motors seen: {len(motor_states)}')
        print(f'   Expected motors: {len(MOTOR_MAPPING)}')
        
        if motor_states:
            print(f'\n🤖 Final Robot State:')
            display_robot_state(motor_states)
        
    except ConnectionRefusedError:
        print('❌ Connection refused: Motor server is not running or not accessible')
    except OSError as e:
        print(f'❌ Network error: {e}')
    except Exception as e:
        print(f'❌ Unexpected error: {e}')


def display_robot_state(motor_states):
    """显示当前机器人状态"""
    # 初始化关节数组
    joint_positions = [0.0] * 16
    joint_velocities = [0.0] * 16
    joint_torques = [0.0] * 16
    
    # 填充关节数据
    for motor_key, feedback in motor_states.items():
        if motor_key in MOTOR_MAPPING:
            joint_index = MOTOR_MAPPING[motor_key]
            joint_positions[joint_index] = feedback.angle
            joint_velocities[joint_index] = feedback.velocity
            joint_torques[joint_index] = feedback.torque
    
    # 按腿部分组显示
    leg_names = ['FL', 'FR', 'RL', 'RR']
    for i, leg_name in enumerate(leg_names):
        start_idx = i * 3
        pos = joint_positions[start_idx:start_idx+3]
        vel = joint_velocities[start_idx:start_idx+3]
        print(f'   {leg_name} leg: pos=[{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}], '
              f'vel=[{vel[0]:6.3f}, {vel[1]:6.3f}, {vel[2]:6.3f}]')
    
    # 显示轮子状态
    wheel_pos = joint_positions[12:16]
    wheel_vel = joint_velocities[12:16]
    print(f'   Wheels:  pos=[{wheel_pos[0]:6.3f}, {wheel_pos[1]:6.3f}, {wheel_pos[2]:6.3f}, {wheel_pos[3]:6.3f}], '
          f'vel=[{wheel_vel[0]:6.3f}, {wheel_vel[1]:6.3f}, {wheel_vel[2]:6.3f}, {wheel_vel[3]:6.3f}]')


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\n🛑 Test interrupted by user')
    except Exception as e:
        print(f'\n💥 Unexpected error: {e}') 