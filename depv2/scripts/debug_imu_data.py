#!/usr/bin/env python3
"""
Debug IMU Raw Data
Debug IMU raw data to check if parsing is correct
"""

import time
import numpy as np
import struct
import sys
import os

# Add parent directory to sys.path to import lib modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.hardware.hipnuc_imu_interface import HipnucIMUInterface
from lib.config.config_manager import get_default_config

def debug_quaternion_data():
    """Debug quaternion data"""
    config = get_default_config()
    imu = HipnucIMUInterface(config['imu'])
    
    print("=" * 80)
    print("🔍 IMU Raw Data Debug")
    print("=" * 80)
    
    if not imu.start():
        print("❌ Failed to start IMU")
        return
    
    print("✅ IMU started successfully, waiting for data...")
    time.sleep(2)
    
    # Collect 10 data packets for analysis
    for i in range(10):
        imu_data = imu.get_latest_data()
        if imu_data is not None:
            quat = imu_data.orientation
            gyro = imu_data.angular_velocity
            accel = imu_data.linear_acceleration
            
            # Calculate quaternion magnitude
            quat_norm = np.linalg.norm(quat)
            
            print(f"\n📦 Data packet {i+1}:")
            print(f"  🔄 Quaternion: [{quat[0]:+8.4f}, {quat[1]:+8.4f}, {quat[2]:+8.4f}, {quat[3]:+8.4f}]")
            print(f"  📏 Quaternion magnitude: {quat_norm:8.4f} (expected: 1.0000)")
            print(f"  📐 Angular velocity: [{gyro[0]:+8.4f}, {gyro[1]:+8.4f}, {gyro[2]:+8.4f}] rad/s")
            print(f"  🏃 Acceleration: [{accel[0]:+8.4f}, {accel[1]:+8.4f}, {accel[2]:+8.4f}] m/s²")
            
            # If quaternion magnitude is close to 1, try normalization
            if quat_norm > 0.1:
                normalized_quat = quat / quat_norm
                print(f"  ✅ Normalized quaternion: [{normalized_quat[0]:+8.4f}, {normalized_quat[1]:+8.4f}, {normalized_quat[2]:+8.4f}, {normalized_quat[3]:+8.4f}]")
                
                # Test gravity projection
                from lib.state.robot_state import quat_to_projected_gravity
                proj_gravity = quat_to_projected_gravity(normalized_quat)
                print(f"  🌍 Projected gravity: [{proj_gravity[0]:+8.4f}, {proj_gravity[1]:+8.4f}, {proj_gravity[2]:+8.4f}]")
            else:
                print(f"  ❌ Quaternion magnitude too small, possible parsing error")
        
        time.sleep(0.5)
    
    imu.stop()
    print("\n🏁 Debug completed")

def debug_raw_packet_parsing():
    """Debug raw packet parsing"""
    print("\n" + "=" * 80)
    print("🔬 Raw Packet Analysis")
    print("=" * 80)
    
    config = get_default_config()
    
    # Create a modified IMU interface for debugging
    class DebugIMUInterface(HipnucIMUInterface):
        def _parse_packet(self):
            """Override parsing function with debug information"""
            try:
                if len(self.parse_buffer) < self.PACKET_MIN_SIZE:
                    return None
                
                # Check packet header
                if (self.parse_buffer[0] != self.PACKET_START1 or 
                    self.parse_buffer[1] != self.PACKET_START2):
                    return None
                
                print(f"\n📦 Packet found:")
                print(f"  Header: 0x{self.parse_buffer[0]:02X} 0x{self.parse_buffer[1]:02X}")
                
                # Read packet length
                packet_length = self.parse_buffer[2]
                total_length = packet_length + 6
                print(f"  Packet length: {packet_length}, total length: {total_length}")
                
                if len(self.parse_buffer) < total_length:
                    return None
                
                # Show raw data
                raw_bytes = self.parse_buffer[:min(50, len(self.parse_buffer))]
                hex_str = ' '.join([f'{b:02X}' for b in raw_bytes])
                print(f"  Raw data: {hex_str}...")
                
                # Try parsing quaternion at different offsets
                for offset in [6, 8, 10, 12, 14, 16]:
                    if len(self.parse_buffer) >= offset + 16:  # 4 float32 = 16 bytes
                        try:
                            quat_data = struct.unpack('<4f', self.parse_buffer[offset:offset+16])
                            quat_norm = np.linalg.norm(quat_data)
                            print(f"  Offset{offset:2d}: [{quat_data[0]:+8.4f}, {quat_data[1]:+8.4f}, {quat_data[2]:+8.4f}, {quat_data[3]:+8.4f}] (magnitude:{quat_norm:6.4f})")
                        except:
                            pass
                
                # Use original parsing logic
                return super()._parse_packet()
                
            except Exception as e:
                print(f"  ❌ Parse error: {e}")
                return None
    
    # Use debug version of IMU interface
    debug_imu = DebugIMUInterface(config['imu'])
    
    if not debug_imu.start():
        print("❌ Failed to start debug IMU")
        return
    
    print("✅ Debug IMU started successfully, analyzing packets...")
    time.sleep(5)  # Run for 5 seconds to collect debug information
    
    debug_imu.stop()
    print("🏁 Raw packet analysis completed")

if __name__ == "__main__":
    # First debug quaternion data
    debug_quaternion_data()
    
    # Then debug raw packets
    debug_raw_packet_parsing() 