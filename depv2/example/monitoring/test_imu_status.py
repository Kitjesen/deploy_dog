#!/usr/bin/env python3
"""
IMU Status Monitoring Script
Connects to the robot via ThunderRobotInterface and continuously displays IMU status.
"""

import asyncio
import os
import sys
import time
import numpy as np

# Add the parent directory to the path to allow local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from thunder_robot_interface import ThunderRobotInterface
from config_manager import get_default_config

async def monitor_imu_status(duration_sec: int = 20):
    """
    Initializes the robot interface, connects, and monitors IMU status.
    """
    print("🤖 Initializing IMU Status Monitor...")
    
    # 1. Load configuration and initialize the interface
    config = get_default_config()
    robot_interface = ThunderRobotInterface(config)

    # 2. Connect to the robot
    print("🔌 Connecting to robot hardware...")
    if not await robot_interface.connect():
        print("❌ Failed to connect to the robot. Aborting.")
        return

    print("✅ Connection successful. Starting IMU monitoring...")
    print("=" * 60)

    try:
        # 3. Monitoring loop
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            # Get the latest system status and observations
            system_status = robot_interface.get_system_status()
            observations = robot_interface.get_observations()

            # Extract IMU specific data
            imu_status = system_status.get('state_receiver', {}).get('imu', {})
            
            # Clear the console for a clean display
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"--- Thunder IMU Status @ {time.strftime('%H:%M:%S')} ---")
            
            if imu_status.get('connected'):
                print("🟢 IMU STATUS: CONNECTED")
                print("-" * 20)
                print(f"  - Data Rate          : {imu_status.get('data_rate', 0):.2f} Hz")
                print(f"  - Packets Received   : {imu_status.get('packets_received', 0)}")
                print(f"  - Packets Parsed     : {imu_status.get('packets_parsed', 0)}")
                print(f"  - Parse Errors       : {imu_status.get('parse_errors', 0)}")
                print(f"  - Data Freshness     : {imu_status.get('data_age', -1):.3f} s")
                
                print("\n--- Real-time Observation Data ---")
                ang_vel = observations.get('base_ang_vel', np.zeros(3))
                gravity = observations.get('projected_gravity', np.zeros(3))
                
                print(f"  - Angular Velocity   : [x={ang_vel[0]:.3f}, y={ang_vel[1]:.3f}, z={ang_vel[2]:.3f}] rad/s")
                print(f"  - Projected Gravity  : [x={gravity[0]:.3f}, y={gravity[1]:.3f}, z={gravity[2]:.3f}]")

            else:
                print("🔴 IMU STATUS: DISCONNECTED or DISABLED")
                print("   - Check hardware connection and configuration.")

            print("=" * 60)
            print(f"Monitoring for {duration_sec} seconds... Press Ctrl+C to exit early.")
            
            await asyncio.sleep(0.5)  # Update display twice per second

    except KeyboardInterrupt:
        print("\n👋 User interrupted. Shutting down.")
    except Exception as e:
        print(f"\n❌ An error occurred during monitoring: {e}")
    finally:
        # 4. Graceful shutdown
        print("\n🔌 Disconnecting from robot hardware...")
        await robot_interface.disconnect()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(monitor_imu_status())
    except Exception as e:
        print(f"Error running script: {e}") 