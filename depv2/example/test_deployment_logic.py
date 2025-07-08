#!/usr/bin/env python3
"""
Test Script for ThunderFlatDeployer Logic
Verifies the core control loop of the deployment script by mocking hardware interfaces.
"""

import asyncio
import logging
import time
import numpy as np
import torch
import os
import sys

# Add the parent directory to the path to allow local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from thunder_flat_deploy import ThunderFlatDeployer
from safety_monitor import ThunderSafetyMonitor

# --- Mock Hardware and Safety Components ---

class MockRobotInterface:
    """A mock of the ThunderRobotInterface for testing without hardware."""
    def __init__(self):
        self.logger = logging.getLogger('MockRobotInterface')
        self.last_applied_actions = None
        self.previous_actions = np.zeros(16) # Add this to store previous actions
        # Start with a realistic initial state
        self._joint_positions = np.array([
            0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5,
            0.0, 0.0, 0.0, 0.0
        ])
        self._joint_velocities = np.zeros(16)
        self.logger.info("Mock Robot Interface created.")

    async def connect(self) -> bool:
        self.logger.info("Mock connect successful.")
        return True

    async def disconnect(self):
        self.logger.info("Mock disconnect successful.")

    async def emergency_stop(self):
        self.logger.warning("MOCK EMERGENCY STOP TRIGGERED")

    def get_observations(self) -> dict:
        """Provide simulated sensor data."""
        # Simulate some minor fluctuations for realism
        self._joint_positions += np.zeros(16) * 0.005
        self._joint_velocities += np.zeros(16) * 0.01

        obs = {
            'base_ang_vel':np.array([0.0, 0.0, 0.0]),
            'projected_gravity': np.array([0.0, 0.0, -1.0]),
            'joint_pos': self._joint_positions,
            'joint_vel': self._joint_velocities,
            'timestamp': time.time()
        }
        return obs

    async def apply_actions(self, processed_actions: np.ndarray):
        """'Apply' actions by storing them and logging."""
        self.logger.info(f"Applying actions. Leg Pos: {processed_actions}")
        self.previous_actions = self.last_applied_actions if self.last_applied_actions is not None else np.zeros(16)
        self.last_applied_actions = processed_actions
        # Simulate robot moving based on commands
        self._joint_positions[:12] = processed_actions[:12] * 0.1 + self._joint_positions[:12] * 0.9 # Inertia
        self._joint_velocities[12:] = processed_actions[12:] # Wheels directly take velocity

class MockSafetyMonitor(ThunderSafetyMonitor):
    """A mock of the safety monitor that always reports the system is safe."""
    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = logging.getLogger('MockSafetyMonitor')
        self.logger.info("Mock Safety Monitor created. Will always report SAFE.")

    def check_safety(self, robot_state: dict) -> bool:
        # Override to always return True for testing the main loop
        return True

# --- Main Test Execution ---

async def test_deployment_logic(num_steps: int = 10):
    """
    Runs the deployment logic test.
    """
    print("\n" + "="*60)
    print("🤖 STARTING DEPLOYMENT LOGIC TEST")
    print("="*60 + "\n")

    # 1. Initialize the deployer
    model_path = "exported/policy.pt" 
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}, creating a dummy one.")
        # Create a dummy model that mimics the real policy's output shape
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                # Per user feedback, the model output should be 16 dimensions
                return torch.randn(1, 16)
        
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        dummy_jit = torch.jit.script(DummyModel())
        dummy_jit.save(model_path)
        print("Created a dummy model for testing with output shape (1, 16).")

    deployer = ThunderFlatDeployer(model_path)
    
    # 2. Replace real components with mocks
    print("🔧 Replacing hardware interfaces with mocks...")
    mock_interface = MockRobotInterface()
    mock_safety = MockSafetyMonitor(deployer.config)
    
    deployer.robot_interface = mock_interface  # type: ignore
    deployer.safety_monitor = mock_safety      # type: ignore
    print("✅ Mocks installed.\n")

    # 3. Run the logic for a few steps
    print(f"🚀 Running control loop for {num_steps} steps...")
    velocity_command = np.array([0.0, 0.0, 0.0]) # Command: move forward

    for i in range(num_steps):
        print(f"\n--- STEP {i+1}/{num_steps} ---")
        
        # Set safe_mode to False to test the action application path
        deployer.safe_mode = False 
        
        success = await deployer.step(velocity_command)

        if not success:
            print("❌ Step failed!")
            break

        # Log results for verification
        print(f"  ▶️  Observation generated for model.")
        if deployer.obs_processor.last_processed_observation is not None:
            obs_numpy = deployer.obs_processor.last_processed_observation.numpy()
            print(f"  📥  Full Observation (shape {obs_numpy.shape}):")
            print(f"    - Ang Vel (3):      {np.round(obs_numpy[0:3], 6)}")
            print(f"    - Gravity (3):      {np.round(obs_numpy[3:6], 6)}")
            print(f"    - Commands (3):     {np.round(obs_numpy[6:9], 6)}")
            print(f"    - Joint Pos (16):   {np.round(obs_numpy[9:25], 6)}")
            print(f"    - Joint Vel (16):   {np.round(obs_numpy[25:41], 6)}")
            print(f"    - Last Actions (16):{np.round(obs_numpy[41:57], 6)}")

        print(f"  🧠  Model RAW Output:                {np.round(deployer.raw_model_output.numpy(), 6)}")
        if mock_interface.last_applied_actions is not None:
            print(f"  ⚙️  Processed Actions (to motors):   {np.round(mock_interface.last_applied_actions, 6)}")
            action_diff = mock_interface.last_applied_actions - mock_interface.previous_actions
            print(f"  📊  Action Diff (Current - Prev):    {np.round(action_diff, 6)}")
        else:
            print(f"  ⚙️  No actions were applied in this step.")
        
        await asyncio.sleep(0.1) # Simulate time between steps

    print("\n" + "="*60)
    print("🏁 TEST COMPLETE")
    print("="*60)
    print("Summary: The script ran through the perception-policy-action pipeline.")
    print("Check the logs above to verify that observations, model outputs, and applied actions look reasonable.")


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s')
    
    try:
        asyncio.run(test_deployment_logic())
    except Exception as e:
        logging.error(f"An error occurred during the test run: {e}", exc_info=True) 