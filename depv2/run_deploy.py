#!/usr/bin/env python3
"""
Thunder Robot Deployment System v2.1.0 - Refactored
Main deployment script using the new modular architecture
"""

import asyncio
import logging
import os
import sys
import signal
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the new thunder deployer
from lib.core.thunder_deployer import ThunderDeployer


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def main() -> None:
    """Main deployment function."""
    setup_logging()
    logger = logging.getLogger('DeploymentMain')
    
    # Configuration
    model_path = "exported/policy.pt"
    config_path = "config/thunder_flat_config.yaml"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.info("Please ensure the model file is in the correct location.")
        return
    
    deployer = None
    
    def signal_handler(signum, frame):
        """Handle signals to trigger graceful stop."""
        print(f"\nReceived signal {signum}, initiating graceful stop...")
        if deployer is not None:
            deployer.request_graceful_stop(return_to_zero=True)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize deployer
        logger.info("Initializing Thunder Robot Deployment System...")
        deployer = ThunderDeployer(model_path, config_path)
        
        # Set velocity commands (stationary by default)
        velocity_commands = deployer.set_velocity_commands(0.0, 0.0, 0.0)

        
        current_state = deployer.get_current_state()
        config_summary = current_state['config_summary']
        
        logger.info(f"Control frequency: {config_summary['control_frequency']} Hz")
        logger.info(f"Observation dimension: {config_summary['observation_dim']}")
        logger.info(f"Action dimension: {config_summary['action_dim']}")
        logger.info(f"Return-to-zero duration: 5.0 seconds")
        
        # Display operation mode
        safe_mode = True  # Default to safe mode
        if safe_mode:
            logger.info("⚠️  Starting in SAFE MODE - motor commands will NOT be sent")
            logger.info("   This allows testing without robot movement")
        else:
            logger.info("🚨 Starting in ACTIVE MODE - robot WILL move!")
            logger.info("   Ensure robot is in safe position!")
        
        logger.info("=" * 50)
        logger.info("Press Ctrl+C to trigger graceful stop with return-to-zero")
        logger.info("Starting deployment...")
        
        # Start deployment
        await deployer.start(velocity_commands, safe_mode=safe_mode)
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Deployment error: {e}")
        raise
    finally:
        if deployer is not None:
            logger.info("Cleaning up...")
            await deployer.stop()
            
            # Display final system health
            health_status = deployer.get_system_health()
            performance_summary = deployer.get_performance_summary()
            
            logger.info("=== Final System Status ===")
            basic_metrics = performance_summary['basic_metrics']
            if basic_metrics.get('total_steps', 0) > 0:
                logger.info(f"Total steps: {basic_metrics['total_steps']}")
                logger.info(f"Runtime: {basic_metrics.get('runtime', 0):.1f}s")
                logger.info(f"Performance score: {deployer.statistics.get_current_performance_score():.1f}%")
            
            # Show any final warnings
            system_health = performance_summary['system_health']
            if not system_health['is_healthy']:
                logger.warning("System health issues detected:")
                for alert in system_health['performance_alerts']:
                    logger.warning(f"  - {alert}")
            
        logger.info("Deployment completed")


async def run_interactive_mode() -> None:
    """Run deployment in interactive mode with user commands."""
    setup_logging()
    logger = logging.getLogger('InteractiveMode')
    
    model_path = "exported/policy.pt"
    config_path = "config/thunder_flat_config.yaml"
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    deployer = ThunderDeployer(model_path, config_path)
    
    logger.info("=== Interactive Thunder Deployment ===")
    logger.info("System Commands:")
    logger.info("  S      - Start safe mode")
    logger.info("  A      - Start active mode") 
    logger.info("  x      - Stop gracefully")
    logger.info("  !      - Immediate stop")
    logger.info("  ?      - Show status")
    logger.info("  i      - Show stats")
    logger.info("  m      - Toggle mode safe/active")
    logger.info("  q      - Quit")
    logger.info("")
    logger.info("Movement Commands:")
    logger.info("  0      - Stop movement (0,0,0)")
    logger.info("  w      - Forward (0.5,0,0)")
    logger.info("  s      - Backward (-0.5,0,0)") 
    logger.info("  d      - Right (0,0.5,0)")
    logger.info("  a      - Left (0,-0.5,0)")
    logger.info("  e      - Turn right (0,0,-0.5)")
    logger.info("  r      - Turn left (0,0,0.5)")
    logger.info("")
    logger.info("Advanced: f <hz>, v <x> <y> <z>, freq, vel, mode [safe|active]")
    
    running = False
    
    try:
        while True:
            try:
                command = input("> ").strip().split()
                
                if not command:
                    continue
                
                cmd = command[0]  # Keep original case for S/A distinction
                
                # Handle single-letter shortcuts
                if cmd.lower() == "q":
                    cmd = "quit"
                elif cmd == "S":  # Capital S for start safe
                    cmd, command = "start", ["start", "safe"]
                elif cmd == "A":  # Capital A for start active
                    cmd, command = "start", ["start", "active"]
                elif cmd.lower() == "x":
                    cmd = "stop"
                elif cmd == "!":
                    cmd = "immediate"
                elif cmd == "?":
                    cmd = "status"
                elif cmd.lower() == "i":
                    cmd = "stats"
                elif cmd.lower() == "m":
                    cmd = "mode"
                elif cmd.lower() == "f":
                    cmd = "freq"
                elif cmd.lower() == "v":
                    cmd = "vel"
                # Movement shortcuts
                elif cmd == "0":
                    cmd, command = "vel", ["vel", "0", "0", "0"]
                elif cmd.lower() == "w":
                    cmd, command = "vel", ["vel", "0.5", "0", "0"]
                elif cmd.lower() == "s":
                    cmd, command = "vel", ["vel", "-0.5", "0", "0"]
                elif cmd.lower() == "d":
                    cmd, command = "vel", ["vel", "0", "0.5", "0"]
                elif cmd.lower() == "a":
                    cmd, command = "vel", ["vel", "0", "-0.5", "0"]
                elif cmd.lower() == "e":
                    cmd, command = "vel", ["vel", "0", "0", "-0.5"]
                elif cmd.lower() == "r":
                    cmd, command = "vel", ["vel", "0", "0", "0.5"]
                
                cmd = cmd.lower()  # Convert to lowercase for processing
                
                if cmd == "quit":
                    break
                elif cmd == "start":
                    if running:
                        logger.warning("Deployment already running")
                        continue
                    
                    mode = command[1] if len(command) > 1 else "safe"
                    safe_mode = mode.lower() == "safe"
                    
                    logger.info(f"Starting deployment in {mode.upper()} mode...")
                    # Run deployment in background task
                    velocity_commands = deployer.set_velocity_commands(0.0, 0.0, 0.0)
                    asyncio.create_task(deployer.start(velocity_commands, safe_mode))
                    running = True
                    
                elif cmd == "stop":
                    if not running:
                        logger.warning("Deployment not running")
                        continue
                    deployer.request_graceful_stop(return_to_zero=True)
                    logger.info("Graceful stop requested")
                    
                elif cmd == "immediate":
                    if not running:
                        logger.warning("Deployment not running")
                        continue
                    deployer.request_immediate_stop()
                    logger.info("Immediate stop requested")
                    
                elif cmd == "status":
                    state = deployer.get_current_state()
                    logger.info(f"Running: {state['deployment_state']['is_running']}")
                    logger.info(f"Mode: {state['deployment_state']['current_mode']}")
                    logger.info(f"Steps: {state['deployment_state']['step_count']}")
                    logger.info(f"Performance: {state['performance_score']:.1f}%")
                    
                elif cmd == "stats":
                    summary = deployer.get_performance_summary()
                    logger.info("Performance Statistics:")
                    for key, value in summary['basic_metrics'].items():
                        logger.info(f"  {key}: {value}")
                    
                elif cmd == "mode":
                    if len(command) < 2:
                        # Toggle mode if no argument provided (for 'm' shortcut)
                        current_state = deployer.get_current_state()
                        current_mode = current_state['deployment_state']['current_mode']
                        new_mode = 'active' if current_mode == 'safe' else 'safe'
                        deployer.set_mode(new_mode)
                        logger.info(f"Mode toggled to {new_mode.upper()}")
                        continue
                    mode = command[1].lower()
                    if mode in ['safe', 'active']:
                        deployer.set_mode(mode)
                        logger.info(f"Mode set to {mode.upper()}")
                    else:
                        logger.warning("Mode must be 'safe' or 'active'")
                
                elif cmd == "freq":
                    if len(command) < 2:
                        logger.warning("Usage: freq <frequency_hz>")
                        continue
                    try:
                        freq = float(command[1])
                        deployer.set_target_frequency(freq)
                        logger.info(f"Target frequency set to {freq} Hz")
                    except ValueError:
                        logger.warning("Invalid frequency value")
                
                elif cmd == "vel":
                    if len(command) < 4:
                        logger.warning("Usage: vel <vx> <vy> <wz>")
                        continue
                    try:
                        vx, vy, wz = map(float, command[1:4])
                        vel_cmd = deployer.set_velocity_commands(vx, vy, wz)
                        logger.info(f"Velocity commands set to: {vel_cmd}")
                    except ValueError:
                        logger.warning("Invalid velocity values")
                
                else:
                    logger.warning(f"Unknown command: {cmd}")
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                if running:
                    deployer.request_graceful_stop(return_to_zero=True)
                    logger.info("Graceful stop requested via Ctrl+C")
                else:
                    break
    
    finally:
        if running:
            await deployer.stop()


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(run_interactive_mode())
    else:
        asyncio.run(main()) 