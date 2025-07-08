#!/usr/bin/env python3
"""
Frequency Control Configuration Example
This file shows how to configure and control dynamic frequency adjustment
"""

import sys
import os

# Add parent directory to sys.path to import lib modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.core.frequency_controller import FrequencyConfig
from run_deploy import ThunderFlatDeployer


# =============================================================================
# 🎯 Main Control Switch - Enable/Disable Dynamic Frequency Adjustment Here
# =============================================================================

def create_frequency_config_conservative():
    """Conservative configuration - disable dynamic adjustment"""
    return FrequencyConfig(
        target_frequency=50.0,
        enable_dynamic_adjustment=False,  # ❌ Disable dynamic adjustment
        max_consecutive_delays=5,
        adjustment_factor=0.9
    )


def create_frequency_config_adaptive():
    """Adaptive configuration - enable dynamic adjustment"""
    return FrequencyConfig(
        target_frequency=50.0,
        enable_dynamic_adjustment=True,   # ✅ Enable dynamic adjustment
        max_consecutive_delays=5,         # Trigger frequency reduction after 5 consecutive delays
        adjustment_factor=0.9,            # Reduce frequency to 90%
        improvement_factor=1.05,          # Increase by 5% during recovery
        delay_threshold_ratio=0.1,        # Warning when delay exceeds 10%
        severe_delay_threshold_ratio=0.5  # Error when delay exceeds 50%
    )


def create_frequency_config_aggressive():
    """Aggressive configuration - fast response adjustment"""
    return FrequencyConfig(
        target_frequency=50.0,
        enable_dynamic_adjustment=True,   # ✅ Enable dynamic adjustment
        max_consecutive_delays=3,         # Trigger after 3 consecutive delays
        adjustment_factor=0.8,            # Reduce frequency to 80%
        improvement_factor=1.1,           # Increase by 10% during recovery
        delay_threshold_ratio=0.05,       # Warning when delay exceeds 5%
        severe_delay_threshold_ratio=0.3  # Error when delay exceeds 30%
    )


# =============================================================================
# Usage Examples
# =============================================================================

def example_conservative_deployment():
    """Example: Conservative deployment - no dynamic adjustment"""
    print("=== Conservative Mode Deployment ===")
    
    # Create ThunderFlatDeployer with conservative configuration
    deployer = ThunderFlatDeployer("exported/policy.pt")
    
    # Manually set to conservative configuration
    deployer.frequency_controller.config = create_frequency_config_conservative()
    deployer.frequency_controller.enable_dynamic_adjustment(False)
    
    print("✅ Dynamic frequency adjustment disabled")
    print("💡 System will always run at 50Hz, no automatic adjustment")
    
    return deployer


def example_adaptive_deployment():
    """Example: Adaptive deployment - use standard dynamic adjustment"""
    print("=== Adaptive Mode Deployment ===")
    
    # Create ThunderFlatDeployer with adaptive configuration
    deployer = ThunderFlatDeployer("exported/policy.pt")
    
    # Manually set to adaptive configuration
    deployer.frequency_controller.config = create_frequency_config_adaptive()
    deployer.frequency_controller.enable_dynamic_adjustment(True)
    
    print("✅ Dynamic frequency adjustment enabled")
    print("💡 System will automatically adjust frequency based on load")
    
    return deployer


def example_runtime_control():
    """Example: Runtime control of dynamic adjustment"""
    print("=== Runtime Control Example ===")
    
    deployer = ThunderFlatDeployer("exported/policy.pt")
    
    # Enable dynamic adjustment at runtime
    print("1. Enable dynamic adjustment")
    deployer.enable_frequency_adjustment(True)
    
    # Adjust parameters
    print("2. Adjust parameters")
    deployer.set_frequency_adjustment_params(
        max_consecutive_delays=3,
        adjustment_factor=0.85
    )
    
    # Get status
    print("3. Check status")
    health = deployer.get_frequency_health_status()
    print(f"   Dynamic adjustment: {'Enabled' if health['dynamic_adjustment_enabled'] else 'Disabled'}")
    print(f"   Target frequency: {health['target_frequency']:.1f} Hz")
    print(f"   Current frequency: {health['current_frequency']:.1f} Hz")
    
    # Use new method to get detailed status
    print("4. Get detailed status")
    status = deployer.get_frequency_status_summary()
    print(f"   Status summary: {status}")
    
    timing_stats = deployer.get_frequency_timing_stats()
    if timing_stats:
        print(f"   On-time execution rate: {timing_stats['on_time_percentage']:.1f}%")
    
    # Disable dynamic adjustment at runtime
    print("5. Disable dynamic adjustment")
    deployer.enable_frequency_adjustment(False)
    
    # Force set specific frequency
    print("6. Force set frequency to 40Hz")
    deployer.force_frequency(40.0)
    
    # Reset counters
    print("7. Reset performance counters")
    deployer.reset_frequency_counters()
    
    return deployer


# =============================================================================
# Quick Setup Functions
# =============================================================================

def quick_setup_no_adjustment():
    """Quick setup: disable dynamic adjustment"""
    deployer = ThunderFlatDeployer("exported/policy.pt")
    deployer.enable_frequency_adjustment(False)
    print("🔒 Dynamic frequency adjustment disabled")
    return deployer


def quick_setup_with_adjustment():
    """Quick setup: enable dynamic adjustment"""
    deployer = ThunderFlatDeployer("exported/policy.pt")
    deployer.enable_frequency_adjustment(True)
    deployer.set_frequency_adjustment_params(
        max_consecutive_delays=5,
        adjustment_factor=0.9
    )
    print("🔧 Dynamic frequency adjustment enabled")
    return deployer


def example_comprehensive_control():
    """Example: Comprehensive control and monitoring"""
    print("=== Comprehensive Control Example ===")
    
    deployer = ThunderFlatDeployer("exported/policy.pt")
    
    # Configure frequency control
    print("1. Configure frequency control")
    deployer.set_target_frequency(60.0)  # Set target frequency to 60Hz
    deployer.set_frequency_display_interval(25)  # Display every 25 steps
    deployer.enable_frequency_adjustment(True)
    
    # Get configuration summary
    print("2. Get configuration summary")
    config = deployer.get_frequency_config_summary()
    print(f"   Target frequency: {config['target_frequency']:.1f} Hz")
    print(f"   Dynamic adjustment: {'Enabled' if config['dynamic_adjustment_enabled'] else 'Disabled'}")
    print(f"   Adjustment factor: {config['adjustment_factor']:.2f}")
    
    # Get status summary
    print("3. Get status summary")
    status = deployer.get_frequency_status_summary()
    print(f"   {status}")
    
    # Simulate post-run statistics
    print("4. Simulate post-run statistics")
    performance_report = deployer.export_frequency_performance_report()
    print(f"   Configuration info: {performance_report['configuration']}")
    
    # Frequency control operations
    print("5. Frequency control operations")
    deployer.force_frequency(45.0)  # Force set to 45Hz
    print(f"   Status after forced frequency: {deployer.get_frequency_status_summary()}")
    
    # Reset to target frequency
    print("6. Reset to target frequency")
    deployer.set_target_frequency(60.0)
    print(f"   Status after reset: {deployer.get_frequency_status_summary()}")
    
    return deployer


if __name__ == "__main__":
    print("Frequency Control Configuration Examples")
    print("=====================================")
    print()
    
    print("1. Conservative configuration (disable dynamic adjustment):")
    example_conservative_deployment()
    print()
    
    print("2. Adaptive configuration (enable dynamic adjustment):")
    example_adaptive_deployment()
    print()
    
    print("3. Runtime control:")
    example_runtime_control()
    print()
    
    print("4. Comprehensive control example:")
    example_comprehensive_control()
    print()
    
    print("🎯 Main Control Methods Summary:")
    print("   ✅ Enable/Disable dynamic adjustment:")
    print("      - In code: thunder_flat_deploy.py line 61 enable_dynamic_adjustment=True/False")
    print("      - At runtime: deployer.enable_frequency_adjustment(True/False)")
    print()
    print("   ⚙️ Get status information:")
    print("      - deployer.get_frequency_health_status()  # Health status")
    print("      - deployer.get_frequency_timing_stats()   # Timing statistics")
    print("      - deployer.get_frequency_status_summary() # Status summary")
    print()
    print("   🔧 Control frequency:")
    print("      - deployer.set_target_frequency(50.0)     # Set target frequency")
    print("      - deployer.force_frequency(40.0)          # Force set frequency")
    print("      - deployer.reset_frequency_counters()     # Reset counters")
    print()
    print("   📊 Display control:")
    print("      - deployer.set_frequency_display_interval(25) # Display interval") 