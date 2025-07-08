#!/usr/bin/env python3
"""
Thunder Robot Hardware Connection Checker

This module provides comprehensive hardware diagnostics for the Thunder robot,
including IMU, motor server, network connectivity, and system permission checks.
"""

import glob
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import serial


class CheckStatus(Enum):
    """Check result status enumeration."""
    PASS = "PASS"
    WARNING = "WARNING" 
    FAIL = "FAIL"
    ERROR = "ERROR"


@dataclass
class CheckResult:
    """Represents the result of a hardware check."""
    status: CheckStatus
    message: str
    details: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None
    
    def __post_init__(self):
        """Initialize empty lists if None."""
        if self.details is None:
            self.details = []
        if self.suggestions is None:
            self.suggestions = []


@dataclass
class HardwareConfig:
    """Hardware configuration parameters."""
    imu_port: str = '/dev/ttyUSB0'
    imu_baud: int = 115200
    motor_host: str = '192.168.66.159'
    motor_port: int = 12345
    connection_timeout: float = 5.0
    imu_data_timeout: float = 3.0
    ping_count: int = 3
    ping_timeout: int = 2000


class HardwareChecker:
    """
    Comprehensive hardware connection checker for Thunder robot.
    
    This class performs various hardware diagnostics including:
    - Serial port availability and permissions
    - IMU device connectivity and data reception
    - Motor server TCP connection
    - Network connectivity tests
    - USB device enumeration
    - System permission verification
    """
    
    def __init__(self, config: Optional[HardwareConfig] = None, verbose: bool = False):
        """
        Initialize the hardware checker.
        
        Args:
            config: Hardware configuration parameters
            verbose: Enable verbose logging output
        """
        self.config = config or HardwareConfig()
        self.verbose = verbose
        self.logger = self._setup_logger()
        self.results: Dict[str, CheckResult] = {}
        
        self._print_header()
    
    def _setup_logger(self) -> logging.Logger:
        """
        Setup logging configuration.
        
        Returns:
            Configured logger instance
        """
        level = logging.DEBUG if self.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('HardwareChecker')
    
    def _print_header(self) -> None:
        """Print the application header."""
        print("Thunder Robot Hardware Connection Checker")
        print("=" * 50)
        if self.verbose:
            print("Running in verbose mode...")
        print()
    
    def run_all_checks(self) -> Dict[str, CheckResult]:
        """
        Execute all hardware checks.
        
        Returns:
            Dictionary mapping check names to their results
        """
        print("Starting comprehensive hardware checks...\n")
        
        # Define all checks to perform
        checks = [
            ("Serial Port Availability", self._check_serial_ports),
            ("IMU Connection", self._check_imu_connection),
            ("Motor Server TCP Connection", self._check_motor_server),
            ("Network Connectivity", self._check_network),
            ("USB Devices", self._check_usb_devices),
            ("System Permissions", self._check_permissions)
        ]
        
        # Execute each check
        for check_name, check_func in checks:
            self._run_single_check(check_name, check_func)
        
        # Print summary and recommendations
        self._print_summary()
        
        return self.results
    
    def _run_single_check(self, check_name: str, check_func) -> None:
        """
        Execute a single hardware check with error handling.
        
        Args:
            check_name: Name of the check being performed
            check_func: Function to execute the check
        """
        print(f"🔍 {check_name}...")
        
        try:
            result = check_func()
            self.results[check_name] = result
            self._print_check_result(check_name, result)
        except Exception as e:
            self.logger.exception(f"Unexpected error in {check_name}")
            error_result = CheckResult(
                status=CheckStatus.ERROR,
                message=f"Unexpected error: {str(e)}",
                suggestions=["Check system logs for more details"]
            )
            self.results[check_name] = error_result
            self._print_check_result(check_name, error_result)
        
        print()
    
    def _check_serial_ports(self) -> CheckResult:
        """
        Check serial port device availability and permissions.
        
        Returns:
            Check result with port status and suggestions
        """
        # Check if specified port exists
        if not os.path.exists(self.config.imu_port):
            return CheckResult(
                status=CheckStatus.FAIL,
                message=f"Serial port {self.config.imu_port} does not exist",
                suggestions=[
                    "Check if IMU device is connected",
                    "Try other USB ports",
                    "Check with 'ls /dev/ttyUSB*' or 'ls /dev/ttyACM*'",
                    "Verify device drivers are installed"
                ]
            )
        
        # Check port permissions
        permission_error = self._check_port_permissions()
        if permission_error:
            return permission_error
        
        # Find all available USB serial ports
        available_ports = self._find_usb_serial_ports()
        
        result = CheckResult(
            status=CheckStatus.PASS,
            message=f"Serial port {self.config.imu_port} is accessible",
            details=[
                f"Primary device: {self.config.imu_port}",
                f"Available USB serial ports: {', '.join(available_ports) if available_ports else 'None'}"
            ]
        )
        
        return result
    
    def _check_port_permissions(self) -> Optional[CheckResult]:
        """
        Check if the user has permission to access the serial port.
        
        Returns:
            CheckResult if permission error, None if permissions are OK
        """
        try:
            with open(self.config.imu_port, 'r'):
                pass
        except PermissionError:
            return CheckResult(
                status=CheckStatus.FAIL,
                message=f"Permission denied accessing {self.config.imu_port}",
                suggestions=[
                    f"Add user to dialout group: sudo usermod -a -G dialout $USER",
                    f"Set permissions: sudo chmod 666 {self.config.imu_port}",
                    "Log out and log back in after adding to group",
                    "Check udev rules for the device"
                ]
            )
        except (FileNotFoundError, OSError):
            # These are expected if device doesn't exist or other issues
            pass
        
        return None
    
    def _find_usb_serial_ports(self) -> List[str]:
        """
        Find all available USB serial ports.
        
        Returns:
            List of available USB serial port paths
        """
        usb_ports = []
        port_patterns = ['/dev/ttyUSB*', '/dev/ttyACM*', '/dev/cu.usb*']
        
        for pattern in port_patterns:
            try:
                usb_ports.extend(glob.glob(pattern))
            except Exception as e:
                self.logger.debug(f"Error finding ports with pattern {pattern}: {e}")
        
        return sorted(usb_ports)
    
    def _check_imu_connection(self) -> CheckResult:
        """
        Test IMU device connectivity and data reception.
        
        Returns:
            Check result with connection status and data reception info
        """
        if not os.path.exists(self.config.imu_port):
            return CheckResult(
                status=CheckStatus.FAIL,
                message=f"IMU serial port {self.config.imu_port} not found",
                suggestions=[
                    "Ensure IMU device is connected",
                    "Check USB cable connections",
                    "Verify device drivers"
                ]
            )
        
        try:
            return self._test_serial_communication()
        except serial.SerialException as e:
            return CheckResult(
                status=CheckStatus.FAIL,
                message=f"Serial connection failed: {str(e)}",
                suggestions=[
                    "Check if another program is using the port",
                    "Verify IMU device is connected properly",
                    "Try different baud rates (9600, 57600, 115200)",
                    "Check cable for damage"
                ]
            )
        except Exception as e:
            return CheckResult(
                status=CheckStatus.ERROR,
                message=f"Unexpected IMU test error: {str(e)}"
            )
    
    def _test_serial_communication(self) -> CheckResult:
        """
        Test actual serial communication with IMU.
        
        Returns:
            Check result with communication test details
        """
        with serial.Serial(
            port=self.config.imu_port,
            baudrate=self.config.imu_baud,
            timeout=self.config.connection_timeout
        ) as ser:
            
            details = [f"Serial port opened: {self.config.imu_port}@{self.config.imu_baud}"]
            
            # Attempt to read data
            data_received = self._wait_for_serial_data(ser)
            
            if data_received:
                details.append(f"Successfully received {data_received} bytes")
                return CheckResult(
                    status=CheckStatus.PASS,
                    message="IMU connection successful - data received",
                    details=details
                )
            else:
                return CheckResult(
                    status=CheckStatus.WARNING,
                    message="Serial port accessible but no data received",
                    details=details,
                    suggestions=[
                        "Check if IMU is powered on",
                        "Verify baud rate configuration",
                        "Check cable connections",
                        "Ensure IMU is sending data"
                    ]
                )
    
    def _wait_for_serial_data(self, ser: serial.Serial) -> int:
        """
        Wait for data from serial port with timeout.
        
        Args:
            ser: Open serial port instance
            
        Returns:
            Number of bytes received
        """
        start_time = time.time()
        total_bytes = 0
        
        while time.time() - start_time < self.config.imu_data_timeout:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                total_bytes += len(data)
                if total_bytes > 0:
                    break
            time.sleep(0.1)
        
        return total_bytes
    
    def _check_motor_server(self) -> CheckResult:
        """
        Test TCP connection to motor server.
        
        Returns:
            Check result with connection status and timing
        """
        try:
            return self._test_tcp_connection()
        except socket.timeout:
            return CheckResult(
                status=CheckStatus.FAIL,
                message="Connection to motor server timed out",
                suggestions=[
                    f"Check if motor server is running on {self.config.motor_host}:{self.config.motor_port}",
                    "Verify network connectivity",
                    "Check firewall settings",
                    "Increase connection timeout if network is slow"
                ]
            )
        except ConnectionRefusedError:
            return CheckResult(
                status=CheckStatus.FAIL,
                message="Connection refused by motor server",
                suggestions=[
                    f"Motor server may not be running on {self.config.motor_host}:{self.config.motor_port}",
                    "Check server status and logs",
                    "Verify port number configuration",
                    "Ensure server is accepting connections"
                ]
            )
        except socket.gaierror:
            return CheckResult(
                status=CheckStatus.FAIL,
                message=f"Cannot resolve hostname {self.config.motor_host}",
                suggestions=[
                    "Check network connection",
                    "Verify IP address or hostname",
                    "Check DNS settings",
                    "Try using IP address instead of hostname"
                ]
            )
        except Exception as e:
            return CheckResult(
                status=CheckStatus.ERROR,
                message=f"Unexpected network error: {str(e)}"
            )
    
    def _test_tcp_connection(self) -> CheckResult:
        """
        Perform the actual TCP connection test.
        
        Returns:
            Check result with connection timing details
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.config.connection_timeout)
            
            start_time = time.time()
            sock.connect((self.config.motor_host, self.config.motor_port))
            connect_time = time.time() - start_time
            
            return CheckResult(
                status=CheckStatus.PASS,
                message="Motor server connection successful",
                details=[
                    f"Host: {self.config.motor_host}:{self.config.motor_port}",
                    f"Connection time: {connect_time:.3f}s"
                ]
            )
    
    def _check_network(self) -> CheckResult:
        """
        Test network connectivity using ping.
        
        Returns:
            Check result with ping statistics
        """
        try:
            ping_result = subprocess.run(
                ['ping', '-c', str(self.config.ping_count), '-W', str(self.config.ping_timeout), self.config.motor_host],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if ping_result.returncode == 0:
                return self._parse_ping_results(ping_result.stdout)
            else:
                return CheckResult(
                    status=CheckStatus.FAIL,
                    message=f"Cannot ping {self.config.motor_host}",
                    suggestions=[
                        "Check network cable connection",
                        "Verify IP address configuration",
                        "Check if target device is powered on",
                        "Verify network routing"
                    ]
                )
        
        except subprocess.TimeoutExpired:
            return CheckResult(
                status=CheckStatus.WARNING,
                message="Ping test timed out",
                suggestions=["Network may be slow or congested"]
            )
        except FileNotFoundError:
            return CheckResult(
                status=CheckStatus.WARNING,
                message="Ping command not available on this system"
            )
        except Exception as e:
            return CheckResult(
                status=CheckStatus.WARNING,
                message=f"Network test error: {str(e)}"
            )
    
    def _parse_ping_results(self, ping_output: str) -> CheckResult:
        """
        Parse ping command output for statistics.
        
        Args:
            ping_output: Raw ping command output
            
        Returns:
            Check result with parsed ping statistics
        """
        details = []
        
        for line in ping_output.split('\n'):
            line = line.strip()
            if 'packet loss' in line:
                details.append(f"Ping: {line}")
            elif 'min/avg/max' in line or 'rtt' in line:
                details.append(f"Latency: {line}")
        
        return CheckResult(
            status=CheckStatus.PASS,
            message=f"Network connectivity to {self.config.motor_host} is good",
            details=details
        )
    
    def _check_usb_devices(self) -> CheckResult:
        """
        Enumerate USB devices and identify potential IMU devices.
        
        Returns:
            Check result with USB device information
        """
        try:
            lsusb_result = subprocess.run(
                ['lsusb'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if lsusb_result.returncode == 0:
                return self._parse_usb_devices(lsusb_result.stdout)
            else:
                return CheckResult(
                    status=CheckStatus.WARNING,
                    message="Could not enumerate USB devices"
                )
        
        except FileNotFoundError:
            return CheckResult(
                status=CheckStatus.WARNING,
                message="lsusb command not available on this system",
                suggestions=["Install usbutils package: sudo apt install usbutils"]
            )
        except subprocess.TimeoutExpired:
            return CheckResult(
                status=CheckStatus.WARNING,
                message="USB enumeration timed out"
            )
        except Exception as e:
            return CheckResult(
                status=CheckStatus.WARNING,
                message=f"USB check error: {str(e)}"
            )
    
    def _parse_usb_devices(self, lsusb_output: str) -> CheckResult:
        """
        Parse lsusb output to identify devices.
        
        Args:
            lsusb_output: Raw lsusb command output
            
        Returns:
            Check result with USB device analysis
        """
        usb_devices = [line.strip() for line in lsusb_output.strip().split('\n') if line.strip()]
        details = [f"Found {len(usb_devices)} USB devices"]
        
        # Search for potential IMU/serial devices
        imu_keywords = ['serial', 'ftdi', 'prolific', 'cp210x', 'ch340', 'silicon labs', 'future technology']
        potential_imu = []
        
        for device in usb_devices:
            device_lower = device.lower()
            for keyword in imu_keywords:
                if keyword in device_lower:
                    potential_imu.append(device.strip())
                    break
        
        if potential_imu:
            details.append("Potential IMU/Serial devices:")
            details.extend([f"  - {device}" for device in potential_imu])
        else:
            details.append("No obvious serial/IMU devices found")
        
        return CheckResult(
            status=CheckStatus.PASS,
            message="USB device enumeration successful",
            details=details
        )
    
    def _check_permissions(self) -> CheckResult:
        """
        Check user group memberships for hardware access.
        
        Returns:
            Check result with permission status and recommendations
        """
        try:
            groups_result = subprocess.run(
                ['groups'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if groups_result.returncode == 0:
                return self._analyze_user_groups(groups_result.stdout.strip())
            else:
                return CheckResult(
                    status=CheckStatus.WARNING,
                    message="Could not determine user groups"
                )
        
        except Exception as e:
            return CheckResult(
                status=CheckStatus.WARNING,
                message=f"Permission check error: {str(e)}"
            )
    
    def _analyze_user_groups(self, groups_output: str) -> CheckResult:
        """
        Analyze user group memberships.
        
        Args:
            groups_output: Raw groups command output
            
        Returns:
            Check result with group analysis
        """
        user_groups = groups_output.split()
        details = [f"User groups: {', '.join(user_groups)}"]
        
        # Check for required groups
        required_groups = ['dialout', 'tty', 'uucp']
        missing_groups = [group for group in required_groups if group not in user_groups]
        
        if missing_groups:
            return CheckResult(
                status=CheckStatus.WARNING,
                message=f"User not in required groups: {', '.join(missing_groups)}",
                details=details,
                suggestions=[
                    f"Add to groups: sudo usermod -a -G {','.join(missing_groups)} $USER",
                    "Log out and log back in for changes to take effect",
                    "Alternatively, use sudo to run hardware monitor"
                ]
            )
        else:
            return CheckResult(
                status=CheckStatus.PASS,
                message="User has appropriate group permissions",
                details=details
            )
    
    def _print_check_result(self, check_name: str, result: CheckResult) -> None:
        """
        Print formatted check result.
        
        Args:
            check_name: Name of the check performed
            result: Check result to display
        """
        # Status icons
        status_icons = {
            CheckStatus.PASS: "✅",
            CheckStatus.WARNING: "⚠️",
            CheckStatus.FAIL: "❌",
            CheckStatus.ERROR: "💥"
        }
        
        icon = status_icons.get(result.status, "❓")
        print(f"  {icon} {result.message}")
        
        # Print details if available and in verbose mode
        if self.verbose and result.details:
            for detail in result.details:
                print(f"     {detail}")
        
        # Print suggestions for non-passing results
        if result.suggestions and result.status != CheckStatus.PASS:
            print(f"     💡 Suggestions:")
            for suggestion in result.suggestions:
                print(f"        - {suggestion}")
    
    def _print_summary(self) -> None:
        """Print comprehensive check summary."""
        print("\n" + "="*60)
        print("HARDWARE CHECK SUMMARY")
        print("="*60)
        
        # Count results by status
        status_counts = self._count_results_by_status()
        total_checks = len(self.results)
        
        print(f"Total Checks: {total_checks}")
        print(f"✅ Passed: {status_counts[CheckStatus.PASS]}")
        print(f"⚠️  Warnings: {status_counts[CheckStatus.WARNING]}")
        print(f"❌ Failed: {status_counts[CheckStatus.FAIL]}")
        print(f"💥 Errors: {status_counts[CheckStatus.ERROR]}")
        
        # Overall assessment
        self._print_overall_assessment(status_counts)
        
        # Next steps
        self._print_next_steps(status_counts)
    
    def _count_results_by_status(self) -> Dict[CheckStatus, int]:
        """
        Count check results by status.
        
        Returns:
            Dictionary mapping status to count
        """
        counts = {status: 0 for status in CheckStatus}
        
        for result in self.results.values():
            counts[result.status] += 1
        
        return counts
    
    def _print_overall_assessment(self, status_counts: Dict[CheckStatus, int]) -> None:
        """
        Print overall system assessment.
        
        Args:
            status_counts: Dictionary of status counts
        """
        failed = status_counts[CheckStatus.FAIL]
        errors = status_counts[CheckStatus.ERROR]
        warnings = status_counts[CheckStatus.WARNING]
        
        if failed > 0 or errors > 0:
            print(f"\n🚨 Hardware connection issues detected!")
            print(f"   Please resolve failed checks before running hardware monitor.")
        elif warnings > 0:
            print(f"\n⚠️  Some warnings detected.")
            print(f"   Hardware monitor may work but performance could be affected.")
        else:
            print(f"\n🎉 All checks passed!")
            print(f"   Hardware should be ready for monitoring.")
    
    def _print_next_steps(self, status_counts: Dict[CheckStatus, int]) -> None:
        """
        Print recommended next steps.
        
        Args:
            status_counts: Dictionary of status counts
        """
        failed = status_counts[CheckStatus.FAIL]
        errors = status_counts[CheckStatus.ERROR]
        
        print("\nNext steps:")
        if failed == 0 and errors == 0:
            print("  - Run: python thunder_hardware_monitor.py")
            print("  - Run: python thunder_flat_deploy.py")
        else:
            print("  - Fix the failed checks above")
            print("  - Re-run this checker to verify fixes")
            print("  - Then run: python thunder_hardware_monitor.py")
    
    def save_results(self, filepath: str) -> None:
        """
        Save check results to a file.
        
        Args:
            filepath: Path to save results file
        """
        try:
            with open(filepath, 'w') as f:
                f.write("Thunder Robot Hardware Check Results\n")
                f.write("=" * 40 + "\n\n")
                
                for check_name, result in self.results.items():
                    f.write(f"Check: {check_name}\n")
                    f.write(f"Status: {result.status.value}\n")
                    f.write(f"Message: {result.message}\n")
                    
                    if result.details:
                        f.write("Details:\n")
                        for detail in result.details:
                            f.write(f"  - {detail}\n")
                    
                    if result.suggestions:
                        f.write("Suggestions:\n")
                        for suggestion in result.suggestions:
                            f.write(f"  - {suggestion}\n")
                    
                    f.write("\n" + "-" * 40 + "\n\n")
            
            print(f"Results saved to: {filepath}")
        except Exception as e:
            print(f"Error saving results: {e}")


def create_config_from_args(args) -> HardwareConfig:
    """
    Create hardware configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Hardware configuration instance
    """
    return HardwareConfig(
        imu_port=getattr(args, 'imu_port', '/dev/ttyUSB0'),
        imu_baud=getattr(args, 'imu_baud', 115200),
        motor_host=getattr(args, 'motor_host', '192.168.66.159'),
        motor_port=getattr(args, 'motor_port', 12345)
    )


def main() -> int:
    """
    Main entry point for the hardware checker.
    
    Returns:
        Exit code: 0 for success, 1 for failures
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Thunder Robot Hardware Connection Checker"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--save-results',
        metavar='FILE',
        help='Save results to specified file'
    )
    parser.add_argument(
        '--imu-port',
        default='/dev/ttyUSB0',
        help='IMU serial port (default: /dev/ttyUSB0)'
    )
    parser.add_argument(
        '--imu-baud',
        type=int,
        default=115200,
        help='IMU baud rate (default: 115200)'
    )
    parser.add_argument(
        '--motor-host',
        default='192.168.66.159',
        help='Motor server host (default: 192.168.66.159)'
    )
    parser.add_argument(
        '--motor-port',
        type=int,
        default=12345,
        help='Motor server port (default: 12345)'
    )
    
    args = parser.parse_args()
    
    # Create configuration and checker
    config = create_config_from_args(args)
    checker = HardwareChecker(config=config, verbose=args.verbose)
    
    # Run all checks
    results = checker.run_all_checks()
    
    # Save results if requested
    if args.save_results:
        checker.save_results(args.save_results)
    
    # Determine exit code
    has_failures = any(
        result.status in [CheckStatus.FAIL, CheckStatus.ERROR]
        for result in results.values()
    )
    
    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main()) 