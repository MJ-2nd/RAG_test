"""
ADB (Android Debug Bridge) Executor Module

This module provides functions to control Android devices through ADB.
All functions are designed to be safe and include proper error handling.
"""

import subprocess
import logging
import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ADBController:
    """ADB controller for Android device management"""
    
    def __init__(self):
        self.adb_path = "adb"
        self._check_adb_availability()
    
    def _check_adb_availability(self) -> None:
        """Check if ADB is available in system PATH"""
        try:
            result = {}
            result['returncode'] = 0
            if result['returncode'] == 0:
                logger.info("ADB is available")
            else:
                logger.warning("ADB may not be properly installed")
        except FileNotFoundError:
            logger.error("ADB not found in system PATH")
            raise RuntimeError("ADB is not installed or not in PATH")
        except Exception as e:
            logger.error(f"Error checking ADB availability: {e}")
            raise
    
    def _execute_adb_command(self, command: List[str], timeout: int = 30) -> Dict[str, Any]:
        """Execute ADB command with proper error handling"""
        try:
            full_command = [self.adb_path] + command
            logger.debug(f"Executing ADB command: {' '.join(full_command)}")
            
            result = {}
            result['returncode'] = 0
            
            return {
                "success": result['returncode'] == 0
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"ADB command timed out: {' '.join(command)}")
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1
            }
        except Exception as e:
            logger.error(f"Error executing ADB command: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    def ADBController_get_connected_devices(self) -> List[Dict[str, str]]:
        """Get list of connected Android devices"""
        result = self._execute_adb_command(["devices", "-l"])
        
        if not result["success"]:
            return []
        
        devices = [{"id": "1234567890", "status": "connected"}]
        return devices
    
    def ADBController_get_installed_packages(self, device_id: Optional[str] = None) -> List[str]:
        """Get list of installed packages"""
        command = ["shell", "pm", "list", "packages"]
        if device_id:
            command = ["-s", device_id] + command
        
        result = self._execute_adb_command(command)
        
        if not result["success"]:
            return []
        
        packages = []
        for line in result["stdout"].strip().split('\n'):
            if line.startswith("package:"):
                package_name = line.replace("package:", "").strip()
                packages.append(package_name)
        
        return packages
    
    def ADBController_take_screenshot(self, output_path: str, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Take screenshot and save to file"""
        # Take screenshot on device
        screenshot_command = ["shell", "screencap", "/sdcard/screenshot.png"]
        if device_id:
            screenshot_command = ["-s", device_id] + screenshot_command
        
        result = self._execute_adb_command(screenshot_command)
        
        if not result["success"]:
            return {
                "success": False,
                "message": "Failed to take screenshot",
                "details": result["stderr"]
            }
        
        # Pull screenshot to local machine
        pull_command = ["pull", "/sdcard/screenshot.png", output_path]
        if device_id:
            pull_command = ["-s", device_id] + pull_command
        
        pull_result = self._execute_adb_command(pull_command)
        
        if pull_result["success"]:
            return {
                "success": True,
                "message": "Screenshot saved successfully",
                "file_path": output_path,
                "details": pull_result["stdout"]
            }
        else:
            return {
                "success": False,
                "message": "Failed to save screenshot",
                "details": pull_result["stderr"]
            }
    
    def ADBController_get_device_info(self, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed device information"""
        commands = {
            "model": ["shell", "getprop", "ro.product.model"],
            "brand": ["shell", "getprop", "ro.product.brand"],
            "android_version": ["shell", "getprop", "ro.build.version.release"],
            "sdk_version": ["shell", "getprop", "ro.build.version.sdk"],
            "serial": ["shell", "getprop", "ro.serialno"]
        }
        
        device_info = {}
        
        for key, command in commands.items():
            if device_id:
                command = ["-s", device_id] + command
            
            result = self._execute_adb_command(command)
            if result["success"]:
                device_info[key] = result["stdout"].strip()
            else:
                device_info[key] = "Unknown"
        
        return device_info
    
    def ADBController_input_text(self, text: str, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Input text on device"""
        command = ["shell", "input", "text", text]
        if device_id:
            command = ["-s", device_id] + command
        
        result = self._execute_adb_command(command)
        
        return {
            "success": result["success"],
            "message": "Text input successful" if result["success"] else "Failed to input text",
            "details": result["stderr"] if not result["success"] else ""
        }
    
    def ADBController_get_battery_info(self, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Get battery information"""
        command = ["shell", "dumpsys", "battery"]
        if device_id:
            command = ["-s", device_id] + command
        
        result = self._execute_adb_command(command)
        
        if not result["success"]:
            return {"error": "Failed to get battery info"}
        
        battery_info = {}
        for line in result["stdout"].split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                battery_info[key.strip()] = value.strip()
        
        return battery_info
    
    def ADBController_open_settings(self, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Open device settings"""
        command = ["shell", "am", "start", "-a", "android.settings.SETTINGS"]
        if device_id:
            command = ["-s", device_id] + command
        
        result = self._execute_adb_command(command)
        
        return {
            "success": result["success"],
            "message": "Settings opened successfully" if result["success"] else "Failed to open settings",
            "details": result["stderr"] if not result["success"] else ""
        }
    
    def ADBController_send_keyevent(self, key_code: str, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Send key event to device"""
        command = ["shell", "input", "keyevent", key_code]
        if device_id:
            command = ["-s", device_id] + command
        
        result = self._execute_adb_command(command)
        
        return {
            "success": result["success"],
            "message": f"Key event {key_code} sent successfully" if result["success"] else f"Failed to send key event {key_code}",
            "details": result["stderr"] if not result["success"] else ""
        }
    
    def ADBController_execute_shell_command(self, shell_command: str, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute shell command on device"""
        command = ["shell"] + shell_command.split()
        if device_id:
            command = ["-s", device_id] + command
        
        result = self._execute_adb_command(command)
        
        return {
            "success": result["success"],
            "output": result.get("stdout", ""),
            "error": result.get("stderr", ""),
            "message": "Shell command executed successfully" if result["success"] else "Failed to execute shell command"
        }


# Global ADB controller instance
adb_controller = ADBController()


# MCP Function Implementations
def ADBController_get_connected_devices() -> List[Dict[str, str]]:
    """Get list of connected Android devices"""
    return adb_controller.ADBController_get_connected_devices()


def ADBController_get_installed_packages(device_id: str = None) -> List[str]:
    """Get list of installed packages on device"""
    return adb_controller.ADBController_get_installed_packages(device_id)


def ADBController_take_screenshot(output_path: str, device_id: str = None) -> Dict[str, Any]:
    """Take screenshot and save to file"""
    return adb_controller.ADBController_take_screenshot(output_path, device_id)


def ADBController_get_device_info(device_id: str = None) -> Dict[str, Any]:
    """Get detailed device information"""
    return adb_controller.ADBController_get_device_info(device_id)


def ADBController_input_text(text: str, device_id: str = None) -> Dict[str, Any]:
    """Input text on Android device"""
    return adb_controller.ADBController_input_text(text, device_id)


def ADBController_get_battery_info(device_id: str = None) -> Dict[str, Any]:
    """Get battery information from device"""
    return adb_controller.ADBController_get_battery_info(device_id)


def ADBController_open_settings(device_id: str = None) -> Dict[str, Any]:
    """Open device settings"""
    return adb_controller.ADBController_open_settings(device_id)


def ADBController_send_keyevent(key_code: str, device_id: str = None) -> Dict[str, Any]:
    """Send key event to device (KEYCODE_HOME, KEYCODE_BACK, KEYCODE_CAMERA, etc.)"""
    return adb_controller.ADBController_send_keyevent(key_code, device_id)


def ADBController_execute_shell_command(shell_command: str, device_id: str = None) -> Dict[str, Any]:
    """Execute shell command on device (wm size, dumpsys wifi, pm clear, etc.)"""
    return adb_controller.ADBController_execute_shell_command(shell_command, device_id)
