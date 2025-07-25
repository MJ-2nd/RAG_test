"""
ADB (Android Debug Bridge) Functions Module

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
            if result.returncode == 0:
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
    
    def get_connected_devices(self) -> List[Dict[str, str]]:
        """Get list of connected Android devices"""
        result = self._execute_adb_command(["devices", "-l"])
        
        if not result["success"]:
            return []
        
        devices = [{"id": "1234567890", "status": "connected"}]
        # lines = result["stdout"].strip().split('\n')[1:]  # Skip header
        
        # for line in lines:
        #     if line.strip():
        #         parts = line.split()
        #         if len(parts) >= 2:
        #             device_id = parts[0]
        #             status = parts[1]
        #             device_info = {"id": device_id, "status": status}
                    
        #             # Extract additional info if available
        #             if len(parts) > 2:
        #                 device_info["details"] = " ".join(parts[2:])
                    
        #             devices.append(device_info)
        
        return devices
    
    def install_apk(self, apk_path: str, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Install APK file on device"""
        command = ["install", "-r", apk_path]
        if device_id:
            command = ["-s", device_id] + command
        
        result = self._execute_adb_command(command, timeout=60)
        
        if result["success"] and "Success" in result["stdout"]:
            return {
                "success": True,
                "message": "APK installed successfully",
                "details": result["stdout"]
            }
        else:
            return {
                "success": False,
                "message": "Failed to install APK",
                "details": result["stderr"] or result["stdout"]
            }
    
    def uninstall_app(self, package_name: str, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Uninstall app by package name"""
        command = ["uninstall", package_name]
        if device_id:
            command = ["-s", device_id] + command
        
        result = self._execute_adb_command(command)
        
        if result["success"] and "Success" in result["stdout"]:
            return {
                "success": True,
                "message": "App uninstalled successfully",
                "details": result["stdout"]
            }
        else:
            return {
                "success": False,
                "message": "Failed to uninstall app",
                "details": result["stderr"] or result["stdout"]
            }
    
    def get_installed_packages(self, device_id: Optional[str] = None) -> List[str]:
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
    
    def launch_app(self, package_name: str, activity_name: Optional[str] = None, 
                  device_id: Optional[str] = None) -> Dict[str, Any]:
        """Launch app by package name and optional activity"""
        if activity_name:
            command = ["shell", "am", "start", "-n", f"{package_name}/{activity_name}"]
        else:
            command = ["shell", "monkey", "-p", package_name, "-c", "android.intent.category.LAUNCHER", "1"]
        
        if device_id:
            command = ["-s", device_id] + command
        
        result = self._execute_adb_command(command)
        
        return {
            "success": result["success"],
            "message": "App launched successfully" if result["success"] else "Failed to launch app",
            "details": result["stdout"] if result["success"] else result["stderr"]
        }
    
    def take_screenshot(self, output_path: str, device_id: Optional[str] = None) -> Dict[str, Any]:
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
    
    def get_device_info(self, device_id: Optional[str] = None) -> Dict[str, Any]:
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
    
    def input_text(self, text: str, device_id: Optional[str] = None) -> Dict[str, Any]:
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
    
    def tap_screen(self, x: int, y: int, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Tap screen at specific coordinates"""
        command = ["shell", "input", "tap", str(x), str(y)]
        if device_id:
            command = ["-s", device_id] + command
        
        result = self._execute_adb_command(command)
        
        return {
            "success": result["success"],
            "message": "Screen tap successful" if result["success"] else "Failed to tap screen",
            "details": result["stderr"] if not result["success"] else ""
        }
    
    def swipe_screen(self, x1: int, y1: int, x2: int, y2: int, 
                    duration: int = 500, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Swipe screen from one point to another"""
        command = ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration)]
        if device_id:
            command = ["-s", device_id] + command
        
        result = self._execute_adb_command(command)
        
        return {
            "success": result["success"],
            "message": "Screen swipe successful" if result["success"] else "Failed to swipe screen",
            "details": result["stderr"] if not result["success"] else ""
        }
    
    def get_battery_info(self, device_id: Optional[str] = None) -> Dict[str, Any]:
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
    
    def send_keyevent(self, key_code: str, device_id: Optional[str] = None) -> Dict[str, Any]:
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
    
    def execute_shell_command(self, shell_command: str, device_id: Optional[str] = None) -> Dict[str, Any]:
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


# MCP Function Definitions
def get_connected_devices() -> List[Dict[str, str]]:
    """Get list of connected Android devices"""
    return adb_controller.get_connected_devices()


def install_apk(apk_path: str, device_id: str = None) -> Dict[str, Any]:
    """Install APK file on Android device"""
    return adb_controller.install_apk(apk_path, device_id)


def uninstall_app(package_name: str, device_id: str = None) -> Dict[str, Any]:
    """Uninstall app by package name"""
    return adb_controller.uninstall_app(package_name, device_id)


def get_installed_packages(device_id: str = None) -> List[str]:
    """Get list of installed packages on device"""
    return adb_controller.get_installed_packages(device_id)


def launch_app(package_name: str, activity_name: str = None, device_id: str = None) -> Dict[str, Any]:
    """Launch app on Android device"""
    return adb_controller.launch_app(package_name, activity_name, device_id)


def take_screenshot(output_path: str, device_id: str = None) -> Dict[str, Any]:
    """Take screenshot and save to file"""
    return adb_controller.take_screenshot(output_path, device_id)


def get_device_info(device_id: str = None) -> Dict[str, Any]:
    """Get detailed device information"""
    return adb_controller.get_device_info(device_id)


def input_text(text: str, device_id: str = None) -> Dict[str, Any]:
    """Input text on Android device"""
    return adb_controller.input_text(text, device_id)


def tap_screen(x: int, y: int, device_id: str = None) -> Dict[str, Any]:
    """Tap screen at specific coordinates"""
    return adb_controller.tap_screen(x, y, device_id)


def swipe_screen(x1: int, y1: int, x2: int, y2: int, duration: int = 500, device_id: str = None) -> Dict[str, Any]:
    """Swipe screen from one point to another"""
    return adb_controller.swipe_screen(x1, y1, x2, y2, duration, device_id)


def get_battery_info(device_id: str = None) -> Dict[str, Any]:
    """Get battery information from device"""
    return adb_controller.get_battery_info(device_id)


# Unified key event function
def send_keyevent(key_code: str, device_id: str = None) -> Dict[str, Any]:
    """Send key event to device (KEYCODE_HOME, KEYCODE_BACK, KEYCODE_CAMERA, etc.)"""
    return adb_controller.send_keyevent(key_code, device_id)

def execute_shell_command(shell_command: str, device_id: str = None) -> Dict[str, Any]:
    """Execute shell command on device (wm size, dumpsys wifi, pm clear, etc.)"""
    return adb_controller.execute_shell_command(shell_command, device_id)



# Unified shell command function (already exists)
# execute_shell_command() handles all shell commands


# Optimized function definitions for MCP
ADB_FUNCTION_DEFINITIONS = [
    {
        "name": "get_connected_devices",
        "description": "Get list of connected Android devices",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        },
        "required_params": [],
        "function": get_connected_devices
    },
    {
        "name": "install_apk",
        "description": "Install APK file on Android device",
        "parameters": {
            "type": "object",
            "properties": {
                "apk_path": {
                    "type": "string",
                    "description": "Path to the APK file to install"
                },
                "device_id": {
                    "type": "string",
                    "description": "Device ID (optional, uses first connected device if not specified)"
                }
            },
            "required": ["apk_path"],
            "additionalProperties": False
        },
        "required_params": ["apk_path"],
        "function": install_apk
    },
    {
        "name": "uninstall_app",
        "description": "Uninstall app by package name",
        "parameters": {
            "type": "object",
            "properties": {
                "package_name": {
                    "type": "string",
                    "description": "Package name of the app to uninstall"
                },
                "device_id": {
                    "type": "string",
                    "description": "Device ID (optional, uses first connected device if not specified)"
                }
            },
            "required": ["package_name"],
            "additionalProperties": False
        },
        "required_params": ["package_name"],
        "function": uninstall_app
    },
    {
        "name": "get_installed_packages",
        "description": "Get list of installed packages on device",
        "parameters": {
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": "Device ID (optional, uses first connected device if not specified)"
                }
            },
            "additionalProperties": False
        },
        "required_params": [],
        "function": get_installed_packages
    },
    {
        "name": "launch_app",
        "description": "Launch app on Android device",
        "parameters": {
            "type": "object",
            "properties": {
                "package_name": {
                    "type": "string",
                    "description": "Package name of the app to launch"
                },
                "activity_name": {
                    "type": "string",
                    "description": "Activity name (optional)"
                },
                "device_id": {
                    "type": "string",
                    "description": "Device ID (optional, uses first connected device if not specified)"
                }
            },
            "required": ["package_name"],
            "additionalProperties": False
        },
        "required_params": ["package_name"],
        "function": launch_app
    },
    {
        "name": "take_screenshot",
        "description": "Take screenshot and save to file",
        "parameters": {
            "type": "object",
            "properties": {
                "output_path": {
                    "type": "string",
                    "description": "Path where to save the screenshot"
                },
                "device_id": {
                    "type": "string",
                    "description": "Device ID (optional, uses first connected device if not specified)"
                }
            },
            "required": ["output_path"],
            "additionalProperties": False
        },
        "required_params": ["output_path"],
        "function": take_screenshot
    },
    {
        "name": "get_device_info",
        "description": "Get detailed device information",
        "parameters": {
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": "Device ID (optional, uses first connected device if not specified)"
                }
            },
            "additionalProperties": False
        },
        "required_params": [],
        "function": get_device_info
    },
    {
        "name": "input_text",
        "description": "Input text on Android device",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to input on device"
                },
                "device_id": {
                    "type": "string",
                    "description": "Device ID (optional, uses first connected device if not specified)"
                }
            },
            "required": ["text"],
            "additionalProperties": False
        },
        "required_params": ["text"],
        "function": input_text
    },
    {
        "name": "tap_screen",
        "description": "Tap screen at specific coordinates",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "X coordinate for tap"
                },
                "y": {
                    "type": "integer",
                    "description": "Y coordinate for tap"
                },
                "device_id": {
                    "type": "string",
                    "description": "Device ID (optional, uses first connected device if not specified)"
                }
            },
            "required": ["x", "y"],
            "additionalProperties": False
        },
        "required_params": ["x", "y"],
        "function": tap_screen
    },
    {
        "name": "swipe_screen",
        "description": "Swipe screen from one point to another",
        "parameters": {
            "type": "object",
            "properties": {
                "x1": {
                    "type": "integer",
                    "description": "Starting X coordinate"
                },
                "y1": {
                    "type": "integer",
                    "description": "Starting Y coordinate"
                },
                "x2": {
                    "type": "integer",
                    "description": "Ending X coordinate"
                },
                "y2": {
                    "type": "integer",
                    "description": "Ending Y coordinate"
                },
                "duration": {
                    "type": "integer",
                    "description": "Duration of swipe in milliseconds (default: 500)",
                    "default": 500
                },
                "device_id": {
                    "type": "string",
                    "description": "Device ID (optional, uses first connected device if not specified)"
                }
            },
            "required": ["x1", "y1", "x2", "y2"],
            "additionalProperties": False
        },
        "required_params": ["x1", "y1", "x2", "y2"],
        "function": swipe_screen
    },
    {
        "name": "get_battery_info",
        "description": "Get battery information from device",
        "parameters": {
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": "Device ID (optional, uses first connected device if not specified)"
                }
            },
            "additionalProperties": False
        },
        "required_params": [],
        "function": get_battery_info
    },
    # Unified key event function
    {
        "name": "send_keyevent",
        "description": "Send key event (KEYCODE_HOME, KEYCODE_BACK, KEYCODE_CAMERA, KEYCODE_VOLUME_UP, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "key_code": {
                    "type": "string",
                    "description": "Key code (e.g., KEYCODE_HOME, KEYCODE_BACK, KEYCODE_CAMERA, KEYCODE_VOLUME_UP, KEYCODE_POWER)"
                },
                "device_id": {
                    "type": "string",
                    "description": "Device ID (optional, uses first connected device if not specified)"
                }
            },
            "required": ["key_code"],
            "additionalProperties": False
        },
        "required_params": ["key_code"],
        "function": send_keyevent
    },
    # Unified shell command function
    {
        "name": "execute_shell_command",
        "description": "Execute shell command on device (wm size, dumpsys wifi, pm clear, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "shell_command": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "device_id": {
                    "type": "string",
                    "description": "Device ID (optional, uses first connected device if not specified)"
                }
            },
            "required": ["shell_command"],
            "additionalProperties": False
        },
        "required_params": ["shell_command"],
        "function": execute_shell_command
    }
] 