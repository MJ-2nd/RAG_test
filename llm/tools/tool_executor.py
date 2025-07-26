"""
Tool execution service
"""

import ast
import logging
import operator
from datetime import datetime
from typing import Dict, Any, Optional

from .tool_manager import ToolCall
from .adb_executor import (
    get_connected_devices, get_installed_packages, take_screenshot,
    get_device_info, input_text, get_battery_info, open_settings,
    send_keyevent, execute_shell_command
)
from ..constants import ToolNames, ErrorMessages, Defaults

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Handles tool execution"""
    
    def __init__(self, android_controller=None):
        self.android_controller = android_controller
    
    async def execute_tool(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Execute a tool call"""
        try:
            tool_name = tool_call.name
            
            # Basic tools
            if tool_name == ToolNames.CALCULATE:
                return self._execute_calculation(tool_call)
            elif tool_name == ToolNames.GET_CURRENT_TIME:
                return self._execute_get_current_time(tool_call)
            
            # ADB tools
            elif tool_name == "get_connected_devices":
                return {"result": get_connected_devices()}
            elif tool_name == "get_installed_packages":
                device_id = tool_call.arguments.get('device_id')
                return {"result": get_installed_packages(device_id)}
            elif tool_name == "take_screenshot":
                output_path = tool_call.arguments.get('output_path')
                device_id = tool_call.arguments.get('device_id')
                return take_screenshot(output_path, device_id)
            elif tool_name == "get_device_info":
                device_id = tool_call.arguments.get('device_id')
                return {"result": get_device_info(device_id)}
            elif tool_name == "input_text":
                text = tool_call.arguments.get('text')
                device_id = tool_call.arguments.get('device_id')
                return input_text(text, device_id)
            elif tool_name == "get_battery_info":
                device_id = tool_call.arguments.get('device_id')
                return {"result": get_battery_info(device_id)}
            elif tool_name == "open_settings":
                device_id = tool_call.arguments.get('device_id')
                return open_settings(device_id)
            elif tool_name == "send_keyevent":
                key_code = tool_call.arguments.get('key_code')
                device_id = tool_call.arguments.get('device_id')
                return send_keyevent(key_code, device_id)
            elif tool_name == "execute_shell_command":
                shell_command = tool_call.arguments.get('shell_command')
                device_id = tool_call.arguments.get('device_id')
                return execute_shell_command(shell_command, device_id)
            
            # Legacy Android control (for compatibility)
            elif tool_name == ToolNames.CONTROL_ANDROID_DEVICE:
                return self._execute_android_control(tool_call)
            
            else:
                return {"error": ErrorMessages.UNKNOWN_TOOL.format(tool_name)}
                
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"error": ErrorMessages.TOOL_EXECUTION_FAILED.format(str(e))}
    
    
    def _execute_calculation(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Execute mathematical calculation"""
        expression = tool_call.arguments.get('expression', '')
        logger.info(f"Calculating: {expression}")
        
        try:
            # 안전한 계산을 위해 허용된 연산만 사용
            allowed_ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }
            
            def eval_expr(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return allowed_ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return allowed_ops[type(node.op)](eval_expr(node.operand))
                else:
                    raise TypeError(node)
            
            result = eval_expr(ast.parse(expression, mode='eval').body)
            return {"result": result}
            
        except Exception as e:
            return {"error": f"Calculation error: {str(e)}"}
    
    def _execute_get_time(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Execute get current time"""
        logger.info("Getting current time")
        return {"result": datetime.now().isoformat()}
    
    def _execute_android_control(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Execute Android device control"""
        logger.info(f"Control Android device: {tool_call.arguments}")
        
        if not self.android_controller:
            return {"error": "Android controller not available"}
        
        try:
            result = self.android_controller.control_android_device(tool_call.arguments)
            return result
        except Exception as e:
            return {"error": f"Android device control error: {str(e)}"} 