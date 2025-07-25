"""
MCP (Model Context Protocol) Package

This package provides a complete implementation of MCP for LLM function calling.
It includes function definitions, parsing, prompt building, and execution handling.
"""

from .functions import MCPFunctionRegistry, FunctionDefinition, function_registry
from .parser import MCPResponseParser, FunctionCall, response_parser
from .prompt_builder import MCPPromptBuilder
from .handler import MCPHandler
from .adb_functions import ADB_FUNCTION_DEFINITIONS

__all__ = [
    'MCPFunctionRegistry',
    'FunctionDefinition', 
    'function_registry',
    'MCPResponseParser',
    'FunctionCall',
    'response_parser',
    'MCPPromptBuilder',
    'MCPHandler',
    'ADB_FUNCTION_DEFINITIONS'
]

# Version information
__version__ = "1.0.0"
__author__ = "MCP Implementation Team"
__description__ = "Model Context Protocol implementation for LLM function calling" 