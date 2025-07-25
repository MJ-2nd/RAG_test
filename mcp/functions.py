"""
MCP (Model Context Protocol) Functions Module

This module defines functions that can be called by the LLM through MCP.
Each function is designed to be self-contained and easily extensible.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FunctionDefinition:
    """Function definition for MCP protocol"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]
    function: callable


class MCPFunctionRegistry:
    """Registry for MCP functions with clean, extensible design"""
    
    def __init__(self):
        self._functions: Dict[str, FunctionDefinition] = {}
        self._register_default_functions()
    
    def register_function(self, func_def: FunctionDefinition) -> None:
        """Register a new function in the registry"""
        self._functions[func_def.name] = func_def
        logger.info(f"Registered MCP function: {func_def.name}")
    
    def get_function(self, name: str) -> Optional[FunctionDefinition]:
        """Get a function by name"""
        return self._functions.get(name)
    
    def list_functions(self) -> List[FunctionDefinition]:
        """List all registered functions"""
        return list(self._functions.values())
    
    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """Get function schemas for MCP protocol"""
        schemas = []
        for func_def in self._functions.values():
            schema = {
                "name": func_def.name,
                "description": func_def.description,
                "parameters": func_def.parameters,
                "required": func_def.required_params
            }
            schemas.append(schema)
        return schemas
    
    def execute_function(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function with given arguments"""
        func_def = self.get_function(name)
        if not func_def:
            return {"error": f"Function '{name}' not found"}
        
        try:
            # Validate required parameters
            missing_params = [param for param in func_def.required_params 
                            if param not in arguments]
            if missing_params:
                return {"error": f"Missing required parameters: {missing_params}"}
            
            # Execute function
            result = func_def.function(**arguments)
            logger.info(f"Executed function '{name}' with result: {result}")
            return {"result": result}
            
        except Exception as e:
            logger.error(f"Error executing function '{name}': {e}")
            return {"error": str(e)}
    
    def _register_default_functions(self) -> None:
        """Register default functions"""
        self._register_time_functions()
    
    def _register_time_functions(self) -> None:
        """Register time-related functions"""
        
        def get_current_time() -> str:
            """Get current date and time in ISO format"""
            return datetime.now().isoformat()
        
        def get_current_date() -> str:
            """Get current date in YYYY-MM-DD format"""
            return datetime.now().strftime("%Y-%m-%d")
        
        def get_current_time_formatted(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
            """Get current time in custom format"""
            return datetime.now().strftime(format_str)
        
        # Register time functions
        time_functions = [
            FunctionDefinition(
                name="get_current_time",
                description="Get current date and time in ISO format",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                },
                required_params=[],
                function=get_current_time
            ),
            FunctionDefinition(
                name="get_current_date",
                description="Get current date in YYYY-MM-DD format",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                },
                required_params=[],
                function=get_current_date
            ),
            FunctionDefinition(
                name="get_current_time_formatted",
                description="Get current time in custom format",
                parameters={
                    "type": "object",
                    "properties": {
                        "format_str": {
                            "type": "string",
                            "description": "Time format string (e.g., '%Y-%m-%d %H:%M:%S')",
                            "default": "%Y-%m-%d %H:%M:%S"
                        }
                    },
                    "additionalProperties": False
                },
                required_params=[],
                function=get_current_time_formatted
            )
        ]
        
        for func_def in time_functions:
            self.register_function(func_def)


# Global function registry instance
function_registry = MCPFunctionRegistry() 