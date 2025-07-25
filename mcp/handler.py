"""
MCP (Model Context Protocol) Handler Module

This module handles the execution of MCP function calls and result processing.
Provides a clean interface for function execution and response formatting.
"""

import logging
from typing import List, Dict, Any, Optional
from .functions import MCPFunctionRegistry
from .parser import FunctionCall, MCPResponseParser

logger = logging.getLogger(__name__)


class MCPHandler:
    """Handler for MCP function calls and execution"""
    
    def __init__(self, function_registry: MCPFunctionRegistry):
        self.function_registry = function_registry
        self.parser = MCPResponseParser()
    
    def process_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Process LLM response and execute any function calls"""
        
        logger.info("Processing LLM response for function calls")
        
        # Parse function calls from response
        function_calls, cleaned_text = self.parser.extract_function_calls_with_context(response_text)
        
        # Execute functions if any
        function_results = []
        if function_calls:
            logger.info(f"Executing {len(function_calls)} function calls")
            function_results = self._execute_function_calls(function_calls)
        else:
            logger.info("No function calls found in response")
        
        return {
            "response_text": cleaned_text,
            "function_calls": function_calls,
            "function_results": function_results,
            "has_functions": len(function_calls) > 0
        }
    
    def _execute_function_calls(self, function_calls: List[FunctionCall]) -> List[Dict[str, Any]]:
        """Execute a list of function calls"""
        results = []
        
        for i, func_call in enumerate(function_calls):
            logger.info(f"Executing function call {i+1}/{len(function_calls)}: {func_call.name}")
            
            try:
                # Execute the function
                result = self.function_registry.execute_function(
                    func_call.name, 
                    func_call.arguments
                )
                
                # Add metadata to result
                result_with_metadata = {
                    "function_name": func_call.name,
                    "arguments": func_call.arguments,
                    "confidence": func_call.confidence,
                    "execution_result": result,
                    "success": "error" not in result
                }
                
                results.append(result_with_metadata)
                logger.info(f"Function {func_call.name} executed successfully")
                
            except Exception as e:
                logger.error(f"Error executing function {func_call.name}: {e}")
                results.append({
                    "function_name": func_call.name,
                    "arguments": func_call.arguments,
                    "confidence": func_call.confidence,
                    "execution_result": {"error": str(e)},
                    "success": False
                })
        
        return results
    
    def format_function_results_for_llm(self, function_results: List[Dict[str, Any]]) -> str:
        """Format function results for LLM consumption"""
        
        if not function_results:
            return ""
        
        formatted_results = []
        for result in function_results:
            func_name = result["function_name"]
            success = result["success"]
            exec_result = result["execution_result"]
            
            if success:
                formatted_results.append(
                    f"Function '{func_name}' returned: {exec_result.get('result', 'No result')}"
                )
            else:
                formatted_results.append(
                    f"Function '{func_name}' failed: {exec_result.get('error', 'Unknown error')}"
                )
        
        return "\n".join(formatted_results)
    
    def get_available_functions_summary(self) -> str:
        """Get a summary of available functions"""
        functions = self.function_registry.list_functions()
        
        if not functions:
            return "No functions available."
        
        summary_parts = ["Available functions:"]
        for func_def in functions:
            # Format parameters
            param_desc = "no parameters"
            if func_def.parameters.get("properties"):
                props = func_def.parameters["properties"]
                param_list = [f"{name}: {prop.get('type', 'any')}" 
                            for name, prop in props.items()]
                param_desc = ", ".join(param_list)
            
            summary_parts.append(f"- {func_def.name}({param_desc}): {func_def.description}")
        
        return "\n".join(summary_parts)
    
    def validate_function_call(self, func_call: FunctionCall) -> Dict[str, Any]:
        """Validate a function call before execution"""
        
        # Check if function exists
        func_def = self.function_registry.get_function(func_call.name)
        if not func_def:
            return {
                "valid": False,
                "error": f"Function '{func_call.name}' not found"
            }
        
        # Check required parameters
        missing_params = []
        for param in func_def.required_params:
            if param not in func_call.arguments:
                missing_params.append(param)
        
        if missing_params:
            return {
                "valid": False,
                "error": f"Missing required parameters: {missing_params}"
            }
        
        # Check parameter types (basic validation)
        validation_errors = []
        for param_name, param_value in func_call.arguments.items():
            if param_name in func_def.parameters.get("properties", {}):
                param_schema = func_def.parameters["properties"][param_name]
                expected_type = param_schema.get("type")
                
                if expected_type == "string" and not isinstance(param_value, str):
                    validation_errors.append(f"Parameter '{param_name}' should be string")
                elif expected_type == "number" and not isinstance(param_value, (int, float)):
                    validation_errors.append(f"Parameter '{param_name}' should be number")
                elif expected_type == "boolean" and not isinstance(param_value, bool):
                    validation_errors.append(f"Parameter '{param_name}' should be boolean")
        
        if validation_errors:
            return {
                "valid": False,
                "error": f"Parameter validation errors: {'; '.join(validation_errors)}"
            }
        
        return {"valid": True}
    
    def get_function_statistics(self) -> Dict[str, Any]:
        """Get statistics about available functions"""
        functions = self.function_registry.list_functions()
        
        total_functions = len(functions)
        functions_with_params = sum(1 for f in functions 
                                  if f.parameters.get("properties"))
        functions_without_params = total_functions - functions_with_params
        
        return {
            "total_functions": total_functions,
            "functions_with_parameters": functions_with_params,
            "functions_without_parameters": functions_without_params,
            "function_names": [f.name for f in functions]
        } 