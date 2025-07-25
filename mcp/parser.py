"""
MCP (Model Context Protocol) Parser Module

This module handles parsing of function calls from LLM responses.
Supports multiple formats for maximum compatibility.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FunctionCall:
    """Parsed function call from LLM response"""
    name: str
    arguments: Dict[str, Any]
    confidence: float = 1.0


class MCPResponseParser:
    """Parser for MCP function calls from LLM responses"""
    
    def __init__(self):
        self._patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> List[Tuple[str, str]]:
        """Initialize regex patterns for different function call formats"""
        return [
            # XML format: <function_call>{"name": "...", "arguments": {...}}</function_call>
            (r'<function_call>\s*(\{.*?\})\s*</function_call>', 'xml'),
            
            # JSON code block: ```json {"name": "...", "arguments": {...}} ```
            (r'```json\s*(\{.*?"name".*?\})\s*```', 'json_block'),
            
            # Tool call format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
            (r'<tool_call>\s*(\{.*?\})\s*</tool_call>', 'tool_call'),
            
            # Simple JSON: {"name": "...", "arguments": {...}}
            (r'(\{[^{}]*"name"[^{}]*\})', 'simple_json'),
            
            # Function format: function_name(arg1, arg2)
            (r'(\w+)\s*\(\s*([^)]*)\s*\)', 'function_call'),
        ]
    
    def parse_function_calls(self, text: str) -> List[FunctionCall]:
        """Parse function calls from LLM response text"""
        function_calls = []
        
        logger.info(f"Parsing function calls from text: {text[:200]}...")
        
        for pattern, format_type in self._patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                try:
                    if format_type == 'function_call':
                        # Handle function_name(arg1, arg2) format
                        func_call = self._parse_function_call_format(match[0], match[1])
                    else:
                        # Handle JSON-based formats
                        func_call = self._parse_json_format(match, format_type)
                    
                    if func_call:
                        function_calls.append(func_call)
                        logger.info(f"Parsed function call: {func_call.name}")
                        
                except Exception as e:
                    logger.warning(f"Failed to parse function call with pattern {format_type}: {e}")
                    continue
        
        logger.info(f"Total function calls parsed: {len(function_calls)}")
        return function_calls
    
    def _parse_json_format(self, json_str: str, format_type: str) -> Optional[FunctionCall]:
        """Parse JSON-based function call formats"""
        try:
            # Clean up the JSON string
            json_str = json_str.strip()
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Extract function name and arguments
            name = data.get('name', '').strip()
            arguments = data.get('arguments', {})
            
            if not name:
                logger.warning(f"No function name found in {format_type} format")
                return None
            
            # Validate arguments is a dictionary
            if not isinstance(arguments, dict):
                arguments = {}
            
            return FunctionCall(
                name=name,
                arguments=arguments,
                confidence=1.0
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error in {format_type} format: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error parsing {format_type} format: {e}")
            return None
    
    def _parse_function_call_format(self, func_name: str, args_str: str) -> Optional[FunctionCall]:
        """Parse function_name(arg1, arg2) format"""
        try:
            # Parse arguments string
            arguments = {}
            
            if args_str.strip():
                # Simple argument parsing (can be extended for complex cases)
                args_parts = args_str.split(',')
                for i, part in enumerate(args_parts):
                    part = part.strip()
                    if '=' in part:
                        # Named argument: key=value
                        key, value = part.split('=', 1)
                        key = key.strip()
                        value = self._parse_value(value.strip())
                        arguments[key] = value
                    else:
                        # Positional argument: use index as key
                        value = self._parse_value(part)
                        arguments[f"arg_{i}"] = value
            
            return FunctionCall(
                name=func_name.strip(),
                arguments=arguments,
                confidence=0.8  # Lower confidence for this format
            )
            
        except Exception as e:
            logger.warning(f"Error parsing function call format: {e}")
            return None
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse a value string to appropriate type"""
        value_str = value_str.strip()
        
        # Remove quotes if present
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]
        
        # Try to parse as number
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass
        
        # Try to parse as boolean
        if value_str.lower() in ['true', 'false']:
            return value_str.lower() == 'true'
        
        # Return as string
        return value_str
    
    def extract_function_calls_with_context(self, text: str) -> Tuple[List[FunctionCall], str]:
        """Extract function calls and return remaining text"""
        function_calls = self.parse_function_calls(text)
        
        # Remove function call blocks from text
        cleaned_text = text
        for pattern, _ in self._patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up extra whitespace
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text).strip()
        
        return function_calls, cleaned_text


# Global parser instance
response_parser = MCPResponseParser() 