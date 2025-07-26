"""
Tool management for LLM server with RAG-based tool retrieval
"""

import json
import logging
import os
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from ..constants import ToolNames, ErrorMessages
from ..config import ConfigManager
from .rag_tool_retriever import RAGToolRetriever

logger = logging.getLogger(__name__)


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


class ToolManager:
    """Handles tool loading and management with RAG-based retrieval"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.rag_retriever = RAGToolRetriever(config_manager.config_path)
        # Keep minimal basic tools
        self._basic_tools = self._create_basic_tools()
    
    def _create_basic_tools(self) -> Dict[str, Dict[str, Any]]:
        """Create minimal basic tools that are always available"""
        return {
            ToolNames.CALCULATE: {
                "name": ToolNames.CALCULATE,
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                    },
                    "required": ["expression"]
                }
            },
            ToolNames.GET_CURRENT_TIME: {
                "name": ToolNames.GET_CURRENT_TIME,
                "description": "Get current date and time",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    
    def parse_tool_calls(self, text: str) -> List[ToolCall]:
        """Parse tool calls from generated text (model-agnostic)"""
        tool_calls = []
        try:
            # 다양한 tool-calling 형식 지원
            patterns = [
                r'<tool_call>\s*(\{.*?\})\s*</tool_call>',  # XML 형식
                r'```json\s*(\{.*?"name".*?\})\s*```',      # JSON 코드 블록
                r'<function_call>\s*(\{.*?\})\s*</function_call>',  # Function call 형식
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                for match in matches:
                    try:
                        tool_data = json.loads(match)
                        if 'name' in tool_data:
                            tool_calls.append(ToolCall(
                                name=tool_data.get('name', ''),
                                arguments=tool_data.get('arguments', {})
                            ))
                    except json.JSONDecodeError:
                        continue
            
        except Exception as e:
            logger.warning(f"Failed to parse tool calls: {e}")
        
        return tool_calls
    
    def get_relevant_tools_for_query(self, user_query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get relevant tools based on user query using RAG"""
        logger.info(f"Getting relevant tools for query: '{user_query}'")
        
        # Get RAG-based relevant tools
        relevant_tools = self.rag_retriever.get_relevant_tools(user_query, top_k)
        
        # Always include basic tools
        basic_tools = list(self._basic_tools.values())
        
        # Convert RAG results to standard tool format
        formatted_tools = basic_tools.copy()
        
        for tool in relevant_tools:
            formatted_tool = {
                "name": tool['name'],
                "description": tool['description'],
                "parameters": {
                    "type": "object",
                    "properties": self._convert_parameters(tool.get('parameters', {})),
                    "required": self._get_required_parameters(tool.get('parameters', {}))
                }
            }
            formatted_tools.append(formatted_tool)
        
        logger.info(f"Returning {len(formatted_tools)} tools ({len(basic_tools)} basic + {len(relevant_tools)} relevant)")
        return formatted_tools
    
    def _convert_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert RAG parameters to OpenAPI format"""
        converted = {}
        for param_name, param_info in parameters.items():
            if isinstance(param_info, dict):
                converted[param_name] = {
                    "type": "string",  # Default to string
                    "description": param_info.get('description', '')
                }
            else:
                converted[param_name] = {
                    "type": "string",
                    "description": str(param_info)
                }
        return converted
    
    def _get_required_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Extract required parameters"""
        required = []
        for param_name, param_info in parameters.items():
            if isinstance(param_info, dict) and param_info.get('required', False):
                required.append(param_name)
        return required
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of basic tools (deprecated - use get_relevant_tools_for_query)"""
        logger.warning("get_available_tools is deprecated. Use get_relevant_tools_for_query instead.")
        return list(self._basic_tools.values())
    
    def get_tool_definition(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool definition by name"""
        # Check basic tools first
        if tool_name in self._basic_tools:
            return self._basic_tools[tool_name]
        
        # For ADB tools, we don't maintain a static registry
        # Tool definitions come from RAG retrieval
        return None
    
    def is_tool_available(self, tool_name: str) -> bool:
        """Check if tool is available"""
        # Basic tools are always available
        if tool_name in self._basic_tools:
            return True
        
        # ADB tools are available if mentioned in documentation
        available_functions = self.rag_retriever.get_all_available_functions()
        return tool_name in available_functions
    
    @property
    def tools_registry(self) -> Dict[str, Dict[str, Any]]:
        """Legacy property for compatibility"""
        return self._basic_tools 