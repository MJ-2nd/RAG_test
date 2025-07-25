"""
MCP (Model Context Protocol) Prompt Builder Module

This module builds prompts that include function definitions for MCP.
Designed to work with various LLM models and formats.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from .functions import MCPFunctionRegistry

logger = logging.getLogger(__name__)


class MCPPromptBuilder:
    """Builder for MCP-enabled prompts with function definitions"""
    
    def __init__(self, function_registry: MCPFunctionRegistry):
        self.function_registry = function_registry
    
    def build_prompt(self, 
                    user_message: str, 
                    model_type: str = "generic",
                    include_functions: bool = True,
                    context: Optional[str] = None,
                    history: Optional[str] = None) -> str:
        """Build a complete prompt with MCP function definitions"""
        
        logger.info(f"Building MCP prompt for model type: {model_type}")
        
        # Get function schemas
        function_schemas = []
        if include_functions:
            function_schemas = self.function_registry.get_function_schemas()
            logger.info(f"Including {len(function_schemas)} functions in prompt")
        
        # Build system message
        system_message = self._build_system_message(
            function_schemas, model_type, context, history
        )
        
        # Build complete prompt based on model type
        prompt = self._format_prompt_for_model(
            system_message, user_message, model_type
        )
        
        logger.info(f"Built prompt with {len(prompt)} characters")
        return prompt
    
    def _build_system_message(self, 
                             function_schemas: List[Dict[str, Any]],
                             model_type: str,
                             context: Optional[str] = None,
                             history: Optional[str] = None) -> str:
        """Build the system message with function definitions"""
        
        # Base system message
        system_parts = [
            "You are a helpful AI assistant with advanced reasoning and function-calling capabilities.",
            "You can use available functions to perform specific tasks when needed."
        ]
        
        # Add history if provided
        if history and history.strip():
            system_parts.append(f"\nPrevious conversation:\n{history}")
        
        # Add context if provided
        if context:
            system_parts.append(
                f"\nPlease refer to the following context to answer:\n```\n{context}\n```"
            )
        
        # Add function definitions
        if function_schemas:
            function_section = self._build_function_section(function_schemas, model_type)
            system_parts.append(function_section)
        
        return "\n".join(system_parts)
    
    def _build_function_section(self, 
                               function_schemas: List[Dict[str, Any]], 
                               model_type: str) -> str:
        """Build the function definitions section"""
        
        # Format function schemas as JSON
        functions_json = json.dumps(function_schemas, indent=2, ensure_ascii=False)
        
        # Get function calling instructions based on model type
        calling_instructions = self._get_calling_instructions(model_type)
        
        return f"""

Available functions:
{functions_json}

{calling_instructions}

You can use multiple functions by including multiple function call blocks.
When you need to use a function, respond with the appropriate function call format.
"""
    
    def _get_calling_instructions(self, model_type: str) -> str:
        """Get function calling instructions for specific model types"""
        
        if model_type in ["deepseek", "qwen", "smollm"]:
            return """To use a function, respond with:
<function_call>
{"name": "function_name", "arguments": {"param": "value"}}
</function_call>"""
        
        elif model_type == "llama":
            return """To use a function, respond with:
<function_call>
{"name": "function_name", "arguments": {"param": "value"}}
</function_call>"""
        
        elif model_type == "mistral":
            return """To use a function, respond with:
<function_call>
{"name": "function_name", "arguments": {"param": "value"}}
</function_call>"""
        
        else:
            return """To use a function, respond with a JSON object in a code block:
```json
{"name": "function_name", "arguments": {"param": "value"}}
```"""
    
    def _format_prompt_for_model(self, 
                                system_message: str, 
                                user_message: str, 
                                model_type: str) -> str:
        """Format prompt according to model-specific chat template"""
        
        if model_type in ["deepseek", "qwen", "smollm"]:
            # ChatML format
            return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
        
        elif model_type == "llama":
            # Llama format
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        elif model_type == "mistral":
            # Mistral format
            return f"""<s>[INST] {system_message}

{user_message} [/INST]"""
        
        else:
            # Generic format
            return f"""System: {system_message}

User: {user_message}
Assistant:"""
    
    def get_function_summary(self) -> str:
        """Get a human-readable summary of available functions"""
        functions = self.function_registry.list_functions()
        
        if not functions:
            return "No functions available."
        
        summary_parts = ["Available functions:"]
        for func_def in functions:
            summary_parts.append(f"- {func_def.name}: {func_def.description}")
        
        return "\n".join(summary_parts) 