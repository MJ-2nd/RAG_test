"""
Prompt formatting service
"""

import json
import logging
from typing import Dict, Any, List, Optional

from ..constants import ModelType, ChatTemplate
from ..config import ConfigManager

logger = logging.getLogger(__name__)


class PromptFormatter:
    """Handles prompt formatting based on model type"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
    
    def _get_chat_template(self, model_type: str) -> str:
        """Get appropriate chat template based on model type"""
        template_mapping = {
            ModelType.KIMI: ChatTemplate.CHATML,
            ModelType.QWEN: ChatTemplate.CHATML,
            ModelType.DEEPSEEK: ChatTemplate.CHATML,
            ModelType.SMOLLM: ChatTemplate.CHATML,
            ModelType.LLAMA: ChatTemplate.LLAMA,
            ModelType.MISTRAL: ChatTemplate.MISTRAL,
            ModelType.GENERIC: ChatTemplate.CHATML
        }
        return template_mapping.get(model_type, ChatTemplate.CHATML)
    
    def _create_tools_section(self, tools: List[Dict[str, Any]], model_type: str) -> str:
        """Create tools section for the prompt with relevant tools only"""
        if not tools or not self.config_manager.is_tool_calling_enabled():
            return ""
        
        logger.info(f"Adding {len(tools)} relevant tools to prompt")
        
        # Create simplified tool descriptions (to reduce prompt size)
        simplified_tools = []
        for tool in tools:
            simplified_tool = {
                "name": tool["name"],
                "description": tool["description"]
            }
            
            # Add required parameters only
            if "parameters" in tool and "required" in tool["parameters"]:
                required_params = tool["parameters"]["required"]
                if required_params:
                    simplified_tool["required_params"] = required_params
            
            simplified_tools.append(simplified_tool)
        
        tools_json = json.dumps(simplified_tools, indent=2, ensure_ascii=False)
        
        # 모델 타입에 따른 tool-calling 형식 안내
        if model_type in [ModelType.KIMI, ModelType.QWEN, ModelType.DEEPSEEK]:
            tool_format = """
To use a tool, respond with:
<tool_call>
{"name": "tool_name", "arguments": {"param": "value"}}
</tool_call>"""
            logger.info("Using ChatML tool format")
        else:
            tool_format = """
To use a tool, respond with a JSON object in a code block:
```json
{"name": "tool_name", "arguments": {"param": "value"}}
```"""
            logger.info("Using JSON tool format")
        
        return f"""

Available tools (selected based on your request):
{tools_json}
{tool_format}

You can use multiple tools by including multiple tool call blocks.
"""
    
    def _create_history_section(self, history: Optional[str]) -> str:
        """Create history section for the prompt"""
        if not history or not history.strip():
            return ""
        
        logger.info("Added history section")
        return f"""
Previous conversation:
{history}

"""
    
    def _create_context_section(self, context: Optional[str]) -> str:
        """Create context section for the prompt"""
        if not context:
            return ""
        
        logger.info("Added context to system message")
        return f"""

Please refer to the following context to answer:
``` "The Context"
{context}
```

Provide accurate and useful answers based on the above context.
If the context doesn't contain relevant information, you can use available tools or provide answers based on your knowledge."""
    
    def _create_system_message(self, context: Optional[str], history: Optional[str], 
                              tools_section: str) -> str:
        """Create system message"""
        history_section = self._create_history_section(history)
        context_section = self._create_context_section(context)
        
        base_message = "You are a helpful AI assistant with advanced reasoning and tool-calling capabilities."
        
        if context:
            return f"""{base_message}

{history_section}Answer the user's question based on the above conversation history.
{context_section}
{tools_section}"""
        else:
            return f"""{base_message}

{history_section}Answer the user's question based on the above conversation history.
{tools_section}"""
    
    def _format_chatml_prompt(self, system_message: str, user_message: str) -> str:
        """Format prompt using ChatML template"""
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
    
    def _format_llama_prompt(self, system_message: str, user_message: str) -> str:
        """Format prompt using Llama template"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def _format_mistral_prompt(self, system_message: str, user_message: str) -> str:
        """Format prompt using Mistral template"""
        return f"""<s>[INST] {system_message}

{user_message} [/INST]"""
    
    def _format_generic_prompt(self, system_message: str, user_message: str) -> str:
        """Format prompt using generic template"""
        return f"""System: {system_message}

User: {user_message}
"""
    
    def format_chat_prompt(self, user_message: str, model_type: str,
                          context: Optional[str] = None, 
                          history: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Format chat prompt with appropriate template based on model type"""
        logger.info("=== FORMATTING CHAT PROMPT ===")
        logger.info(f"User message: {user_message}")
        logger.info(f"Model type: {model_type}")
        logger.info(f"Tools provided: {len(tools) if tools else 0}")
        
        # Create sections
        tools_section = self._create_tools_section(tools, model_type)
        system_message = self._create_system_message(context, history, tools_section)
        
        # Apply template
        template_type = self._get_chat_template(model_type)
        logger.info(f"Using template type: {template_type}")
        
        if template_type == ChatTemplate.CHATML:
            chat_prompt = self._format_chatml_prompt(system_message, user_message)
        elif template_type == ChatTemplate.LLAMA:
            chat_prompt = self._format_llama_prompt(system_message, user_message)
        elif template_type == ChatTemplate.MISTRAL:
            chat_prompt = self._format_mistral_prompt(system_message, user_message)
        else:
            chat_prompt = self._format_generic_prompt(system_message, user_message)
        
        logger.info(f"Final prompt length: {len(chat_prompt)} characters")
        logger.info("=== CHAT PROMPT FORMATTING COMPLETE ===")
        return chat_prompt 