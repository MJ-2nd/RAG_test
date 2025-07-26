"""
Core LLM server implementation
"""

import logging
from typing import Dict, Any, List, Optional

from adb_implementation_example import AndroidDeviceController

from ..config import ConfigManager
from ..models import ModelManager
from ..tools import ToolManager, ToolExecutor
from ..services import PromptFormatter, TextGenerator
from ..constants import Defaults, ErrorMessages

logger = logging.getLogger(__name__)


class LLMServer:
    """Main LLM server class"""
    
    def __init__(self, config_path: str = "llm/config.yaml"):
        # Initialize configuration
        self.config_manager = ConfigManager(config_path)
        
        # Initialize components
        self.model_manager = ModelManager(self.config_manager)
        self.tool_manager = ToolManager(self.config_manager)
        self.prompt_formatter = PromptFormatter(self.config_manager)
        
        # Initialize Android controller (optional)
        self.android_controller = AndroidDeviceController(default_timeout=Defaults.TIMEOUT)
        self.tool_executor = ToolExecutor(self.android_controller)
        
        # Initialize text generator
        self.text_generator = TextGenerator(
            self.config_manager, 
            self.model_manager, 
            self.tool_manager
        )
        
        logger.info("LLM Server components initialized")
    
    def initialize(self) -> None:
        """Initialize the LLM server"""
        logger.info("=== LLM SERVER INITIALIZATION START ===")
        
        try:
            # Initialize model
            self.model_manager.initialize()
            
            logger.info(f"Model type: {self.model_manager.model_type}")
            logger.info(f"Tool calling enabled: {self.config_manager.is_tool_calling_enabled()}")
            logger.info(f"Available tools: {len(self.tool_manager.tools_registry)}")
            logger.info("=== LLM SERVER INITIALIZATION COMPLETE ===")
            
        except Exception as e:
            logger.error(f"LLM server initialization failed: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None, top_p: Optional[float] = None,
                tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Generate text with optional tool-calling"""
        if not self.model_manager.is_initialized():
            raise RuntimeError(ErrorMessages.LLM_NOT_INITIALIZED)
        
        return self.text_generator.generate(prompt, max_tokens, temperature, top_p, tools)
    
    def format_chat_prompt(self, user_message: str, context: Optional[str] = None, 
                          history: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Format chat prompt with appropriate template and relevant tools"""
        # If tools not provided, get relevant tools based on user query
        if tools is None:
            tools = self.get_relevant_tools_for_query(user_message)
        
        return self.prompt_formatter.format_chat_prompt(
            user_message, 
            self.model_manager.model_type,
            context, 
            history, 
            tools
        )
    
    def get_relevant_tools_for_query(self, user_query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get relevant tools for user query using RAG"""
        return self.tool_manager.get_relevant_tools_for_query(user_query, top_k)
    
    async def execute_tool(self, tool_call) -> Dict[str, Any]:
        """Execute a tool call"""
        return await self.tool_executor.execute_tool(tool_call)
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools"""
        return self.tool_manager.get_available_tools()
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU cache to free VRAM"""
        self.model_manager.clear_gpu_cache()
    
    @property
    def is_initialized(self) -> bool:
        """Check if server is initialized"""
        return self.model_manager.is_initialized()
    
    @property
    def model_name(self) -> str:
        """Get model name"""
        return self.config_manager.model_name
    
    @property
    def tool_calling_enabled(self) -> bool:
        """Check if tool calling is enabled"""
        return self.config_manager.is_tool_calling_enabled()
    
    @property
    def mcp_enabled(self) -> bool:
        """Check if MCP is enabled"""
        return self.config_manager.is_mcp_enabled() 