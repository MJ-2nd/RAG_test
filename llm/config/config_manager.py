"""
Configuration management for LLM server
"""

import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ..constants import Defaults

logger = logging.getLogger(__name__)


class ConfigManager:
    """Handles configuration loading and management"""
    
    def __init__(self, config_path: str = "llm/config.yaml"):
        self.config_path = config_path
        self._config = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
                
            with open(config_file, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary"""
        return self._config
    
    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self._config.get('llm', {})
    
    @property
    def server_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        return self._config.get('server', {})
    
    @property
    def tool_config(self) -> Dict[str, Any]:
        """Get tool calling configuration"""
        return self.llm_config.get('tool_calling', {})
    
    @property
    def mcp_config(self) -> Dict[str, Any]:
        """Get MCP configuration"""
        return self._config.get('mcp', {})
    
    @property
    def model_name(self) -> str:
        """Get model name"""
        return self.llm_config.get('model_name', '')
    
    @property
    def generation_config(self) -> Dict[str, Any]:
        """Get generation configuration"""
        return self.llm_config.get('generation', {})
    
    @property
    def quantization_config(self) -> Dict[str, Any]:
        """Get quantization configuration"""
        return self.llm_config.get('quantization', {})
    
    def get_generation_kwargs(self, 
                            max_tokens: Optional[int] = None,
                            temperature: Optional[float] = None,
                            top_p: Optional[float] = None) -> Dict[str, Any]:
        """Get generation kwargs with optional overrides"""
        gen_config = self.generation_config
        
        return {
            'max_new_tokens': max_tokens or gen_config.get('max_tokens', Defaults.MAX_TOKENS),
            'temperature': temperature or gen_config.get('temperature', Defaults.TEMPERATURE),
            'top_p': top_p or gen_config.get('top_p', Defaults.TOP_P),
            'repetition_penalty': gen_config.get('repetition_penalty', Defaults.REPETITION_PENALTY),
            'do_sample': gen_config.get('do_sample', True)
        }
    
    @property
    def stop_sequences(self) -> list:
        """Get stop sequences"""
        return self.generation_config.get('stop_sequences', [])
    
    def is_tool_calling_enabled(self) -> bool:
        """Check if tool calling is enabled"""
        return self.tool_config.get('enabled', False)
    
    def is_mcp_enabled(self) -> bool:
        """Check if MCP is enabled"""
        return self.mcp_config.get('enabled', False)
    
    def get_tools_registry_path(self) -> str:
        """Get tools registry path"""
        return self.mcp_config.get('tools_registry', './tools/') 