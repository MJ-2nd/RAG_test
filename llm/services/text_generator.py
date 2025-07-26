"""
Text generation service
"""

import logging
from typing import Dict, Any, Optional, List

from ..config import ConfigManager
from ..models import ModelManager
from ..tools import ToolManager

logger = logging.getLogger(__name__)


class TextGenerator:
    """Handles text generation with tool calling support"""
    
    def __init__(self, config_manager: ConfigManager, model_manager: ModelManager, tool_manager: ToolManager):
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.tool_manager = tool_manager
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None, top_p: Optional[float] = None,
                tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Generate text with optional tool-calling"""
        logger.info("=== GENERATE METHOD START ===")
        
        try:
            # 생성 파라미터 설정
            gen_kwargs = self.config_manager.get_generation_kwargs(max_tokens, temperature, top_p)
            
            # 토크나이저의 토큰 설정
            gen_kwargs['pad_token_id'] = self.model_manager.tokenizer.eos_token_id
            gen_kwargs['eos_token_id'] = self.model_manager.tokenizer.eos_token_id
            
            logger.info(f"Generation kwargs: {gen_kwargs}")
            
            # 텍스트 생성
            outputs = self.model_manager.pipeline(
                prompt,
                **gen_kwargs,
                return_full_text=False
            )
            
            # 결과 추출
            generated_text = outputs[0]['generated_text'].strip()
            logger.info(f"Raw generated text: '{generated_text}'")
            
            # Stop sequences 처리
            generated_text = self._apply_stop_sequences(generated_text)
            
            # Tool-calling 파싱
            tool_calls = []
            if tools and self.config_manager.is_tool_calling_enabled():
                tool_calls = self.tool_manager.parse_tool_calls(generated_text)
                logger.info(f"Parsed {len(tool_calls)} tool calls")
            
            logger.info("=== GENERATE METHOD COMPLETE ===")
            return {
                'response': generated_text,
                'tool_calls': tool_calls
            }
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _apply_stop_sequences(self, text: str) -> str:
        """Apply stop sequences to generated text"""
        stop_sequences = self.config_manager.stop_sequences
        if not stop_sequences:
            return text
        
        for stop_seq in stop_sequences:
            if stop_seq in text:
                text = text.split(stop_seq)[0].strip()
                logger.info(f"Stopped at sequence: {stop_seq}")
                break
        
        return text 