"""
Model management for LLM server
"""

import logging
import torch
from typing import Tuple, Optional, Dict, Any
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)

from ..constants import ModelType, QuantizationMethod
from ..config import ConfigManager

logger = logging.getLogger(__name__)


class ModelManager:
    """Handles model loading and management"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_type = None
        
    def detect_model_type(self, model_name: str) -> str:
        """Detect model type from model name for optimized handling"""
        model_name_lower = model_name.lower()
        
        type_mapping = {
            ModelType.KIMI: ["kimi", "moonshotai"],
            ModelType.QWEN: ["qwen"],
            ModelType.DEEPSEEK: ["deepseek"],
            ModelType.SMOLLM: ["smollm"],
            ModelType.LLAMA: ["llama"],
            ModelType.MISTRAL: ["mistral", "mixtral"]
        }
        
        for model_type, keywords in type_mapping.items():
            if any(keyword in model_name_lower for keyword in keywords):
                return model_type
        
        return ModelType.GENERIC
    
    def _get_quantization_config(self, model_name: str) -> Tuple[Optional[BitsAndBytesConfig], Optional[str]]:
        """Get quantization configuration based on model type"""
        quantization_config = self.config_manager.quantization_config
        model_name_lower = model_name.lower()
        
        # AWQ 모델 자동 감지
        if "awq" in model_name_lower:
            return None, QuantizationMethod.AWQ
        
        # MoE 모델들 (이미 최적화됨)
        if any(keyword in model_name_lower for keyword in ["kimi", "moonshotai", "mixtral"]):
            return None, QuantizationMethod.MOE_OPTIMIZED
        
        # 기타 모델들의 양자화 설정
        if quantization_config.get('enabled', False):
            if quantization_config.get('method') == 'bitsandbytes':
                bits = quantization_config.get('bits', 4)
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=(bits == 4),
                    load_in_8bit=(bits == 8),
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                return bnb_config, QuantizationMethod.BITSANDBYTES
        
        return None, None
    
    def _setup_gpu_configuration(self) -> Dict[str, Any]:
        """Setup GPU configuration for model loading"""
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        logger.info(f"Available GPUs: {device_count}")
        
        model_kwargs = {
            'trust_remote_code': True,
            'torch_dtype': torch.float16,
            'low_cpu_mem_usage': True,
        }
        
        # Flash Attention 지원 체크 (지원하는 모델만)
        if device_count > 0 and self.model_type in [ModelType.KIMI, ModelType.QWEN, ModelType.DEEPSEEK, ModelType.LLAMA]:
            model_kwargs['attn_implementation'] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled")
        
        # 멀티 GPU 설정
        if device_count > 1:
            model_kwargs['device_map'] = "auto"
            logger.info(f"Using {device_count} GPUs with device_map='auto'")
        elif device_count == 1:
            model_kwargs['device_map'] = {"": 0}
            logger.info("Using single GPU")
        else:
            model_kwargs['device_map'] = "cpu"
            logger.info("Using CPU")
        
        return model_kwargs, device_count
    
    def _load_tokenizer(self, model_name: str) -> None:
        """Load and configure tokenizer"""
        logger.info("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        logger.info("Tokenizer loaded successfully")
    
    def _load_model(self, model_name: str, model_kwargs: Dict[str, Any], 
                   quantization_config: Optional[BitsAndBytesConfig]) -> None:
        """Load the actual model"""
        if quantization_config:
            model_kwargs['quantization_config'] = quantization_config
            logger.info("Quantization config added")
        
        logger.info("Loading model with AutoModelForCausalLM")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        logger.info("✅ Model loaded successfully")
    
    def _create_pipeline(self, device_count: int) -> None:
        """Create generation pipeline"""
        logger.info("Creating generation pipeline")
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto" if device_count > 1 else (0 if device_count == 1 else -1),
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        logger.info("Pipeline created successfully")
    
    def initialize(self) -> None:
        """Initialize the model and tokenizer"""
        logger.info("=== MODEL INITIALIZATION START ===")
        
        try:
            model_name = self.config_manager.model_name
            self.model_type = self.detect_model_type(model_name)
            
            logger.info(f"Loading model: {model_name}")
            logger.info(f"Model type detected: {self.model_type}")
            
            # 양자화 설정
            quantization_config, quant_method = self._get_quantization_config(model_name)
            logger.info(f"Optimization method: {quant_method}")
            
            # 토크나이저 로드
            self._load_tokenizer(model_name)
            
            # GPU 설정
            model_kwargs, device_count = self._setup_gpu_configuration()
            
            # 모델 로드
            self._load_model(model_name, model_kwargs, quantization_config)
            
            # Pipeline 생성
            self._create_pipeline(device_count)
            
            logger.info(f"Model loaded: {model_name}")
            logger.info("=== MODEL INITIALIZATION COMPLETE ===")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU cache to free VRAM"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")
    
    def is_initialized(self) -> bool:
        """Check if model is initialized"""
        return self.model is not None and self.tokenizer is not None and self.pipeline is not None 