from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yaml
import uvicorn
from typing import List, Optional, Dict, Any
import logging
import os
import torch

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG LLM Server", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

class QueryResponse(BaseModel):
    response: str
    generation_info: dict

class LLMServer:
    """Transformers-based LLM server with RTX 5070 Ti support"""
    
    def __init__(self, config_path: str = "llm/config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialize_llm()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_quantization_config(self, model_name: str) -> tuple:
        """Get quantization configuration"""
        quantization_config = self.config['llm'].get('quantization', {})
        
        # AWQ 모델 자동 감지
        if "awq" in model_name.lower():
            return None, "awq"  # AWQ는 별도 config 불필요
        
        # BitsAndBytes 양자화 설정
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
                return bnb_config, "bitsandbytes"
        
        return None, None
    
    def _initialize_llm(self):
        """Initialize LLM model with Transformers"""
        try:
            llm_config = self.config['llm']
            
            # config에서 모델명 가져오기
            model_name = llm_config['model_name']
            
            # 양자화 설정
            quantization_config, quant_method = self._get_quantization_config(model_name)
            
            logger.info(f"Loading model: {model_name}")
            logger.info(f"Quantization method: {quant_method}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # GPU 설정
            device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16,
                'low_cpu_mem_usage': True,
            }
            
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
            
            # 양자화 설정 추가
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            logger.info("✅ Model loaded successfully")
            
            # Generation pipeline 생성
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if device_count > 1 else (0 if device_count == 1 else -1),
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # 기본 generation 설정
            gen_config = llm_config['generation']
            self.default_gen_kwargs = {
                'max_new_tokens': gen_config['max_tokens'],
                'temperature': gen_config['temperature'],
                'top_p': gen_config['top_p'],
                'repetition_penalty': gen_config['repetition_penalty'],
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }
            
            # 로그 출력
            logger.info(f"Model loaded: {model_name}")
            logger.info(f"Quantization: {quant_method}")
            logger.info(f"Device count: {device_count}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None, top_p: Optional[float] = None) -> str:
        """Generate text"""
        try:
            # 동적 generation 파라미터 설정
            gen_kwargs = self.default_gen_kwargs.copy()
            if max_tokens:
                gen_kwargs['max_new_tokens'] = max_tokens
            if temperature:
                gen_kwargs['temperature'] = temperature
            if top_p:
                gen_kwargs['top_p'] = top_p
            
            # 텍스트 생성
            outputs = self.pipeline(
                prompt,
                **gen_kwargs,
                return_full_text=False  # 입력 프롬프트 제외하고 생성된 부분만 반환
            )
            
            # 결과 추출
            generated_text = outputs[0]['generated_text']
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    def format_chat_prompt(self, user_message: str, context: str = None, history: List[Dict[str, Any]] = None) -> str:
        """Format Qwen chat prompt"""
        if context:
            system_message = f"""You are a helpful AI assistant that answers questions based on the given context.

                            This is chat history so far:
                            ``` "The Chat History"
                            {history}
                            ```
                            Answer the user's question based on the above "The Chat History".

                            Please refer to the following context to answer:
                            ``` "The Context"
                            {context}
                            ```

                            Provide accurate and useful answers based on the above "The Context".

                            If "The Context" is not matched with your knowledge, you must answer based on the context.
                            """
        else:
            system_message = f"""You are a helpful AI assistant.

                            This is chat history so far:
                            ``` "The Chat History"
                            {history}
                            ```
                            Answer the user's question based on the above "The Chat History".
                            
                            Please answer the user's question based on the above history.
                            """
        
        # Qwen 채팅 템플릿 사용
        chat_prompt = f"""<|im_start|>system
                    {system_message}<|im_end|>
                    <|im_start|>user
                    {user_message}<|im_end|>
                    <|im_start|>assistant
                    """
        
        return chat_prompt

# Global LLM server instance
llm_server = None

@app.on_event("startup")
async def startup_event():
    """Initialize LLM server on app startup"""
    global llm_server
    try:
        llm_server = LLMServer()
        logger.info("LLM server initialization complete")
    except Exception as e:
        logger.error(f"LLM server initialization failed: {e}")
        raise

@app.post("/generate", response_model=QueryResponse)
async def generate_text(request: QueryRequest):
    """Text generation endpoint"""
    if llm_server is None:
        raise HTTPException(status_code=500, detail="LLM server not initialized.")
    
    try:
        # Format prompt
        prompt = llm_server.format_chat_prompt(request.query)
        
        # Generate text
        response = llm_server.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return QueryResponse(
            response=response,
            generation_info={
                "max_tokens": request.max_tokens or llm_server.default_gen_kwargs['max_new_tokens'],
                "temperature": request.temperature or llm_server.default_gen_kwargs['temperature'],
                "top_p": request.top_p or llm_server.default_gen_kwargs['top_p']
            }
        )
        
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": llm_server is not None}

if __name__ == "__main__":
    # Load configuration
    with open("llm/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    server_config = config['server']
    
    uvicorn.run(
        app,
        host=server_config['host'],
        port=server_config['port'],
        workers=server_config['workers']
    )
