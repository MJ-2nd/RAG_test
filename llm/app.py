from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import yaml
import uvicorn
from typing import List, Optional, Dict, Any
import logging
import os
import torch

# MCP imports
from mcp import MCPHandler, MCPPromptBuilder, function_registry, ADB_FUNCTION_DEFINITIONS

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG LLM Server", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용 - 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 (웹 인터페이스)
app.mount("/web", StaticFiles(directory="web_interface"), name="web")

class QueryRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

class QueryResponse(BaseModel):
    response: str
    function_calls: Optional[List[Dict[str, Any]]] = None
    function_results: Optional[List[Dict[str, Any]]] = None
    generation_info: dict

class LLMServer:
    """Transformers-based LLM server with MCP support"""
    
    def __init__(self, config_path: str = "llm/config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_type = None
        
        # MCP components
        self.mcp_handler = MCPHandler(function_registry)
        self.mcp_prompt_builder = MCPPromptBuilder(function_registry)
        
        self._initialize_llm()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _detect_model_type(self, model_name: str) -> str:
        """Detect model type from model name"""
        model_name_lower = model_name.lower()
        
        if "deepseek" in model_name_lower:
            return "deepseek"
        elif "qwen" in model_name_lower:
            return "qwen"
        elif "llama" in model_name_lower:
            return "llama"
        elif "mistral" in model_name_lower:
            return "mistral"
        elif "smollm" in model_name_lower:
            return "smollm"
        else:
            return "generic"
    
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
            self.model_type = self._detect_model_type(model_name)
            
            # 양자화 설정
            quantization_config, quant_method = self._get_quantization_config(model_name)
            
            logger.info(f"Loading model: {model_name}")
            logger.info(f"Model type: {self.model_type}")
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
                temperature: Optional[float] = None, top_p: Optional[float] = None) -> Dict[str, Any]:
        """Generate text with MCP support"""
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
            generated_text = outputs[0]['generated_text'].strip()
            
            # MCP 처리
            mcp_result = self.mcp_handler.process_llm_response(generated_text)
            
            return {
                'response': mcp_result['response_text'],
                'function_calls': mcp_result['function_calls'],
                'function_results': mcp_result['function_results'],
                'has_functions': mcp_result['has_functions']
            }
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free VRAM"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")
    
    def format_chat_prompt(self, user_message: str, context: str = None, history: str = None) -> str:
        """Format chat prompt with MCP support"""
        return self.mcp_prompt_builder.build_prompt(
            user_message=user_message,
            model_type=self.model_type,
            include_functions=True,
            context=context,
            history=history
        )

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
    """Text generation endpoint with MCP support"""
    if llm_server is None:
        raise HTTPException(status_code=500, detail="LLM server not initialized.")
    
    try:
        # Format prompt with MCP
        prompt = llm_server.format_chat_prompt(request.query)
        
        # Generate text with MCP processing
        result = llm_server.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return QueryResponse(
            response=result['response'],
            function_calls=[fc.__dict__ for fc in result['function_calls']] if result['function_calls'] else None,
            function_results=result['function_results'],
            generation_info={
                "max_tokens": request.max_tokens or llm_server.default_gen_kwargs['max_new_tokens'],
                "temperature": request.temperature or llm_server.default_gen_kwargs['temperature'],
                "top_p": request.top_p or llm_server.default_gen_kwargs['top_p'],
                "has_functions": result['has_functions']
            }
        )
        
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if llm_server is None:
        raise HTTPException(status_code=503, detail="LLM server not initialized.")
    
    return {
        "status": "healthy",
        "model_name": llm_server.config['llm']['model_name'],
        "model_type": llm_server.model_type,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "mcp_enabled": True,
        "available_functions": llm_server.mcp_handler.get_function_statistics()
    }


@app.get("/functions")
async def list_functions():
    """List available MCP functions"""
    if llm_server is None:
        raise HTTPException(status_code=503, detail="LLM server not initialized.")
    
    return {
        "functions": llm_server.mcp_handler.get_function_statistics(),
        "function_schemas": function_registry.get_function_schemas(),
        "summary": llm_server.mcp_handler.get_available_functions_summary()
    }


# 웹 인터페이스 관련 엔드포인트들
@app.get("/")
async def root():
    """Redirect to web interface"""
    return FileResponse("web_interface/index.html")


@app.post("/execute_function")
async def execute_function(request: Dict[str, Any]):
    """Execute a specific function"""
    try:
        function_name = request.get("function_name")
        arguments = request.get("arguments", {})
        
        if not function_name:
            raise HTTPException(status_code=400, detail="function_name is required")
        
        result = function_registry.execute_function(function_name, arguments)
        return result
        
    except Exception as e:
        logger.error(f"Error executing function {function_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/web_interface")
async def get_web_interface():
    """Serve the web interface"""
    return FileResponse("web_interface/index.html")


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
