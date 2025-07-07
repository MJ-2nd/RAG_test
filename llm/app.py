from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yaml
import uvicorn
from typing import List, Optional
import logging
import os

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
    """VLLM-based LLM server"""
    
    def __init__(self, config_path: str = "llm/config.yaml"):
        self.config = self._load_config(config_path)
        self.llm = None
        self.sampling_params = None
        self._initialize_llm()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _initialize_llm(self):
        """Initialize LLM model"""
        try:
            llm_config = self.config['llm']
            
            # Set model path
            model_path = llm_config['model_path']
            if not os.path.exists(model_path):
                logger.warning(f"Local model path {model_path} not found. Downloading from HuggingFace.")
                model_path = llm_config['model_name']
            
            # VLLM configuration
            vllm_config = llm_config['vllm']
            
            # Quantization configuration
            quantization_config = llm_config.get('quantization', {})
            quantization = "awq" if quantization_config.get('enabled', False) else None
            
            # Load VLLM model
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=vllm_config['tensor_parallel_size'],
                max_model_len=vllm_config['max_model_len'],
                gpu_memory_utilization=vllm_config['gpu_memory_utilization'],
                enforce_eager=vllm_config['enforce_eager'],
                quantization=quantization,
                trust_remote_code=True  # Required for Qwen models
            )
            
            # Set sampling parameters
            gen_config = llm_config['generation']
            self.sampling_params = SamplingParams(
                max_tokens=gen_config['max_tokens'],
                temperature=gen_config['temperature'],
                top_p=gen_config['top_p'],
                repetition_penalty=gen_config['repetition_penalty']
            )
            
            logger.info("LLM model loaded successfully")
            logger.info(f"Model: {model_path}")
            logger.info(f"Quantization: {quantization}")
            logger.info(f"GPU memory utilization: {vllm_config['gpu_memory_utilization']}")
            logger.info(f"Tensor parallel size: {vllm_config['tensor_parallel_size']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None, top_p: Optional[float] = None) -> str:
        """Generate text"""
        try:
            # Set dynamic sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens or self.sampling_params.max_tokens,
                temperature=temperature or self.sampling_params.temperature,
                top_p=top_p or self.sampling_params.top_p,
                repetition_penalty=self.sampling_params.repetition_penalty
            )
            
            # Execute generation
            outputs = self.llm.generate([prompt], sampling_params)
            
            # Extract result
            generated_text = outputs[0].outputs[0].text
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    def format_chat_prompt(self, user_message: str, context: str = None) -> str:
        """Format Qwen2.5 chat prompt"""
        if context:
            system_message = f"""You are a helpful AI assistant that answers questions based on the given context.

Please refer to the following context to answer:
{context}

Provide accurate and useful answers based on the above context."""
        else:
            system_message = "You are a helpful AI assistant."
        
        # Use Qwen2.5 chat template
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
                "max_tokens": request.max_tokens or llm_server.sampling_params.max_tokens,
                "temperature": request.temperature or llm_server.sampling_params.temperature,
                "top_p": request.top_p or llm_server.sampling_params.top_p
            }
        )
        
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check"""
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
