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
from typing import List, Optional, Dict, Any, Union
import logging
import os
import torch
import json
import asyncio
from datetime import datetime

from adb_implementation_example import AndroidDeviceController

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG LLM Server with Tool-calling", version="2.0.0")

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class QueryRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto"

class QueryResponse(BaseModel):
    response: str
    tool_calls: Optional[List[ToolCall]] = None
    generation_info: dict

class LLMServer:
    """Universal LLM server with tool-calling and MCP support for various models"""
    
    def __init__(self, config_path: str = "llm/config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.tools_registry = {}
        self.model_type = None  # 모델 타입 저장
        
        # Android 디바이스 컨트롤러 초기화 (선택사항)
        self.android_controller = AndroidDeviceController(default_timeout=30)
        logger.info("Android device controller initialized")
        
        self._initialize_llm()
        self._load_tools()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _detect_model_type(self, model_name: str) -> str:
        """Detect model type from model name for optimized handling"""
        model_name_lower = model_name.lower()
        
        if "kimi" in model_name_lower or "moonshotai" in model_name_lower:
            return "kimi"
        elif "qwen" in model_name_lower:
            return "qwen"
        elif "deepseek" in model_name_lower:
            return "deepseek"
        elif "smollm" in model_name_lower:
            return "smollm"
        elif "llama" in model_name_lower:
            return "llama"
        elif "mistral" in model_name_lower or "mixtral" in model_name_lower:
            return "mistral"
        else:
            return "generic"
    
    def _get_quantization_config(self, model_name: str) -> tuple:
        """Get quantization configuration based on model type"""
        quantization_config = self.config['llm'].get('quantization', {})
        model_name_lower = model_name.lower()
        
        # AWQ 모델 자동 감지
        if "awq" in model_name_lower:
            return None, "awq"
        
        # MoE 모델들 (이미 최적화됨)
        if any(keyword in model_name_lower for keyword in ["kimi", "moonshotai", "mixtral"]):
            return None, "moe_optimized"
        
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
                return bnb_config, "bitsandbytes"
        
        return None, None
    
    def _initialize_llm(self):
        """Initialize LLM model with tool-calling support for various model types"""
        logger.info("=== LLM INITIALIZATION START ===")
        try:
            llm_config = self.config['llm']
            
            # config에서 모델명 가져오기
            model_name = llm_config['model_name']
            self.model_type = self._detect_model_type(model_name)
            
            # 양자화 설정
            quantization_config, quant_method = self._get_quantization_config(model_name)
            
            logger.info(f"Loading model: {model_name}")
            logger.info(f"Model type detected: {self.model_type}")
            logger.info(f"Optimization method: {quant_method}")
            
            logger.info("Loading tokenizer")
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            logger.info("Tokenizer loaded successfully")
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            logger.info("Setting up GPU configuration")
            # GPU 설정
            device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            logger.info(f"Available GPUs: {device_count}")
            
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16,
                'low_cpu_mem_usage': True,
            }
            
            # Flash Attention 지원 체크 (지원하는 모델만)
            if device_count > 0 and self.model_type in ["kimi", "qwen", "deepseek", "llama"]:
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
            
            # 양자화 설정 추가
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
                logger.info("Quantization config added")
            
            logger.info("Loading model with AutoModelForCausalLM")
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            logger.info("✅ Model loaded successfully")
            
            logger.info("Creating generation pipeline")
            # Generation pipeline 생성
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if device_count > 1 else (0 if device_count == 1 else -1),
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            logger.info("Pipeline created successfully")
            
            logger.info("Setting up generation parameters")
            # 기본 generation 설정 (Tool-calling 최적화)
            gen_config = llm_config['generation']
            self.default_gen_kwargs = {
                'max_new_tokens': gen_config['max_tokens'],
                'temperature': gen_config['temperature'],
                'top_p': gen_config['top_p'],
                'repetition_penalty': gen_config['repetition_penalty'],
                'do_sample': gen_config['do_sample'],
                'pad_token_id': self.tokenizer.eos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }
            logger.info(f"Generation kwargs: {self.default_gen_kwargs}")
            
            # Stop sequences는 별도로 저장 (pipeline에서 직접 사용하지 않음)
            self.stop_sequences = gen_config.get('stop_sequences', [])
            logger.info(f"Stop sequences: {self.stop_sequences}")
            
            # Tool-calling 설정
            self.tool_config = llm_config.get('tool_calling', {})
            
            logger.info(f"Model loaded: {model_name}")
            logger.info(f"Tool-calling enabled: {self.tool_config.get('enabled', False)}")
            logger.info(f"MCP support: {self.config.get('mcp', {}).get('enabled', False)}")
            logger.info("=== LLM INITIALIZATION COMPLETE ===")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _load_tools(self):
        """Load MCP tools from tools registry"""
        try:
            tools_dir = self.config.get('mcp', {}).get('tools_registry', './tools/')
            if os.path.exists(tools_dir):
                for tool_file in os.listdir(tools_dir):
                    if tool_file.endswith('.json'):
                        with open(os.path.join(tools_dir, tool_file), 'r') as f:
                            tool_def = json.load(f)
                            self.tools_registry[tool_def['name']] = tool_def
                logger.info(f"Loaded {len(self.tools_registry)} tools from registry")
            else:
                logger.info("No tools registry found, using default tools")
                self._create_default_tools()
        except Exception as e:
            logger.warning(f"Failed to load tools: {e}")
            self._create_default_tools()
    
    def _create_default_tools(self):
        """Create default tools for demonstration"""
        self.tools_registry = {
            "search_documents": {
                "name": "search_documents",
                "description": "Search through documents using RAG",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "description": "Number of results", "default": 5}
                    },
                    "required": ["query"]
                }
            },
            "calculate": {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                    },
                    "required": ["expression"]
                }
            },
            "get_current_time": {
                "name": "get_current_time",
                "description": "Get current date and time",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            "control_android_device": {
                "name": "control_android_device",
                "description": "Control Android device using ADB shell commands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command_type": {"type": "string", "description": "Type of ADB command"},
                        "shell_command": {"type": "string", "description": "Custom shell command"}
                    },
                    "required": ["command_type"]
                }
            }
        }
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None, top_p: Optional[float] = None,
                tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Generate text with optional tool-calling"""
        logger.info("=== GENERATE METHOD START ===")
        try:
            logger.info("Setting up generation parameters")
            # 동적 generation 파라미터 설정
            gen_kwargs = self.default_gen_kwargs.copy()
            if max_tokens:
                gen_kwargs['max_new_tokens'] = max_tokens
            if temperature:
                gen_kwargs['temperature'] = temperature
            if top_p:
                gen_kwargs['top_p'] = top_p
            
            logger.info(f"Generation kwargs: {gen_kwargs}")
            logger.info(f"Pipeline available: {self.pipeline is not None}")
            
            logger.info("Calling pipeline for text generation")
            # 텍스트 생성
            outputs = self.pipeline(
                prompt,
                **gen_kwargs,
                return_full_text=False  # 입력 프롬프트 제외하고 생성된 부분만 반환
            )
            logger.info("Pipeline call completed")
            
            logger.info("Extracting generated text")
            # 결과 추출
            generated_text = outputs[0]['generated_text'].strip()
            logger.info(f"Raw generated text: '{generated_text}'")
            
            logger.info("Processing stop sequences")
            # Stop sequences 처리 (생성 후 수동으로 제거)
            if hasattr(self, 'stop_sequences') and self.stop_sequences:
                for stop_seq in self.stop_sequences:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0].strip()
                        logger.info(f"Stopped at sequence: {stop_seq}")
                        break
            
            logger.info("Parsing tool calls")
            # Tool-calling 파싱 (모델 타입에 따라 다를 수 있음)
            tool_calls = []
            if tools and self.tool_config.get('enabled', False):
                tool_calls = self._parse_tool_calls(generated_text)
                logger.info(f"Parsed {len(tool_calls)} tool calls")
            else:
                logger.info("Tool calling disabled or no tools provided")
            
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
    
    def _parse_tool_calls(self, text: str) -> List[ToolCall]:
        """Parse tool calls from generated text (model-agnostic)"""
        tool_calls = []
        try:
            import re
            
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
    
    async def execute_tool(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Execute a tool call"""
        try:
            tool_name = tool_call.name
            
            if tool_name == "search_documents":
                # RAG 검색 실행 (실제 구현 필요)
                logger.info(f"RAG search results for: {tool_call.arguments.get('query')}")
                return {"result": f"RAG search results for: {tool_call.arguments.get('query')}"}
            
            elif tool_name == "calculate":
                # 계산 실행
                logger.info(f"Calculating: {tool_call.arguments.get('expression')}")
                expression = tool_call.arguments.get('expression', '')
                try:
                    # 안전한 계산을 위해 eval 대신 더 안전한 방법 사용
                    import ast
                    import operator
                    
                    # 간단한 수학 연산만 허용
                    allowed_ops = {
                        ast.Add: operator.add,
                        ast.Sub: operator.sub,
                        ast.Mult: operator.mul,
                        ast.Div: operator.truediv,
                        ast.Pow: operator.pow,
                        ast.USub: operator.neg,
                    }
                    
                    def eval_expr(node):
                        if isinstance(node, ast.Num):
                            return node.n
                        elif isinstance(node, ast.BinOp):
                            return allowed_ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                        elif isinstance(node, ast.UnaryOp):
                            return allowed_ops[type(node.op)](eval_expr(node.operand))
                        else:
                            raise TypeError(node)
                    
                    result = eval_expr(ast.parse(expression, mode='eval').body)
                    return {"result": result}
                    
                except Exception as e:
                    return {"error": f"Calculation error: {str(e)}"}
            
            elif tool_name == "get_current_time":
                # 현재 시간 반환
                logger.info(f"Getting current time")
                return {"result": datetime.now().isoformat()}
            
            elif tool_name == "control_android_device":                
                logger.info(f"Control Android device: {tool_call.arguments}")
                try:
                    # AndroidDeviceController 클래스 사용
                    result = self.android_controller.control_android_device(tool_call.arguments)
                    return result
                    
                except Exception as e:
                    return {"error": f"Android device control error: {str(e)}"}
            
            else:
                return {"error": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"error": str(e)}
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free VRAM"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")
    
    def _get_chat_template(self, model_type: str) -> str:
        """Get appropriate chat template based on model type"""
        templates = {
            "kimi": "chatml",      # ChatML format for Kimi
            "qwen": "chatml",      # ChatML format for Qwen
            "deepseek": "chatml",  # ChatML format for DeepSeek
            "llama": "llama",      # Llama format
            "mistral": "mistral",  # Mistral format
            "smollm": "chatml",    # ChatML format
            "generic": "chatml"    # Default to ChatML
        }
        return templates.get(model_type, "chatml")
    
    def format_chat_prompt(self, user_message: str, context: str = None, history: str = None,
                          tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Format chat prompt with appropriate template based on model type"""
        logger.info("=== FORMATTING CHAT PROMPT ===")
        logger.info(f"User message: {user_message}")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Tools provided: {len(tools) if tools else 0}")
        
        # Tool definitions을 시스템 메시지에 포함
        tools_section = ""
        if tools and self.tool_config.get('enabled', False):
            logger.info("Adding tools section to prompt")
            tools_json = json.dumps(tools, indent=2, ensure_ascii=False)
            
            # 모델 타입에 따른 tool-calling 형식 안내
            if self.model_type in ["kimi", "qwen", "deepseek"]:
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
            
            tools_section = f"""

Available tools:
{tools_json}
{tool_format}

You can use multiple tools by including multiple tool call blocks.
"""
        else:
            logger.info("No tools section added")
        
        # 기본 시스템 메시지
        history_section = ""
        if history and history.strip():
            history_section = f"""
Previous conversation:
{history}

"""
            logger.info("Added history section")
        
        if context:
            system_message = f"""You are a helpful AI assistant with advanced reasoning and tool-calling capabilities.

{history_section}Answer the user's question based on the above conversation history.

Please refer to the following context to answer:
``` "The Context"
{context}
```

Provide accurate and useful answers based on the above context.
If the context doesn't contain relevant information, you can use available tools or provide answers based on your knowledge.
{tools_section}"""
            logger.info("Added context to system message")
        else:
            system_message = f"""You are a helpful AI assistant with advanced reasoning and tool-calling capabilities.

{history_section}Answer the user's question based on the above conversation history.
{tools_section}"""
            logger.info("Using basic system message")
        
        # 모델 타입에 따른 채팅 템플릿 적용
        template_type = self._get_chat_template(self.model_type)
        logger.info(f"Using template type: {template_type}")
        
        if template_type == "chatml":
            # ChatML 형식 (Kimi, Qwen, DeepSeek, SmolLM 등)
            chat_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
        elif template_type == "llama":
            # Llama 형식
            chat_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        elif template_type == "mistral":
            # Mistral 형식
            chat_prompt = f"""<s>[INST] {system_message}

{user_message} [/INST]"""
        else:
            # 기본 형식
            chat_prompt = f"""System: {system_message}

User: {user_message}
"""
        
        logger.info(f"Final prompt length: {len(chat_prompt)} characters")
        logger.info("=== CHAT PROMPT FORMATTING COMPLETE ===")
        return chat_prompt

# Global LLM server instance
llm_server = None

@app.on_event("startup")
async def startup_event():
    """Initialize LLM server on app startup"""
    global llm_server
    logger.info("=== SERVER STARTUP BEGIN ===")
    try:
        logger.info("Creating LLMServer instance")
        llm_server = LLMServer()
        logger.info("LLM server initialization complete")
        logger.info(f"Model type: {llm_server.model_type}")
        logger.info(f"Tool calling enabled: {llm_server.tool_config.get('enabled', False)}")
        logger.info(f"Available tools: {len(llm_server.tools_registry)}")
        logger.info("=== SERVER STARTUP COMPLETE ===")
    except Exception as e:
        logger.error(f"LLM server initialization failed: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

@app.post("/generate", response_model=QueryResponse)
async def generate_text(request: QueryRequest):
    """Text generation endpoint with tool-calling support"""
    logger.info(f"=== REQUEST RECEIVED ===")
    logger.info(f"Query: {request.query}")
    logger.info(f"Max tokens: {request.max_tokens}")
    logger.info(f"Temperature: {request.temperature}")
    logger.info(f"Top_p: {request.top_p}")
    
    if llm_server is None:
        logger.error("LLM server not initialized")
        raise HTTPException(status_code=500, detail="LLM server not initialized.")
    
    try:
        logger.info("=== PREPARING TOOLS ===")
        # Available tools 준비
        available_tools = None
        if request.tools:
            available_tools = request.tools
            logger.info(f"Using provided tools: {len(request.tools)} tools")
        elif llm_server.tool_config.get('enabled', False):
            available_tools = list(llm_server.tools_registry.values())
            logger.info(f"Using registry tools: {len(available_tools)} tools")
        else:
            logger.warning("No tools available")
        
        logger.info("=== FORMATTING PROMPT ===")
        # Format prompt
        prompt = llm_server.format_chat_prompt(
            request.query, 
            tools=available_tools
        )
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Prompt preview: {prompt[:200]}...")
        
        logger.info("=== GENERATING TEXT ===")
        # Generate text
        result = llm_server.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            tools=available_tools
        )
        logger.info(f"Generated text length: {len(result['response'])} characters")
        logger.info(f"Generated text: {result['response']}")
        logger.info(f"Tool calls found: {len(result['tool_calls'])}")
        
        # Execute tools if any
        executed_tools = []
        if result['tool_calls']:
            logger.info("=== EXECUTING TOOLS ===")
            for i, tool_call in enumerate(result['tool_calls']):
                logger.info(f"Executing tool {i+1}/{len(result['tool_calls'])}: {tool_call.name}")
                tool_result = await llm_server.execute_tool(tool_call)
                logger.info(f"Tool {tool_call.name} result: {tool_result}")
                executed_tools.append({
                    'tool_call': tool_call.dict(),
                    'result': tool_result
                })
        else:
            logger.info("No tools to execute")
        
        logger.info("=== PREPARING RESPONSE ===")
        response = QueryResponse(
            response=result['response'],
            tool_calls=result['tool_calls'],
            generation_info={
                "max_tokens": request.max_tokens or llm_server.default_gen_kwargs['max_new_tokens'],
                "temperature": request.temperature or llm_server.default_gen_kwargs['temperature'],
                "top_p": request.top_p or llm_server.default_gen_kwargs['top_p'],
                "tools_executed": len(executed_tools),
                "tool_results": executed_tools
            }
        )
        logger.info("=== RESPONSE READY ===")
        logger.info(f"Final response length: {len(response.response)} characters")
        return response
        
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools")
async def list_tools():
    """List available tools"""
    if llm_server is None:
        raise HTTPException(status_code=500, detail="LLM server not initialized.")
    
    return {"tools": list(llm_server.tools_registry.values())}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": llm_server is not None,
        "model_name": llm_server.config['llm']['model_name'] if llm_server else None,
        "tool_calling_enabled": llm_server.tool_config.get('enabled', False) if llm_server else False,
        "mcp_enabled": llm_server.config.get('mcp', {}).get('enabled', False) if llm_server else False
    }

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
