"""
Refactored LLM Server Application with Clean Architecture

이 파일은 기존 app.py를 클린 코드 원칙에 따라 리팩토링한 결과입니다.
- 단일 책임 원칙 (SRP)
- 의존성 주입
- 모듈화
- 가독성 향상
- 확장 가능성
"""

import logging
import uvicorn
import yaml
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException

from .constants import ErrorMessages
from .core import LLMServer
from .api import QueryRequest, QueryResponse, ToolCall

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(title="RAG LLM Server with Tool-calling", version="2.0.0")

# Global LLM server instance
llm_server: LLMServer = None


@app.on_event("startup")
async def startup_event():
    """Initialize LLM server on app startup"""
    global llm_server
    logger.info("=== SERVER STARTUP BEGIN ===")
    
    try:
        logger.info("Creating LLMServer instance")
        llm_server = LLMServer()
        llm_server.initialize()
        
        logger.info("LLM server initialization complete")
        logger.info(f"Model name: {llm_server.model_name}")
        logger.info(f"Tool calling enabled: {llm_server.tool_calling_enabled}")
        logger.info(f"MCP enabled: {llm_server.mcp_enabled}")
        logger.info(f"Available tools: {len(llm_server.get_available_tools())}")
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
    logger.info("=== REQUEST RECEIVED ===")
    logger.info(f"Query: {request.query}")
    logger.info(f"Max tokens: {request.max_tokens}")
    logger.info(f"Temperature: {request.temperature}")
    logger.info(f"Top_p: {request.top_p}")
    
    
    try:
        # Prepare tools
        available_tools = _prepare_tools(request)
        
        # Format prompt
        prompt = llm_server.format_chat_prompt(
            request.query, 
            tools=available_tools
        )
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        # Generate text
        result = llm_server.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            tools=available_tools
        )
        logger.info(f"Generated text length: {len(result['response'])} characters")
        logger.info(f"Tool calls found: {len(result['tool_calls'])}")
        
        # Execute tools if any
        executed_tools = await _execute_tools(result['tool_calls'])
        
        # Prepare response
        response = QueryResponse(
            response=result['response'],
            tool_calls=result['tool_calls'],
            generation_info=_create_generation_info(request, executed_tools)
        )
        
        logger.info("=== RESPONSE READY ===")
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
        raise HTTPException(status_code=500, detail=ErrorMessages.LLM_NOT_INITIALIZED)
    
    return {"tools": llm_server.get_available_tools()}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": llm_server is not None and llm_server.is_initialized,
        "model_name": llm_server.model_name if llm_server else None,
        "tool_calling_enabled": llm_server.tool_calling_enabled if llm_server else False,
        "mcp_enabled": llm_server.mcp_enabled if llm_server else False
    }


def _prepare_tools(request: QueryRequest) -> List[Dict[str, Any]]:
    """Prepare relevant tools for the request using RAG"""
    logger.info("=== PREPARING TOOLS ===")
    
    if request.tools:
        logger.info(f"Using provided tools: {len(request.tools)} tools")
        return request.tools
    elif llm_server.tool_calling_enabled:
        # Use RAG to get relevant tools based on user query
        relevant_tools = llm_server.get_relevant_tools_for_query(request.query)
        logger.info(f"Using RAG-selected tools: {len(relevant_tools)} tools")
        return relevant_tools
    else:
        logger.info("No tools available")
        return []


async def _execute_tools(tool_calls: List[ToolCall]) -> List[Dict[str, Any]]:
    """Execute tool calls and return results"""
    executed_tools = []
    
    if not tool_calls:
        logger.info("No tools to execute")
        return executed_tools
    
    logger.info("=== EXECUTING TOOLS ===")
    for i, tool_call in enumerate(tool_calls):
        logger.info(f"Executing tool {i+1}/{len(tool_calls)}: {tool_call.name}")
        tool_result = await llm_server.execute_tool(tool_call)
        logger.info(f"Tool {tool_call.name} result: {tool_result}")
        
        executed_tools.append({
            'tool_call': tool_call.dict(),
            'result': tool_result
        })
    
    return executed_tools


def _create_generation_info(request: QueryRequest, executed_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create generation info for response"""
    return {
        "max_tokens": request.max_tokens or llm_server.config_manager.generation_config.get('max_tokens'),
        "temperature": request.temperature or llm_server.config_manager.generation_config.get('temperature'),
        "top_p": request.top_p or llm_server.config_manager.generation_config.get('top_p'),
        "tools_executed": len(executed_tools),
        "tool_results": executed_tools
    }


def main():
    """Main entry point"""
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


if __name__ == "__main__":
    main() 