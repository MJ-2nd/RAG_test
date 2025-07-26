"""
API models for LLM server
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel


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