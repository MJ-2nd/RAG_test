"""
RAG-based Tool Retriever

Uses RAG system to retrieve relevant tools based on user query,
reducing prompt size and improving efficiency.
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ...rag.retriever import FAISSRetriever

logger = logging.getLogger(__name__)


class RAGToolRetriever:
    """RAG-based tool retriever for dynamic tool selection"""
    
    def __init__(self, config_path: str = "llm/config.yaml"):
        self.config_path = config_path
        self.retriever = None
        self.is_initialized = False
        self.function_registry: Dict[str, Dict[str, Any]] = {}
        self._load_function_registry()
        self._initialize_rag()
    
    def _load_function_registry(self) -> None:
        """Load all functions from JSON into registry for stable prompt generation"""
        try:
            json_path = "documents/adb_mcp_functions.json"
            if not os.path.exists(json_path):
                logger.warning(f"Tool documentation not found: {json_path}")
                return
            
            with open(json_path, 'r', encoding='utf-8') as f:
                tool_data = json.load(f)
            
            # Store all functions in registry
            for func in tool_data.get('functions', []):
                self.function_registry[func['name']] = func
            
            logger.info(f"Loaded {len(self.function_registry)} functions into registry")
            
        except Exception as e:
            logger.error(f"Failed to load function registry: {e}")
    
    def _initialize_rag(self) -> None:
        """Initialize RAG system for tool retrieval"""
        try:
            # Initialize retriever
            self.retriever = FAISSRetriever(self.config_path)
            
            # Check if index exists
            index_path = "models/tool_functions_index"
            if os.path.exists(f"{index_path}.faiss") and os.path.exists(f"{index_path}.pkl"):
                logger.info("Loading existing tool functions index")
                self.retriever.load_index(index_path)
                self.is_initialized = True
            else:
                logger.info("Building tool functions index")
                self._build_tool_index()
                
        except Exception as e:
            logger.warning(f"Failed to initialize RAG tool retriever: {e}")
            logger.warning("Falling back to static tool list")
    
    def _build_tool_index(self) -> None:
        """Build index from JSON tool documentation"""
        try:
            # Load JSON tool documentation
            json_path = "documents/adb_mcp_functions.json"
            if not os.path.exists(json_path):
                logger.warning(f"Tool documentation not found: {json_path}")
                return
            
            # Load JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                tool_data = json.load(f)
            
            # Create function-specific chunks from JSON
            chunks, metadata = self._create_chunks_from_json(tool_data)
            
            # Build index
            if chunks:
                self.retriever.build_index(chunks, metadata)
                
                # Save index
                index_path = "models/tool_functions_index"
                os.makedirs(os.path.dirname(index_path), exist_ok=True)
                self.retriever.save_index(index_path)
                
                self.is_initialized = True
                logger.info(f"Tool index built successfully with {len(chunks)} function descriptions")
            else:
                logger.warning("No function chunks found in JSON documentation")
                
        except Exception as e:
            logger.error(f"Failed to build tool index: {e}")
    
    def _create_chunks_from_json(self, tool_data: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Create searchable chunks from JSON tool data"""
        chunks = []
        metadata = []
        
        # Create chunks for individual functions
        for func in tool_data.get('functions', []):
            # Main function chunk with all details
            func_chunk = self._create_function_chunk(func)
            chunks.append(func_chunk)
            
            metadata.append({
                'chunk_id': len(chunks) - 1,
                'function_name': func['name'],
                'category': func.get('category', 'general'),
                'type': 'function',
                'keywords': func.get('keywords', []),
                'content_preview': func['description'][:100] + "..." if len(func['description']) > 100 else func['description']
            })
            
            # Create additional chunks for examples and common commands if available
            if 'examples' in func and func['examples']:
                example_chunk = f"Function {func['name']} examples: " + ". ".join(func['examples'])
                chunks.append(example_chunk)
                metadata.append({
                    'chunk_id': len(chunks) - 1,
                    'function_name': func['name'],
                    'category': func.get('category', 'general'),
                    'type': 'examples',
                    'content_preview': example_chunk[:100] + "..."
                })
            
            # For execute_shell_command, add common commands as separate chunks
            if func['name'] == 'execute_shell_command' and 'common_commands' in func:
                for cmd in func['common_commands']:
                    cmd_chunk = f"Shell command: {cmd['command']} - {cmd['description']}. Use execute_shell_command function."
                    chunks.append(cmd_chunk)
                    metadata.append({
                        'chunk_id': len(chunks) - 1,
                        'function_name': func['name'],
                        'category': 'shell_commands',
                        'type': 'command_example',
                        'command': cmd['command'],
                        'content_preview': cmd_chunk[:100] + "..."
                    })
        
        # Add usage patterns as chunks
        for pattern in tool_data.get('usage_patterns', []):
            pattern_chunk = f"Usage pattern for {pattern['pattern']}: " + " → ".join(pattern['steps'])
            chunks.append(pattern_chunk)
            metadata.append({
                'chunk_id': len(chunks) - 1,
                'function_name': None,
                'category': 'usage_pattern',
                'type': 'pattern',
                'pattern_name': pattern['pattern'],
                'content_preview': pattern_chunk[:100] + "..."
            })
        
        return chunks, metadata
    
    def _create_function_chunk(self, func: Dict[str, Any]) -> str:
        """Create a comprehensive text chunk for a function"""
        chunk_parts = [
            f"Function: {func['name']}",
            f"Description: {func['description']}",
            f"Category: {func.get('category', 'general')}",
            f"Usage: {func.get('usage', '')}"
        ]
        
        # Add parameters info
        if func.get('parameters'):
            param_parts = []
            for param_name, param_info in func['parameters'].items():
                if isinstance(param_info, dict):
                    required = "required" if param_info.get('required', False) else "optional"
                    param_parts.append(f"{param_name} ({required}): {param_info.get('description', '')}")
                
            if param_parts:
                chunk_parts.append(f"Parameters: {', '.join(param_parts)}")
        
        # Add keywords for better matching
        if func.get('keywords'):
            chunk_parts.append(f"Keywords: {', '.join(func['keywords'])}")
        
        # Add examples
        if func.get('examples'):
            chunk_parts.append(f"Examples: {'. '.join(func['examples'])}")
        
        return "\n".join(chunk_parts)
    
    def get_relevant_tools(self, user_query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get relevant tools based on user query"""
        if not self.is_initialized:
            logger.warning("RAG tool retriever not initialized, returning empty list")
            return []
        
        try:
            # Search for relevant tool descriptions
            results = self.retriever.search(user_query, top_k=top_k)
            
            relevant_tools = []
            seen_functions = set()
            
            for content, similarity, metadata in results:
                function_name = metadata.get('function_name')
                
                # Only include function descriptions, not general documentation
                if function_name and function_name not in seen_functions:
                    seen_functions.add(function_name)
                    
                    # Get complete function info from registry instead of parsing RAG content
                    tool_info = self._get_function_from_registry(function_name)
                    if tool_info:
                        relevant_tools.append(tool_info)
            
            logger.info(f"Found {len(relevant_tools)} relevant tools for query: '{user_query}'")
            return relevant_tools
            
        except Exception as e:
            logger.error(f"Failed to retrieve relevant tools: {e}")
            return []
    
    def _get_function_from_registry(self, function_name: str) -> Optional[Dict[str, Any]]:
        """Get complete function information from registry"""
        if function_name in self.function_registry:
            return self.function_registry[function_name]
        else:
            logger.warning(f"Function {function_name} not found in registry")
            return None
    
    def get_all_available_functions(self) -> List[str]:
        """Get list of all available function names"""
        return list(self.function_registry.keys()) 