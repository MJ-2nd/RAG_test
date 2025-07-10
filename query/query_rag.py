import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rag.retriever import FAISSRetriever
from llm.app import LLMServer
from query.query_history import QueryHistory
import yaml
import argparse
from typing import List, Tuple, Dict, Any
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG System - Retrieval-Augmented Generation"""
    
    def __init__(self, config_path: str = "llm/config.yaml", index_path: str = "models/faiss_index"):
        self.config = self._load_config(config_path)
        self.retriever = FAISSRetriever(config_path)
        self.llm_server = LLMServer(config_path)
        self.index_path = index_path
        
        # Initialize query history
        self.query_history = QueryHistory(max_size=10)
        
        # Try to load existing index
        try:
            self.retriever.load_index(index_path)
            logger.info("Existing index loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            logger.info("New index needs to be built.")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def retrieve_context(self, query: str, top_k: int = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Retrieve relevant documents"""
        if not self.retriever.is_built:
            raise ValueError("Index not built. Please build the index first.")
        
        return self.retriever.search(query, top_k)
    
    def format_context(self, retrieved_docs: List[Tuple[str, float, Dict[str, Any]]]) -> str:
        """Format retrieved documents as context"""
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, (doc_text, similarity, metadata) in enumerate(retrieved_docs, 1):
            filename = metadata.get('filename', 'Unknown')
            chunk_id = metadata.get('chunk_id', 0)
            
            context_parts.append(f"""
Document {i} (File: {filename}, Chunk: {chunk_id}, Similarity: {similarity:.3f}):
{doc_text}
""")
        
        return "\n".join(context_parts)
    
    def query(self, user_query: str, top_k: int = None, include_context: bool = True) -> Dict[str, Any]:
        """Process RAG query"""
        try:
            # 1. Retrieve relevant documents
            retrieved_docs = self.retrieve_context(user_query, top_k)
            
            if not retrieved_docs:
                # Generate general LLM response if no relevant documents found
                logger.info("No relevant documents found, generating general LLM response.")
                prompt = self.llm_server.format_chat_prompt(user_query, history=self.query_history.get_history())
                response = self.llm_server.generate(prompt)
                
                # Store in history queue
                self.query_history.store(user_query, None, response)
                
                return {
                    "query": user_query,
                    "response": response,
                    "retrieved_docs": [],
                    "context_used": False
                }
            
            # 2. Format context
            context = self.format_context(retrieved_docs)
            
            # 3. Generate RAG prompt and response
            if include_context:
                prompt = self.llm_server.format_chat_prompt(user_query, context, history=self.query_history.get_history())
            else:
                prompt = self.llm_server.format_chat_prompt(user_query, history=self.query_history.get_history())
            
            response = self.llm_server.generate(prompt)
            
            # Store in history queue
            self.query_history.store(user_query, context if include_context else None, response)
            
            return {
                "query": user_query,
                "response": response,
                "retrieved_docs": [
                    {
                        "text": doc_text,
                        "similarity": similarity,
                        "metadata": metadata
                    }
                    for doc_text, similarity, metadata in retrieved_docs
                ],
                "context_used": include_context,
                "context": context if include_context else None
            }
            
        except Exception as e:
            logger.error(f"RAG query processing failed: {e}")
            raise
    
    def interactive_mode(self):
        """Interactive mode"""
        print("=== RAG System Interactive Mode ===")
        print("Enter your questions (exit: 'quit' or 'exit')")
        print("Commands:")
        print("  /stats - Show index statistics")
        print("  /context on/off - Toggle context usage")
        print("  /topk <number> - Set number of documents to retrieve")
        print()
        
        include_context = True
        top_k = None
        
        while True:
            try:
                user_input = input("Question: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Shutting down RAG system.")
                    break
                
                if user_input.startswith('/'):
                    # Handle commands
                    if user_input == '/stats':
                        stats = self.retriever.get_stats()
                        print("=== Index Statistics ===")
                        for key, value in stats.items():
                            print(f"{key}: {value}")
                        continue
                    
                    elif user_input.startswith('/context'):
                        parts = user_input.split()
                        if len(parts) > 1:
                            if parts[1].lower() == 'on':
                                include_context = True
                                print("Context usage enabled.")
                            elif parts[1].lower() == 'off':
                                include_context = False
                                print("Context usage disabled.")
                            else:
                                print("Usage: /context on or /context off")
                        else:
                            print(f"Current context usage: {'ON' if include_context else 'OFF'}")
                        continue
                    
                    elif user_input.startswith('/topk'):
                        parts = user_input.split()
                        if len(parts) > 1:
                            try:
                                top_k = int(parts[1])
                                print(f"Number of documents to retrieve set to {top_k}.")
                            except ValueError:
                                print("Please enter a valid number.")
                        else:
                            print(f"Current number of documents to retrieve: {top_k or 'default'}")
                        continue
                    
                    else:
                        print("Unknown command.")
                        continue
                
                if not user_input:
                    continue
                
                # Process RAG query
                print("Processing...")
                result = self.query(user_input, top_k, include_context)
                
                print(f"\nAnswer: {result['response']}")
                
                if result['context_used'] and result['retrieved_docs']:
                    print(f"\nReference documents ({len(result['retrieved_docs'])}):")
                    for i, doc in enumerate(result['retrieved_docs'], 1):
                        filename = doc['metadata'].get('filename', 'Unknown')
                        similarity = doc['similarity']
                        print(f"  {i}. {filename} (Similarity: {similarity:.3f})")
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\nShutting down RAG system.")
                break
            except Exception as e:
                print(f"Error occurred: {e}")
                continue

def main():
    parser = argparse.ArgumentParser(description="RAG system query")
    parser.add_argument("--config", default="llm/config.yaml", help="Configuration file path")
    parser.add_argument("--index", default="models/faiss_index", help="Index path")
    parser.add_argument("--query", help="Execute single query")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--no-context", action="store_true", help="Run without context")
    parser.add_argument("--top-k", type=int, help="Number of documents to retrieve")
    
    args = parser.parse_args()
    
    # Initialize RAG system
    try:
        rag_system = RAGSystem(args.config, args.index)
    except Exception as e:
        logger.error(f"RAG system initialization failed: {e}")
        return
    
    if args.interactive:
        # Interactive mode
        rag_system.interactive_mode()
    elif args.query:
        # Single query execution
        try:
            result = rag_system.query(
                args.query, 
                top_k=args.top_k,
                include_context=not args.no_context
            )
            
            print(f"Question: {result['query']}")
            print(f"Answer: {result['response']}")
            
            if result['context_used'] and result['retrieved_docs']:
                print(f"\nReference documents ({len(result['retrieved_docs'])}):")
                for i, doc in enumerate(result['retrieved_docs'], 1):
                    filename = doc['metadata'].get('filename', 'Unknown')
                    similarity = doc['similarity']
                    print(f"  {i}. {filename} (Similarity: {similarity:.3f})")
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
    else:
        print("Use --query or --interactive option.")
        print("Usage: python query/query_rag.py --interactive")
        print("       python query/query_rag.py --query 'your question'")

if __name__ == "__main__":
    main()
