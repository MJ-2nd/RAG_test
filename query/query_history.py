from collections import deque
from typing import List, Dict, Any, Optional

class QueryHistory:
    """Query history management class"""
    
    def __init__(self, max_size: int = 10):
        """Initialize query history with maximum size"""
        self.history_queue = deque(maxlen=max_size)
    
    def store(self, query: str, context: Optional[str] = None, response: Optional[str] = None):
        """Store query, context, and response in history queue"""
        history_entry = {
            "query": query,
            "context": context,
            "response": response
        }
        self.history_queue.append(history_entry)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Retrieve all history entries"""
        return list(self.history_queue)
