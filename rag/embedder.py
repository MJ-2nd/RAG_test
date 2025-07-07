import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import yaml
import os

class BGEEmbedder:
    """BGE-M3 embedding model class - CPU optimized"""
    
    def __init__(self, config_path: str = "llm/config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.device = "cpu"  # Run on CPU to save VRAM
        self._load_model()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_model(self):
        """Load BGE-M3 model"""
        try:
            model_name = self.config['embedding']['model_name']
            
            # Load from local path if exists, otherwise download from HuggingFace
            model_path = self.config['embedding']['model_path']
            if os.path.exists(model_path):
                self.model = SentenceTransformer(model_path, device=self.device)
            else:
                print(f"Local model path {model_path} does not exist. Downloading from HuggingFace...")
                self.model = SentenceTransformer(model_name, device=self.device)
            
            print(f"Embedding model loaded: {model_name} on {self.device}")
            
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = None) -> np.ndarray:
        """Convert text to embedding vectors"""
        if isinstance(texts, str):
            texts = [texts]
        
        if batch_size is None:
            batch_size = self.config['embedding']['batch_size']
        
        # Efficient batch processing on CPU
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity calculation
        )
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Return embedding dimension"""
        return self.model.get_sentence_embedding_dimension()
    
    def encode_query(self, query: str) -> np.ndarray:
        """Query-specific embedding (optimized for retrieval)"""
        # BGE-M3 can add special prompts for queries
        query_prompt = f"Represent this sentence for searching relevant passages: {query}"
        return self.encode(query_prompt)
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Document-specific embedding (optimized for indexing)"""
        return self.encode(documents)
    
    def compute_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between embeddings"""
        # Since embeddings are already normalized, we only need dot product
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = np.dot(query_embedding, doc_embeddings.T)
        return similarities.flatten()
