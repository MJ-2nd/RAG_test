import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Any
from .embedder import BGEEmbedder
import yaml

class FAISSRetriever:
    """FAISS-based high-performance retrieval system"""
    
    def __init__(self, config_path: str = "llm/config.yaml"):
        self.config = self._load_config(config_path)
        self.embedder = BGEEmbedder(config_path)
        self.index = None
        self.documents = []
        self.document_metadata = []
        self.is_built = False
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def build_index(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """Build document index"""
        print("Generating document embeddings...")
        
        # Generate document embeddings (runs on CPU)
        doc_embeddings = self.embedder.encode_documents(documents)
        
        # Create FAISS index
        dimension = doc_embeddings.shape[1]
        
        # Choose efficient index type for CPU
        if len(documents) < 1000:
            # Small scale: exact search
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        else:
            # Large scale: approximate search for speed
            nlist = min(int(np.sqrt(len(documents))), 100)
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatIP(dimension), 
                dimension, 
                nlist
            )
            # Training required
            self.index.train(doc_embeddings.astype(np.float32))
        
        # Add embeddings to index
        self.index.add(doc_embeddings.astype(np.float32))
        
        # Store documents and metadata
        self.documents = documents
        self.document_metadata = metadata or [{} for _ in documents]
        self.is_built = True
        
        print(f"Index built successfully: {len(documents)} documents")
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search query"""
        if not self.is_built:
            raise ValueError("Index not built. Please call build_index() first.")
        
        if top_k is None:
            top_k = self.config['rag']['top_k']
        
        # Generate query embedding
        query_embedding = self.embedder.encode_query(query)
        
        # FAISS search
        similarities, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32), 
            top_k
        )
        
        # Process results
        results = []
        similarity_threshold = self.config['rag']['similarity_threshold']
        
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity < similarity_threshold:
                continue
                
            results.append((
                self.documents[idx],
                float(similarity),
                self.document_metadata[idx]
            ))
        
        return results
    
    def save_index(self, index_path: str):
        """Save index"""
        if not self.is_built:
            raise ValueError("No index to save.")
        
        # Save FAISS index
        faiss.write_index(self.index, f"{index_path}.faiss")
        
        # Save documents and metadata
        with open(f"{index_path}.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.document_metadata
            }, f)
        
        print(f"Index saved: {index_path}")
    
    def load_index(self, index_path: str):
        """Load index"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{index_path}.faiss")
            
            # Load documents and metadata
            with open(f"{index_path}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.document_metadata = data['metadata']
            
            self.is_built = True
            print(f"Index loaded successfully: {len(self.documents)} documents")
            
        except Exception as e:
            print(f"Failed to load index: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics"""
        if not self.is_built:
            return {"status": "Index not built"}
        
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embedder.get_embedding_dimension(),
            "index_type": type(self.index).__name__,
            "memory_usage_mb": f"{self.index.ntotal * self.embedder.get_embedding_dimension() * 4 / 1024 / 1024:.1f}"
        }
