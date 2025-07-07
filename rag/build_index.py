import os
import re
from typing import List, Dict, Any, Tuple
from .retriever import FAISSRetriever
import argparse

class DocumentProcessor:
    """Document processing and chunking class"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_documents(self, doc_dir: str) -> List[Dict[str, Any]]:
        """Load documents from directory"""
        documents = []
        
        for filename in os.listdir(doc_dir):
            filepath = os.path.join(doc_dir, filename)
            
            if filename.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                documents.append({
                    'filename': filename,
                    'filepath': filepath,
                    'content': content
                })
        
        return documents
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        # Split by sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check size when adding sentence to current chunk
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start new chunk
                current_chunk = sentence
        
        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process documents and return chunks with metadata"""
        all_chunks = []
        all_metadata = []
        
        for doc in documents:
            chunks = self.chunk_text(doc['content'])
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'filename': doc['filename'],
                    'filepath': doc['filepath'],
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                })
        
        return all_chunks, all_metadata

def build_index(doc_dir: str, index_path: str, config_path: str = "llm/config.yaml"):
    """Main function for building document index"""
    print(f"Document directory: {doc_dir}")
    print(f"Index save path: {index_path}")
    
    # Initialize document processor
    processor = DocumentProcessor()
    
    # Load documents
    documents = processor.load_documents(doc_dir)
    print(f"Loaded documents: {len(documents)}")
    
    # Chunk documents
    chunks, metadata = processor.process_documents(documents)
    print(f"Generated chunks: {len(chunks)}")
    
    # Initialize retrieval system
    retriever = FAISSRetriever(config_path)
    
    # Build index
    retriever.build_index(chunks, metadata)
    
    # Save index
    retriever.save_index(index_path)
    
    # Print statistics
    stats = retriever.get_stats()
    print("\n=== Index Build Complete ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG index")
    parser.add_argument("--doc_dir", default="documents", help="Document directory path")
    parser.add_argument("--index_path", default="models/faiss_index", help="Index save path")
    parser.add_argument("--config", default="llm/config.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    build_index(args.doc_dir, args.index_path, args.config)
