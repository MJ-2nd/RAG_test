"""
RAG (Retrieval-Augmented Generation) module

This module provides document retrieval and embedding functionality.
"""

from .embedder import BGEEmbedder
from .retriever import FAISSRetriever
from .build_index import DocumentProcessor, build_index

__all__ = [
    'BGEEmbedder',
    'FAISSRetriever', 
    'DocumentProcessor',
    'build_index'
] 