"""
RAGProcessor - High-performance document processing and semantic search pipeline.

This package provides tools for processing various document types, creating optimized
text chunks, and performing semantic search using vector databases.
"""

__version__ = "1.0.1"
__author__ = "Michał Kamiński"

from .vector_store_service import VectorStoreService
from .text_splitter import TextSplitter, TextChunk, TextChunkMetadata, TextChunkIndexes
from .document_service import DocumentService

__all__ = [
    "VectorStoreService",
    "TextSplitter", 
    "TextChunk",
    "TextChunkMetadata", 
    "TextChunkIndexes",
    "DocumentService"
]
