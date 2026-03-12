"""app.services.file — File handling, security scanning and RAG retrieval."""

from ..file_handler import add_file, extract_text_from_binary_file
from ..file_processor import DynamicFileProcessor as FileProcessor
from ..file_security_scanner import FileSecurityScanner, SecurityConfig, ThreatLevel
from ..structured_file_engine import StructuredFileEngine
from ..lightweight_rag import LightweightRAG
from ..pgvector_file_retriever import PgVectorFileRetriever, VectorSearchConfig
from ..embedding_retriever import EmbeddingBasedRetriever as EmbeddingRetriever
from ..embedding_service import EmbeddingService
from ..rag_context_optimizer import RAGContextOptimizer
from ..rag_context_retriever import RAGContextRetriever

__all__ = [
    "add_file", "extract_text_from_binary_file", "FileProcessor",
    "FileSecurityScanner", "SecurityConfig", "ThreatLevel",
    "StructuredFileEngine", "LightweightRAG", "PgVectorFileRetriever",
    "VectorSearchConfig", "EmbeddingRetriever", "EmbeddingService",
    "RAGContextOptimizer", "RAGContextRetriever",
]
