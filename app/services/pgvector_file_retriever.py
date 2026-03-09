"""
PGVECTOR-BASED FILE RETRIEVAL (P1 - Zero Hardcoding)

Indexed vector similarity search for files:
- Store embeddings in pgvector with HNSW index
- Query by vector similarity (cosine distance)
- Session/user filtering
- Lazy evaluation for performance

Replaces in-memory similarity matching with production DB-indexed search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class VectorIndexStrategy(str, Enum):
    """Vector index types supported by pgvector."""
    HNSW = "hnsw"  # Hierarchical Navigable Small World (default, better recall)
    IVFFLAT = "ivfflat"  # Faster for large datasets, lower recall


@dataclass
class Vector:
    """Wrapper for vector operations."""
    values: List[float]
    dimension: int = None
    
    def __post_init__(self):
        if self.dimension is None:
            self.dimension = len(self.values)
    
    @staticmethod
    def from_embedding(embedding: Any) -> Vector:
        """Create from numpy array or list."""
        if isinstance(embedding, np.ndarray):
            return Vector(values=embedding.tolist())
        elif isinstance(embedding, list):
            return Vector(values=embedding)
        else:
            raise ValueError(f"Unsupported embedding type: {type(embedding)}")
    
    def to_list(self) -> List[float]:
        return self.values


@dataclass
class FileChunkMetadata:
    """Metadata for a file chunk (for citations)."""
    chunk_id: str
    file_id: str
    file_name: str
    chunk_index: int
    source_type: str  # "page" | "sheet" | "row" | "paragraph"
    source_locator: str  # "page 5" | "sheet '<name>'" | "row 42"
    char_start: int
    char_end: int
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "file_id": self.file_id,
            "file_name": self.file_name,
            "chunk_index": self.chunk_index,
            "source_type": self.source_type,
            "source_locator": self.source_locator,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class RetrievedChunk:
    """Result from vector search."""
    metadata: FileChunkMetadata
    content: str
    similarity_score: float  # 1.0 = perfect match, 0.0 = opposite
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "content": self.content,
            "similarity_score": round(self.similarity_score, 4),
        }


@dataclass
class VectorSearchConfig:
    """Configuration for vector search."""
    similarity_threshold: float = 0.5  # Min similarity to return results
    top_k: int = 5  # Return top-k results
    search_timeout_ms: int = 5000  # Query timeout
    index_strategy: VectorIndexStrategy = VectorIndexStrategy.HNSW
    
    # For HNSW index creation
    m: int = 16  # Max connections per node
    ef_construction: int = 64  # Construction parameter
    
    # For IVFFlat index creation
    lists: int = 100  # Number of clustering
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "similarity_threshold": self.similarity_threshold,
            "top_k": self.top_k,
            "search_timeout_ms": self.search_timeout_ms,
            "index_strategy": self.index_strategy.value,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "lists": self.lists,
        }


class PgVectorFileRetriever:
    """
    File retrieval using pgvector indexed search.
    
    KEY PRINCIPLE: All queries are parameterized, session/user scoped.
    """
    
    def __init__(self, db, embedding_model, config: Optional[VectorSearchConfig] = None):
        """
        Args:
            db: AsyncSessionLocal or similar
            embedding_model: Callable that returns Vector from text
            config: VectorSearchConfig
        """
        self.db = db
        self.embedding_model = embedding_model
        self.config = config or VectorSearchConfig()
        logger.info(f"PgVectorFileRetriever initialized: strategy={self.config.index_strategy}, "
                   f"k={self.config.top_k}, threshold={self.config.similarity_threshold}")
    
    async def setup_index(self, table_name: str = "file_chunks") -> None:
        """
        Create pgvector extension and index (idempotent).
        
        Call once at app startup.
        """
        from sqlalchemy import text
        
        try:
            async with self.db() as session:
                # Create extension
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                
                # Create index based on strategy
                if self.config.index_strategy == VectorIndexStrategy.HNSW:
                    index_sql = f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_vector_hnsw 
                    ON {table_name} USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {self.config.m}, ef_construction = {self.config.ef_construction})
                    """
                else:  # IVFFlat
                    index_sql = f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_vector_ivf 
                    ON {table_name} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {self.config.lists})
                    """
                
                await session.execute(text(index_sql))
                await session.commit()
                logger.info(f"Vector index created: {self.config.index_strategy.value}")
        
        except Exception as e:
            logger.warning(f"Index setup error (may already exist): {e}")
    
    async def search(
        self,
        query_text: str,
        session_id: str,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Vector similarity search for chunks.
        
        Args:
            query_text: User query
            session_id: Scope to this session
            user_id: Optional additional scope to user
            limit: Override top_k
            
        Returns:
            List of RetrievedChunk sorted by similarity (highest first)
        """
        from sqlalchemy import text
        
        k = limit or self.config.top_k
        threshold = self.config.similarity_threshold
        
        # Generate query embedding
        try:
            query_vector = self.embedding_model(query_text)
            if not isinstance(query_vector, Vector):
                query_vector = Vector.from_embedding(query_vector)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []
        
        # Build parameterized query (NO string interpolation)
        # Using PostgreSQL vector distance operator <-> (cosine distance)
        query_sql = """
        SELECT 
            fc.chunk_id,
            fc.file_id,
            fc.file_name,
            fc.chunk_index,
            fc.source_type,
            fc.source_locator,
            fc.char_start,
            fc.char_end,
            fc.created_at,
            fc.content,
            -- Convert distance to similarity (1 - distance)
            (1 - (fc.embedding <-> :query_vector::vector)) as similarity_score
        FROM file_chunks fc
        WHERE fc.session_id = :session_id
        """
        
        # Optional user scoping
        if user_id:
            query_sql += " AND fc.user_id = :user_id"
        
        query_sql += f"""
        AND (1 - (fc.embedding <-> :query_vector::vector)) >= :threshold
        ORDER BY fc.embedding <-> :query_vector::vector
        LIMIT :limit
        """
        
        try:
            async with self.db() as session:
                result = await session.execute(
                    text(query_sql),
                    {
                        "query_vector": str(query_vector.to_list()),  # Convert to string for pgvector
                        "session_id": session_id,
                        "user_id": user_id,
                        "threshold": threshold,
                        "limit": k,
                    }
                )
                
                rows = result.fetchall()
                
                chunks = []
                for row in rows:
                    metadata = FileChunkMetadata(
                        chunk_id=row.chunk_id,
                        file_id=row.file_id,
                        file_name=row.file_name,
                        chunk_index=row.chunk_index,
                        source_type=row.source_type,
                        source_locator=row.source_locator,
                        char_start=row.char_start,
                        char_end=row.char_end,
                        created_at=row.created_at,
                    )
                    
                    chunks.append(RetrievedChunk(
                        metadata=metadata,
                        content=row.content,
                        similarity_score=row.similarity_score,
                    ))
                
                logger.info(f"Vector search: query_len={len(query_text)}, "
                           f"session={session_id}, results={len(chunks)}")
                return chunks
        
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    async def add_chunk(
        self,
        chunk_id: str,
        file_id: str,
        file_name: str,
        content: str,
        chunk_index: int,
        source_type: str,
        source_locator: str,
        char_start: int,
        char_end: int,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Add a chunk with embedding to the index.
        
        Args:
            All parameters for the chunk
            
        Returns:
            True if successful
        """
        from sqlalchemy import text, insert
        from sqlalchemy.orm import declarative_base
        
        try:
            # Generate embedding
            embedding = self.embedding_model(content)
            if not isinstance(embedding, Vector):
                embedding = Vector.from_embedding(embedding)
            
            # Insert chunk with embedding (parameterized)
            # Assuming file_chunks table exists with columns:
            # (chunk_id, file_id, file_name, content, chunk_index, source_type, source_locator,
            #  char_start, char_end, embedding, session_id, user_id, created_at)
            
            insert_sql = """
            INSERT INTO file_chunks 
            (chunk_id, file_id, file_name, content, chunk_index, source_type, source_locator,
             char_start, char_end, embedding, session_id, user_id, created_at)
            VALUES (:chunk_id, :file_id, :file_name, :content, :chunk_index, :source_type,
                    :source_locator, :char_start, :char_end, :embedding::vector, :session_id,
                    :user_id, :created_at)
            ON CONFLICT (chunk_id) DO UPDATE SET
                embedding = :embedding::vector,
                updated_at = :created_at
            """
            
            async with self.db() as session:
                await session.execute(
                    text(insert_sql),
                    {
                        "chunk_id": chunk_id,
                        "file_id": file_id,
                        "file_name": file_name,
                        "content": content,
                        "chunk_index": chunk_index,
                        "source_type": source_type,
                        "source_locator": source_locator,
                        "char_start": char_start,
                        "char_end": char_end,
                        "embedding": str(embedding.to_list()),
                        "session_id": session_id,
                        "user_id": user_id,
                        "created_at": datetime.utcnow(),
                    }
                )
                await session.commit()
                logger.info(f"Chunk added: {chunk_id} (session={session_id})")
                return True
        
        except Exception as e:
            logger.error(f"Error adding chunk: {e}")
            return False
    
    async def cleanup_session(self, session_id: str) -> int:
        """
        Delete all chunks for a session (privacy/cleanup).
        
        Returns:
            Number of chunks deleted
        """
        from sqlalchemy import text
        
        try:
            async with self.db() as session:
                result = await session.execute(
                    text("DELETE FROM file_chunks WHERE session_id = :session_id"),
                    {"session_id": session_id}
                )
                await session.commit()
                deleted = result.rowcount
                logger.info(f"Cleaned up {deleted} chunks for session {session_id}")
                return deleted
        
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")
            return 0
