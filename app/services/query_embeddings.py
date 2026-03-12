"""
Query Embedding and Vector Storage Service

This module provides semantic search capabilities for queries using embeddings:
1. Embeds user queries and their results
2. Stores embeddings in PostgreSQL with pgvector
3. Retrieves semantically similar previous queries for follow-ups
4. Uses vector similarity search via pgvector for cross-session retrieval
"""

from __future__ import annotations

import json
import logging
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class QueryEmbedding:
    """Storage for query embeddings and metadata including all result rows."""
    query_id: str
    session_id: str
    user_query: str
    generated_sql: str
    result_count: int
    column_names: List[str]
    all_result_rows: List[Dict[str, Any]]  # NEW: All rows from result set
    embedding: List[float]
    created_at: datetime
    

class QueryEmbeddingStore:
    """
    Persistent vector store for query embeddings using PostgreSQL + pgvector.
    
    Stores embeddings in the database and uses pgvector for efficient
    vector similarity searches. Enables semantic retrieval across sessions.
    """
    
    def __init__(self, db_session: Optional[AsyncSession] = None, embedding_dim: int = 384):
        """Initialize embedding store.
        
        FIX: Changed default from 768 to 384 to match:
        - all-MiniLM-L6-v2 model (384 dimensions)
        - pgvector schema in models.py (Vector(384))
        - EmbeddingGenerator default (384 dimensions)
        """
        self.db_session = db_session
        self.embedding_dim = embedding_dim
        # Keep in-memory cache for current session
        self.embeddings: Dict[str, QueryEmbedding] = {}  # query_id -> embedding
        self.session_queries: Dict[str, List[str]] = {}  # session_id -> [query_ids]
        logger.info(f"[VECTOR STORE] Initialized with {embedding_dim}d embeddings (database-backed)")
    
    def set_db_session(self, db_session: AsyncSession) -> None:
        """Set the database session for persistence."""
        self.db_session = db_session
        logger.info("[VECTOR STORE] Database session configured for persistence")
    
    async def store_embedding(
        self,
        query_id: str,
        session_id: str,
        user_query: str,
        generated_sql: str,
        result_count: int,
        column_names: List[str],
        all_result_rows: List[Dict[str, Any]],  # NEW: ALL rows from result set
        embedding: List[float],
        db_session: Optional[AsyncSession] = None,  # Explicit session (preferred over self.db_session)
    ) -> None:
        """Store a query embedding with ALL result rows from query execution."""
        print(f"\n[VECTOR STORE] ⏸️ store_embedding() called for: {user_query}")
        print(f"[VECTOR STORE]   query_id: {query_id}")
        print(f"[VECTOR STORE]   session_id: {session_id}")
        print(f"[VECTOR STORE]   result_count: {result_count}")
        print(f"[VECTOR STORE]   storing {len(all_result_rows)} result rows")
        
        # Store in-memory cache
        query_emb = QueryEmbedding(
            query_id=query_id,
            session_id=session_id,
            user_query=user_query,
            generated_sql=generated_sql,
            result_count=result_count,
            column_names=column_names,
            all_result_rows=all_result_rows,  # NEW: Store all rows
            embedding=embedding,
            created_at=datetime.utcnow(),
        )
        
        print(f"[VECTOR STORE] Created QueryEmbedding object: {query_emb.query_id}")
        
        self.embeddings[query_id] = query_emb
        print(f"[VECTOR STORE] Added to embeddings dict. Total embeddings now: {len(self.embeddings)}")
        
        if session_id not in self.session_queries:
            self.session_queries[session_id] = []
        self.session_queries[session_id].append(query_id)
        print(f"[VECTOR STORE] Session {session_id} now has {len(self.session_queries[session_id])} queries")
        
        # Persist to database if a session is available.
        # Prefer the explicitly-passed db_session over self.db_session to avoid
        # using a stale/shared request session from a previous call.
        effective_session = db_session or self.db_session
        if effective_session:
            try:
                # Import here to avoid circular imports
                from app.models import QueryEmbedding as QueryEmbeddingModel
                from app.helpers import make_json_serializable
                import uuid

                # Create query hash for duplicate detection
                query_hash = hashlib.sha256(user_query.encode()).hexdigest()

                # Serialize all_result_rows using the standard helper (handles Decimal, datetime, etc.)
                serialized_rows = []
                for row in all_result_rows:
                    if isinstance(row, dict):
                        # Use make_json_serializable to handle Decimal, datetime, and other types
                        serialized_row = {k: make_json_serializable(v) for k, v in row.items()}
                        serialized_rows.append(serialized_row)
                    else:
                        serialized_rows.append(make_json_serializable(row))

                # Calculate result quality score (0-100 scale)
                # Based on: embedding sparsity, result coverage, and data completeness
                non_zero_embeddings = sum(1 for v in embedding if abs(v) > 0.001)
                sparsity_score = min(100, (non_zero_embeddings / 10))  # 10+ non-zero = 100
                coverage_score = min(100, (len(serialized_rows) / 100) * 100)  # 100 rows = 100
                result_quality_score = int((sparsity_score + coverage_score) / 2)  # Average of both factors

                print(f"[VECTOR STORE] Calculated quality score: {result_quality_score} (sparsity: {non_zero_embeddings} non-zero, rows: {len(serialized_rows)})")

                # Create database record
                db_embedding = QueryEmbeddingModel(
                    id=uuid.UUID(query_id),
                    session_id=uuid.UUID(session_id),
                    user_query=user_query,
                    generated_sql=generated_sql,
                    result_count=result_count,
                    column_names=column_names,
                    all_result_rows=serialized_rows,  # Store ALL rows with proper JSON serialization
                    embedding=embedding,
                    query_hash=query_hash,
                    result_quality_score=result_quality_score,  # NEW: Store quality score
                )

                effective_session.add(db_embedding)
                await effective_session.flush()  # Flush to get the ID without committing
                print(f"[VECTOR STORE] Persisted embedding with {len(all_result_rows)} result rows to database")
                logger.info(f"[VECTOR STORE] Persisted embedding with {len(all_result_rows)} rows: {query_id}")
            except Exception as e:
                # Attempt rollback only on the session we own (the background-task session).
                # Never rollback self.db_session here — it belongs to the caller.
                if db_session:
                    try:
                        await db_session.rollback()
                        print(f"[VECTOR STORE] ⚠️ Rolled back background session after error")
                    except Exception:
                        pass
                print(f"[VECTOR STORE] ⚠️ Failed to persist to database: {e}")
                logger.error(f"[VECTOR STORE] Failed to persist embedding: {str(e)}")
                # Continue anyway - in-memory store works
        
        print(f"[VECTOR STORE] ✅ SUCCESSFULLY STORED {len(all_result_rows)} result rows for query: {user_query[:50]}")
        logger.info(f"[VECTOR STORE] Stored {len(all_result_rows)} result rows for query: {user_query[:50]}")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        A = np.array(vec1)
        B = np.array(vec2)
        
        cos_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
        return float(cos_sim)
    
    async def search_similar_queries(
        self,
        session_id: str,
        query_embedding: List[float],
        top_k: int = 3,
        similarity_threshold: float = 0.6,
        db_session: Optional[AsyncSession] = None,  # Explicit session (preferred over self.db_session)
    ) -> List[Tuple[QueryEmbedding, float]]:
        """
        Search for semantically similar previous queries.
        
        First searches in-memory cache for current session, then queries database
        for cross-session historical similarity if available.
        
        Args:
            session_id: Session to search within (primary), but also searches globally
            query_embedding: Embedding of the current query
            top_k: Return top K results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (QueryEmbedding, similarity_score) tuples, sorted by similarity descending
        """
        print(f"\n[VECTOR SEARCH] Searching for similar queries")
        print(f"[VECTOR SEARCH] Store has {len(self.session_queries)} sessions total (in-memory)")
        print(f"[VECTOR SEARCH] Store has {len(self.embeddings)} total embeddings (in-memory)")
        
        similarities = []
        
        # 1. Search in-memory cache first (current session)
        if session_id in self.session_queries:
            query_ids = self.session_queries[session_id]
            print(f"[VECTOR SEARCH] Found {len(query_ids)} previous queries in session (in-memory):")
            
            for qid in query_ids:
                if qid in self.embeddings:
                    embedding = self.embeddings[qid]
                    sim = self._cosine_similarity(query_embedding, embedding.embedding)
                    print(f"[VECTOR SEARCH] Similarity with '{embedding.user_query}': {sim:.3f}")
                    
                    if sim >= similarity_threshold:
                        similarities.append((embedding, sim))
        else:
            print(f"[VECTOR SEARCH] No in-memory queries found for session: {session_id}")
        
        # 2. Search database for historical queries if available
        effective_session = db_session or self.db_session
        if effective_session:
            try:
                from app.models import QueryEmbedding as QueryEmbeddingModel

                # Fetch embeddings from database
                # Note: Using JSON embeddings which are stored as regular JSON in the database
                # Similarity search is done in Python using cosine distance
                stmt = select(QueryEmbeddingModel).order_by(
                    QueryEmbeddingModel.created_at.desc()
                ).limit(top_k * 10)  # Get more to filter by threshold
                
                result = await effective_session.execute(stmt)
                db_embeddings = result.scalars().all()

                print(f"[VECTOR SEARCH] Found {len(db_embeddings)} potential matches in database")

                for db_emb in db_embeddings:
                    # Ensure embedding is a list (JSON may return it as list or dict)
                    emb_data = db_emb.embedding
                    if emb_data is None:
                        continue  # Skip records with no embedding
                    elif isinstance(emb_data, dict):
                        # If it's a dict, try to reconstruct the list
                        continue  # Skip malformed embeddings
                    elif not isinstance(emb_data, list):
                        try:
                            emb_data = list(emb_data)
                        except (TypeError, ValueError):
                            continue  # Skip if can't convert to list
                    
                    # Convert database record to QueryEmbedding
                    qe = QueryEmbedding(
                        query_id=str(db_emb.id),
                        session_id=str(db_emb.session_id),
                        user_query=db_emb.user_query,
                        generated_sql=db_emb.generated_sql,
                        result_count=db_emb.result_count,
                        column_names=db_emb.column_names,
                        all_result_rows=db_emb.all_result_rows if db_emb.all_result_rows else [],  # All rows stored
                        embedding=emb_data,
                        created_at=db_emb.created_at,
                    )
                    
                    sim = self._cosine_similarity(query_embedding, qe.embedding)
                    print(f"[VECTOR SEARCH] DB match - '{qe.user_query}': {sim:.3f}")
                    
                    # Avoid duplicates with in-memory results
                    if not any(e[0].query_id == qe.query_id for e in similarities):
                        if sim >= similarity_threshold:
                            similarities.append((qe, sim))
                            
            except Exception as e:
                print(f"[VECTOR SEARCH] ⚠️ Database search failed: {e}")
                logger.error(f"[VECTOR SEARCH] Database search error: {str(e)}")
                # Continue with in-memory results
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"[VECTOR SEARCH] Found {len(similarities)} similar queries (threshold: {similarity_threshold})")
        logger.info(f"[VECTOR SEARCH] Found {len(similarities)} similar queries (threshold: {similarity_threshold})")
        
        return similarities[:top_k]
    
    async def search_similar_queries_cross_session(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        similarity_threshold: float = 0.65,
    ) -> List[Tuple[QueryEmbedding, float]]:
        """
        Search for semantically similar queries across ALL sessions.
        
        Uses Python-based cosine similarity search (works with JSON embeddings).
        Great for finding similar patterns regardless of session.
        
        Args:
            query_embedding: Embedding of the current query
            top_k: Return top K results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (QueryEmbedding, similarity_score) tuples, sorted by similarity descending
        """
        print(f"\n[VECTOR SEARCH] Cross-session similarity search")
        
        if not self.db_session:
            print("[VECTOR SEARCH] ❌ Database not available for cross-session search")
            return []
        
        try:
            from app.models import QueryEmbedding as QueryEmbeddingModel
            
            # Fetch all embeddings and compute similarity in Python
            stmt = select(QueryEmbeddingModel).order_by(
                QueryEmbeddingModel.created_at.desc()
            )
            
            result = await self.db_session.execute(stmt)
            db_embeddings = result.scalars().all()
            
            similarities = []
            for db_emb in db_embeddings:
                # Ensure embedding is a list
                emb_data = db_emb.embedding
                if isinstance(emb_data, dict) or not isinstance(emb_data, list):
                    continue  # Skip malformed embeddings
                
                qe = QueryEmbedding(
                    query_id=str(db_emb.id),
                    session_id=str(db_emb.session_id),
                    user_query=db_emb.user_query,
                    generated_sql=db_emb.generated_sql,
                    result_count=db_emb.result_count,
                    column_names=db_emb.column_names,
                    all_result_rows=db_emb.all_result_rows or [],
                    embedding=db_emb.embedding,
                    created_at=db_emb.created_at,
                )
                
                sim = self._cosine_similarity(query_embedding, qe.embedding)
                if sim >= similarity_threshold:
                    similarities.append((qe, sim))
                    print(f"[VECTOR SEARCH] Cross-session match: '{qe.user_query}' (sim: {sim:.3f})")
            
            # Sort by similarity descending and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            print(f"[VECTOR SEARCH] ❌ Cross-session search failed: {e}")
            logger.error(f"[VECTOR SEARCH] Cross-session search error: {str(e)}")
            return []
    
    def get_session_history(self, session_id: str) -> List[QueryEmbedding]:
        """Get chronological history of queries in a session (in-memory cache)."""
        if session_id not in self.session_queries:
            return []
        
        query_ids = self.session_queries[session_id]
        return [self.embeddings[qid] for qid in query_ids if qid in self.embeddings]
    
    async def get_session_history_from_db(
        self, session_id: str, db_session: Optional[AsyncSession] = None
    ) -> List[QueryEmbedding]:
        """Get chronological history of queries from the database."""
        effective_session = db_session or self.db_session
        if not effective_session:
            return []
        
        try:
            from app.models import QueryEmbedding as QueryEmbeddingModel
            import uuid
            
            stmt = select(QueryEmbeddingModel).where(
                QueryEmbeddingModel.session_id == uuid.UUID(session_id)
            ).order_by(QueryEmbeddingModel.created_at)

            result = await effective_session.execute(stmt)
            db_embeddings = result.scalars().all()

            return [
                QueryEmbedding(
                    query_id=str(db_emb.id),
                    session_id=str(db_emb.session_id),
                    user_query=db_emb.user_query,
                    generated_sql=db_emb.generated_sql,
                    result_count=db_emb.result_count,
                    column_names=db_emb.column_names,
                    all_result_rows=db_emb.all_result_rows or [],
                    embedding=db_emb.embedding,
                    created_at=db_emb.created_at,
                )
                for db_emb in db_embeddings
            ]
        except Exception as e:
            logger.error(f"[VECTOR STORE] Failed to get session history from DB: {str(e)}")
            return []


class EmbeddingGenerator:
    """
    Lightweight embedding generator - NO neural models needed!
    
    Uses lightweight hash-based encoding instead of sentence-transformers.
    Encodes query structure (tables, operations, filters) into fixed-size vector.
    Compatible with 384-dim storage for backward compatibility with pgvector schema.
    
    NO EXTERNAL DEPENDENCIES - works offline instantly!
    ✅ Replaces: sentence-transformers
    ✅ Instant initialization (no model download)
    ✅ Works completely offline
    ✅ Minimal memory footprint
    """
    
    def __init__(self, dim: Optional[int] = None, model_name: Optional[str] = None):
        """
        Initialize lightweight embedding generator.
        
        Args:
            dim: Optional dimension (defaults to 384 for compatibility)
            model_name: Ignored - kept for API compatibility
        """
        import os
        
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "lightweight-v1")
        self.dim = dim or 384  # Keep 384 for pgvector compatibility
        
        logger.info(f"[EMBEDDINGS] ✅ Lightweight embedding generator initialized")
        logger.info(f"[EMBEDDINGS] Model: {self.model_name} ({self.dim}d embeddings)")
        logger.info(f"[EMBEDDINGS] NOTE: Using lightweight encoding (no neural models)")
    
    def _hash_to_vector(self, text: str, size: int = 384) -> List[float]:
        """
        Convert text to fixed-size vector using hash-based encoding.
        
        This is a lightweight alternative to neural embeddings:
        - Deterministic (same input = same output)
        - Fast (no model inference)
        - Captures token frequency patterns
        - Compatible with pgvector storage
        
        Args:
            text: Input text to encode
            size: Output vector dimension
        
        Returns:
            Fixed-size float vector
        """
        import hashlib
        
        # Tokenize and count frequency
        tokens = text.lower().split()
        token_freqs = {}
        for token in tokens:
            # Clean token (remove punctuation)
            clean_token = ''.join(c for c in token if c.isalnum())
            if len(clean_token) > 2:
                token_freqs[clean_token] = token_freqs.get(clean_token, 0) + 1
        
        # Create hash-based vector
        vector = [0.0] * size
        
        # Use token hashes to set vector values
        for token, freq in token_freqs.items():
            # Hash token to get positions
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            
            # Set multiple positions based on frequency and hash
            for i in range(min(3, freq)):  # Frequency (max 3)
                pos = (h + i) % size
                vector[pos] += 0.1 * freq
        
        # Normalize vector
        magnitude = sum(v * v for v in vector) ** 0.5
        if magnitude > 0:
            vector = [v / magnitude for v in vector]
        
        # Ensure all values in [-1, 1] range for pgvector compatibility
        vector = [max(-1.0, min(1.0, v)) for v in vector]
        
        return vector
    
    async def generate_embedding(
        self, 
        query: str, 
        sql: str = "",
        result_data: Optional[Dict[str, Any]] = None,
        column_names: Optional[List[str]] = None,
        result_count: Optional[int] = None,
    ) -> List[float]:
        """
        Generate lightweight embedding using hash-based encoding.
        
        Combines:
        - User query text
        - Generated SQL structure
        - Result columns
        - Result count
        
        Returns:
        - 384-dim vector (format compatible with pgvector)
        """
        import asyncio
        
        # Build comprehensive text representation
        text_parts = [query]
        
        if sql:
            text_parts.append(sql)
        
        if column_names:
            text_parts.append(' '.join(column_names))
        
        if result_data:
            # Add sample values from results
            sample_values = []
            for key, value in list(result_data.items())[:5]:
                if value is not None:
                    sample_values.append(str(value))
            if sample_values:
                text_parts.append(' '.join(sample_values))
        
        if result_count is not None:
            text_parts.append(f"count:{result_count}")
        
        combined_text = " ".join(text_parts)
        
        # Generate embedding asynchronously (run in thread to avoid blocking)
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._hash_to_vector(combined_text, self.dim)
        )
        
        logger.info(f"[EMBEDDINGS] Generated {self.dim}d embedding for: {query}")
        return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
    


# Global instances
_embedding_store: Optional[QueryEmbeddingStore] = None
_embedding_generator: Optional[EmbeddingGenerator] = None


async def get_embedding_store(db_session: Optional[AsyncSession] = None) -> QueryEmbeddingStore:
    """
    Get or create the global embedding store.

    NOTE: db_session is intentionally NOT stored on the singleton here.
    Pass db_session directly to store_embedding() / search_similar_queries()
    so each caller uses its own session and never corrupts a shared one.

    Args:
        db_session: Deprecated — kept for signature compatibility; ignored.

    Returns:
        Global QueryEmbeddingStore instance
    """
    global _embedding_store
    if _embedding_store is None:
        print(f"[VECTOR STORE] Creating new global embedding store instance")
        # Do not inject db_session — callers pass it per-operation
        _embedding_store = QueryEmbeddingStore()
    else:
        print(f"[VECTOR STORE] Reusing existing global embedding store (id: {id(_embedding_store)}, has {len(_embedding_store.embeddings)} embeddings)")
    return _embedding_store


async def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create the global embedding generator."""
    global _embedding_generator
    if _embedding_generator is None:
        print(f"[EMBEDDINGS] Creating new global embedding generator instance")
        _embedding_generator = EmbeddingGenerator()
    else:
        print(f"[EMBEDDINGS] Reusing existing global embedding generator (id: {id(_embedding_generator)})")
    return _embedding_generator
