"""
RAG Context Retriever for Follow-Up Queries

This service retrieves semantically relevant context from previous queries
using vector embeddings, enabling follow-ups to work even with intervening queries.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession

from .query_embeddings import (
    get_embedding_store,
    get_embedding_generator,
    QueryEmbedding,
)

logger = logging.getLogger(__name__)


@dataclass
class RAGContext:
    """Retrieved context from previous similar queries."""
    is_relevant: bool
    similarity_score: float
    previous_query: str
    previous_sql: str
    result_count: int
    column_names: List[str]
    sample_row: Optional[Dict[str, Any]]
    context_text: str  # Formatted text for LLM


class RAGContextRetriever:
    """
    Retrieves context from previous queries for follow-ups using RAG.
    
    Pure semantic approach:
    1. Generate embedding for current query (token-based hashing, database-agnostic)
    2. Find similar queries via cosine similarity
    3. Fallback to most recent query if no semantic match found
    
    Works with:
    - Single and multi-table queries (embeddings capture all tokens)
    - Any database schema (no domain-specific hardcoding)
    - Follow-ups across intervening queries
    """
    
    def __init__(self, similarity_threshold: float = 0.15, top_k: int = 3, db_session: Optional[AsyncSession] = None):
        """
        Initialize RAG retriever.
        
        Args:
            similarity_threshold: Minimum cosine similarity for semantic matching (0.15 for token hashing)
            top_k: Return top K results
            db_session: Optional database session for persistence
        """
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.db_session = db_session
        logger.info(f"[RAG RETRIEVER] Initialized - Pure semantic mode (threshold={similarity_threshold}, top_k={top_k}, db_backed={db_session is not None})")
    
    async def retrieve_context_for_followup(
        self,
        session_id: str,
        current_query: str,
    ) -> Optional[RAGContext]:
        """
        Retrieve context for a follow-up query using pure semantic analysis.
        
        Strategy:
        1. Generate embedding for current query via token-based hashing
        2. Find semantically similar previous queries using cosine similarity
        3. Fallback to most recent query if no semantic match (for implicit follow-ups)
        
        Semantic Embedding Mechanism:
        - Each query token is hashed: hash(token) % embedding_dim
        - This captures semantic relationships generically
        - Works for any database, any schema, any language
        - Multi-table queries: embeddings include all table/column names, so JOINs are preserved
        
        Args:
            session_id: Session ID to search within
            current_query: The follow-up query
            
        Returns:
            RAGContext with most relevant previous query, or None if nothing found
        """
        print(f"\n[RAG] Semantic retrieval for: {current_query}")
        
        # Step 1: Generate semantic embedding for current query
        # Uses token hashing - database-agnostic, works with any schema
        generator = await get_embedding_generator()
        current_embedding = await generator.generate_embedding(current_query)
        
        # Step 2: Search for semantically similar queries in session history
        store = await get_embedding_store()
        similar_queries = await store.search_similar_queries(
            session_id=session_id,
            query_embedding=current_embedding,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold,
            db_session=self.db_session,  # Pass explicitly — never stored on singleton
        )
        
        # Step 3: If semantic search finds results, return best match
        if similar_queries:
            best_match, score = similar_queries[0]
            
            print(f"[RAG] Semantic match found (score: {score:.2f})")
            print(f"[RAG]   Previous: {best_match.user_query}")
            print(f"[RAG]   SQL: {best_match.generated_sql}")
            print(f"[RAG]   Results: {best_match.result_count} rows")
            
            context_text = self._format_context(best_match, score)
            
            return RAGContext(
                is_relevant=True,
                similarity_score=score,
                previous_query=best_match.user_query,
                previous_sql=best_match.generated_sql,
                result_count=best_match.result_count,
                column_names=best_match.column_names,
                sample_row=best_match.all_result_rows[0] if best_match.all_result_rows else None,
                context_text=context_text,
            )
        
        # Step 4: Fallback - use most recent query if available
        # This handles implicit follow-ups like "only those approved" after a previous data query
        # The semantic embeddings of both queries may be low-similarity, but recent context is still valuable
        print(f"[RAG] No semantic match at threshold {self.similarity_threshold}, falling back to recency...")

        session_history = store.get_session_history(session_id)
        # After re-login/server restart, in-memory store is empty — check database
        if not session_history and self.db_session:
            print(f"[RAG] No in-memory history, checking database...")
            session_history = await store.get_session_history_from_db(
                session_id, db_session=self.db_session
            )
        if not session_history:
            print(f"[RAG] No previous queries in session")
            return None
        
        # Use most recent query as context
        best_match = session_history[-1]
        
        print(f"[RAG] Using most recent query (recency fallback):")
        print(f"[RAG]   Previous: {best_match.user_query}")
        print(f"[RAG]   SQL: {best_match.generated_sql}")
        
        # Estimate similarity based on recency
        recency_score = 0.6  # Conservative score for recency-based fallback
        context_text = self._format_context(best_match, recency_score)
        
        return RAGContext(
            is_relevant=True,
            similarity_score=recency_score,
            previous_query=best_match.user_query,
            previous_sql=best_match.generated_sql,
            result_count=best_match.result_count,
            column_names=best_match.column_names,
            sample_row=best_match.all_result_rows[0] if best_match.all_result_rows else None,
            context_text=context_text,
        )
    
    def _format_context(self, embedding: QueryEmbedding, similarity: float) -> str:
        """Format embedding context as text for LLM."""
        context = f"""
PREVIOUS QUERY CONTEXT (similarity: {similarity:.1%}):
  Your previous question: {embedding.user_query}
  Generated SQL: {embedding.generated_sql}
  Results: {embedding.result_count} rows returned
  Columns: {', '.join(embedding.column_names[:5])}{'...' if len(embedding.column_names) > 5 else ''}
  
This suggests you're asking about the same entity/table. Consider:
- Refining the filter or scope based on this context
- Keeping the same table structure for consistency
"""
        return context.strip()
    
    async def retrieve_all_context_for_session(
        self,
        session_id: str,
    ) -> List[QueryEmbedding]:
        """Get all previous queries in a session with their embeddings."""
        store = await get_embedding_store(db_session=self.db_session)
        history = store.get_session_history(session_id)
        logger.info(f"[RAG] Retrieved {len(history)} previous queries from session")
        return history


# Global instance
_rag_retriever: Optional[RAGContextRetriever] = None


async def get_rag_retriever(
    similarity_threshold: float = 0.65,
    top_k: int = 3,
    db_session: Optional[AsyncSession] = None,
) -> RAGContextRetriever:
    """
    Get or create global RAG context retriever.
    
    Args:
        similarity_threshold: Minimum cosine similarity for semantic matching
        top_k: Return top K results
        db_session: Optional database session for persistence
        
    Returns:
        RAGContextRetriever instance
    """
    global _rag_retriever
    if _rag_retriever is None:
        _rag_retriever = RAGContextRetriever(
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            db_session=db_session,
        )
    elif db_session and not _rag_retriever.db_session:
        # Update database session if provided
        _rag_retriever.db_session = db_session
    return _rag_retriever
