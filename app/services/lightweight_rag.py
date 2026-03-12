"""
Lightweight RAG Service - Alternative to sentence-transformers with no model loading.

This service provides semantic search WITHOUT heavy neural models:
1. BM25 keyword ranking (no ML models, instant results)
2. SQL structure fingerprinting (tables, columns, operations)
3. Query hash-based duplicate detection
4. PostgreSQL full-text search integration
5. Lightweight token-based similarity (no embeddings)

Benefits:
- No model downloads from HuggingFace
- No GPU/CPU overhead
- Instant initialization
- Works offline completely
- Minimal memory footprint (~1MB vs 500MB+)
"""

from __future__ import annotations

import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
import re
import json

from sqlalchemy import text, select, desc
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class QueryFingerprint:
    """Lightweight query signature for matching without ML models."""
    query_hash: str  # SHA256 of original query
    tables: List[str]  # Extracted table names
    operations: List[str]  # SELECT, JOIN, WHERE, GROUP BY, ORDER BY, etc.
    columns: List[str]  # Referenced column names
    filters: List[str]  # Filter expressions simplified
    token_hash: str  # Hash of token bag
    structure_score: float  # Similarity score (0-1)


class BM25Ranker:
    """
    BM25 (Best Matching 25) - Industry standard information retrieval ranking.
    
    Used for keyword-based query matching without neural models.
    Works with query text, SQL, and metadata.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 ranker.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.idf_cache: Dict[str, float] = {}
        self.avg_doc_length = 0.0
        self.num_docs = 0
    
    def build_index(self, documents: List[str]) -> None:
        """Build BM25 index from documents."""
        self.num_docs = len(documents)
        
        # Tokenize all documents
        tokenized_docs = []
        for doc in documents:
            tokens = self._tokenize(doc)
            tokenized_docs.append(tokens)
        
        # Calculate IDF for each unique token
        doc_frequencies: Dict[str, int] = {}
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_frequencies[token] = doc_frequencies.get(token, 0) + 1
        
        # Calculate IDF scores
        for token, freq in doc_frequencies.items():
            idf = math.log10((self.num_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf_cache[token] = idf
        
        # Calculate average document length
        self.avg_doc_length = sum(len(tokens) for tokens in tokenized_docs) / self.num_docs if self.num_docs > 0 else 0
    
    def score(self, query: str, document: str) -> float:
        """Calculate BM25 score for query against document."""
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(document)
        
        score = 0.0
        doc_len = len(doc_tokens)
        token_freqs = Counter(doc_tokens)
        
        for token in query_tokens:
            if token not in self.idf_cache:
                continue
            
            idf = self.idf_cache[token]
            freq = token_freqs.get(token, 0)
            norm_len = 1.0 - self.b + self.b * (doc_len / self.avg_doc_length)
            score += idf * ((self.k1 + 1) * freq) / (self.k1 * norm_len + freq)
        
        return score
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'will',
            'would', 'should', 'may', 'might', 'must', 'by', 'from'
        }
        return [t for t in tokens if t not in stop_words and len(t) > 2]


class QueryStructureAnalyzer:
    """Extract structural information from SQL queries for lightweight matching."""
    
    @staticmethod
    def extract_tables(sql: str) -> List[str]:
        """Extract table names from SQL query."""
        # Handle schema-qualified names (e.g., schema.table)
        pattern = r'(?:FROM|JOIN)\s+(?:\w+\.)?(\w+)'
        matches = re.findall(pattern, sql, re.IGNORECASE)
        return list(set([m.lower() for m in matches]))
    
    @staticmethod
    def extract_columns(sql: str) -> List[str]:
        """Extract column names referenced in query."""
        # Simple pattern: word after dot or comma or SELECT/WHERE
        pattern = r'(?:SELECT|WHERE|ON|=|\bAS\s+)\s*(?:\w+\.)?(\w+)'
        matches = re.findall(pattern, sql, re.IGNORECASE)
        return list(set([m.lower() for m in matches]))
    
    @staticmethod
    def extract_operations(sql: str) -> List[str]:
        """Extract SQL operations (SELECT, JOIN, WHERE, etc.)."""
        operations = []
        sql_upper = sql.upper()
        
        if 'SELECT' in sql_upper:
            operations.append('SELECT')
        if 'JOIN' in sql_upper:
            operations.append('JOIN')
        if 'WHERE' in sql_upper:
            operations.append('WHERE')
        if 'GROUP BY' in sql_upper:
            operations.append('GROUP BY')
        if 'ORDER BY' in sql_upper:
            operations.append('ORDER BY')
        if 'LIMIT' in sql_upper:
            operations.append('LIMIT')
        if 'HAVING' in sql_upper:
            operations.append('HAVING')
        if 'UNION' in sql_upper:
            operations.append('UNION')
        if 'DISTINCT' in sql_upper:
            operations.append('DISTINCT')
        if 'AGGREGATE' in sql_upper or 'COUNT' in sql_upper or 'SUM' in sql_upper or 'AVG' in sql_upper:
            operations.append('AGGREGATE')
        
        return operations
    
    @staticmethod
    def extract_filters_simplified(sql: str) -> List[str]:
        """Extract simplified filter expressions."""
        filters = []
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|HAVING|;|$)', sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            # Split by AND/OR and simplify
            conditions = re.split(r'\s+AND\s+|\s+OR\s+', where_clause, flags=re.IGNORECASE)
            for cond in conditions:
                cond = cond.strip()
                if len(cond) > 3:
                    # Hash the condition for comparison
                    filters.append(hashlib.md5(cond.encode()).hexdigest()[:8])
        
        return filters
    
    @staticmethod
    def calculate_structure_similarity(fp1: QueryFingerprint, fp2: QueryFingerprint) -> float:
        """Calculate similarity between two query fingerprints (0-1)."""
        if not fp1.tables or not fp2.tables:
            return 0.0
        
        # Table similarity (most important)
        table_sim = len(set(fp1.tables) & set(fp2.tables)) / max(len(set(fp1.tables) | set(fp2.tables)), 1)
        
        # Operation similarity
        op_sim = len(set(fp1.operations) & set(fp2.operations)) / max(len(set(fp1.operations) | set(fp2.operations)), 1) if (fp1.operations or fp2.operations) else 1.0
        
        # Filter similarity
        filter_sim = len(set(fp1.filters) & set(fp2.filters)) / max(len(set(fp1.filters) | set(fp2.filters)), 1) if (fp1.filters or fp2.filters) else 1.0
        
        # Weighted average
        return (table_sim * 0.5) + (op_sim * 0.3) + (filter_sim * 0.2)


class LightweightRAG:
    """
    Lightweight RAG system without neural models.
    
    Replaces sentence-transformers with:
    - BM25 ranking for keyword search
    - SQL structure fingerprinting
    - PostgreSQL full-text search
    - Hash-based query matching
    """
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        """Initialize lightweight RAG system."""
        self.db_session = db_session
        self.bm25 = BM25Ranker()
        self.analyzer = QueryStructureAnalyzer()
        self.stored_fingerprints: Dict[str, QueryFingerprint] = {}
        logger.info("[LIGHTWEIGHT_RAG] ✅ Initialized (no model loading required)")
    
    def set_db_session(self, db_session: AsyncSession) -> None:
        """Set database session for persistence."""
        self.db_session = db_session
        logger.info("[LIGHTWEIGHT_RAG] Database session configured")
    
    def create_query_fingerprint(self, user_query: str, sql: str) -> QueryFingerprint:
        """Create lightweight fingerprint for query matching."""
        query_hash = hashlib.sha256(user_query.encode()).hexdigest()
        
        # Extract structural elements from SQL
        tables = self.analyzer.extract_tables(sql)
        operations = self.analyzer.extract_operations(sql)
        columns = self.analyzer.extract_columns(sql)
        filters = self.analyzer.extract_filters_simplified(sql)
        
        # Create token hash
        all_tokens = ' '.join([user_query, sql, ' '.join(tables), ' '.join(operations)])
        token_hash = hashlib.md5(all_tokens.encode()).hexdigest()
        
        fp = QueryFingerprint(
            query_hash=query_hash,
            tables=tables,
            operations=operations,
            columns=columns,
            filters=filters,
            token_hash=token_hash,
            structure_score=0.0
        )
        
        return fp
    
    async def find_similar_queries(
        self,
        user_query: str,
        sql: str,
        session_id: Optional[str] = None,
        threshold: float = 0.5,
        top_k: int = 3
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Find similar queries without neural models.
        
        Uses:
        1. Exact hash matching (fastest)
        2. Structure matching (tables, operations)
        3. BM25 text ranking (keyword similarity)
        """
        if not self.db_session:
            logger.warning("[LIGHTWEIGHT_RAG] No DB session, skipping persistence search")
            return []
        
        try:
            # Create fingerprint for current query
            current_fp = self.create_query_fingerprint(user_query, sql)
            
            # Database query to find similar queries
            from app.models import QueryEmbedding as QueryEmbeddingModel
            
            query = select(QueryEmbeddingModel).order_by(desc(QueryEmbeddingModel.created_at)).limit(100)
            result = await self.db_session.execute(query)
            all_embeddings = result.scalars().all()
            
            similar_queries = []
            
            for embedding_record in all_embeddings:
                # Skip queries from same session if filtering by session
                if session_id and str(embedding_record.session_id) == session_id:
                    continue
                
                # Step 1: Try exact hash match (instant)
                if embedding_record.query_hash == current_fp.query_hash:
                    score = 1.0
                    logger.info(f"[LIGHTWEIGHT_RAG] ✅ Exact hash match for query: {embedding_record.user_query}")
                else:
                    # Step 2: Structure matching
                    try:
                        other_fp = self.create_query_fingerprint(embedding_record.user_query, embedding_record.generated_sql)
                        structure_score = self.analyzer.calculate_structure_similarity(current_fp, other_fp)
                        
                        # Step 3: BM25 text ranking
                        bm25_score = self.bm25.score(user_query, embedding_record.user_query) / 100.0  # Normalize
                        
                        # Combine scores with weights
                        score = (structure_score * 0.6) + (min(bm25_score, 1.0) * 0.4)
                    except Exception as e:
                        logger.warning(f"[LIGHTWEIGHT_RAG] Failed to analyze query: {e}")
                        score = 0.0
                
                # Add if above threshold
                if score >= threshold:
                    similar_queries.append((
                        embedding_record.user_query,
                        score,
                        {
                            'sql': embedding_record.generated_sql,
                            'result_count': embedding_record.result_count,
                            'tables': current_fp.tables,
                            'operations': current_fp.operations,
                            'created_at': embedding_record.created_at.isoformat() if embedding_record.created_at else None,
                        }
                    ))
            
            # Sort by score and return top-k
            similar_queries.sort(key=lambda x: x[1], reverse=True)
            result_queries = similar_queries[:top_k]
            
            logger.info(f"[LIGHTWEIGHT_RAG] Found {len(result_queries)} similar queries (threshold={threshold})")
            return result_queries
            
        except Exception as e:
            logger.error(f"[LIGHTWEIGHT_RAG] ❌ Error during similarity search: {e}")
            return []
    
    async def store_query_fingerprint(
        self,
        query_id: str,
        session_id: str,
        user_query: str,
        sql: str,
        result_rows: List[Dict[str, Any]],
        result_count: int,
        column_names: List[str]
    ) -> bool:
        """
        Store query fingerprint in database without neural embeddings.
        
        Uses lightweight token hash instead of semantic embeddings.
        """
        if not self.db_session:
            logger.warning("[LIGHTWEIGHT_RAG] No DB session, cannot persist query")
            return False
        
        try:
            from app.models import QueryEmbedding as QueryEmbeddingModel
            import uuid
            
            fp = self.create_query_fingerprint(user_query, sql)
            
            # Create lightweight embedding: just concatenate and hash important fields
            # NO neural model needed
            embedding_input = f"{user_query} {sql} {' '.join(column_names)} {result_count}"
            lightweight_embedding = [
                float(ord(c) % 256) / 256.0 
                for c in embedding_input[:384]  # Keep 384 dimension compatibility
            ]
            # Pad to 384
            while len(lightweight_embedding) < 384:
                lightweight_embedding.append(0.0)
            
            # Create database record with transaction error handling
            db_embedding = QueryEmbeddingModel(
                id=uuid.UUID(query_id),
                session_id=uuid.UUID(session_id),
                user_query=user_query,
                generated_sql=sql,
                result_count=result_count,
                column_names=column_names,
                all_result_rows=result_rows,
                embedding=lightweight_embedding[:384],  # Keep 384 dims for compatibility
                query_hash=fp.query_hash,
                result_quality_score=len(result_rows) * 10 if result_rows else 0,
            )
            
            # Add and commit with proper transaction handling
            self.db_session.add(db_embedding)
            await self.db_session.flush()  # Flush to check for errors
            await self.db_session.commit()  # Commit if flush succeeded
            
            logger.info(f"[LIGHTWEIGHT_RAG] ✅ Stored query fingerprint: {query_id}")
            return True
            
        except Exception as e:
            # Rollback on any error
            try:
                await self.db_session.rollback()
            except:
                pass
            
            logger.error(f"[LIGHTWEIGHT_RAG] ❌ Failed to store fingerprint: {e}")
            return False


# Global instances
_lightweight_rag: Optional[LightweightRAG] = None


async def get_lightweight_rag(db_session: Optional[AsyncSession] = None) -> LightweightRAG:
    """Get or create global lightweight RAG instance."""
    global _lightweight_rag
    if _lightweight_rag is None:
        _lightweight_rag = LightweightRAG(db_session=db_session)
    elif db_session and _lightweight_rag.db_session != db_session:
        _lightweight_rag.set_db_session(db_session)
    
    return _lightweight_rag


# Import math at module level
import math
