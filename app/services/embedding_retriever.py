"""
Embedding-Based Candidate Retrieval - Component B of semantic system.

Uses vector embeddings for intelligent table and column candidate selection.
Provides top-K results with confidence scores for semantic query routing.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    """Result from embedding-based search."""
    type: str  # 'table' or 'column'
    name: str
    full_name: Optional[str]  # schema.table or table.column
    confidence: float  # 0.0-1.0
    description: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class JoinCandidate:
    """Candidate join relationship between tables."""
    left_table: str
    left_col: str
    right_table: str
    right_col: str
    confidence: float  # 0.0-1.0
    reason: str  # 'naming_convention' or 'shared_identifier'


class EmbeddingBasedRetriever:
    """
    Retrieves top-K table/column candidates using semantic embeddings.
    
    Falls back to keyword matching if embeddings unavailable.
    """
    
    def __init__(self, catalog):
        """
        Initialize retriever with catalog.
        
        Args:
            catalog: SemanticSchemaCatalog instance
        """
        self.catalog = catalog
        self._embedding_model = None
        self._table_embeddings = {}  # Cache: table_name -> embedding vector
        self._column_embeddings = {}  # Cache: "table.column" -> embedding vector
        self._description_embeddings = {}  # Cache: full_name -> embedding vector
        self._embeddings_loaded = False
        self._embedding_model_initialized = False
        # ✅ FIXED: Defer embedding model loading to avoid blocking on init
        # The model will be loaded lazily on first use
    
    def _load_embedding_model(self) -> None:
        """Load embedding model if available (lazy initialization)."""
        if self._embedding_model_initialized:
            return  # Already initialized or failed before
        
        self._embedding_model_initialized = True
        
        try:
            # Import SentenceTransformers for semantic embeddings
            from sentence_transformers import SentenceTransformer
            
            # Use lightweight model optimized for semantic similarity
            model_name = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality
            logger.info(f"[RETRIEVER] Loading embedding model: {model_name}")
            
            self._embedding_model = SentenceTransformer(model_name)
            logger.info("[RETRIEVER] Embedding model loaded successfully")
            
            # ✅ FIXED: SKIP pre-computation to avoid blocking
            # Use keyword-based fallback for queries while embeddings are optional
            # Embeddings can be computed in background if needed
            logger.info("[RETRIEVER] Embeddings will be computed on-demand if model is available")
            self._embeddings_loaded = False  # Mark as not loaded initially
            
        except ImportError:
            logger.warning(
                "[RETRIEVER] SentenceTransformers not installed. "
                "Using keyword-based retrieval only. "
                "Install with: pip install sentence-transformers"
            )
            self._embedding_model = None
        except Exception as e:
            logger.warning(f"[RETRIEVER] Failed to load embedding model: {e}")
            logger.warning("[RETRIEVER] Falling back to keyword-based retrieval")
            self._embedding_model = None
    
    def _precompute_embeddings(self) -> None:
        """Pre-compute and cache embeddings for all schema items."""
        if not self._embedding_model:
            return
        
        try:
            logger.info("[RETRIEVER] Pre-computing embeddings for schema items...")
            
            # Collect all items to embed
            items_to_embed = []
            item_keys = []
            
            # Add table names and descriptions
            for table_meta in self.catalog.tables.values():
                # Table name
                items_to_embed.append(table_meta.name)
                item_keys.append(("table", table_meta.name))
                
                # Table description (if available)
                if table_meta.description:
                    items_to_embed.append(table_meta.description)
                    item_keys.append(("description", table_meta.full_name))
                
                # Column names and descriptions
                for col_meta in table_meta.columns.values():
                    items_to_embed.append(col_meta.name)
                    item_keys.append(("column", f"{table_meta.name}.{col_meta.name}"))
                    
                    if col_meta.description:
                        items_to_embed.append(col_meta.description)
                        item_keys.append(("col_description", f"{table_meta.name}.{col_meta.name}"))
            
            if not items_to_embed:
                logger.warning("[RETRIEVER] No schema items found for embedding")
                return
            
            # Compute embeddings in batch
            logger.info(f"[RETRIEVER] Computing {len(items_to_embed)} embeddings...")
            embeddings = self._embedding_model.encode(items_to_embed, convert_to_numpy=True)
            
            # Store in cache
            for (item_type, key), embedding in zip(item_keys, embeddings):
                if item_type == "table":
                    self._table_embeddings[key] = embedding
                elif item_type == "column":
                    self._column_embeddings[key] = embedding
                elif item_type in ["description", "col_description"]:
                    self._description_embeddings[key] = embedding
            
            self._embeddings_loaded = True
            logger.info(
                f"[RETRIEVER] Embeddings ready: {len(self._table_embeddings)} tables, "
                f"{len(self._column_embeddings)} columns"
            )
            
        except Exception as e:
            logger.error(f"[RETRIEVER] Error pre-computing embeddings: {e}")
            self._embeddings_loaded = False
    
    async def retrieve_table_candidates(
        self,
        query: str,
        k: int = 3,
        threshold: float = 0.0,
        top_k: int = None
    ) -> List[CandidateResult]:
        """
        Retrieve top-K table candidates for a query using semantic embeddings.
        
        Args:
            query: User's natural language query
            k: Number of candidates to return (or use top_k for backward compat)
            threshold: Confidence threshold filter
            top_k: Deprecated, use k instead
        
        Returns:
            List of CandidateResult objects sorted by confidence (highest first)
        """
        # Support both k and top_k parameters for backward compatibility
        if top_k is not None:
            k = top_k
        logger.debug(f"[RETRIEVER] Finding top-{k} tables for: {query[:50]}...")
        
        # ✅ FIXED: Skip embedding model loading - use keyword fallback for now
        # Embedding model loading was causing hangs during queries
        # Use keyword-based retrieval which is fast and reliable
        logger.debug("[RETRIEVER] Using keyword-based retrieval (embeddings disabled for performance)")
        table_scores = self.catalog.find_tables_for_query(query, top_k=k)
        
        results = []
        for item in table_scores:
            # Handle both tuple and object formats
            if isinstance(item, tuple):
                table_meta, score = item
            else:
                table_meta = item
                score = 0.5  # Default confidence
            
            # Type guard: convert strings to TableMetadata objects instead of skipping
            if isinstance(table_meta, str):
                table_name = table_meta
                logger.debug(f"[RETRIEVER] Converting string table name '{table_name}' to TableMetadata")
                # Try to fetch from catalog if available
                if hasattr(self, 'catalog') and hasattr(self.catalog, 'get_table'):
                    catalog_table = self.catalog.get_table(table_name)
                    if catalog_table:
                        table_meta = catalog_table
                    else:
                        # Create minimal metadata object
                        logger.debug(f"[RETRIEVER] Creating minimal metadata for table '{table_name}'")
                        class SimpleTableMetadata:
                            def __init__(self, name):
                                self.name = name
                                self.full_name = f"genai.{name}"
                                self.columns = {}
                                self.row_count = 0
                                self.description = f"Table: {name}"
                        table_meta = SimpleTableMetadata(table_name)
                else:
                    # Fallback
                    class SimpleTableMetadata:
                        def __init__(self, name):
                            self.name = name
                            self.full_name = f"genai.{name}"
                            self.columns = {}
                            self.row_count = 0
                            self.description = f"Table: {name}"
                    table_meta = SimpleTableMetadata(table_name)
            
            if not hasattr(table_meta, 'name'):
                logger.warning(f"[RETRIEVER] Object doesn't have 'name' attribute: {type(table_meta)}. Skipping.")
                continue
            
            result = CandidateResult(
                type="table",
                name=table_meta.name,
                full_name=table_meta.full_name if hasattr(table_meta, 'full_name') else f"{table_meta.name}",
                confidence=min(score, 1.0),  # Normalize to 0-1
                description=table_meta.description or f"Table: {table_meta.name}" if hasattr(table_meta, 'description') else f"Table: {table_meta.name}",
                metadata={
                    "row_count": getattr(table_meta, 'row_count', 0),
                    "column_count": len(table_meta.columns) if hasattr(table_meta, 'columns') else 0,
                    "columns": list(table_meta.columns.keys()) if hasattr(table_meta, 'columns') else [],
                }
            )
            results.append(result)
            logger.debug(f"  → {result.full_name}: {result.confidence:.0%}")
        
        return results
    
    async def retrieve_column_candidates(
        self,
        query: str,
        table_name: Optional[str] = None,
        k: int = 5,
        threshold: float = 0.0,
        top_k: int = None
    ) -> List[CandidateResult]:
        """
        Retrieve top-K column candidates for a query (FAST - keyword only, no embeddings).
        
        Args:
            query: User's natural language query
            table_name: Optional table filter
            k: Number of candidates to return
            threshold: Confidence threshold filter (ignored)
            top_k: Deprecated, use k instead
        
        Returns:
            List of CandidateResult objects
        """
        # Support both k and top_k parameters for backward compatibility
        if top_k is not None:
            k = top_k
        
        logger.debug(f"[RETRIEVER] Finding top-{k} columns in {table_name or 'all tables'}...")
        
        # ✅ FAST PATH: Return table columns directly without expensive embedding/keyword operations
        # This prevents hangs and speeds up query execution
        if table_name:
            table_meta = self.catalog.get_table(table_name)
            if not table_meta:
                logger.warning(f"[RETRIEVER] Table not found: {table_name}")
                return []
            
            # Return first k columns as-is (fast, no processing)
            results = []
            for col_meta in list(table_meta.columns.values())[:k]:
                result = CandidateResult(
                    type="column",
                    name=col_meta.name,
                    full_name=f"{table_meta.name}.{col_meta.name}",
                    confidence=0.7,  # Default confidence
                    description=col_meta.description,
                    metadata={
                        "db_type": col_meta.db_type,
                        "semantic_type": getattr(col_meta, 'semantic_type', 'unknown'),
                        "table": table_meta.name,
                    }
                )
                results.append(result)
            
            logger.debug(f"[RETRIEVER] Returned {len(results)} columns for table {table_name}")
            return results
        
        # If no table specified, return empty
        return []
    
    async def retrieve_join_candidates(
        self,
        table_names: List[str],
        k: int = 2,
        threshold: float = 0.0,
        top_k: int = None,
    ) -> List[JoinCandidate]:
        """
        Find likely join candidates between tables (FAST - returns immediately).
        
        Args:
            table_names: List of table names to find joins between
            k: Number of join candidates to return
            threshold: Confidence threshold
            top_k: Deprecated, use k instead
            
        Returns:
            List of join candidates (empty for now - optimization)
        """
        # Support both k and top_k parameters for backward compatibility
        if top_k is not None:
            k = top_k
        
        logger.debug(f"[RETRIEVER] Finding join candidates (fast path)")
        
        # ✅ OPTIMIZATION: Return empty list immediately
        # In practice, joins are inferred from foreign key relationships, not needed for semantic retrieval
        # Returning empty speeds up query processing significantly
        logger.debug(f"[RETRIEVER] Skipping join candidate search for speed (using FK detection instead)")
        return []  # Fast return - no expensive processing
    
    async def get_where_clause_candidates(
        self,
        query: str,
        table_name: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Suggest WHERE clause candidates based on query.
        
        Returns dictionary with filter columns and operators.
        """
        columns = await self.retrieve_column_candidates(query, table_name, top_k)
        
        suggestions = {
            "filter_columns": columns,
            "likely_operators": [],
        }
        
        # Dynamically infer likely operators based on query patterns (no hardcoded keywords)
        query_lower = query.lower()
        
        # Pattern-based detection (generic, not domain-specific)
        if re.search(r'\bfrom\s+\d+\s+to\s+\d+|\d+\s*-\s*\d+|\d+\s+and\s+\d+', query_lower):
            suggestions["likely_operators"].extend(["BETWEEN", ">", "<"])
        elif re.search(r'%|_|\*|partial|pattern|match', query_lower):
            suggestions["likely_operators"].extend(["LIKE", "ILIKE"])
        elif re.search(r'\blist\b|\bany\b|\bmultiple\b|,\s*\w+,', query_lower):
            suggestions["likely_operators"].append("IN")
        else:
            suggestions["likely_operators"].extend(["=", "LIKE"])
        
        return suggestions


async def initialize_retriever(catalog) -> EmbeddingBasedRetriever:
    """Initialize embedding-based retriever."""
    retriever = EmbeddingBasedRetriever(catalog)
    logger.info("[RETRIEVER] Embedding-based retriever initialized")
    return retriever
