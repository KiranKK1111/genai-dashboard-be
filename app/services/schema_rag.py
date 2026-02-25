"""
Schema RAG Service - Retrieves relevant schema for user queries.

Instead of dumping full DB schema to LLM, this service:
1. Embeds user query
2. Finds top-k similar tables/columns from schema
3. Returns compact schema context to LLM

This is the "RAG" (Retrieval Augmented Generation) pattern applied to schema.
Keeps LLM context clean and focused.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..services.embedding_service import EmbeddingService
from .schema_discovery import SchemaCatalog, TableInfo, ColumnInfo

logger = logging.getLogger(__name__)


class SchemaRAG:
    """
    Retrieves relevant schema context for user queries.
    
    Uses embeddings to find tables/columns semantically similar to user intent.
    """
    
    def __init__(self, catalog: SchemaCatalog, embeddings: EmbeddingService):
        self.catalog = catalog
        self.embeddings = embeddings
        self._table_embeddings: Dict[str, np.ndarray] = {}
        self._column_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    
    async def initialize(self) -> None:
        """Pre-compute embeddings for all tables and columns."""
        logger.info("Initializing Schema RAG embeddings...")
        
        tables = self.catalog.get_all_tables()
        
        for table_name, table_info in tables.items():
            # Embed table name + signature
            table_desc = f"{table_name} with columns: " + ", ".join(
                sorted(table_info.columns.keys())[:10]
            )
            embedding = await self.embeddings.embed_text(table_desc)
            self._table_embeddings[table_name] = embedding
            
            # Embed columns
            self._column_embeddings[table_name] = {}
            for col_name, col_info in table_info.columns.items():
                col_desc = f"{col_name} ({col_info.data_type})"
                col_embedding = await self.embeddings.embed_text(col_desc)
                self._column_embeddings[table_name][col_name] = col_embedding
        
        logger.info(f"✓ Schema RAG initialized: {len(tables)} tables embedded")
    
    async def retrieve_relevant_tables(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find top-k tables relevant to user query.
        
        Args:
            query: User's natural language query
            top_k: Number of tables to return
        
        Returns:
            List of (table_name, similarity_score) tuples
        """
        query_embedding = await self.embeddings.embed_text(query)
        
        similarities: Dict[str, float] = {}
        
        for table_name, table_embedding in self._table_embeddings.items():
            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, table_embedding)
            similarities[table_name] = similarity
        
        # Sort and return top-k
        ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    
    async def retrieve_relevant_columns(
        self,
        query: str,
        table_name: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find top-k columns in a table relevant to query.
        
        Args:
            query: User's natural language query
            table_name: Table to search within
            top_k: Number of columns to return
        
        Returns:
            List of (column_name, similarity_score) tuples
        """
        if table_name not in self._column_embeddings:
            return []
        
        query_embedding = await self.embeddings.embed_text(query)
        col_embeddings = self._column_embeddings[table_name]
        
        similarities: Dict[str, float] = {}
        
        for col_name, col_embedding in col_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, col_embedding)
            similarities[col_name] = similarity
        
        ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    
    async def retrieve_schema_for_query(
        self,
        query: str,
        top_tables: int = 3
    ) -> str:
        """
        Get compact schema context for a user query.
        
        This returns only the most relevant tables/columns, not the full schema.
        Reduces LLM context while keeping it relevant.
        
        Args:
            query: User's natural language query
            top_tables: Number of tables to include
        
        Returns:
            Formatted schema string for LLM context
        """
        # Get relevant tables
        relevant_tables = await self.retrieve_relevant_tables(query, top_k=top_tables)
        
        lines = []
        lines.append("Relevant Database Schema:")
        lines.append("-" * 60)
        
        for table_name, table_score in relevant_tables:
            table_info = self.catalog.get_table(table_name)
            if not table_info:
                continue
            
            lines.append(f"\nTable: {table_name} (relevance: {table_score:.2f}, {table_info.row_count:,} rows)")
            
            # Get relevant columns for this table
            relevant_cols = await self.retrieve_relevant_columns(query, table_name, top_k=8)
            
            for col_name, col_score in relevant_cols:
                col_info = table_info.columns.get(col_name)
                if col_info:
                    pk_marker = " [PRIMARY KEY]" if col_info.is_primary_key else ""
                    lines.append(f"  - {col_name}: {col_info.data_type}{pk_marker}")
            
            # Add primary keys info
            if table_info.primary_keys:
                lines.append(f"  Primary Key: {', '.join(table_info.primary_keys)}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


async def create_schema_rag(
    catalog: SchemaCatalog,
    embeddings: EmbeddingService
) -> SchemaRAG:
    """Factory function to create and initialize Schema RAG."""
    rag = SchemaRAG(catalog, embeddings)
    await rag.initialize()
    return rag
