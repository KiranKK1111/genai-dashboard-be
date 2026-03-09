"""
RAG Context Optimizer - Multi-factor relevance scoring.

Instead of: score = BM25(query, doc)
Use: score = 0.3*semantic + 0.25*structure + 0.2*entity + 0.15*filter + 0.1*temporal

This significantly improves RAG retrieval quality and relevance.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class RAGScoreFactors:
    """Individual scoring factors for RAG retrieval."""
    semantic_sim: float  # BM25 semantic match (0-1)
    structure_sim: float  # SQL structure match (0-1)
    entity_overlap: float  # Table/column overlap (0-1)
    filter_compat: float  # Filter compatibility (0-1)
    temporal_relevance: float  # Recency score (0-1)
    total_score: float  # Weighted sum


class RAGContextOptimizer:
    """
    Enhanced RAG scoring with multiple factors.
    Improves retrieval quality beyond simple BM25.
    """
    
    def __init__(
        self,
        semantic_weight: float = 0.30,
        structure_weight: float = 0.25,
        entity_weight: float = 0.20,
        filter_weight: float = 0.15,
        temporal_weight: float = 0.10,
        half_life_days: float = 3.0,
    ):
        """
        Initialize RAG optimizer with custom weights.
        
        Args:
            semantic_weight: BM25 text similarity
            structure_weight: SQL structure match
            entity_weight: Table/column overlap
            filter_weight: Filter compatibility
            temporal_weight: Recency weighting
            half_life_days: Temporal decay half-life
        """
        self.weights = {
            "semantic": semantic_weight,
            "structure": structure_weight,
            "entity": entity_weight,
            "filter": filter_weight,
            "temporal": temporal_weight,
        }
        self.half_life_days = half_life_days
    
    async def score_context_relevance(
        self,
        current_query: str,
        stored_query: str,
        stored_sql: str,
        creation_timestamp: datetime,
    ) -> RAGScoreFactors:
        """
        Calculate multi-factor relevance score.
        
        Args:
            current_query: User's current natural language query
            stored_query: Previous query from RAG store
            stored_sql: SQL that was generated for stored_query
            creation_timestamp: When stored_query was created
        
        Returns:
            RAGScoreFactors with individual scores and total
        """
        
        # FACTOR 1: Semantic Similarity (BM25)
        semantic_score = await self._calculate_semantic_similarity(
            current_query,
            stored_query
        )
        
        # FACTOR 2: SQL Structure Similarity
        structure_score = await self._calculate_structure_similarity(
            current_query,
            stored_sql
        )
        
        # FACTOR 3: Entity Overlap (tables/columns)
        entity_score = await self._calculate_entity_overlap(
            current_query,
            stored_sql
        )
        
        # FACTOR 4: Filter Compatibility
        filter_score = await self._calculate_filter_compatibility(
            current_query,
            stored_sql
        )
        
        # FACTOR 5: Temporal Relevance (recency)
        temporal_score = await self._calculate_temporal_relevance(
            creation_timestamp
        )
        
        # Weighted total
        total_score = (
            semantic_score * self.weights["semantic"] +
            structure_score * self.weights["structure"] +
            entity_score * self.weights["entity"] +
            filter_score * self.weights["filter"] +
            temporal_score * self.weights["temporal"]
        )
        
        factors = RAGScoreFactors(
            semantic_sim=semantic_score,
            structure_sim=structure_score,
            entity_overlap=entity_score,
            filter_compat=filter_score,
            temporal_relevance=temporal_score,
            total_score=total_score,
        )
        
        return factors
    
    async def _calculate_semantic_similarity(
        self,
        query1: str,
        query2: str
    ) -> float:
        """
        BM25 semantic similarity.
        """
        try:
            from .lightweight_rag import BM25Ranker
            
            ranker = BM25Ranker()
            ranker.build_index([query2])
            score = ranker.score(query1, query2)
            
            # Normalize to 0-1
            return min(max(score / 100.0, 0.0), 1.0)
        except Exception as e:
            logger.warning(f"[SEMANTIC] Error: {e}")
            return 0.0
    
    async def _calculate_structure_similarity(
        self,
        query: str,
        sql: str
    ) -> float:
        """
        SQL structure similarity.
        Do they have same operations, joins, aggregations?
        """
        
        # Extract structure from natural language query
        query_lower = query.lower()
        query_keywords = {
            "has_join": any(j in query_lower for j in ["join", "with", "and their", "including"]),
            "has_where": any(w in query_lower for w in ["where", "filter", "only", "from"]),
            "has_aggregate": any(a in query_lower for a in ["total", "count", "sum", "average", "mean"]),
            "has_group": any(g in query_lower for g in ["group", "by", "grouped"]),
            "has_order": any(o in query_lower for o in ["sort", "order", "top", "least"]),
        }
        
        # Extract structure from SQL
        sql_upper = sql.upper()
        sql_keywords = {
            "has_join": "JOIN" in sql_upper or "LEFT" in sql_upper,
            "has_where": "WHERE" in sql_upper,
            "has_aggregate": "GROUP BY" in sql_upper or any(
                agg in sql_upper for agg in ["COUNT", "SUM", "AVG", "MAX", "MIN"]
            ),
            "has_group": "GROUP BY" in sql_upper,
            "has_order": "ORDER BY" in sql_upper,
        }
        
        # Calculate overlap
        matches = sum(
            1 for k in query_keywords
            if query_keywords[k] == sql_keywords[k]
        )
        
        structure_score = matches / len(query_keywords)
        
        logger.debug(
            f"[STRUCTURE] Query={query_keywords}, SQL={sql_keywords}, "
            f"Score={structure_score:.2f}"
        )
        
        return structure_score
    
    async def _calculate_entity_overlap(
        self,
        query: str,
        sql: str
    ) -> float:
        """
        Entity overlap: how many tables/columns in common?
        """
        
        # Extract table names from SQL
        table_pattern = r"(?:FROM|JOIN|INNER|LEFT|RIGHT|FULL|CROSS)\s+(\w+)"
        sql_tables = set(re.findall(table_pattern, sql, re.IGNORECASE))
        
        # Extract table names from query (heuristic)
        query_lower = query.lower()
        possible_tables = set()
        
        # Dynamic table matching: find query words that match table names from SQL
        # NO HARDCODED TABLE KEYWORDS - match against actual tables in SQL
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        for sql_table in sql_tables:
            table_lower = sql_table.lower()
            # Check if table name or its singular/plural form appears in query
            if table_lower in query_words:
                possible_tables.add(table_lower)
            elif table_lower.rstrip('s') in query_words:  # plural -> singular
                possible_tables.add(table_lower)
            elif table_lower + 's' in query_words:  # singular -> plural
                possible_tables.add(table_lower)
        
        # Remove duplicates
        sql_tables = {t.lower() for t in sql_tables}
        possible_tables = {t.lower() for t in possible_tables}
        
        # Calculate Jaccard similarity
        if sql_tables or possible_tables:
            overlap = len(sql_tables & possible_tables)
            union = len(sql_tables | possible_tables)
            entity_score = overlap / union if union > 0 else 0.0
        else:
            entity_score = 0.0
        
        logger.debug(
            f"[ENTITY] SQL tables={sql_tables}, Query tables={possible_tables}, "
            f"Score={entity_score:.2f}"
        )
        
        return entity_score
    
    async def _calculate_filter_compatibility(
        self,
        query: str,
        sql: str
    ) -> float:
        """
        Filter compatibility: can we reuse filters?
        """
        
        # Extract WHERE clause from SQL
        where_match = re.search(
            r"WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)",
            sql,
            re.IGNORECASE | re.DOTALL
        )
        sql_filters = where_match.group(1).strip() if where_match else ""
        
        # Heuristic: look for common filter patterns in query (not domain-specific)
        query_lower = query.lower()
        filter_keywords = [
            "status", "type", "category",
            "month", "year", "date", "between",
            "only", "except", "exclude", "include",
            "where", "filter", "having"
        ]
        
        query_has_filters = any(kw in query_lower for kw in filter_keywords)
        sql_has_filters = len(sql_filters) > 5
        
        # Compatibility: if both have or don't have filters
        if query_has_filters == sql_has_filters:
            filter_score = 0.8
        else:
            filter_score = 0.4
        
        logger.debug(
            f"[FILTER] Query filters={query_has_filters}, SQL filters={sql_has_filters}, "
            f"Score={filter_score:.2f}"
        )
        
        return filter_score
    
    async def _calculate_temporal_relevance(
        self,
        creation_timestamp: datetime,
    ) -> float:
        """
        Temporal relevance: recent queries are more valuable.
        Uses exponential decay: score = 0.5^(age/half_life)
        """
        
        age_days = (datetime.now() - creation_timestamp).days
        
        # Exponential decay
        decay_score = 0.5 ** (age_days / self.half_life_days)
        
        # Cap at reasonable bounds
        temporal_score = max(min(decay_score, 1.0), 0.0)
        
        logger.debug(
            f"[TEMPORAL] Age={age_days} days, Half-life={self.half_life_days}, "
            f"Score={temporal_score:.2f}"
        )
        
        return temporal_score
    
    async def rank_context_results(
        self,
        current_query: str,
        context_results: List[Dict],
        top_k: int = 5,
        minimum_score_threshold: float = 0.5,
    ) -> List[Tuple[Dict, RAGScoreFactors]]:
        """
        Rank and filter context results by multi-factor score.
        
        Args:
            current_query: User's current query
            context_results: List of stored queries from RAG
            top_k: Top K results to return
            minimum_score_threshold: Minimum score to include
        
        Returns:
            List of (result, scores) tuples sorted by total score
        """
        
        scored_results = []
        
        for result in context_results:
            try:
                factors = await self.score_context_relevance(
                    current_query,
                    result.get("user_query", ""),
                    result.get("generated_sql", ""),
                    result.get("created_at", datetime.now()),
                )
                
                if factors.total_score >= minimum_score_threshold:
                    scored_results.append((result, factors))
                    
            except Exception as e:
                logger.warning(f"[RANK] Error scoring result: {e}")
        
        # Sort by total score
        scored_results.sort(key=lambda x: x[1].total_score, reverse=True)
        
        # Return top-k
        top_results = scored_results[:top_k]
        
        logger.info(
            f"[RAG-RANK] Scored {len(context_results)} results, "
            f"returned top {len(top_results)} (threshold={minimum_score_threshold:.2f})"
        )
        
        return top_results
    
    async def apply_temporal_decay(
        self,
        context_items: List[Dict],
        current_time: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Apply temporal decay to context scores.
        Older context becomes less relevant over time.
        """
        
        if current_time is None:
            current_time = datetime.now()
        
        decayed_items = []
        
        for item in context_items:
            # Get or estimate creation time
            created_at = item.get("created_at", current_time)
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            
            age_days = (current_time - created_at).days
            
            # Exponential decay
            decay_factor = 0.5 ** (age_days / self.half_life_days)
            
            # Apply to score if present
            if "relevance_score" in item:
                item["relevance_score"] *= decay_factor
            else:
                item["decay_factor"] = decay_factor
            
            logger.debug(
                f"[DECAY] Age={age_days}d, Decay={decay_factor:.2f}"
            )
            
            decayed_items.append(item)
        
        # Sort by decayed score
        if any("relevance_score" in item for item in decayed_items):
            decayed_items.sort(
                key=lambda x: x.get("relevance_score", 0),
                reverse=True
            )
        
        return decayed_items


# Singleton instance
_rag_optimizer: Optional[RAGContextOptimizer] = None


async def get_rag_context_optimizer(
    semantic_weight: float = 0.30,
    structure_weight: float = 0.25,
    entity_weight: float = 0.20,
    filter_weight: float = 0.15,
    temporal_weight: float = 0.10,
) -> RAGContextOptimizer:
    """Get or create RAG context optimizer."""
    global _rag_optimizer
    
    if _rag_optimizer is None:
        _rag_optimizer = RAGContextOptimizer(
            semantic_weight=semantic_weight,
            structure_weight=structure_weight,
            entity_weight=entity_weight,
            filter_weight=filter_weight,
            temporal_weight=temporal_weight,
        )
    
    return _rag_optimizer
