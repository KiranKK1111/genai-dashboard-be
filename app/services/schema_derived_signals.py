"""
Schema-Derived Signals for DB-Agnostic Intent Routing.

Replaces keyword heuristics with DETERMINISTIC FACTS from actual database schema.

NO hardcoded lists. Instead:
- FK relationships → identify joinable entities
- Column types → detect aggregatable metrics
- Enum values → detect filterable categories
- Cardinality → detect dimension vs fact tables

All signals derived from schema_discovery, not domain assumptions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of query analysis (derived from schema, not keywords)."""
    EXPLORATORY = "exploratory"  # SELECT all columns, no filters
    AGGREGATION = "aggregation"  # GROUP BY, SUM/COUNT/AVG on numeric columns
    FILTERING = "filtering"  # WHERE clause with categorical columns
    JOINING = "joining"  # References entities via FK relationships
    DIMENSIONAL = "dimensional"  # Multiple JOINs, GROUP BY on dimensions
    TIME_SERIES = "time_series"  # GROUP BY on timestamp columns
    CONCEPTUAL = "conceptual"  # No concrete entities mentioned
    UNKNOWN = "unknown"


@dataclass
class ColumnSignature:
    """Deterministic facts about a column (schema-only, no NLP)."""
    column_name: str
    table_name: str
    data_type: str
    is_numeric: bool  # Can aggregate? (INT, FLOAT, DECIMAL, etc.)
    is_categorical: bool  # Useful for WHERE clause? (low cardinality)
    is_temporal: bool  # Is it DATE/TIMESTAMP?
    is_id: bool  # Is it a PK or FK?
    cardinality: Optional[int] = None  # distinct values (if sampled)
    enum_values: Optional[List[str]] = None  # If enum type
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_name": self.column_name,
            "table_name": self.table_name,
            "data_type": self.data_type,
            "is_numeric": self.is_numeric,
            "is_categorical": self.is_categorical,
            "is_temporal": self.is_temporal,
            "is_id": self.is_id,
            "cardinality": self.cardinality,
            "enum_values": self.enum_values,
        }


@dataclass
class EntityReference:
    """
    A concrete reference to a table (NOT based on domain keyword matching).
    
    Derived from:
    - Explicit schema match: "products" in query → products table exists
    - FK traversal: "entity_a related to entity_b" → join via FK
    - Context: last_sql_exists → continue with previous tables
    """
    table_name: str
    confidence: float  # 0-1: how confident we can match this entity
    reason: str  # "explicit: exact table match" | "fk_path: 2 hops via FK" | "context: last_sql_used"
    fk_path: Optional[List[str]] = None  # Tables traversed via FKs if indirect reference
    columns_referenced: Optional[List[str]] = None  # Columns explicitly mentioned in query
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "confidence": self.confidence,
            "reason": self.reason,
            "fk_path": self.fk_path,
            "columns_referenced": self.columns_referenced,
        }


@dataclass
class SchemaDerivedSignals:
    """
    Pure schema facts. No NLP heuristics. No domain assumptions.
    
    These signals deterministically indicate query analysis type (aggregation vs joining, etc.)
    without keyword matching or regex patterns.
    """
    
    # Entity detection (schema-based, not keyword-based)
    explicit_tables: List[EntityReference]  # Tables whose names appear in query
    referenced_columns: List[ColumnSignature]  # Columns mentioned or column name patterns detected
    joinable_entities: List[Tuple[str, str]]  # (table1, table2) pairs reachable via FK
    
    # Capability signals (from column types and cardinality)
    aggregatable_columns: List[ColumnSignature]  # Numeric columns (can SUM/AVG/COUNT on)
    filterable_columns: List[ColumnSignature]  # Low-cardinality or categorical columns
    temporal_columns: List[ColumnSignature]  # DATE/TIMESTAMP columns (time series)
    
    # Analysis signals (derived from capabilities, not keywords)
    can_do_aggregation: bool = False
    can_do_grouping: bool = False
    can_do_joining: bool = False
    can_do_filtering: bool = False
    can_do_time_series: bool = False
    
    # Ambiguity indicators
    entity_count: int = 0  # How many concrete tables could be relevant?
    column_ambiguity: float = 0.0  # 0-1: how ambiguous are referenced columns?
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "explicit_tables": [t.to_dict() for t in self.explicit_tables],
            "referenced_columns": [c.to_dict() for c in self.referenced_columns],
            "aggregatable_columns": [c.to_dict() for c in self.aggregatable_columns],
            "filterable_columns": [c.to_dict() for c in self.filterable_columns],
            "temporal_columns": [c.to_dict() for c in self.temporal_columns],
            "can_do_aggregation": self.can_do_aggregation,
            "can_do_grouping": self.can_do_grouping,
            "can_do_joining": self.can_do_joining,
            "can_do_filtering": self.can_do_filtering,
            "can_do_time_series": self.can_do_time_series,
            "entity_count": self.entity_count,
            "column_ambiguity": self.column_ambiguity,
        }


class SchemaDerivedSignalExtractor:
    """
    Extract deterministic signals from database schema.
    
    This replaces keyword heuristics. For example:
    
    OLD (keyword-based, fails in new domains):
    ```
    if "<entity_term>" in query.lower():
        table = "<table_name>"
    ```
    
    NEW (schema-based, works everywhere):
    ```
    explicit_tables = [t for t in schema.tables if t.name in query.lower()]
    joinable = get_fk_reachable_tables(explicit_tables, schema)
    aggregatable = [c for c in schema.columns if c.is_numeric]
    ```
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._schema_cache: Optional[Dict[str, Any]] = None
    
    async def extract_signals(
        self,
        user_query: str,
        schema_discovery_service,  # Will accept schema_discovery.get_schema()
    ) -> SchemaDerivedSignals:
        """
        Extract deterministic signals from schema that match the user query.
        
        Args:
            user_query: The user's question
            schema_discovery_service: Schema discovery service with table/column metadata
            
        Returns:
            SchemaDerivedSignals with deterministic schema facts
        """
        
        # Step 1: Get all tables and columns from schema
        all_tables = await self._get_all_tables(schema_discovery_service)
        
        # Step 2: Find EXPLICIT table matches (no guessing, schema-based only)
        explicit_tables = self._find_explicit_table_matches(user_query, all_tables)
        
        # Step 3: Find referenced columns (exact name matches only)
        referenced_columns = self._find_referenced_columns(user_query, all_tables)
        
        # Step 4: Build capability signals (what operations are possible?)
        aggregatable = self._find_aggregatable_columns(all_tables)
        filterable = self._find_filterable_columns(all_tables)
        temporal = self._find_temporal_columns(all_tables)
        joinable = self._find_joinable_pairs(all_tables)
        
        # Step 5: Derive analysis capabilities from schema facts
        can_aggregate = len(aggregatable) > 0
        can_group = len(temporal) > 0 or len(filterable) > 0
        can_join = len(joinable) > 0
        can_filter = len(filterable) > 0
        can_timeseries = len(temporal) > 0 and can_aggregate
        
        # Step 6: Calculate ambiguity scores
        entity_count = len(explicit_tables)
        column_ambiguity = self._compute_column_ambiguity(referenced_columns, all_tables)
        
        return SchemaDerivedSignals(
            explicit_tables=explicit_tables,
            referenced_columns=referenced_columns,
            joinable_entities=joinable,
            aggregatable_columns=aggregatable,
            filterable_columns=filterable,
            temporal_columns=temporal,
            can_do_aggregation=can_aggregate,
            can_do_grouping=can_group,
            can_do_joining=can_join,
            can_do_filtering=can_filter,
            can_do_time_series=can_timeseries,
            entity_count=entity_count,
            column_ambiguity=column_ambiguity,
        )
    
    async def _get_all_tables(self, schema_discovery_service) -> Dict[str, Dict[str, Any]]:
        """Get schema of all tables (from schema_discovery service)."""
        # Placeholder: would call schema_discovery.get_schema()
        # For now, returns empty - integration point
        return {}
    
    def _find_explicit_table_matches(
        self,
        user_query: str,
        all_tables: Dict[str, Dict[str, Any]]
    ) -> List[EntityReference]:
        """
        Find tables whose EXACT names appear in the query.
        
        NO fuzzy matching. NO keyword assumptions.
        Example: "SELECT * FROM <table>" → explicit match on "<table>" table
        """
        matches = []
        query_lower = user_query.lower()
        
        for table_name in all_tables.keys():
            # Exact substring match (word boundary)
            if f" {table_name} " in f" {query_lower} ":
                matches.append(EntityReference(
                    table_name=table_name,
                    confidence=1.0,
                    reason=f"explicit: exact table name match in query"
                ))
        
        return matches
    
    def _find_referenced_columns(
        self,
        user_query: str,
        all_tables: Dict[str, Dict[str, Any]]
    ) -> List[ColumnSignature]:
        """
        Find columns whose EXACT names appear in the query.
        
        Example: "What is the total amount?" + schema has "amount" column
                 → match on "amount", derive is_numeric from schema
        """
        columns = []
        query_words = set(user_query.lower().split())
        
        for table_name, table_info in all_tables.items():
            for col_name, col_info in table_info.get("columns", {}).items():
                # Exact word match in query
                if col_name.lower() in query_words:
                    columns.append(ColumnSignature(
                        column_name=col_name,
                        table_name=table_name,
                        data_type=col_info.get("data_type", "unknown"),
                        is_numeric=self._is_numeric_type(col_info.get("data_type")),
                        is_categorical=self._is_categorical_type(col_info.get("data_type")),
                        is_temporal=self._is_temporal_type(col_info.get("data_type")),
                        is_id=col_info.get("is_primary_key", False) or col_info.get("is_foreign_key", False),
                    ))
        
        return columns
    
    def _find_aggregatable_columns(
        self,
        all_tables: Dict[str, Dict[str, Any]]
    ) -> List[ColumnSignature]:
        """Find all numeric columns (can SUM, AVG, COUNT on these)."""
        columns = []
        
        for table_name, table_info in all_tables.items():
            for col_name, col_info in table_info.get("columns", {}).items():
                if self._is_numeric_type(col_info.get("data_type")):
                    columns.append(ColumnSignature(
                        column_name=col_name,
                        table_name=table_name,
                        data_type=col_info.get("data_type"),
                        is_numeric=True,
                        is_categorical=False,
                        is_temporal=False,
                        is_id=False,
                    ))
        
        return columns
    
    def _find_filterable_columns(
        self,
        all_tables: Dict[str, Dict[str, Any]]
    ) -> List[ColumnSignature]:
        """Find low-cardinality or categorical columns (useful for WHERE clauses)."""
        columns = []
        
        for table_name, table_info in all_tables.items():
            for col_name, col_info in table_info.get("columns", {}).items():
                # Categorical: enum types, booleans, or low cardinality
                is_cat = self._is_categorical_type(col_info.get("data_type"))
                cardinality = col_info.get("distinct_count", 0)
                low_cardinality = cardinality > 0 and cardinality < 100  # heuristic: < 100 distinct
                
                if is_cat or low_cardinality:
                    columns.append(ColumnSignature(
                        column_name=col_name,
                        table_name=table_name,
                        data_type=col_info.get("data_type"),
                        is_numeric=False,
                        is_categorical=True,
                        is_temporal=False,
                        is_id=False,
                        cardinality=cardinality,
                        enum_values=col_info.get("enum_values"),
                    ))
        
        return columns
    
    def _find_temporal_columns(
        self,
        all_tables: Dict[str, Dict[str, Any]]
    ) -> List[ColumnSignature]:
        """Find DATE and TIMESTAMP columns (for time series analysis)."""
        columns = []
        
        for table_name, table_info in all_tables.items():
            for col_name, col_info in table_info.get("columns", {}).items():
                if self._is_temporal_type(col_info.get("data_type")):
                    columns.append(ColumnSignature(
                        column_name=col_name,
                        table_name=table_name,
                        data_type=col_info.get("data_type"),
                        is_numeric=False,
                        is_categorical=False,
                        is_temporal=True,
                        is_id=False,
                    ))
        
        return columns
    
    def _find_joinable_pairs(
        self,
        all_tables: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[str, str]]:
        """Find all table pairs connected via FK relationships."""
        pairs = []
        
        for table_name, table_info in all_tables.items():
            fks = table_info.get("foreign_keys", {})
            for fk_col, (ref_table, ref_col) in fks.items():
                pairs.append((table_name, ref_table))
                pairs.append((ref_table, table_name))  # Bidirectional
        
        # Remove duplicates
        return list(set(pairs))
    
    def _compute_column_ambiguity(
        self,
        referenced_columns: List[ColumnSignature],
        all_tables: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Compute ambiguity score for referenced columns.
        
        0.0 = clear (referenced columns exist in exactly one table)
        1.0 = highly ambiguous (referenced columns exist in many tables)
        """
        if not referenced_columns:
            return 0.0  # No columns referenced, not ambiguous
        
        # Count how many tables each column appears in
        column_appearances: Dict[str, int] = {}
        for col in referenced_columns:
            col_key = col.column_name
            column_appearances[col_key] = column_appearances.get(col_key, 0) + 1
        
        # Average appearances across referenced columns
        if not column_appearances:
            return 0.0
        
        avg_appearances = sum(column_appearances.values()) / len(column_appearances)
        total_tables = len(all_tables)
        
        # Normalize: 1 appearance = 0.0 ambiguity, max_tables = 1.0 ambiguity
        if total_tables <= 1:
            return 0.0
        
        ambiguity = (avg_appearances - 1) / (total_tables - 1)
        return min(1.0, max(0.0, ambiguity))
    
    @staticmethod
    def _is_numeric_type(data_type: str) -> bool:
        """Check if type is numeric (can aggregate)."""
        numeric_types = {"int", "bigint", "smallint", "float", "double", "decimal", "numeric", "money"}
        return any(t in data_type.lower() for t in numeric_types)
    
    @staticmethod
    def _is_categorical_type(data_type: str) -> bool:
        """Check if type should be used for filtering/grouping."""
        cat_types = {"boolean", "bool", "enum", "varchar", "char", "text"}
        return any(t in data_type.lower() for t in cat_types)
    
    @staticmethod
    def _is_temporal_type(data_type: str) -> bool:
        """Check if type is a date/time type."""
        temporal_types = {"date", "datetime", "timestamp", "time", "interval"}
        return any(t in data_type.lower() for t in temporal_types)
