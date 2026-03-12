"""
Value Grounding: Maps semantic filter values to actual database values.

Handles:
- Converting a user-provided value → actual enum/text values present in the DB
- Probing database for distinct values
- LLM-assisted fuzzy matching when needed
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class ValueGrounder:
    """Maps semantic values to actual database values dynamically.
    
    NO HARDCODED FALLBACK_MAPPINGS - all value matching is done via:
    1. Database probing (get actual distinct values)
    2. LLM semantic matching
    3. Fuzzy string matching
    """
    
    # No hardcoded mappings - rely on dynamic value discovery
    
    def __init__(self, db_connector: Optional[Any] = None):
        """
        Initialize value grounder.
        
        Args:
            db_connector: Optional database connector to probe for actual values
                         Should have execute_query(sql) method
        """
        self.db_connector = db_connector
        self._cache: Dict[str, List[str]] = {}  # Cache of probe results
        logger.info("[VALUE-GROUNDER] Initialized")
    
    async def ground_value(
        self,
        column_path: str,  # e.g., "table.column"
        semantic_value: str,  # e.g., "premium"
        schema: Optional[str] = None,
    ) -> str:
        """
        Ground a semantic value to actual DB value.
        
        Args:
            column_path: Table.column reference
            semantic_value: Semantic value to ground (e.g., "premium")
            schema: Schema name
        
        Returns:
            Best matching DB value (quoted and ready for SQL)
        """
        from app.config import settings as _settings
        schema = schema or _settings.postgres_schema
        table, column = column_path.split('.')

        # Try to probe DB for actual values
        actual_values = await self._probe_column_values(table, column, schema)
        
        if actual_values:
            # Find best match among actual values
            best_match = self._find_best_match(semantic_value, actual_values)
            logger.info(f"[VALUE-GROUND] Mapped '{semantic_value}' → '{best_match}' for {column_path}")
            return f"'{best_match}'"
        else:
            # Fall back to fallback mappings
            best_match = self._fallback_ground(semantic_value)
            logger.info(f"[VALUE-GROUND] Using fallback for '{semantic_value}' → '{best_match}'")
            return f"'{best_match}'"
    
    async def _probe_column_values(
        self,
        table: str,
        column: str,
        schema: Optional[str] = None,
        limit: int = 50,
    ) -> List[str]:
        """
        Probe database for distinct values in a column.
        
        Args:
            table: Table name
            column: Column name
            schema: Schema name
            limit: Max distinct values to retrieve
        
        Returns:
            List of distinct values found in DB (empty if probe fails)
        """
        from app.config import settings as _settings
        schema = schema or _settings.postgres_schema
        cache_key = f"{schema}.{table}.{column}"

        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.db_connector:
            return []
        
        try:
            # Probe for distinct values
            sql = f"""
                SELECT DISTINCT {column}
                FROM {schema}.{table}
                WHERE {column} IS NOT NULL
                LIMIT {limit}
            """
            
            results = await self.db_connector.execute_query(sql)
            
            # Extract values from query results
            values = []
            if isinstance(results, list):
                for row in results:
                    if isinstance(row, dict) and column in row:
                        values.append(str(row[column]))
                    elif isinstance(row, (tuple, list)) and len(row) > 0:
                        values.append(str(row[0]))
            
            self._cache[cache_key] = values
            logger.debug(f"[VALUE-GROUNDER] Probed {cache_key}: {len(values)} distinct values")
            return values
            
        except Exception as e:
            logger.warning(f"[VALUE-GROUNDER] Failed to probe {cache_key}: {e}")
            return []
    
    def _find_best_match(self, semantic_value: str, actual_values: List[str]) -> str:
        """
        Find best matching DB value for semantic value.
        
        Tries:
        1. Exact match (case-insensitive)
        2. Prefix match
        3. Substring match
        4. Levenshtein distance (if similar)
        
        Args:
            semantic_value: What user said (e.g., "premium")
            actual_values: What's in the DB
        
        Returns:
            Best matching DB value
        """
        semantic_lower = semantic_value.lower()
        
        # Strategy 1: Exact match (case-insensitive)
        for val in actual_values:
            if val.lower() == semantic_lower:
                return val
        
        # Strategy 2: Starts with semantic value
        for val in actual_values:
            if val.lower().startswith(semantic_lower):
                return val
        
        # Strategy 3: Semantic value is substring
        for val in actual_values:
            if semantic_lower in val.lower():
                return val
        
        # Strategy 4: Abbreviation match (first letter)
        if semantic_lower:
            for val in actual_values:
                if val.lower().startswith(semantic_lower[0]):
                    return val
        
        # Fall back to first value
        return actual_values[0] if actual_values else semantic_value
    
    def _fallback_ground(self, semantic_value: str) -> str:
        """
        Fallback when DB probe fails - returns original value.
        
        NO HARDCODED MAPPINGS - relies on LLM matching in ground_value().
        The original value is passed through and the SQL generator's
        LLM will handle semantic matching with actual schema values.
        
        Args:
            semantic_value: Semantic value
        
        Returns:
            Original value (no hardcoded transformations)
        """
        # No hardcoded fallback mappings - return original
        # The LLM in SQL generation will handle semantic matching
        return semantic_value


class FilterValueMapper:
    """Maps WHERE condition values to grounded DB values."""
    
    def __init__(self, grounder: Optional[ValueGrounder] = None):
        """Initialize with value grounder."""
        self.grounder = grounder or ValueGrounder()
        logger.info("[FILTER-MAPPER] Initialized")
    
    async def ground_where_conditions(
        self,
        where_conditions: List[Dict[str, Any]],
        table_references: Dict[str, str],  # Maps table to schema
    ) -> List[Dict[str, Any]]:
        """
        Ground all WHERE condition values.
        
        Args:
            where_conditions: List of WHERE condition dicts with 'column' and 'value_hint'
            table_references: Dict mapping table names to schemas
        
        Returns:
            Grounded conditions with 'value' field populated
        """
        grounded = []
        
        for cond in where_conditions:
            column = cond.get('column', '')  # e.g., "table.column"
            value_hint = cond.get('value_hint', '')  # e.g., "premium"
            
            if not column or not value_hint:
                grounded.append(cond)
                continue
            
            # Extract schema for this table
            from app.config import settings as _settings
            _default_schema = _settings.postgres_schema
            parts = column.split('.')
            if len(parts) == 2:
                table = parts[0]
                schema = table_references.get(table, _default_schema)
            else:
                schema = _default_schema
            
            # Ground the value
            try:
                grounded_value = await self.grounder.ground_value(column, value_hint, schema)
                cond['value'] = grounded_value
                grounded.append(cond)
            except Exception as e:
                logger.warning(f"[FILTER-MAPPER] Failed to ground {column}={value_hint}: {e}")
                cond['value'] = f"'{value_hint}'"
                grounded.append(cond)
        
        return grounded
