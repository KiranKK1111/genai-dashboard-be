"""
Universal Value Grounder - Grounds user values to actual DB values dynamically.

Replaces hardcoded FALLBACK_MAPPINGS with:
1. Dynamic enum mining from database
2. LLM-based semantic value matching
3. Smart fallback for unknown values

Supports domain-specific enums: PREMIUM, STANDARD, SETTLED, VERIFIED, etc.
Works across all database types.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from .database_adapter import get_global_adapter, DatabaseType
from .. import llm

logger = logging.getLogger(__name__)


@dataclass
class ValueMatch:
    """Result of matching a user value to a database value."""
    database_value: str
    user_value: str
    similarity_score: float  # 0.0-1.0
    match_type: str  # "exact", "semantic", "abbreviation", "case_insensitive"


class UniversalValueGrounder:
    """Grounds user values to actual DB values WITHOUT hardcoding."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.adapter = get_global_adapter()
        self._enum_cache: Dict[str, List[str]] = {}  # column_path -> values
        self._match_cache: Dict[Tuple[str, str], Optional[ValueMatch]] = {}  # (user_val, col_path) -> match
    
    async def ground_value_to_column(
        self,
        user_value: str,
        column_path: str,  # "table.column" or "schema.table.column"
        threshold: float = 0.7,
        table_name: Optional[str] = None,
        column_name: Optional[str] = None
    ) -> Optional[ValueMatch]:
        """
        Ground a user-provided value to actual DB value.
        
        Examples:
                - ground_value("premium", "users.status")
                    → Checks DB: SELECT DISTINCT status FROM users
                    → Finds: ["PREMIUM", "STANDARD", "BASIC"]
                    → Semantic matches "premium" → "PREMIUM"
                    → Returns: ValueMatch("PREMIUM", "premium", 0.95, "semantic")
        
                - ground_value("premium", "users.tier")
          → Checks DB: ["PREMIER", "STANDARD", "BASIC"]
          → Semantic matches "premium" → "PREMIER"
          → Returns: ValueMatch("PREMIER", "premium", 0.90, "semantic")
        
        - ground_value("xyz123", "any.column")
          → No match found
          → Returns: None (don't add filter)
        
        Args:
            user_value: What the user typed
            column_path: "table.column" or "schema.table.column"
            threshold: Minimum similarity (0.0-1.0) to accept match
            table_name: Optional explicit table name (if column_path is just column name)
            column_name: Optional explicit column name
        
        Returns:
            ValueMatch if confident match found, None otherwise
        """
        
        # Check cache first
        cache_key = (user_value.lower(), column_path.lower())
        if cache_key in self._match_cache:
            return self._match_cache[cache_key]
        
        # Parse column path
        if '.' in column_path:
            parts = column_path.split('.')
            if len(parts) == 2:
                table_name = parts[0]
                column_name = parts[1]
            elif len(parts) == 3:
                schema_name = parts[0]
                table_name = parts[1]
                column_name = parts[2]
        
        if not table_name or not column_name:
            logger.warning(f"Invalid column path: {column_path}")
            return None
        
        try:
            # Step 1: Mine actual enum values from database (NO hardcoding)
            actual_values = await self._mine_column_values(table_name, column_name)
            
            if not actual_values:
                # No enum values found, can't ground
                logger.debug(f"No enum values found for {table_name}.{column_name}")
                return None
            
            logger.debug(f"Found {len(actual_values)} values for {table_name}.{column_name}: {actual_values}")
            
            # Step 2: Use LLM + semantic similarity to find best match
            best_match = await self._find_best_match(
                user_value,
                actual_values,
                threshold
            )
            
            if best_match:
                # Cache the result
                self._match_cache[cache_key] = best_match
            
            return best_match
        
        except Exception as e:
            logger.error(f"Error grounding value '{user_value}' to {column_path}: {e}", exc_info=True)
            return None
    
    async def _mine_column_values(self, table_name: str, column_name: str, limit: int = 100) -> List[str]:
        """
        Mine ACTUAL enum values from database.
        
        Works with:
        - ENUM types (PostgreSQL, MySQL)
        - CHECK constraints
        - Sample data (fallback)
        
        NO hardcoding, uses real database values.
        """
        
        cache_key = f"{table_name}.{column_name}"
        
        if cache_key in self._enum_cache:
            logger.debug(f"Using cached values for {cache_key}")
            return self._enum_cache[cache_key]
        
        logger.debug(f"Mining enum values for {cache_key}")
        
        # Step 1: Try ENUM type discovery (most reliable)
        enum_values = await self._get_enum_type_values(table_name, column_name)
        if enum_values:
            self._enum_cache[cache_key] = enum_values
            logger.debug(f"Found ENUM values for {cache_key}: {enum_values}")
            return enum_values
        
        # Step 2: Try CHECK constraints (second most reliable)
        check_values = await self._get_check_constraint_values(table_name, column_name)
        if check_values:
            self._enum_cache[cache_key] = check_values
            logger.debug(f"Found CHECK constraint values for {cache_key}: {check_values}")
            return check_values
        
        # Step 3: Sample DISTINCT values from table (fallback, slower)
        sample_values = await self._sample_distinct_values(table_name, column_name, limit)
        if sample_values:
            self._enum_cache[cache_key] = sample_values
            logger.debug(f"Found {len(sample_values)} distinct values for {cache_key}")
        
        return sample_values
    
    async def _get_enum_type_values(self, table_name: str, column_name: str) -> List[str]:
        """Query database for ENUM type values (NO assumptions)."""
        
        try:
            if self.adapter.db_type == DatabaseType.POSTGRESQL:
                # PostgreSQL ENUM discovery
                query = """
                SELECT e.enumlabel
                FROM pg_enum e
                JOIN pg_type t ON e.enumtypid = t.oid
                JOIN pg_class c ON t.typrelid = c.oid
                WHERE c.relname = :table AND t.typname LIKE :pattern
                ORDER BY e.enumsortorder
                """
                result = await self.db.execute(
                    text(query),
                    {"table": table_name, "pattern": f"%{column_name}%"}
                )
                return [row[0] for row in result.all()]
            
            elif self.adapter.db_type == DatabaseType.MYSQL:
                # MySQL ENUM parsing
                query = """
                SELECT COLUMN_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = :table AND COLUMN_NAME = :column
                AND TABLE_SCHEMA = DATABASE()
                """
                result = await self.db.execute(
                    text(query),
                    {"table": table_name, "column": column_name}
                )
                row = result.first()
                if row and row[0].startswith('enum('):
                    # Parse: enum('A','B','C')
                    enum_str = row[0][5:-1]  # Remove "enum(" and ")"
                    return [val.strip("'\"") for val in enum_str.split(',')]
            
            # Other databases don't have ENUM type
            return []
        
        except Exception as e:
            logger.debug(f"Error querying ENUM types: {e}")
            return []
    
    async def _get_check_constraint_values(self, table_name: str, column_name: str) -> List[str]:
        """Extract enum values from CHECK constraints."""
        
        try:
            if self.adapter.db_type == DatabaseType.SQL_SERVER:
                # Query SQL Server CHECK constraints
                query = """
                SELECT definition
                FROM sys.check_constraints
                WHERE parent_object_id = OBJECT_ID(:table_name)
                AND definition LIKE :pattern
                """
                result = await self.db.execute(
                    text(query),
                    {
                        "table_name": f"dbo.{table_name}",
                        "pattern": f"%{column_name}%"
                    }
                )
                
                values = []
                for row in result.all():
                    definition = row[0]
                    # Parse patterns like: ([status]='ACTIVE' OR [status]='INACTIVE')
                    import re
                    matches = re.findall(r"'([^']+)'", definition)
                    values.extend(matches)
                return list(set(values))
            
            return []
        
        except Exception as e:
            logger.debug(f"Error querying CHECK constraints: {e}")
            return []
    
    async def _sample_distinct_values(
        self,
        table_name: str,
        column_name: str,
        limit: int = 100
    ) -> List[str]:
        """
        Sample DISTINCT values from table as fallback.
        
        Slower but works for any column with categorical data.
        """
        
        try:
            # Build dialect-specific DISTINCT query
            if self.adapter.db_type == DatabaseType.POSTGRESQL:
                query = f"""
                SELECT DISTINCT {column_name}
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
                LIMIT :limit
                """
            elif self.adapter.db_type == DatabaseType.MYSQL:
                query = f"""
                SELECT DISTINCT {column_name}
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
                LIMIT :limit
                """
            elif self.adapter.db_type == DatabaseType.SQL_SERVER:
                query = f"""
                SELECT DISTINCT TOP :limit {column_name}
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
                """
            elif self.adapter.db_type == DatabaseType.SQLITE:
                query = f"""
                SELECT DISTINCT {column_name}
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
                LIMIT :limit
                """
            else:
                return []
            
            result = await self.db.execute(text(query), {"limit": limit})
            values = [str(row[0]) for row in result.all() if row[0] is not None]
            return values
        
        except Exception as e:
            logger.debug(f"Error sampling distinct values from {table_name}.{column_name}: {e}")
            return []
    
    async def _find_best_match(
        self,
        user_value: str,
        actual_values: List[str],
        threshold: float = 0.7
    ) -> Optional[ValueMatch]:
        """
        Find best match between user value and actual DB values.
        Uses LLM for semantic matching (not simple string distance).
        """
        
        # First try simple matches
        user_lower = user_value.lower()
        
        # Exact match
        for val in actual_values:
            if val.lower() == user_lower:
                return ValueMatch(
                    database_value=val,
                    user_value=user_value,
                    similarity_score=1.0,
                    match_type="exact"
                )
        
        # Case-insensitive match
        for val in actual_values:
            if val.lower() == user_lower:
                return ValueMatch(
                    database_value=val,
                    user_value=user_value,
                    similarity_score=0.99,
                    match_type="case_insensitive"
                )
        
        # Use LLM for semantic matching
        values_str = json.dumps(actual_values)
        
        prompt = f"""
Given:
- User typed: "{user_value}"
- Available database values: {values_str}

Find the BEST MATCHING database value for what the user meant.

Consider:
1. Semantic meaning (user "premium" → database "PREMIUM")
2. Abbreviations (user "pkg" → database "PACKAGE")
3. Synonyms (user "high-value" → database "PREMIUM")
4. Case differences

Respond with ONLY valid JSON:
{{
  "best_match": "the actual DB value that best matches user input",
  "confidence": 0.0-1.0,
  "reasoning": "Why this is the best match",
  "match_type": "semantic|abbreviation|synonym|other"
}}

If no good match exists, set best_match to "NO_MATCH" with confidence 0.0.
"""
        
        try:
            response = await llm.call_llm([
                {
                    "role": "system",
                    "content": "You are a value matcher. Respond with ONLY valid JSON."
                },
                {"role": "user", "content": prompt}
            ], max_tokens=200, temperature=0.2)
            
            response_text = str(response)
            result = json.loads(response_text)
            
            best_match = result.get("best_match")
            confidence = float(result.get("confidence", 0.0))
            
            if best_match == "NO_MATCH" or confidence < threshold:
                logger.debug(
                    f"No match found for '{user_value}' (confidence: {confidence})"
                )
                return None
            
            return ValueMatch(
                database_value=best_match,
                user_value=user_value,
                similarity_score=confidence,
                match_type=result.get("match_type", "semantic")
            )
        
        except Exception as e:
            logger.error(f"Error finding best match: {e}")
            return None
    
    async def ground_multiple_values(
        self,
        column_path: str,
        user_values: List[str],
        threshold: float = 0.7
    ) -> Dict[str, Optional[ValueMatch]]:
        """
        Ground multiple values at once for the same column.
        
        Returns dict mapping user values to their matches.
        """
        
        results = {}
        for user_value in user_values:
            match = await self.ground_value_to_column(user_value, column_path, threshold)
            results[user_value] = match
        
        return results
    
    def clear_cache(self) -> None:
        """Clear all value caches."""
        self._enum_cache.clear()
        self._match_cache.clear()
        logger.debug("Cleared value grounder caches")
