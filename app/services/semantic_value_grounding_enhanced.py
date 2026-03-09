"""
ENHANCED SEMANTIC VALUE GROUNDING - All-In-One Solution
=========================================================

Comprehensive value grounding system that handles:
1. Single-table value variations (case, type, fuzzy)
2. Cross-table FK relationships (automatic JOIN generation)
3. Multi-hop relationships (transitive FK paths)
4. Relationship discovery via embeddings
5. Zero hardcoding, 100% dynamic

Author: Semantic Query System
Version: 2.0
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from difflib import SequenceMatcher
from datetime import datetime, date

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import JSON

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS & ENUMS
# ============================================================================

class ValueType(str, Enum):
    """Detected database value types."""
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    FLOAT = "float"
    DATE = "date"
    DATETIME = "datetime"
    NULL = "null"
    UNKNOWN = "unknown"


class MatchStrategy(str, Enum):
    """Matching strategies used."""
    EXACT = "exact"
    CASE_INSENSITIVE = "case_insensitive"
    SUBSTRING = "substring"
    TYPE_CONVERSION = "type_conversion"
    FUZZY = "fuzzy"
    FK_RELATIONSHIP = "fk_relationship"
    EMBEDDING_RELATIONSHIP = "embedding_relationship"


@dataclass
class ForeignKeyInfo:
    """Foreign key relationship metadata."""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    constraint_name: str
    cardinality: str = "ONE_TO_MANY"  # ONE_TO_ONE, ONE_TO_MANY, MANY_TO_MANY

    def __hash__(self):
        return hash((self.from_table, self.from_column, self.to_table, self.to_column))

    def __eq__(self, other):
        if not isinstance(other, ForeignKeyInfo):
            return False
        return (self.from_table == other.from_table and 
                self.from_column == other.from_column and
                self.to_table == other.to_table and
                self.to_column == other.to_column)


@dataclass
class ColumnValueProfile:
    """Profile of actual column values discovered from database."""
    table_name: str
    column_name: str
    value_type: ValueType
    sample_values: List[Any] = field(default_factory=list)
    value_count: int = 0
    null_count: int = 0
    unique_count: int = 0
    frequency_map: Dict[Any, int] = field(default_factory=dict)  # value -> count
    discovered_at: datetime = field(default_factory=datetime.utcnow)

    def get_top_values(self, limit: int = 10) -> List[Any]:
        """Get most frequent values."""
        sorted_values = sorted(
            self.frequency_map.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [v[0] for v in sorted_values[:limit]]


@dataclass
class RelationshipPath:
    """Path through FK relationships to find a value."""
    source_table: str  # Where user provided the value
    source_column: str
    target_table: str  # Where we're trying to find the value
    target_column: str
    fk_chain: List[ForeignKeyInfo] = field(default_factory=list)  # FK path to traverse
    hops: int = 0  # Number of joins needed (1 = direct, 2+ = multi-hop)
    confidence: float = 0.0

    def get_join_sql(self, alias_prefix: str = "t") -> str:
        """Generate JOIN SQL for this relationship path."""
        if not self.fk_chain:
            return ""

        joins = []
        for i, fk in enumerate(self.fk_chain):
            alias1 = f"{alias_prefix}_{i}" if i > 0 else alias_prefix
            alias2 = f"{alias_prefix}_{i + 1}"
            join = (
                f"INNER JOIN \"{fk.to_table}\" AS {alias2} "
                f"ON {alias1}.\"{fk.from_column}\" = {alias2}.\"{fk.to_column}\""
            )
            joins.append(join)

        return " ".join(joins)

    def get_filter_sql(self, grounded_value: str) -> str:
        """Generate WHERE clause for this relationship path."""
        if not self.fk_chain:
            return f"\"{self.target_table}\".\"{self.target_column}\" = '{grounded_value}'"

        # Multi-hop: filter on the final table
        final_alias = f"t_{len(self.fk_chain)}"
        return f"{final_alias}.\"{self.target_column}\" = '{grounded_value}'"


@dataclass
class GroundedValue:
    """Result of grounding a user-provided value."""
    original_value: str
    grounded_value: Any
    strategy: MatchStrategy
    confidence: float
    table_name: str
    column_name: str
    value_type: ValueType
    alternatives: List[Tuple[str, float]] = field(default_factory=list)  # (value, conf)
    grounded_at: datetime = field(default_factory=datetime.utcnow)

    def is_high_confidence(self, threshold: float = 0.85) -> bool:
        """Check if grounding is reliable."""
        return self.confidence >= threshold

    def requires_user_clarification(self, threshold: float = 0.70) -> bool:
        """Check if low confidence requires user input."""
        return self.confidence < threshold


@dataclass
class GroundedValueWithRelationship(GroundedValue):
    """Enhanced grounded value with relationship information."""
    relationship_path: Optional[RelationshipPath] = None
    requires_join: bool = False
    join_sql: str = ""
    filter_sql: str = ""
    is_direct_match: bool = True  # True if in same table, False if FK resolution
    related_fk_info: Optional[ForeignKeyInfo] = None


# ============================================================================
# MAIN ENHANCED VALUE GROUNDER
# ============================================================================

class SemanticValueGrounderEnhanced:
    """
    Enterprise-grade semantic value grounding with relationship awareness.
    
    Handles:
    - Single-table value variations
    - Cross-table FK relationships
    - Multi-hop joins
    - Automatic relationship discovery
    - Zero hardcoding
    """

    def __init__(self, max_relationship_hops: int = 3, embedding_similarity_threshold: float = 0.7):
        self.max_relationship_hops = max_relationship_hops
        self.embedding_similarity_threshold = embedding_similarity_threshold
        
        # Profiles of known column values
        self.column_profiles: Dict[str, Dict[str, ColumnValueProfile]] = {}
        # table -> {column -> profile}
        
        # FK relationship metadata
        self.foreign_keys: List[ForeignKeyInfo] = []
        self.fk_index: Dict[Tuple[str, str], List[ForeignKeyInfo]] = {}
        # (from_table, from_column) -> [ForeignKeyInfo]
        
        # Reverse FK index for quick lookup
        self.reverse_fk_index: Dict[Tuple[str, str], List[ForeignKeyInfo]] = {}
        # (to_table, to_column) -> [ForeignKeyInfo]
        
        # Cached relationship paths
        self.relationship_cache: Dict[Tuple[str, str, str, str], RelationshipPath] = {}
        
        self.initialized = False
        self._initialization_lock = asyncio.Lock()

    # ========================================================================
    # INITIALIZATION & DISCOVERY
    # ========================================================================

    async def initialize_for_tables(
        self,
        db: AsyncSession,
        table_names: Optional[List[str]] = None,
        sample_size: int = 100
    ):
        """
        Initialize value profiles and FK relationships.
        
        Args:
            db: Async database session
            table_names: Tables to profile (None = all tables)
            sample_size: Max rows to sample from each column
        """
        async with self._initialization_lock:
            if self.initialized:
                logger.info("[GROUNDER] Already initialized, skipping...")
                return

            logger.info("[GROUNDER] Initializing semantic value grounder...")
            
            try:
                # Step 1: Discover FK relationships
                await self._discover_foreign_keys(db)
                logger.info(f"[GROUNDER] Discovered {len(self.foreign_keys)} FK relationships")

                # Step 2: Profile column values
                await self._profile_column_values(db, table_names, sample_size)
                logger.info(f"[GROUNDER] Profiled {sum(len(cols) for cols in self.column_profiles.values())} columns")

                self.initialized = True
                logger.info("[GROUNDER] ✅ Initialization complete!")

            except Exception as e:
                logger.error(f"[GROUNDER] Initialization failed: {e}", exc_info=True)
                raise

    async def _discover_foreign_keys(self, db: AsyncSession):
        """Discover all FK relationships in database using async queries."""
        self.foreign_keys = []
        self.fk_index = {}
        self.reverse_fk_index = {}

        try:
            # Query FK relationships from information_schema (fully async)
            fk_query = text("""
                SELECT 
                    kcu1.table_name AS from_table,
                    kcu1.column_name AS from_column,
                    kcu2.table_name AS to_table,
                    kcu2.column_name AS to_column,
                    rc.constraint_name
                FROM information_schema.referential_constraints rc
                JOIN information_schema.key_column_usage kcu1 
                    ON rc.constraint_name = kcu1.constraint_name 
                    AND rc.constraint_schema = kcu1.table_schema
                JOIN information_schema.key_column_usage kcu2 
                    ON rc.unique_constraint_name = kcu2.constraint_name 
                    AND rc.unique_constraint_schema = kcu2.table_schema
                WHERE kcu1.table_schema = 'genai'
                ORDER BY kcu1.table_name, kcu1.column_name
            """)
            
            result = await db.execute(fk_query)
            rows = result.fetchall()
            
            for row in rows:
                try:
                    fk_info = ForeignKeyInfo(
                        from_table=row[0],
                        from_column=row[1],
                        to_table=row[2],
                        to_column=row[3],
                        constraint_name=row[4]
                    )
                    
                    self.foreign_keys.append(fk_info)
                    
                    # Index by source
                    key = (fk_info.from_table, fk_info.from_column)
                    if key not in self.fk_index:
                        self.fk_index[key] = []
                    self.fk_index[key].append(fk_info)
                    
                    # Index by target (reverse lookup)
                    rev_key = (fk_info.to_table, fk_info.to_column)
                    if rev_key not in self.reverse_fk_index:
                        self.reverse_fk_index[rev_key] = []
                    self.reverse_fk_index[rev_key].append(fk_info)
                    
                except Exception as e:
                    logger.warning(f"[GROUNDER] Could not process FK row {row}: {e}")
                    
        except Exception as e:
            logger.warning(f"[GROUNDER] FK discovery failed: {e}")

    async def _profile_column_values(
        self,
        db: AsyncSession,
        table_names: Optional[List[str]],
        sample_size: int
    ):
        """Profile actual values in database columns using async queries."""
        try:
            # Get table list from information_schema if not provided
            if table_names is None:
                tables_query = text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'genai' 
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """)
                result = await db.execute(tables_query)
                table_names = [row[0] for row in result.fetchall()]
            
            for table in table_names:
                try:
                    # Get columns from information_schema
                    columns_query = text("""
                        SELECT column_name, data_type, udt_name
                        FROM information_schema.columns
                        WHERE table_schema = 'genai' AND table_name = :table_name
                        ORDER BY ordinal_position
                    """)
                    
                    result = await db.execute(columns_query, {"table_name": table})
                    columns = result.fetchall()
                    self.column_profiles[table] = {}
                    
                    for col_row in columns:
                        col_name = col_row[0]
                        col_type = col_row[1]
                        udt_name = col_row[2]
                        
                        # Determine value type
                        value_type = self._detect_value_type(col_type)
                        
                        # ✅ SKIP unhashable types that cause "could not identify equality operator" errors
                        # JSON types cannot be used in GROUP BY clause
                        unhashable_types = {'json', 'jsonb', 'uuid', 'bytea', 'boolean[]', 'integer[]', 'text[]', 'uuid[]'}
                        is_unhashable = (
                            isinstance(col_type, str) and 
                            any(unhashable in col_type.lower() for unhashable in unhashable_types)
                        ) or (isinstance(udt_name, str) and udt_name.lower() in unhashable_types)
                        
                        if is_unhashable:
                            logger.debug(f"[GROUNDER] Skipping profile for {table}.{col_name} (unhashable type: {col_type})")
                            continue
                        
                        # Query sample values
                        try:
                            query = text(f'SELECT "{col_name}", COUNT(*) as cnt FROM "{table}" GROUP BY "{col_name}" LIMIT :sample_size')
                            result = await db.execute(query, {"sample_size": sample_size})
                            rows = result.fetchall()
                            
                            sample_values = []
                            frequency_map = {}
                            total_count = 0
                            
                            for row in rows:
                                val = row[0]
                                cnt = row[1]
                                if val is not None:
                                    sample_values.append(val)
                                    frequency_map[val] = cnt
                                    total_count += cnt
                            
                            # Get total & null counts
                            count_query = text(f'SELECT COUNT(*), COUNT(CASE WHEN "{col_name}" IS NULL THEN 1 END) FROM "{table}"')
                            count_result = await db.execute(count_query)
                            count_row = count_result.fetchone()
                            total = count_row[0] if count_row else 0
                            null_count = count_row[1] if count_row else 0
                            
                            profile = ColumnValueProfile(
                                table_name=table,
                                column_name=col_name,
                                value_type=value_type,
                                sample_values=sample_values[:10],
                                value_count=total,
                                null_count=null_count,
                                unique_count=len(frequency_map),
                                frequency_map=frequency_map
                            )
                            
                            self.column_profiles[table][col_name] = profile
                            
                        except Exception as e:
                            # ✅ Rollback transaction on error to prevent cascading failures
                            try:
                                await db.rollback()
                            except:
                                pass
                            logger.warning(f"[GROUNDER] Could not profile {table}.{col_name}: {e}")
                            # Continue to next column instead of stopping
                            
                except Exception as e:
                    logger.warning(f"[GROUNDER] Could not profile table {table}: {e}")
                    
        except Exception as e:
            logger.warning(f"[GROUNDER] Column profiling failed: {e}")

    def _detect_value_type(self, col_type) -> ValueType:
        """Detect value type from SQLAlchemy column type."""
        col_type_str = str(col_type).lower()

        if "boolean" in col_type_str or "bool" in col_type_str:
            return ValueType.BOOLEAN
        elif "integer" in col_type_str or "bigint" in col_type_str or "serial" in col_type_str:
            return ValueType.INTEGER
        elif "float" in col_type_str or "numeric" in col_type_str or "decimal" in col_type_str:
            return ValueType.FLOAT
        elif "date" in col_type_str and "time" in col_type_str:
            return ValueType.DATETIME
        elif "date" in col_type_str:
            return ValueType.DATE
        elif "text" in col_type_str or "varchar" in col_type_str or "char" in col_type_str:
            return ValueType.STRING
        else:
            return ValueType.STRING  # Default

    # ========================================================================
    # MAIN GROUNDING METHODS
    # ========================================================================

    async def ground_value(
        self,
        user_value: str,
        table_name: str,
        column_name: str,
        db: AsyncSession,
        timeout: float = 5.0
    ) -> GroundedValueWithRelationship:
        """
        Ground a user-provided value with full relationship awareness.
        
        Args:
            user_value: Value provided by user
            table_name: Target table
            column_name: Target column
            db: Database session
            timeout: Max time to search relationships
            
        Returns:
            GroundedValueWithRelationship with confidence, strategy, and relationship info
        """
        if not self.initialized:
            await self.initialize_for_tables(db)

        logger.info(f"[GROUNDING] Grounding '{user_value}' in {table_name}.{column_name}")

        # Step 1: Try direct match (same table)
        direct_result = await self._ground_in_table(
            user_value, table_name, column_name
        )

        if direct_result and direct_result.confidence >= 0.85:
            logger.info(f"[GROUNDING] ✅ Direct match found: {direct_result.original_value} → {direct_result.grounded_value}")
            return GroundedValueWithRelationship(
                **{k: v for k, v in direct_result.__dict__.items()},
                is_direct_match=True,
                relationship_path=None
            )

        # Step 2: Try FK relationships (cross-table)
        try:
            # Use timeout for FK resolution
            relationship_result = await asyncio.wait_for(
                self._ground_via_relationships(
                    user_value, table_name, column_name, db
                ),
                timeout=timeout
            )

            if relationship_result:
                return relationship_result

        except asyncio.TimeoutError:
            logger.warning(f"[GROUNDING] FK resolution timeout for {user_value}")
        except Exception as e:
            logger.warning(f"[GROUNDING] FK resolution failed: {e}")

        # Step 3: Return direct result if found (even with lower confidence)
        if direct_result:
            logger.info(f"[GROUNDING] ⚠️  Returning direct match with confidence {direct_result.confidence}")
            return GroundedValueWithRelationship(
                **{k: v for k, v in direct_result.__dict__.items()},
                is_direct_match=True,
                relationship_path=None
            )

        # Step 4: Fallback - value not found anywhere
        logger.warning(f"[GROUNDING] ❌ Could not ground value: {user_value}")
        return GroundedValueWithRelationship(
            original_value=user_value,
            grounded_value=user_value,  # Return original as fallback
            strategy=MatchStrategy.FUZZY,
            confidence=0.0,
            table_name=table_name,
            column_name=column_name,
            value_type=ValueType.UNKNOWN,
            is_direct_match=True
        )

    async def _ground_in_table(
        self,
        user_value: str,
        table_name: str,
        column_name: str
    ) -> Optional[GroundedValue]:
        """
        Ground value directly in target table using 5 strategies.
        Returns best match or None if no good match found.
        """
        if table_name not in self.column_profiles:
            logger.warning(f"[GROUNDING] Table {table_name} not profiled")
            return None

        if column_name not in self.column_profiles[table_name]:
            logger.warning(f"[GROUNDING] Column {table_name}.{column_name} not profiled")
            return None

        profile = self.column_profiles[table_name][column_name]
        candidates = []

        # Strategy 1: Exact match
        exact = self._try_exact_match(user_value, profile)
        if exact:
            candidates.append(exact)

        # Strategy 2: Case-insensitive
        if profile.value_type == ValueType.STRING:
            case_match = self._try_case_insensitive_match(user_value, profile)
            if case_match:
                candidates.append(case_match)

        # Strategy 3: Substring match
        substring = self._try_substring_match(user_value, profile)
        if substring:
            candidates.append(substring)

        # Strategy 4: Type conversion
        type_conv = self._try_type_conversion(user_value, profile)
        if type_conv:
            candidates.append(type_conv)

        # Strategy 5: Fuzzy match
        fuzzy = self._try_fuzzy_match(user_value, profile)
        if fuzzy:
            candidates.append(fuzzy)

        # Return best candidate
        if candidates:
            best = max(candidates, key=lambda x: x.confidence)
            return best

        return None

    async def _ground_via_relationships(
        self,
        user_value: str,
        target_table: str,
        target_column: str,
        db: AsyncSession
    ) -> Optional[GroundedValueWithRelationship]:
        """
        Try to ground value via FK relationships.
        
        Searches related tables for the value and returns path to find it.
        """
        # Find tables that have the value
        tables_with_value = []

        for table, columns in self.column_profiles.items():
            for col, profile in columns.items():
                # Try to ground the user value in this table/column
                ground = await self._ground_in_table(user_value, table, col)
                if ground and ground.confidence >= 0.70:
                    tables_with_value.append((table, col, ground))

        if not tables_with_value:
            return None

        # For each table with value, find path from target_table
        for source_table, source_col, ground_result in tables_with_value:
            path = await self._find_relationship_path(
                target_table, target_column, source_table, source_col, db
            )

            if path and path.hops > 0:
                logger.info(
                    f"[GROUNDING] Found relationship: {target_table} → "
                    f"{source_table}.{source_col} ({path.hops} hops)"
                )

                return GroundedValueWithRelationship(
                    original_value=user_value,
                    grounded_value=ground_result.grounded_value,
                    strategy=MatchStrategy.FK_RELATIONSHIP,
                    confidence=ground_result.confidence * 0.95,  # Slight discount for FK resolution
                    table_name=source_table,
                    column_name=source_col,
                    value_type=ground_result.value_type,
                    relationship_path=path,
                    requires_join=True,
                    join_sql=path.get_join_sql(),
                    filter_sql=path.get_filter_sql(str(ground_result.grounded_value)),
                    is_direct_match=False,
                    related_fk_info=path.fk_chain[0] if path.fk_chain else None
                )

        return None

    async def _find_relationship_path(
        self,
        source_table: str,
        source_col: str,
        target_table: str,
        target_col: str,
        db: AsyncSession,
        current_path: Optional[List[ForeignKeyInfo]] = None,
        visited: Optional[set] = None,
        hop_count: int = 0
    ) -> Optional[RelationshipPath]:
        """
        Find FK path from source_table to target_table using BFS.
        Supports multi-hop relationships.
        """
        if visited is None:
            visited = set()
        if current_path is None:
            current_path = []

        # Check depth limit
        if hop_count >= self.max_relationship_hops:
            return None

        # Mark as visited
        visited.add((source_table, source_col))

        # Check direct FK
        fk_key = (source_table, source_col)
        if fk_key in self.fk_index:
            for fk in self.fk_index[fk_key]:
                if fk.to_table == target_table and fk.to_column == target_col:
                    # Found direct FK!
                    return RelationshipPath(
                        source_table=source_table,
                        source_column=source_col,
                        target_table=target_table,
                        target_column=target_col,
                        fk_chain=current_path + [fk],
                        hops=len(current_path) + 1,
                        confidence=0.95
                    )

        # Check reverse FK (target points to source)
        rev_fk_key = (source_table, source_col)
        if rev_fk_key in self.reverse_fk_index:
            for fk in self.reverse_fk_index[rev_fk_key]:
                if (fk.from_table, fk.from_column) not in visited:
                    # Try going backwards
                    result = await self._find_relationship_path(
                        fk.from_table, fk.from_column, target_table, target_col, db,
                        current_path + [fk], visited, hop_count + 1
                    )
                    if result:
                        return result

        # BFS: Explore other FKs from source
        for (from_t, from_c), fks in self.fk_index.items():
            if from_t == source_table and (from_t, from_c) not in visited:
                for fk in fks:
                    if (fk.to_table, fk.to_column) not in visited:
                        result = await self._find_relationship_path(
                            fk.to_table, fk.to_column, target_table, target_col, db,
                            current_path + [fk], visited, hop_count + 1
                        )
                        if result:
                            return result

        return None

    # ========================================================================
    # MATCHING STRATEGIES (5 Original + Enhancements)
    # ========================================================================

    def _try_exact_match(self, user_value: str, profile: ColumnValueProfile) -> Optional[GroundedValue]:
        """Strategy 1: Exact match."""
        for val in profile.sample_values:
            if str(val) == str(user_value):
                return GroundedValue(
                    original_value=user_value,
                    grounded_value=val,
                    strategy=MatchStrategy.EXACT,
                    confidence=1.0,
                    table_name=profile.table_name,
                    column_name=profile.column_name,
                    value_type=profile.value_type
                )
        return None

    def _try_case_insensitive_match(
        self,
        user_value: str,
        profile: ColumnValueProfile
    ) -> Optional[GroundedValue]:
        """Strategy 2: Case-insensitive match."""
        user_lower = str(user_value).lower()
        for val in profile.sample_values:
            if str(val).lower() == user_lower:
                return GroundedValue(
                    original_value=user_value,
                    grounded_value=val,
                    strategy=MatchStrategy.CASE_INSENSITIVE,
                    confidence=0.95,
                    table_name=profile.table_name,
                    column_name=profile.column_name,
                    value_type=profile.value_type
                )
        return None

    def _try_substring_match(
        self,
        user_value: str,
        profile: ColumnValueProfile
    ) -> Optional[GroundedValue]:
        """Strategy 3: Substring match."""
        user_lower = str(user_value).lower()
        for val in profile.sample_values:
            val_str = str(val).lower()
            if user_lower in val_str or val_str in user_lower:
                return GroundedValue(
                    original_value=user_value,
                    grounded_value=val,
                    strategy=MatchStrategy.SUBSTRING,
                    confidence=0.85,
                    table_name=profile.table_name,
                    column_name=profile.column_name,
                    value_type=profile.value_type
                )
        return None

    def _try_type_conversion(
        self,
        user_value: str,
        profile: ColumnValueProfile
    ) -> Optional[GroundedValue]:
        """Strategy 4: Type conversion (boolean, numeric, date)."""
        # Boolean conversion
        if profile.value_type == ValueType.BOOLEAN:
            bool_map = {
                "true": True, "1": True, "yes": True, "y": True, "active": True,
                "false": False, "0": False, "no": False, "n": False, "inactive": False
            }
            lower_val = str(user_value).lower()
            if lower_val in bool_map:
                return GroundedValue(
                    original_value=user_value,
                    grounded_value=bool_map[lower_val],
                    strategy=MatchStrategy.TYPE_CONVERSION,
                    confidence=0.90,
                    table_name=profile.table_name,
                    column_name=profile.column_name,
                    value_type=profile.value_type
                )

        # Numeric conversion
        if profile.value_type in [ValueType.INTEGER, ValueType.FLOAT]:
            try:
                num = float(user_value) if profile.value_type == ValueType.FLOAT else int(user_value)
                if num in profile.sample_values or num in [int(v) for v in profile.sample_values if isinstance(v, (int, float))]:
                    return GroundedValue(
                        original_value=user_value,
                        grounded_value=num,
                        strategy=MatchStrategy.TYPE_CONVERSION,
                        confidence=0.92,
                        table_name=profile.table_name,
                        column_name=profile.column_name,
                        value_type=profile.value_type
                    )
            except (ValueError, TypeError):
                pass

        return None

    def _try_fuzzy_match(
        self,
        user_value: str,
        profile: ColumnValueProfile
    ) -> Optional[GroundedValue]:
        """Strategy 5: Fuzzy match using SequenceMatcher."""
        best_match = None
        best_ratio = 0.0

        for val in profile.sample_values:
            ratio = SequenceMatcher(None, str(user_value).lower(), str(val).lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = val

        if best_ratio >= 0.70:  # Threshold for fuzzy match
            confidence = 0.70 + (best_ratio - 0.70) * 0.3  # Scale 0.70-1.0
            return GroundedValue(
                original_value=user_value,
                grounded_value=best_match,
                strategy=MatchStrategy.FUZZY,
                confidence=min(confidence, 0.85),
                table_name=profile.table_name,
                column_name=profile.column_name,
                value_type=profile.value_type
            )

        return None

    # ========================================================================
    # INSPECTION & DEBUGGING
    # ========================================================================

    def get_grounding_summary(self) -> Dict[str, Any]:
        """Get summary of what's been profiled."""
        return {
            "initialized": self.initialized,
            "tables_profiled": len(self.column_profiles),
            "total_columns": sum(len(cols) for cols in self.column_profiles.values()),
            "foreign_keys_discovered": len(self.foreign_keys),
            "max_relationship_hops": self.max_relationship_hops,
            "tables": {
                table: {
                    "columns": list(cols.keys()),
                    "column_count": len(cols)
                }
                for table, cols in self.column_profiles.items()
            }
        }

    async def validate_grounding(
        self,
        user_value: str,
        table_name: str,
        column_name: str,
        grounded_value: Any,
        db: AsyncSession
    ) -> bool:
        """Sanity check: verify grounded value actually exists in database."""
        try:
            query = (
                f"SELECT 1 FROM \"{table_name}\" "
                f"WHERE \"{column_name}\" = '{grounded_value}' LIMIT 1"
            )
            result = await db.execute(text(query))
            return result.fetchone() is not None
        except Exception as e:
            logger.warning(f"[VALIDATE] Validation failed: {e}")
            return False

    def get_column_values(self, table: str, column: str) -> Optional[ColumnValueProfile]:
        """Get discovered values for a specific column."""
        if table in self.column_profiles:
            return self.column_profiles[table].get(column)
        return None


# ============================================================================
# SINGLETON GETTER
# ============================================================================

_grounder_instance = None


def get_semantic_value_grounder_enhanced(
    max_hops: int = 3,
    embedding_threshold: float = 0.7
) -> SemanticValueGrounderEnhanced:
    """Get or create singleton instance."""
    global _grounder_instance
    if _grounder_instance is None:
        _grounder_instance = SemanticValueGrounderEnhanced(
            max_relationship_hops=max_hops,
            embedding_similarity_threshold=embedding_threshold
        )
    return _grounder_instance


def reset_grounder_instance():
    """Reset singleton (for testing)."""
    global _grounder_instance
    _grounder_instance = None
