"""
Plan-First SQL Generator - Stage 3 & 4 of Query Understanding Pipeline

Converts semantic concepts into structured query plans, then renders SQL.
This implements the critical missing piece: structured plan generation BEFORE SQL.

Architecture:
Semantic Intent → Canonical QueryPlan (from query_plan.py) → Compiler → SQL

This module now uses the CANONICAL QueryPlan from query_plan.py directly,
eliminating the legacy duplicate plan model.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession

# ── Value-scan cache ──────────────────────────────────────────────────────────
# Maps (schema, table, value_lower) → column_name | None
# Stable across requests: if 'TG' lives in 'state', that never changes mid-run.
# Cleared by the schema-change detector if DDL is detected.
_VALUE_SCAN_CACHE: Dict[Tuple[str, str, str], Optional[str]] = {}


def clear_value_scan_cache() -> None:
    """Called by the schema-change detector when DDL changes are detected."""
    global _VALUE_SCAN_CACHE
    _VALUE_SCAN_CACHE = {}

from .semantic_concept_extractor import SemanticIntent, FilterConcept, OperatorType, IntentType

# ============================================================================
# CANONICAL IMPORTS - Use the ONE true QueryPlan model
# ============================================================================
from .query_plan import (
    QueryPlan,
    SelectClause,
    FromClause,
    JoinClause as CanonicalJoinClause,
    BinaryCondition,
    ColumnRef,
    Literal as LiteralValue,
    GroupByClause,
    OrderByClause,
    OrderByField,
)
from .query_plan_compiler import compile_query_plan

logger = logging.getLogger(__name__)


def _rank_columns_by_concept(candidate_cols: List[str], concept: str) -> List[str]:
    """Rank candidate columns by semantic similarity to the filter concept.

    Uses token-overlap scoring between the concept name (e.g. "city",
    "account_status") and each column name so that the most relevant
    columns are probed first during value scanning.  No hardcoded
    suffix lists — the ranking is derived purely from the concept text.
    """
    from difflib import SequenceMatcher

    # Normalise: split on underscores / camelCase boundaries → lowercase tokens
    import re as _re

    def _tokens(name: str) -> set:
        parts = _re.split(r"[_\s]+", name.lower())
        return {p for p in parts if p}

    concept_tokens = _tokens(concept)

    def _score(col: str) -> float:
        col_tokens = _tokens(col)
        # Token overlap gives a strong signal (e.g. concept="city", col="city")
        overlap = len(concept_tokens & col_tokens)
        # Sequence similarity covers partial / substring matches
        seq = SequenceMatcher(None, concept.lower(), col.lower()).ratio()
        return -(overlap + seq)  # Negative so lower = better (sorted ascending)

    return sorted(candidate_cols, key=_score)


@dataclass
class ColumnMapping:
    """Maps semantic concepts to actual database columns"""
    concept: str           # Semantic concept (e.g., 'birth_month')
    table: str            # Database table
    column: str           # Database column 
    data_type: str        # Column data type
    confidence: float = 1.0


class SemanticGrounder:
    """
    Maps semantic concepts to actual database schema.
    Stage 2 of the pipeline - grounding abstract concepts to concrete columns.
    
    DELEGATES to SemanticConceptDiscovery for all heuristic/pattern-based mapping.
    This keeps SQL generation logic clean and focused.
    
    Now produces CANONICAL QueryPlan from query_plan.py.
    """
    
    def __init__(self, db: AsyncSession, schema_name: Optional[str] = None):
        from app.config import settings as _settings
        self.db = db
        self.schema_name = schema_name or _settings.postgres_schema
        self._concept_discovery = None  # Lazy-initialized
        self._init_lock = asyncio.Lock()  # Prevent double-initialization race condition

        logger.info(f"[SEMANTIC_GROUNDER] Initialized for schema: {self.schema_name}")

    async def _get_concept_discovery(self):
        """Get or create concept discovery instance (thread-safe via asyncio.Lock)."""
        async with self._init_lock:
            if self._concept_discovery is None:
                from .semantic_concept_discovery import get_concept_discovery
                self._concept_discovery = await get_concept_discovery(self.db, self.schema_name)
        return self._concept_discovery
    
    async def get_concept_mappings(self) -> Dict[str, List[str]]:
        """Get concept mappings via SemanticConceptDiscovery.
        
        Returns simple dict format for backward compatibility.
        """
        discovery = await self._get_concept_discovery()
        all_concepts = await discovery.discover_all()
        
        # Convert DiscoveredConcept format to simple dict format
        mappings: Dict[str, List[str]] = {}
        for concept, matches in all_concepts.items():
            mappings[concept] = [m.column if m.column != "*" else m.table for m in matches]
        
        return mappings
    
    async def ground_semantic_intent(self, intent: SemanticIntent) -> Tuple[QueryPlan, List[ColumnMapping]]:
        """
        Ground semantic intent to CANONICAL QueryPlan.
        
        This is the CRITICAL step that maps:
        - "clients" → "customers" table
        - "birth_month" → "dob" column with MONTH extraction
        - "gender" → "gender" column
        
        Returns:
            Tuple of (canonical QueryPlan, column mappings for debugging)
        """
        # Find primary table
        primary_table = await self._find_primary_table(intent.entity)
        if not primary_table:
            raise ValueError(f"Could not find table for entity: {intent.entity}")
        
        # Ground all filter concepts to columns
        mappings = []
        where_conditions = []
        raw_expressions = []  # For temporal expressions that need special SQL
        
        for filter_concept in intent.filters:
            column_mapping = await self._ground_filter_concept(filter_concept, primary_table)
            if column_mapping:
                mappings.append(column_mapping)
                
                # Convert to canonical WHERE condition
                condition, raw_expr = self._create_canonical_condition(filter_concept, column_mapping)
                # If raw_expr is provided, it REPLACES the placeholder condition (for temporal expressions)
                if raw_expr:
                    raw_expressions.append(raw_expr)
                elif condition:
                    # Only add the canonical condition if no raw expression replaces it
                    where_conditions.append(condition)
        
        # Create SELECT clause based on intent
        select_clause = self._create_canonical_select(intent)
        
        # Create FROM clause - no alias for single-table queries
        from_clause = FromClause(
            table=f"{self.schema_name}.{primary_table}",
            alias=None  # Don't use alias for single-table queries - avoids reference issues
        )
        
        # Create GROUP BY if aggregating
        group_by = None
        if intent.intent in (IntentType.COUNT, IntentType.AGGREGATE):
            # Aggregates typically don't need GROUP BY for total counts
            pass
        
        # Build the canonical QueryPlan
        plan = QueryPlan(
            intent="data_query",
            select=select_clause,
            from_=from_clause,
            joins=None,
            where=where_conditions if where_conditions else None,
            group_by=group_by,
            having=None,
            order_by=None,
            limit=1000,  # Default safety limit
            offset=None,
            metadata={
                "semantic_intent": intent.intent.value if hasattr(intent.intent, 'value') else str(intent.intent),
                "raw_expressions": raw_expressions,  # Store temporal expressions in metadata
                "source": "plan_first_sql_generator"
            }
        )
        
        logger.info(f"[SEMANTIC_GROUNDER] Grounded to canonical plan: {len(where_conditions)} conditions, table={primary_table}")
        return plan, mappings
    
    async def _find_primary_table(self, entity: Optional[str]) -> Optional[str]:
        """Find the primary table for a semantic entity.
        
        Delegates to SemanticConceptDiscovery for entity resolution.
        """
        if not entity:
            return None
        
        # Use concept discovery for table resolution
        discovery = await self._get_concept_discovery()
        table = await discovery.get_table_for_entity(entity)
        
        if table:
            logger.debug(f"[SEMANTIC_GROUNDER] Resolved entity '{entity}' -> table '{table}'")
            return table
        
        # Fallback: direct database check
        from sqlalchemy import text
        result = await self.db.execute(
            text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = :schema_name
                ORDER BY table_name
            """),
            {"schema_name": self.schema_name},
        )
        tables = [row[0] for row in result]
        
        # Direct match
        if entity in tables:
            return entity

        # Fuzzy stem/plural match (customers ≈ customer)
        entity_stem = entity.lower().rstrip('s')
        for table in tables:
            if entity_stem == table.rstrip('s').lower():
                return table

        # LLM semantic fallback — no hardcoded synonyms
        from .semantic_concept_discovery import SemanticConceptDiscovery
        discovery = SemanticConceptDiscovery(self.db, self.schema_name)
        discovery._cache.tables = list(tables)  # Reuse already-fetched table list
        resolved = await discovery._llm_resolve_entity(entity, list(tables))
        if resolved:
            logger.info("[SEMANTIC_GROUNDER] LLM resolved: '%s' → table '%s'", entity, resolved)
            return resolved

        return None
    
    async def _ground_filter_concept(self, filter_concept: FilterConcept, table: str) -> Optional[ColumnMapping]:
        """Ground a semantic filter concept to an actual column.
        
        Delegates to SemanticConceptDiscovery for concept-to-column mapping.
        """
        # Try concept discovery first
        discovery = await self._get_concept_discovery()
        discovered = await discovery.get_concept_mapping(filter_concept.concept)
        
        if discovered and discovered.table == table:
            # Get column data type from schema (parameterized to prevent SQL injection)
            from sqlalchemy import text
            result = await self.db.execute(
                text("""
                    SELECT data_type
                    FROM information_schema.columns
                    WHERE table_schema = :schema_name
                      AND table_name = :table_name
                      AND column_name = :column_name
                """),
                {
                    "schema_name": self.schema_name,
                    "table_name": table,
                    "column_name": discovered.column,
                },
            )
            row = result.fetchone()
            data_type = row[0] if row else "text"
            if not row:
                logger.warning(
                    "[SEMANTIC_GROUNDER] Could not resolve data type for %s.%s — defaulting to 'text'",
                    table, discovered.column,
                )
            
            return ColumnMapping(
                concept=filter_concept.concept,
                table=table,
                column=discovered.column,
                data_type=data_type,
                confidence=discovered.confidence
            )
        
        # Fallback: Get column info for table and do matching (parameterized)
        from sqlalchemy import text
        result = await self.db.execute(
            text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = :schema_name AND table_name = :table_name
                ORDER BY column_name
            """),
            {"schema_name": self.schema_name, "table_name": table},
        )
        columns = {row[0]: row[1] for row in result}
        concept_lower = filter_concept.concept.lower()

        # 1. Exact column name match
        if filter_concept.concept in columns:
            return ColumnMapping(
                concept=filter_concept.concept,
                table=table,
                column=filter_concept.concept,
                data_type=columns[filter_concept.concept],
            )

        # 2. Case-insensitive match
        for col_name, col_type in columns.items():
            if col_name.lower() == concept_lower:
                return ColumnMapping(
                    concept=filter_concept.concept,
                    table=table,
                    column=col_name,
                    data_type=col_type,
                )

        # 3. Partial / contains match (e.g., "state" matches "state_code", "state_id")
        for col_name, col_type in columns.items():
            col_lower = col_name.lower()
            if concept_lower in col_lower or col_lower in concept_lower:
                logger.debug(
                    "[SEMANTIC_GROUNDER] Partial match: concept '%s' → column '%s'",
                    filter_concept.concept, col_name,
                )
                return ColumnMapping(
                    concept=filter_concept.concept,
                    table=table,
                    column=col_name,
                    data_type=col_type,
                    confidence=0.7,
                )

        # 4. DB-level value scan — find which string column actually contains the filter value.
        # Results are cached in _VALUE_SCAN_CACHE so the same value is never scanned twice.
        filter_value = filter_concept.value or (
            filter_concept.values[0] if filter_concept.values else None
        )
        if filter_value is not None:
            cache_key = (self.schema_name, table, str(filter_value).lower())

            # Check cache first
            if cache_key in _VALUE_SCAN_CACHE:
                cached_col = _VALUE_SCAN_CACHE[cache_key]
                if cached_col:
                    logger.debug(
                        "[SEMANTIC_GROUNDER] Cache hit: value '%s' → column '%s' (cached)",
                        filter_value, cached_col,
                    )
                    return ColumnMapping(
                        concept=filter_concept.concept,
                        table=table,
                        column=cached_col,
                        data_type=columns.get(cached_col, "text"),
                        confidence=0.85,
                    )
                else:
                    # Cached as "not found" — skip scan
                    logger.debug(
                        "[SEMANTIC_GROUNDER] Cache hit: value '%s' not in any column (cached)",
                        filter_value,
                    )
            else:
                # Only scan low-cardinality string columns (varchar/char/text/enum)
                candidate_cols = [
                    col_name for col_name, col_type in columns.items()
                    if "char" in col_type.lower() or "text" in col_type.lower()
                    or col_type.lower() in ("user-defined",)
                ]
                # Rank columns by semantic similarity to the filter concept name
                # so that concept "city" probes column "city" before "customer_code"
                candidate_cols = _rank_columns_by_concept(
                    candidate_cols, filter_concept.concept
                )
                from sqlalchemy import text as _text
                matched_col = None
                for col_name in candidate_cols:
                    try:
                        probe = await self.db.execute(
                            _text(
                                f"SELECT 1 FROM {self.schema_name}.{table}"
                                f" WHERE LOWER({col_name}) = LOWER(:val) LIMIT 1"
                            ),
                            {"val": str(filter_value)},
                        )
                        if probe.fetchone():
                            matched_col = col_name
                            break
                    except Exception:
                        continue  # Column might not support = comparison; skip

                # Store result (hit or miss) in cache
                _VALUE_SCAN_CACHE[cache_key] = matched_col

                if matched_col:
                    logger.info(
                        "[SEMANTIC_GROUNDER] Value scan matched: value '%s' found in column '%s.%s'",
                        filter_value, table, matched_col,
                    )
                    return ColumnMapping(
                        concept=filter_concept.concept,
                        table=table,
                        column=matched_col,
                        data_type=columns[matched_col],
                        confidence=0.85,
                    )

        # 5. Fuzzy value resolution — exact scan failed.
        #    Phase A: Targeted ILIKE probe — search for the value as a substring
        #             in each candidate column (handles aliases like "vizag" → "Visakhapatnam").
        #    Phase B: LLM-based matching on a sample of distinct values (fallback).
        if filter_value is not None:
            candidate_cols = [
                col_name for col_name, col_type in columns.items()
                if "char" in col_type.lower() or "text" in col_type.lower()
                or col_type.lower() in ("user-defined",)
            ]
            # Rank columns by semantic similarity to the filter concept name
            candidate_cols = _rank_columns_by_concept(
                candidate_cols, filter_concept.concept
            )
            from sqlalchemy import text as _text2

            # Phase A: Targeted ILIKE probe — fast, no LLM call needed
            for col_name in candidate_cols:
                try:
                    like_result = await self.db.execute(
                        _text2(
                            f"SELECT DISTINCT {col_name} FROM {self.schema_name}.{table}"
                            f" WHERE LOWER({col_name}) LIKE :pattern LIMIT 5"
                        ),
                        {"pattern": f"%{str(filter_value).lower()}%"},
                    )
                    matches = [str(r[0]) for r in like_result.fetchall() if r[0]]
                    if matches:
                        resolved = matches[0]
                        filter_concept.value = resolved
                        _VALUE_SCAN_CACHE[(self.schema_name, table, str(filter_value).lower())] = col_name
                        _VALUE_SCAN_CACHE[(self.schema_name, table, resolved.lower())] = col_name
                        logger.info(
                            "[SEMANTIC_GROUNDER] ILIKE probe resolved: '%s' → '%s' in column '%s.%s'",
                            filter_value, resolved, table, col_name,
                        )
                        return ColumnMapping(
                            concept=filter_concept.concept,
                            table=table,
                            column=col_name,
                            data_type=columns[col_name],
                            confidence=0.80,
                        )
                except Exception:
                    continue

            # Phase B: LLM-based fuzzy matching on sampled distinct values
            from app.llm import call_llm as _call_llm

            for idx, col_name in enumerate(candidate_cols):
                try:
                    # Top-ranked columns get a larger sample for better coverage
                    sample_limit = 100 if idx < 2 else 30
                    # Fetch a sample of distinct values from this column
                    sample_result = await self.db.execute(
                        _text2(
                            f"SELECT DISTINCT {col_name} FROM {self.schema_name}.{table}"
                            f" WHERE {col_name} IS NOT NULL LIMIT :lim"
                        ),
                        {"lim": sample_limit},
                    )
                    sample_vals = [str(r[0]) for r in sample_result.fetchall() if r[0]]
                    if not sample_vals:
                        continue

                    sample_list = ", ".join(f'"{v}"' for v in sample_vals)
                    llm_prompt = (
                        f"The user typed '{filter_value}' as a filter value "
                        f"for the concept '{filter_concept.concept}' "
                        f"(column: '{col_name}').\n"
                        f"Actual values in this column: {sample_list}\n\n"
                        f"Which value from the list BEST matches what the user meant?\n"
                        f"IMPORTANT: Only match if the value semantically relates to "
                        f"'{filter_concept.concept}'. Reply 'none' if the column "
                        f"values don't represent {filter_concept.concept} data.\n"
                        f"Reply with ONLY the exact matching value from the list, "
                        f"or 'none' if nothing fits. Do not explain."
                    )
                    raw = await _call_llm(
                        messages=[{"role": "user", "content": llm_prompt}],
                        max_tokens=30,
                        temperature=0.0,
                    )
                    resolved = str(raw).strip().strip("'\"")
                    if resolved and resolved.lower() != "none" and resolved in sample_vals:
                        # Update the filter concept value so the SQL uses the resolved value
                        filter_concept.value = resolved
                        # Cache both the original and resolved value
                        _VALUE_SCAN_CACHE[(self.schema_name, table, str(filter_value).lower())] = col_name
                        _VALUE_SCAN_CACHE[(self.schema_name, table, resolved.lower())] = col_name
                        logger.info(
                            "[SEMANTIC_GROUNDER] Fuzzy value resolved: '%s' → '%s' in column '%s.%s'",
                            filter_value, resolved, table, col_name,
                        )
                        return ColumnMapping(
                            concept=filter_concept.concept,
                            table=table,
                            column=col_name,
                            data_type=columns[col_name],
                            confidence=0.75,
                        )
                except Exception as _fuzz_err:
                    logger.debug("[SEMANTIC_GROUNDER] Fuzzy scan error on column '%s': %s", col_name, _fuzz_err)
                    continue

        # 6. Nothing worked — drop the filter rather than inject a wrong column name.
        # The caller's loop uses `if column_mapping:` so returning None silently skips it.
        logger.warning(
            "[SEMANTIC_GROUNDER] Could not ground concept '%s' (value=%s) to any column in '%s' "
            "— filter dropped to avoid wrong SQL",
            filter_concept.concept, filter_value, table,
        )
        return None
    
    def _create_canonical_condition(
        self, 
        filter_concept: FilterConcept, 
        mapping: ColumnMapping
    ) -> Tuple[Optional[BinaryCondition], Optional[str]]:
        """
        Create canonical BinaryCondition from semantic filter and column mapping.
        
        Returns:
            Tuple of (condition, raw_expression_for_temporal)
        """
        raw_expr = None
        
        if filter_concept.operator == OperatorType.EQUALS:
            condition = BinaryCondition(
                left=ColumnRef(column=mapping.column, table=None),  # No table prefix for single-table queries
                op="=",
                right=LiteralValue(
                    value=filter_concept.value,
                    type="string" if isinstance(filter_concept.value, str) else "number"
                )
            )
            return condition, None
            
        elif filter_concept.operator == OperatorType.IN:
            in_values = [v for v in (filter_concept.values or []) if v is not None]

            # "in AP" — LLM used English "in" as SQL IN; single value → use = instead
            # Also handles empty IN list which would produce invalid SQL `IN ()`
            if len(in_values) == 0:
                # Fall back to the scalar value if present
                scalar = filter_concept.value
                if scalar is None:
                    return None, None  # No value at all — skip filter
                condition = BinaryCondition(
                    left=ColumnRef(column=mapping.column, table=None),
                    op="=",
                    right=LiteralValue(
                        value=scalar,
                        type="string" if isinstance(scalar, str) else "number",
                    ),
                )
            elif len(in_values) == 1:
                # Single value — = is cleaner and avoids `IN ('AP')` verbosity
                condition = BinaryCondition(
                    left=ColumnRef(column=mapping.column, table=None),
                    op="=",
                    right=LiteralValue(
                        value=in_values[0],
                        type="string" if isinstance(in_values[0], str) else "number",
                    ),
                )
            else:
                condition = BinaryCondition(
                    left=ColumnRef(column=mapping.column, table=None),
                    op="in",
                    right=[
                        LiteralValue(value=v, type="string" if isinstance(v, str) else "number")
                        for v in in_values
                    ],
                )
            return condition, None
            
        elif filter_concept.operator == OperatorType.MONTH_EQUALS:
            # Temporal month extraction — emit dialect-aware expression.
            # PostgreSQL: EXTRACT(MONTH FROM col)
            # MySQL / SQL Server: MONTH(col)
            # SQLite: CAST(strftime('%m', col) AS INTEGER)
            dialect = getattr(self, "dialect", "postgresql")
            if dialect in ("mysql", "mssql"):
                raw_expr = f"MONTH({mapping.column}) = {filter_concept.value}"
            elif dialect == "sqlite":
                raw_expr = f"CAST(strftime('%m', {mapping.column}) AS INTEGER) = {filter_concept.value}"
            else:  # postgresql and others
                raw_expr = f"EXTRACT(MONTH FROM {mapping.column}) = {filter_concept.value}"
            # Placeholder condition keeps plan structure intact; real SQL uses raw_expr
            condition = BinaryCondition(
                left=ColumnRef(column=mapping.column, table=None),
                op="=",
                right=LiteralValue(value=filter_concept.value, type="number"),
            )
            return condition, raw_expr
        
        return None, None
    
    def _create_canonical_select(self, intent: SemanticIntent) -> SelectClause:
        """Create canonical SELECT clause based on intent"""
        if intent.intent == IntentType.COUNT:
            return SelectClause(fields=["COUNT(*)"], distinct=False)
        elif intent.intent == IntentType.LIST:
            return SelectClause(fields=["*"], distinct=False)
        else:
            return SelectClause(fields=["*"], distinct=False)


class PlanFirstSQLGenerator:
    """
    Generates SQL from CANONICAL QueryPlan using the official compiler.
    
    This is a thin wrapper that:
    1. Receives canonical QueryPlan (from query_plan.py)
    2. Handles special cases (temporal expressions)
    3. Routes to query_plan_compiler for actual SQL generation
    """
    
    def __init__(self, dialect: str = "postgresql", schema_name: Optional[str] = None):
        """Initialize generator.

        Args:
            dialect: SQL dialect for compilation
            schema_name: Database schema name
        """
        from app.config import settings as _settings
        self.dialect = dialect
        self.schema_name = schema_name or _settings.postgres_schema
    
    def render_sql_from_plan(self, plan: QueryPlan) -> str:
        """
        Render SQL from CANONICAL QueryPlan using compiler.
        
        This is deterministic - same plan always produces same SQL.
        Routes through official query_plan_compiler.
        """
        # Check for raw temporal expressions in metadata
        raw_expressions = []
        if plan.metadata and plan.metadata.get("raw_expressions"):
            raw_expressions = plan.metadata["raw_expressions"]
        
        try:
            # Use canonical compiler
            sql = compile_query_plan(plan, dialect=self.dialect)
            
            # If we have raw expressions (like temporal EXTRACT), inject them
            if raw_expressions:
                sql = self._inject_raw_expressions(sql, raw_expressions)
            
            logger.info(f"[PLAN_SQL_GENERATOR] Generated SQL via canonical compiler")
            return sql
            
        except Exception as e:
            logger.error(f"[PLAN_SQL_GENERATOR] Canonical compiler failed: {e}")
            raise
    
    def _inject_raw_expressions(self, sql: str, raw_expressions: List[str]) -> str:
        """
        Inject raw temporal expressions into WHERE clause.
        
        This handles special cases like EXTRACT(MONTH FROM ...) that
        aren't directly supported by the canonical model.
        """
        if not raw_expressions:
            return sql
        
        sql_upper = sql.upper()
        
        # Find WHERE clause boundaries - ends at GROUP BY, ORDER BY, LIMIT, or end
        where_idx = sql_upper.find("WHERE")
        
        # Find where WHERE clause ends (start of next clause)
        end_markers = ["GROUP BY", "ORDER BY", "HAVING", "LIMIT", ";"]
        where_end = len(sql)
        for marker in end_markers:
            pos = sql_upper.find(marker)
            if pos > where_idx and pos < where_end:
                where_end = pos
        
        if where_idx != -1:
            # There's an existing WHERE clause - find where it ends
            where_content = sql[where_idx + 5:where_end].strip()
            rest_of_sql = sql[where_end:]
            
            # Combine original WHERE conditions with raw expressions
            combined = " AND ".join([where_content] + raw_expressions) if where_content else " AND ".join(raw_expressions)
            sql = sql[:where_idx] + f"WHERE {combined}\n{rest_of_sql}"
        else:
            # No WHERE yet, add one before GROUP BY/ORDER BY/LIMIT
            insert_pos = where_end
            where_clause = f"WHERE {' AND '.join(raw_expressions)}\n"
            sql = sql[:insert_pos] + where_clause + sql[insert_pos:]
        
        return sql.strip()


class PlanFirstQueryHandler:
    """
    Complete plan-first query handler.
    Orchestrates the full pipeline: SemanticIntent → Canonical QueryPlan → SQL
    
    This is the ONLY entry point for plan-first SQL generation.
    All SQL is generated through the canonical pipeline.
    """
    
    def __init__(self, db: AsyncSession, dialect: str = "postgresql", schema_name: Optional[str] = None):
        """Initialize handler.

        Args:
            db: Database session
            dialect: SQL dialect for compilation
            schema_name: Database schema name
        """
        from app.config import settings as _settings
        self.db = db
        self.dialect = dialect
        self.schema_name = schema_name or _settings.postgres_schema
        self.grounder = SemanticGrounder(db, schema_name=self.schema_name)
        self.sql_generator = PlanFirstSQLGenerator(dialect=dialect, schema_name=self.schema_name)

    async def handle_semantic_intent(self, intent: SemanticIntent) -> Tuple[str, Dict[str, Any]]:
        """
        Handle semantic intent end-to-end: ground to canonical plan, compile to SQL.
        
        Returns:
            Tuple of (generated_sql, debug_info)
        """
        try:
            # Ground semantic intent to CANONICAL query plan
            plan, mappings = await self.grounder.ground_semantic_intent(intent)
            
            # Render SQL using canonical compiler
            sql = self.sql_generator.render_sql_from_plan(plan)
            
            debug_info = {
                "semantic_intent": intent.to_dict(),
                "query_plan": plan.to_dict(),
                "column_mappings": [
                    {"concept": m.concept, "table": m.table, "column": m.column}
                    for m in mappings
                ],
                "pipeline_stage": "canonical_plan_first_generation",
                "compiler_used": "query_plan_compiler"
            }
            
            logger.info(f"[PLAN_FIRST_HANDLER] Generated SQL via canonical pipeline with {len(mappings)} concept mappings")
            return sql, debug_info
            
        except Exception as e:
            logger.error(f"[PLAN_FIRST_HANDLER] Error in canonical plan-first generation: {e}")
            raise


async def get_plan_first_handler(
    db: AsyncSession,
    dialect: str = "postgresql",
    schema_name: Optional[str] = None,
) -> PlanFirstQueryHandler:
    """Get plan-first query handler instance.
    
    Args:
        db: Database session
        dialect: SQL dialect for compilation
        schema_name: Database schema name
    
    Returns:
        PlanFirstQueryHandler that uses canonical pipeline
    """
    return PlanFirstQueryHandler(db, dialect=dialect, schema_name=schema_name)
