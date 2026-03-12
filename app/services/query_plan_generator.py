"""
LLM Query Plan Generator - Generates canonical QueryPlan from natural language.

Architecture:
    NL -> LLM -> (internal parsing) -> canonical QueryPlan -> compiler -> SQL

This module uses the CANONICAL QueryPlan from query_plan.py.
The internal GeneratedQueryPlan class is only used for parsing LLM output,
then immediately converted to canonical QueryPlan.

SQL compilation goes directly through:
    query_plan_compiler.py

PUBLIC API:
    - QueryPlanGenerator: Main class for generating query plans from NL
    - generate_query_plan(): Convenience function

INTERNAL TYPES (not for external use):
    - GeneratedQueryPlan: Intermediate LLM parsing structure
    - JoinType, AggregationType, etc.: Internal enums for LLM output parsing

For canonical types, import from query_plan.py instead.
"""

from __future__ import annotations

# Public API exports
__all__ = [
    "QueryPlanGenerator",
    "create_query_plan",  # Convenience function
]

import logging
import random
import string
from typing import Dict, List, Optional, Any, Literal, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

# Import CANONICAL QueryPlan types (including ColumnSelectionIntent/Analysis)
from .query_plan import (
    QueryPlan,
    SelectClause,
    FromClause,
    JoinClause as CanonicalJoinClause,
    JoinCondition as CanonicalJoinCondition,
    BinaryCondition,
    GroupByClause,
    OrderByClause,
    OrderByField as CanonicalOrderByField,
    ColumnSelectionIntent,
    ColumnSelectionAnalysis,
    CTEClause as CanonicalCTEClause,
    SetOperation as CanonicalSetOperation,
)

# Import value grounding for column discovery
from .value_based_column_grounding import ValueBasedColumnGrounder


def generate_random_alias(table_name: str = "") -> str:
    """Generate a random unique table alias (e.g., 't_aBc123').
    
    Args:
        table_name: Optional table name for context (not used for generation for randomness)
        
    Returns:
        Random alias like 't_abc123xyz'
    """
    random_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"t_{random_chars}"


class JoinType(str, Enum):
    """SQL join types."""
    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"
    CROSS = "CROSS"


class AggregationType(str, Enum):
    """Aggregation functions."""
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    STDDEV = "STDDEV"


class OrderDirection(str, Enum):
    """Sort directions."""
    ASC = "ASC"
    DESC = "DESC"


class RightKind(str, Enum):
    """Type of right-hand side in a condition."""
    LITERAL = "literal"        # "AP", 100, True
    LIST = "list"              # ["AP","TS"]
    RANGE = "range"            # ["2024-01-01","2024-12-31"]
    COLUMN = "column"          # other_col reference
    SUBQUERY = "subquery"      # "SELECT ..."
    RAW = "raw"                # "NOW()", "DATE_TRUNC(...)"


class SetOperation(str, Enum):
    """Set operation types."""
    UNION = "UNION"
    UNION_ALL = "UNION ALL"
    INTERSECT = "INTERSECT"
    EXCEPT = "EXCEPT"


# Re-export from canonical query_plan for backward compatibility
# These are now defined in query_plan.py as the canonical location
# ColumnSelectionIntent = ColumnSelectionIntent  # Already imported
# ColumnSelectionAnalysis = ColumnSelectionAnalysis  # Already imported


@dataclass
class JoinCondition:
    """Single join condition (left_col = right_col)."""
    left_col: str  # table.column
    right_col: str  # table.column
    operator: str = "="  # =, <, >, etc.


@dataclass
class JoinClause:
    """JOIN clause (all joins are left joins by default for safety)."""
    join_type: JoinType = JoinType.INNER
    right_table: str = ""
    right_schema: Optional[str] = None
    conditions: List[JoinCondition] = field(default_factory=list)
    
    @property
    def full_table_name(self) -> str:
        """Schema-qualified table name."""
        if self.right_schema:
            return f"{self.right_schema}.{self.right_table}"
        return self.right_table


@dataclass
class AggregateField:
    """Aggregated column (COUNT(*), SUM(amount), etc.)."""
    function: AggregationType
    column: Optional[str] = None  # None for COUNT(*)
    alias: Optional[str] = None


@dataclass
class WhereCondition:
    """Single WHERE condition."""
    left: str  # column name or expression
    operator: str  # =, <, >, LIKE, IN, BETWEEN, EXISTS, IS NULL, etc.
    right: Any = None  # value, list of values, subquery, or range
    right_kind: RightKind = RightKind.LITERAL  # Type of right-hand side
    is_literal: bool = True  # Legacy field (deprecated - use right_kind instead)


@dataclass
class OrderByField:
    """ORDER BY field."""
    column: str
    direction: OrderDirection = OrderDirection.ASC


@dataclass
class GeneratedQueryPlan:
    """
    LLM-generated query plan with clarification support.
    
    This is the output format from LLM plan generation. It includes fields
    specific to the LLM interaction (clarification_needed, clarification_question,
    column_selection intent, etc.) that are not part of the canonical QueryPlan.
    
    Architecture:
        NL -> LLM -> GeneratedQueryPlan -> convert -> canonical QueryPlan -> compiler -> SQL
    
    For the canonical QueryPlan model (used for SQL compilation), see query_plan.py.
    Conversion is done via query_plan_unifier.convert_from_generator_plan().
    """
    # SELECT clause
    select_expressions: List[str] = field(default_factory=list)  # column names or expressions
    select_aggregates: List[AggregateField] = field(default_factory=list)  # sum(x), count(*), etc.
    distinct: bool = False  # SELECT DISTINCT
    
    # FROM clause
    from_table: str = ""
    from_schema: Optional[str] = None
    from_alias: Optional[str] = None
    
    # JOIN clauses
    joins: List[JoinClause] = field(default_factory=list)
    
    # WHERE clause
    where_conditions: List[WhereCondition] = field(default_factory=list)
    
    # GROUP BY
    group_by: List[str] = field(default_factory=list)
    
    # HAVING
    having_conditions: List[WhereCondition] = field(default_factory=list)
    
    # ORDER BY
    order_by: List[OrderByField] = field(default_factory=list)
    
    # PAGINATION
    limit: Optional[int] = None  # LIMIT clause
    offset: Optional[int] = None  # OFFSET clause
    
    # WINDOW FUNCTIONS (optional - can also be in select_expressions)
    window_expressions: List[str] = field(default_factory=list)  # ROW_NUMBER(), RANK(), etc.
    
    # CTEs and SET OPERATIONS (composable for future expansion)
    ctes: List[Dict[str, Any]] = field(default_factory=list)  # [{"name": "...", "sql": "..."}]
    set_op: Optional[Dict[str, Any]] = None  # {"type": "UNION", "right_sql": "..."}
    
    # CLARIFICATION FIELDS (LLM-driven)
    clarification_needed: bool = False  # Does LLM need user input?
    clarification_question: Optional[str] = None  # The question to ask
    clarification_options: List[str] = field(default_factory=list)  # Multiple choice options
    
    # SEMANTIC ANALYSIS - Column Selection Intent (LLM-determined, not hardcoded)
    column_selection: Optional[ColumnSelectionAnalysis] = None
    
    # Metadata
    intent: str = ""  # "list_all", "aggregate", "filter", etc.
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def from_table_qualified(self) -> str:
        """Schema-qualified table name."""
        if self.from_schema:
            return f"{self.from_schema}.{self.from_table}"
        return self.from_table
    
    def is_valid(self) -> Tuple[bool, str]:
        """Validate plan structure."""
        if not self.from_table:
            return False, "No FROM table specified"
        
        if not self.select_expressions and not self.select_aggregates:
            return False, "No SELECT columns or aggregates specified"
        
        return True, ""
    
    def to_canonical(self) -> QueryPlan:
        """
        Convert this GeneratedQueryPlan to canonical QueryPlan.
        
        This is the bridge from LLM output format to SQL compilation format.
        """
        # Build SELECT clause
        select_fields = list(self.select_expressions) if self.select_expressions else []
        
        # Handle aggregates - convert to string expressions
        for agg in (self.select_aggregates or []):
            func = agg.function.value if hasattr(agg.function, 'value') else str(agg.function)
            col = agg.column or "*"
            expr = f"{func}({col})"
            if agg.alias:
                expr += f" AS {agg.alias}"
            select_fields.append(expr)
        
        select_clause = SelectClause(
            fields=select_fields if select_fields else ["*"],
            distinct=self.distinct
        )
        
        # Build FROM clause
        from_clause = FromClause(
            table=self.from_table,
            alias=self.from_alias
        )
        
        # Build JOIN clauses
        joins = []
        for gen_join in (self.joins or []):
            # Map join type
            join_type_map = {
                "INNER": "inner",
                "LEFT": "left", 
                "RIGHT": "right",
                "FULL": "full",
                "CROSS": "cross"
            }
            join_type = join_type_map.get(
                gen_join.join_type.value if hasattr(gen_join.join_type, 'value') else str(gen_join.join_type).upper(),
                "inner"
            )
            
            # Build join conditions
            on_conditions = []
            for cond in (gen_join.conditions or []):
                on_conditions.append(CanonicalJoinCondition(
                    left=cond.left_col,
                    op=cond.operator,
                    right=cond.right_col
                ))
            
            joins.append(CanonicalJoinClause(
                type=join_type,
                table=gen_join.right_table,
                alias=None,
                on=on_conditions if on_conditions else None
            ))
        
        # Build WHERE conditions
        where_conditions = []
        for cond in (self.where_conditions or []):
            where_conditions.append(BinaryCondition(
                left=cond.left,
                op=cond.operator,
                right=cond.right
            ))
        
        # Build GROUP BY
        group_by = None
        if self.group_by:
            group_by = GroupByClause(fields=list(self.group_by))
        
        # Build ORDER BY
        order_by = None
        if self.order_by:
            order_fields = []
            for ob in self.order_by:
                direction = ob.direction.value.lower() if hasattr(ob.direction, 'value') else str(ob.direction).lower()
                order_fields.append(CanonicalOrderByField(expr=ob.column, direction=direction))
            order_by = OrderByClause(fields=order_fields)
        
        # Window functions — append as raw string expressions to SELECT fields
        # (they are already embedded in select_expressions by the LLM, but
        #  window_expressions provides an explicit list as a fallback)
        for wexpr in (self.window_expressions or []):
            if wexpr and wexpr not in select_clause.fields:
                select_clause.fields.append(wexpr)

        # CTEs
        canonical_ctes = None
        if self.ctes:
            canonical_ctes = []
            for cte in self.ctes:
                if isinstance(cte, dict):
                    canonical_ctes.append(CanonicalCTEClause(
                        name=cte.get("name", "cte"),
                        raw_sql=cte.get("sql", cte.get("raw_sql", "")),
                        recursive=cte.get("recursive", False),
                    ))

        # Set operation
        canonical_set_op = None
        if self.set_op and isinstance(self.set_op, dict):
            _op_map = {
                "UNION": "union", "UNION ALL": "union_all",
                "INTERSECT": "intersect", "EXCEPT": "except",
            }
            _op = _op_map.get(str(self.set_op.get("type", "UNION")).upper(), "union")
            _right_sql = self.set_op.get("right_sql", self.set_op.get("raw_sql", ""))
            if _right_sql:
                canonical_set_op = CanonicalSetOperation(op=_op, raw_sql=_right_sql)

        # Build metadata dict
        metadata = {
            "from_schema": self.from_schema,
            "select_aggregates": [
                {"function": a.function.value if hasattr(a.function, 'value') else str(a.function),
                 "column": a.column, "alias": a.alias}
                for a in (self.select_aggregates or [])
            ],
            "column_selection": self.column_selection,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        
        return QueryPlan(
            intent="data_query",
            select=select_clause,
            from_=from_clause,
            joins=joins if joins else None,
            where=where_conditions if where_conditions else None,
            group_by=group_by,
            having=None,
            order_by=order_by,
            limit=self.limit,
            offset=self.offset,
            ctes=canonical_ctes,
            set_op=canonical_set_op,
            clarification_needed=self.clarification_needed,
            clarification_question=self.clarification_question,
            clarification_options=self.clarification_options,
            confidence=self.confidence,
            metadata=metadata,
        )


async def create_query_plan(
    user_query: str,
    catalog,
    retriever,
    llm
) -> QueryPlan:
    """
    Generate canonical QueryPlan from natural language.
    
    Returns canonical QueryPlan (not GeneratedQueryPlan).
    """
    # Find relevant tables
    table_candidates = await retriever.retrieve_table_candidates(user_query, top_k=3)
    if not table_candidates:
        raise ValueError("Could not find relevant tables for query")
    
    # Use best matching table
    best_table = table_candidates[0]
    
    # Find relevant columns
    col_candidates = await retriever.retrieve_column_candidates(
        user_query,
        best_table.full_name,
        top_k=10
    )
    
    if not col_candidates:
        # Fall back to SELECT *
        select_exprs = ["*"]
    else:
        select_exprs = [f"{col.full_name}" for col in col_candidates[:5]]
    
    # Create basic plan (internal format)
    internal_plan = GeneratedQueryPlan(
        select_expressions=select_exprs,
        from_table=best_table.name,
        from_schema=best_table.metadata.get("schema", None) if best_table.metadata else None,
        limit=None,
        intent="select",
        confidence=best_table.confidence,
    )
    
    # Convert to canonical and return
    return internal_plan.to_canonical()


class QueryPlanGenerator:
    """
    LLM-based query plan generator for semantic query orchestration.
    
    Generates CANONICAL QueryPlan objects from natural language using LLM.
    
    Architecture:
        NL -> LLM -> canonical QueryPlan -> compiler -> SQL
    """
    
    def __init__(
        self, 
        dialect: str = "postgresql", 
        schema_grounding: Optional[Any] = None, 
        join_graph_builder: Optional[Any] = None,
    ):
        """Initialize plan generator.
        
        Args:
            dialect: SQL dialect (postgresql, mysql, sqlite, etc.)
            schema_grounding: Optional SchemaGroundingContext for FK-aware join resolution
            join_graph_builder: Optional JoinGraphBuilder for auto-filling join conditions
        """
        self.dialect = dialect
        self.schema_grounding = schema_grounding
        self.join_graph_builder = join_graph_builder
        
        logger.info(f"[PLAN GENERATOR] Initialized with dialect: {dialect} (canonical pipeline)")
        if join_graph_builder:
            logger.info(f"[PLAN GENERATOR] FK-aware join resolution enabled")

    
    def generate_basic_plan(
        self,
        primary_table: str,
        selected_columns: Optional[List[str]] = None,
        schema: Optional[str] = None,
        where_conditions: Optional[List[Dict[str, Any]]] = None,
        joins: Optional[List[Dict[str, Any]]] = None,
    ) -> QueryPlan:
        """
        Generate a basic canonical QueryPlan.
        
        Args:
            primary_table: Main table for query
            selected_columns: Columns to select (None = SELECT *)
            schema: Schema name
            where_conditions: List of WHERE conditions
            joins: List of JOINs
            
        Returns:
            Canonical QueryPlan object
        """
        # Build internal plan and convert to canonical
        select_exprs = selected_columns if selected_columns else ["*"]
        
        # Build WHERE conditions
        where_conds = []
        if where_conditions:
            for cond in where_conditions:
                where_conds.append(
                    WhereCondition(
                        left=cond.get("column", ""),
                        operator=cond.get("operator", "="),
                        right=cond.get("value"),
                        is_literal=cond.get("is_literal", True),
                    )
                )
        
        # Build JOIN clauses
        join_clauses = []
        if joins:
            for join in joins:
                conditions = [
                    JoinCondition(
                        left_col=cond.get("left_col"),
                        right_col=cond.get("right_col"),
                        operator=cond.get("operator", "="),
                    )
                    for cond in join.get("conditions", [])
                ]
                join_clauses.append(
                    JoinClause(
                        join_type=JoinType(join.get("type", "INNER")),
                        right_table=join.get("table"),
                        right_schema=join.get("schema"),
                        conditions=conditions,
                    )
                )
        
        # Create internal plan and convert
        internal_plan = GeneratedQueryPlan(
            select_expressions=select_exprs,
            from_table=primary_table,
            from_schema=schema,
            where_conditions=where_conds,
            joins=join_clauses,
            intent="select",
            confidence=1.0,
        )
        
        logger.info(f"[PLAN] Generated canonical plan: {primary_table} with {len(select_exprs)} columns")
        return internal_plan.to_canonical()
    
    def render_plan(self, plan: QueryPlan) -> str:
        """
        Render a canonical QueryPlan to SQL.
        
        Args:
            plan: Canonical QueryPlan to render
        
        Returns:
            SQL query string
        """
        from .query_plan_compiler import compile_query_plan
        
        sql = compile_query_plan(plan, dialect=self.dialect)
        logger.info("[PLAN GENERATOR] Generated SQL via canonical pipeline")
        return sql

    
    async def create_query_plan(
        self,
        user_query: str,
        available_tables: List[str],
        available_columns_per_table: Dict[str, List[str]],
        possible_joins: List[Tuple[str, str]],
        table_metadata: Optional[Dict[str, Any]] = None,
        catalog: Optional[Any] = None,
    ) -> QueryPlan:
        """
        âœ… REQUIREMENT C: Plan-first LLM generation with validation and regeneration.
        
        Generate a CANONICAL QueryPlan from natural language using LLM.
        1. LLM outputs structured plan JSON
        2. Validator enforces semantic coverage rules
        3. If validation fails, regenerate with stronger guidance
        4. Returns canonical QueryPlan (not internal GeneratedQueryPlan)
        
        Args:
            user_query: Natural language user query
            available_tables: List of relevant table names (from semantic retrieval)
            available_columns_per_table: Dict mapping tables to their columns
            possible_joins: List of (table1, table2) join candidates
            table_metadata: Optional dict with table/column metadata including sample_values
            catalog: Optional SemanticSchemaCatalog for sample data
            
        Returns:
            Canonical QueryPlan object (ready for compilation to SQL)
        """
        # Build schema context for LLM (with sample data to prevent hallucination)
        schema_context = self._build_schema_context(
            available_tables,
            available_columns_per_table,
            possible_joins,
            catalog=catalog
        )
        
        # First attempt: Generate plan with standard prompt
        # Smart pre-analysis: temporal, entity, value grounding, ambiguity
        enhanced_context_str: Optional[str] = None
        try:
            from .smart_query_processor import get_smart_query_processor
            _sqp = get_smart_query_processor(llm_client=None)  # heuristic mode
            _enhanced = await _sqp.process(
                user_query=user_query,
                schema_catalog=catalog,
                target_tables=available_tables,
            )
            if _enhanced.needs_clarification:
                # Surface clarification as a QueryPlan with clarification_needed=True
                return QueryPlan(
                    intent="data_query",
                    select=SelectClause(fields=["*"]),
                    from_=FromClause(table=available_tables[0] if available_tables else "unknown"),
                    clarification_needed=True,
                    clarification_question=_enhanced.clarification_question,
                    clarification_options=_enhanced.clarification_options,
                    confidence=0.3,
                )
            enhanced_context_str = _enhanced.to_prompt_context()
        except Exception as _sqp_err:
            logger.debug("[PLAN] SmartQueryProcessor skipped: %s", _sqp_err)

        internal_plan = await self._llm_generate_plan(
            user_query,
            available_tables,
            available_columns_per_table,
            schema_context,
            table_metadata,
            regeneration_hint=None,  # First attempt
            enhanced_context=enhanced_context_str,
        )
        
        # âœ… ENHANCEMENT: Ground WHERE conditions to actual column values in database
        if internal_plan and catalog:
            internal_plan = self._enhance_plan_with_value_grounding(
                internal_plan,
                user_query,
                available_tables,
                available_columns_per_table,
                catalog
            )
        
        if internal_plan:
            # Validate plan against semantic coverage rules
            is_valid, validation_msg, suggestion = self._validate_plan_coverage(
                user_query,
                internal_plan,
                available_tables
            )
            
            if is_valid:
                logger.info(f"[PLAN-FIRST] Generated and validated plan: {internal_plan.from_table}")
                return internal_plan.to_canonical()
            else:
                # Validation failed - regenerate with stronger guidance
                logger.warning(f"[PLAN-VALIDATOR] Validation failed: {validation_msg}")
                logger.info(f"[PLAN-VALIDATOR] Regenerating with guidance: {suggestion}")
                
                # Create stronger prompt that includes validators findings
                internal_plan = await self._llm_generate_plan(
                    user_query,
                    available_tables,
                    available_columns_per_table,
                    schema_context,
                    table_metadata,
                    regeneration_hint=suggestion,
                    enhanced_context=enhanced_context_str,
                )
                
                if internal_plan:
                    logger.info(f"[PLAN-FIRST] Regenerated plan includes: {internal_plan.from_table} + {len(internal_plan.joins)} joins")
                    return internal_plan.to_canonical()
                else:
                    # If regeneration also fails, fall back
                    logger.warning("[PLAN-FIRST] Regeneration failed, using fallback")
                    fallback = self._create_fallback_plan(
                        available_tables,
                        available_columns_per_table,
                        user_query,
                        schema_context,
                        table_metadata
                    )
                    return fallback.to_canonical()
        else:
            # Initial generation failed
            logger.warning("[PLAN-FIRST] Initial generation failed, using fallback")
            fallback = self._create_fallback_plan(
                available_tables,
                available_columns_per_table,
                user_query,
                schema_context,
                table_metadata
            )
            return fallback.to_canonical()
    
    async def _llm_generate_plan(
        self,
        user_query: str,
        available_tables: List[str],
        available_columns_per_table: Dict[str, List[str]],
        schema_context: str,
        table_metadata: Optional[Dict[str, Any]],
        regeneration_hint: Optional[str],
        enhanced_context: Optional[str] = None,
    ) -> Optional[GeneratedQueryPlan]:
        """
        Call LLM to generate plan.

        Args:
            regeneration_hint: If provided, includes guidance for regeneration
            enhanced_context:  Semantic pre-analysis from SmartQueryProcessor

        Returns:
            GeneratedQueryPlan or None if generation fails
        """
        # Extract real table/column names from schema to use in example
        example_table = available_tables[0] if available_tables else "table_name"
        example_columns = available_columns_per_table.get(example_table, [])[:2] if available_columns_per_table else []
        if not example_columns:
            example_columns = ["id", "name"]
        example_metric_column = example_columns[0] if example_columns else "id"
        
        regeneration_guidance = ""
        if regeneration_hint:
            regeneration_guidance = f"\nâš ï¸ REGENERATION GUIDANCE:\n{regeneration_hint}\nMake sure the plan includes ALL required tables for the query context."

        # Semantic pre-analysis block (injected by SmartQueryProcessor)
        semantic_block = ""
        if enhanced_context:
            semantic_block = "\n" + enhanced_context + "\n"

        llm_prompt = f"""RESPOND WITH ONLY THIS JSON STRUCTURE. NO OTHER TEXT. NO EXPLANATIONS.

You are generating a DATABASE-AGNOSTIC QUERY PLAN that supports:
- Single-table queries: DISTINCT, filters, aggregations, pagination, sorting
- Multi-table queries: JOINs (INNER/LEFT/RIGHT/FULL), aggregations across joins, EXISTS/NOT EXISTS
- Complex operators: BETWEEN, IN (literal/subquery), LIKE/ILIKE, IS NULL, EXISTS

âš ï¸ CRITICAL: ONLY include query features that the USER explicitly requests:
- If user says "get all X" â†’ NO joins, NO aggregations, NO filters. Use SELECT * 
- If user says "count X" â†’ use COUNT aggregate
- If user mentions "where/filter/specific" â†’ add WHERE conditions
- If user asks for multiple tables â†’ use JOINs
- DO NOT add features the user didn't ask for - NO HALLUCINATING

EXAMPLE 1 - Simple "Get all" query (most common):
User: "get all {example_table}"
Output:
{{
  "distinct": false,
  "select_expressions": ["*"],
  "select_aggregates": [],
    "from_table": "{example_table}",
  "joins": [],
  "where_conditions": [],
  "group_by": [],
  "having_conditions": [],
  "order_by": [],
  "limit": null,
  "offset": 0,
  "clarification_needed": false,
  "clarification_question": null,
  "clarification_options": []
}}

EXAMPLE 1.5 - SUPERLATIVE QUERY (highest, lowest, top, most, least):
User: "which {example_table} has the highest {example_metric_column}" / "top {example_table}"
Output:
{{
  "distinct": false,
  "select_expressions": ["*"],
  "select_aggregates": [],
    "from_table": "{example_table}",
  "joins": [],
  "where_conditions": [],
  "group_by": [],
  "having_conditions": [],
    "order_by": [{{"column": "{example_metric_column}", "direction": "DESC"}}],
  "limit": 1,
  "offset": 0,
  "clarification_needed": false,
  "clarification_question": null,
  "clarification_options": []
}}
âš ï¸ SUPERLATIVE RULE: Words like "highest", "maximum", "top", "greatest", "largest" â†’ ORDER BY column DESC LIMIT 1
âš ï¸ SUPERLATIVE RULE: Words like "lowest", "minimum", "bottom", "smallest", "least" â†’ ORDER BY column ASC LIMIT 1
âš ï¸ If user asks "which X has highest Y" â†’ MUST use ORDER BY Y DESC LIMIT 1

EXAMPLE 2 - Complex query with multiple features (ONLY if user requests all these):
User: "show me records with status ACTIVE or INACTIVE, grouped by region, sorted by count desc, limit 100"
Output:
{{
  "distinct": false,
  "select_expressions": ["primary_table.id", "primary_table.region"],
  "select_aggregates": [
    {{"fn": "COUNT", "expr": "related_table.id", "alias": "item_count"}},
    {{"fn": "SUM", "expr": "related_table.amount", "alias": "total_amount"}}
  ],
  "from_table": "primary_table",
  "joins": [
    {{
      "right_table": "related_table",
      "join_type": "INNER",
      "on_conditions": [
        {{
          "left_column": "primary_table.id",
          "right_column": "related_table.primary_id",
          "operator": "="
        }}
      ]
    }}
  ],
  "where_conditions": [
    {{"left": "related_table.status", "operator": "IN", "right": ["ACTIVE", "INACTIVE"], "right_kind": "list"}}
  ],
  "group_by": ["primary_table.region"],
  "having_conditions": [],
  "order_by": [
    {{"column": "item_count", "direction": "DESC"}}
  ],
  "limit": 100,
  "offset": 0,
  "clarification_needed": false,
  "clarification_question": null,
  "clarification_options": []
}}

COMPOSABLE RULES:
âœ… Your query plan is a COMPOSITION of these OPTIONAL clauses:
   - SELECT [DISTINCT] columns [+ aggregates]
   - FROM table1
   - [JOIN table2 ON conditions] (0 or more joins)
   - [WHERE conditions] (0 or more conditions ANDed together)
   - [GROUP BY columns] [HAVING conditions]
   - [ORDER BY columns ASC|DESC]
   - [LIMIT n] [OFFSET n]

âœ… OPERATORS & RIGHT_KIND MAPPING (How to structure conditions):
    - Literal value: {{"operator": "=", "right": "VALUE", "right_kind": "literal"}}
    - List of values: {{"operator": "IN", "right": ["VALUE1", "VALUE2"], "right_kind": "list"}}
   - Range (BETWEEN): {{"operator": "BETWEEN", "right": ["2024-01-01", "2024-12-31"], "right_kind": "range"}}
   - Column reference: {{"operator": "=", "right": "other_table.other_column", "right_kind": "column"}}
   - Subquery: {{"operator": "IN", "right": "SELECT id FROM...", "right_kind": "subquery"}}
   - Raw expression: {{"operator": "=", "right": "NOW()", "right_kind": "raw"}}

âœ… SUPPORTED OPERATORS (will be rendered dynamically regardless of dialect):
   Comparison: =, <>, <, >, <=, >=, <>, !=
   Pattern: LIKE, ILIKE, NOT LIKE
   Null: IS NULL, IS NOT NULL
   Range: BETWEEN
   List: IN, NOT IN
   Existence: EXISTS, NOT EXISTS

âš ï¸ WHERE CONDITIONS: Extract filter criteria DYNAMICALLY from user query
- If user mentions "active", "inactive", "premium", "standard", etc. â†’ ADD WHERE conditions
- Match keywords to actual column names from schema (e.g., "status" columns, "type" columns)
- Use REAL column names from AVAILABLE TABLES below
- DO NOT add filters the user didn't mention (zero hardcoding)
- If user says "all records" or no filters mentioned â†’ use empty array []

âš ï¸ CLARIFICATION FIELDS (LLM-driven):
If you need user input BEFORE generating a full query, set:
  "clarification_needed": true,
  "clarification_question": "Which metric would you like to see: record count or total amount?",
  "clarification_options": ["Record Count", "Total Amount", "Both"]

This immediately returns the question to the user (ChatGPT-style), and the query generation retries after user responds.

âš ï¸ AGGREGATES STRUCTURE:
Use "fn" (function name), "expr" (column to aggregate), "alias" (result name):
  [
    {{"fn": "COUNT", "expr": "table.id", "alias": "record_count"}},
    {{"fn": "SUM", "expr": "table.amount", "alias": "total"}},
    {{"fn": "AVG", "expr": "table.value", "alias": "avg_value"}}
  ]

Functions: COUNT, SUM, AVG, MIN, MAX, STDDEV

âš ï¸ NOTE: LIMIT is NOT required - use count-first approach!
LIMIT enforcement is handled automatically by apply_smart_limit() after COUNT probe.

✅ WINDOW FUNCTIONS (ROW_NUMBER, RANK, running totals, partitioned analytics):
Include window expressions as raw strings directly inside "select_expressions".
Examples:
  "ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn"
  "RANK() OVER (ORDER BY total_amount DESC) AS rank"
  "SUM(amount) OVER (PARTITION BY region ORDER BY txn_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total"
Use these when user asks for: "rank", "running total", "cumulative", "row number", "top N per group".

✅ CASE WHEN / COALESCE / expressions — include as raw strings in "select_expressions":
  "CASE WHEN amount > 1000 THEN 'High' WHEN amount > 500 THEN 'Medium' ELSE 'Low' END AS category"
  "COALESCE(email, phone, 'N/A') AS contact"

✅ CTEs (WITH clause) — use "ctes" field:
  "ctes": [{{"name": "cte_name", "sql": "SELECT ...", "recursive": false}}]
  Main from_table can reference the CTE name. Use recursive:true for tree queries.

✅ SET OPERATIONS — use "set_op" field:
  "set_op": {{"type": "UNION ALL", "right_sql": "SELECT ... FROM other_table"}}
  Types: UNION, UNION ALL, INTERSECT, EXCEPT

âš ï¸ RESPONSE CONSTRUCTION RULES:
1. Always include these base fields: distinct, select_expressions, from_table, limit, offset, clarification_needed
2. Use empty arrays/null for optional features NOT requested:
   - select_aggregates: [] (empty if no COUNT/SUM/AVG mentioned)
   - joins: [] (empty if only single table query)
   - where_conditions: [] (empty if no filters mentioned)
   - group_by: [] (empty if no grouping mentioned)
   - having_conditions: [] (empty if no group filtering mentioned)
   - order_by: [] (empty if no sorting mentioned)
3. DO NOT add fields or features beyond what the user requests
4. If "get all X" or "list X" â†’ select_expressions=["*"], no joins, no aggregates
5. If user mentions "count" or "how many" â†’ use select_aggregates with COUNT
6. If user mentions specific columns â†’ list them instead of ["*"]

âš ï¸ IMPORTANT FOR JOINS:
- Use "right_table" field (NOT "table")
- Structure on_conditions as array of objects with "left_column", "right_column", "operator"
- Do NOT use "on_expression" string format
- Join types: INNER, LEFT, RIGHT, FULL, CROSS

CRITICAL RULES:
1. âš ï¸ PREVENT HALLUCINATION: Do NOT invent features
    - If user says "get all {example_table}" â†’ DO NOT add where conditions, joins, or aggregates
   - Only include what user explicitly asks for
   - Empty arrays [] for unused features (not null, not ignored)
   
2. âš ï¸ USE REAL COLUMN NAMES ONLY - Column names are shown with SAMPLE values (in brackets)
    - Example: "some_table: [...id, related_id, status...]" means these are ACTUAL columns
   - Example sample values shown like "status: [ACTIVE, INACTIVE]" are REAL data from database
    - DO NOT invent column names like "status_flag" if not listed with samples
   
3. Match user query keywords to actual columns shown with samples
   - If user says "active" and you see "status: [ACTIVE, INACTIVE]" â†’ use status column
   - If a column isn't shown with sample values, it may not exist in your data sample
   
4. Use ONLY tables and columns from the AVAILABLE TABLES below
5. If the user explicitly names an entity that matches a table in AVAILABLE TABLES, include that table
6. If the query mentions multiple entities that correspond to different tables, include joins to connect them
7. ANALYZE user query for explicit filters ONLY (status, type, category, grade, amount ranges, dates, etc.)
8. DO NOT use placeholder names like "table1", "column1" etc.
9. ALWAYS use the exact field names shown in the EXAMPLE above
10. âœ… VALIDATION: Before using a column in WHERE/GROUP BY/ORDER BY, verify it's listed in schema section below
11. âš ï¸ SUPERLATIVES: If user asks for "highest/top/maximum/greatest" â†’ ORDER BY column DESC LIMIT 1
    If user asks for "lowest/bottom/minimum/smallest/least" â†’ ORDER BY column ASC LIMIT 1
    Match the superlative word to the relevant column (e.g., "highest amount" â†’ ORDER BY amount DESC LIMIT 1)

{semantic_block}USER QUERY: {user_query}

AVAILABLE TABLES AND COLUMNS:
{schema_context}{regeneration_guidance}

REQUIRED: Generate the complete GeneratedQueryPlan JSON ONLY. Use REAL table/column names. Only include fields for features the user requests. Empty arrays [] for unused features. No markdown, no explanations."""
        
        from .. import llm
        
        messages = [
            {
                "role": "system",
                "content": "Output ONLY valid JSON. No markdown, no explanations, no other text. Start immediately with { and end with }."
            },
            {
                "role": "user",
                "content": llm_prompt
            }
        ]
        
        try:
            llm_response = await llm.call_llm(messages, stream=False, max_tokens=512)
            
            # Ensure response is a string
            if hasattr(llm_response, 'content'):
                llm_response = llm_response.content
            
            llm_response = str(llm_response).strip()
            
            if not llm_response:
                return None
            
            # Clean markdown if present
            if llm_response.startswith("```"):
                llm_response = llm_response.split("```")[1]
                if llm_response.startswith("json"):
                    llm_response = llm_response[4:]
                llm_response = llm_response.strip()
            if llm_response.endswith("```"):
                llm_response = llm_response[:-3].strip()
            
            if not llm_response:
                return None
            
            # Parse and validate JSON
            import json
            from .llm_output_schemas import LLMQueryPlanOutput
            raw_dict = json.loads(llm_response)
            logger.debug(f"[PLAN-LLM] Raw JSON from LLM: {json.dumps(raw_dict, indent=2)[:500]}")
            # Pydantic validation — safe defaults for every field
            validated = LLMQueryPlanOutput.model_validate_lenient(raw_dict)
            plan_dict = validated.to_raw_dict()
            plan = self._dict_to_query_plan(plan_dict)
            logger.debug(f"[PLAN-LLM] Converted plan - FROM: {plan.from_table}, JOINS: {[(j.right_table, len(j.conditions)) for j in plan.joins]}")
            return plan
            
        except Exception as e:
            logger.debug(f"[PLAN-FIRST] LLM generation error: {e}")
            return None
    
    def _validate_plan_coverage(
        self,
        user_query: str,
        plan: GeneratedQueryPlan,
        available_tables: List[str],
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Validate that plan covers required tables based on query modifiers.
        
        Returns:
            (is_valid, reason, suggestion)
        """
        from .plan_validator import PlanValidator
        
        validator = PlanValidator(available_tables)
        
        # Get tables in plan
        plan_tables = [plan.from_table]
        for join in plan.joins:
            plan_tables.append(join.right_table)
        
        # Validate
        is_valid, reason, suggestion = validator.validate_plan(
            user_query,
            plan_tables,
            available_tables
        )
        
        return is_valid, reason, suggestion
    
    def _enhance_plan_with_value_grounding(
        self,
        plan: GeneratedQueryPlan,
        user_query: str,
        available_tables: List[str],
        available_columns_per_table: Dict[str, List[str]],
        catalog: Any,
    ) -> GeneratedQueryPlan:
        """
        Enhance WHERE conditions by grounding values mentioned in the query
        to actual columns in the database that contain those values.
        
        Example:
        - User: "Show records with category premium"
        - LLM generates: WHERE status = 'ACTIVE' (generic)
        - Value grounding finds: 'premium' â†’ category column has 'PREMIUM' values
        - Enhanced: Add WHERE category = 'PREMIUM'
        
        Args:
            plan: GeneratedQueryPlan from LLM
            user_query: Original user query
            available_tables: List of accessible tables
            available_columns_per_table: Dict of columns per table (limited to top-k)
            catalog: SemanticSchemaCatalog with sample data
            
        Returns:
            Enhanced GeneratedQueryPlan
        """
        try:
            grounder = ValueBasedColumnGrounder(catalog)
            
            # Get tables in the plan (tables being queried)
            plan_tables = [plan.from_table] + [j.right_table for j in plan.joins]
            
            # âœ… FIX: Build ALL_COLUMNS mapping for value grounding
            # The retriever only returns top-3 columns, but value grounding needs
            # to search ALL columns to find attribute columns that have sample values
            # matching the user's query keywords (e.g., 'premium' -> finds 'PREMIUM' in category)
            all_columns_per_table = {}
            for table_name in plan_tables:
                if catalog and table_name in catalog.tables:
                    table_meta = catalog.tables[table_name]
                    all_columns_per_table[table_name] = list(table_meta.columns.keys())
            
            logger.debug(f"[VALUE GROUNDING] Using {sum(len(c) for c in all_columns_per_table.values())} total columns for value grounding")
            
            # Extract suggested filters based on query values
            grounding_result = grounder.ground_query_values_to_filters(
                user_query,
                available_tables=plan_tables,
                available_columns_per_table=all_columns_per_table
            )
            
            suggested_filters = grounding_result.get("suggested_filters", [])
            
            if not suggested_filters:
                # No value-based filters found, return plan as-is
                logger.debug("[VALUE GROUNDING] No value-based filters found, plan unchanged")
                return plan
            
            # Log what we found
            logger.info(f"[VALUE GROUNDING] Found {len(suggested_filters)} value-based filter suggestions")
            for filt in suggested_filters:
                logger.info(f"  â†’ {filt['table']}.{filt['column']} = '{filt['value']}'")
            
            # âœ… ENHANCEMENT STRATEGY: 
            # If LLM didn't add specific filters, add the value-grounded ones
            if not plan.where_conditions:
                # No WHERE conditions from LLM, add grounded filters
                logger.info("[VALUE GROUNDING] Enhancing plan: LLM had no WHERE conditions, adding value-grounded filters")
                for filt in suggested_filters:
                    plan.where_conditions.append(
                        WhereCondition(
                            left=f"{filt['table']}.{filt['column']}",
                            operator="=",
                            right=filt['value'],
                            is_literal=True
                        )
                    )
            else:
                # LLM has filters, but check if we should enhance with better ones
                logger.debug(f"[VALUE GROUNDING] Plan has {len(plan.where_conditions)} existing WHERE condition(s)")
                
                # Check if LLM's filter is too generic and we have a better grounded one
                for existing_cond in plan.where_conditions:
                    logger.debug(f"  Existing: {existing_cond.left} {existing_cond.operator} {existing_cond.right}")
                    
                    # Check if this is using a generic column
                    col_name = existing_cond.left.split(".")[-1] if "." in existing_cond.left else existing_cond.left
                    generic_cols = {"status", "type", "category", "state", "kind"}
                    
                    if col_name.lower() in generic_cols and suggested_filters:
                        # We have a more specific grounded filter, use it instead
                        best_grounded = suggested_filters[0]
                        logger.info(f"[VALUE GROUNDING] Replacing generic filter '{col_name}' with grounded filter '{best_grounded['column']}'")
                        
                        existing_cond.left = f"{best_grounded['table']}.{best_grounded['column']}"
                        existing_cond.right = best_grounded['value']
            
            return plan
            
        except Exception as e:
            logger.warning(f"[VALUE GROUNDING] Enhancement failed: {e}, continuing with original plan")
            return plan
    
    def _build_schema_context(
        self,
        available_tables: List[str],
        available_columns: Dict[str, List[str]],
        possible_joins: List[Tuple[str, str]],
        catalog: Optional[Any] = None,
    ) -> str:
        """Build human-readable schema context for LLM with sample data to prevent hallucination."""
        schema_text = "TABLES AND COLUMNS (with sample values to prevent column hallucination):\n"
        for table in available_tables:
            cols = available_columns.get(table, [])
            col_list = ", ".join(cols) if cols else "(no metadata)"
            schema_text += f"  â€¢ {table}: {col_list}\n"
            
            # Add sample data if catalog available
            if catalog:
                try:
                    sample_data = catalog.get_table_sample_data(table)
                    if sample_data:
                        for col_name, samples in sample_data.items():
                            # Always show actual sample values so LLM knows they exist
                            sample_values_str = ", ".join(str(s)[:20] for s in samples[:3])
                            schema_text += f"      â””â”€ {col_name}: [{sample_values_str}]\n"
                except Exception as e:
                    logger.debug(f"[SCHEMA-CONTEXT] Could not get samples for {table}: {e}")
        
        if possible_joins:
            schema_text += "\nPOSSIBLE JOINS:\n"
            for table1, table2 in possible_joins:
                schema_text += f"  â€¢ {table1} â†â†’ {table2}\n"
        
        return schema_text
    
    def _dict_to_query_plan(self, plan_dict: Dict[str, Any]) -> GeneratedQueryPlan:
        """
        Convert LLM-generated dict to GeneratedQueryPlan object.
        Handles all fields including new ones: distinct, limit, offset, group_by, having,
        order_by, and clarification fields.
        """
        # Extract fields with defaults
        select_exprs = plan_dict.get("select_expressions", ["*"])
        from_table = plan_dict.get("from_table", "")
        
        if not from_table:
            raise ValueError("LLM plan missing required 'from_table' field")
        
        # âœ… VALIDATION: Detect and reject placeholder table names
        placeholder_patterns = ["table_name", "table1", "table2", "other_table", "unknown", "table"]
        if from_table.lower() in placeholder_patterns:
            raise ValueError(f"LLM returned placeholder table name '{from_table}' instead of actual table - rejecting plan")
        
        # âœ… VALIDATION: Detect and replace placeholder column names
        placeholder_col_patterns = ["column1", "column2", "col_name", "value", "col"]
        cleaned_exprs = []
        for expr in select_exprs:
            if expr.lower() in placeholder_col_patterns or expr == "*":
                cleaned_exprs.append(expr)  # Keep "*" or placeholders for now
            else:
                cleaned_exprs.append(expr)
        
        select_exprs = cleaned_exprs
        
        # âœ… NEW: Parse select_aggregates with dynamic "fn", "expr", "alias" fields
        select_aggregates = []
        for agg in plan_dict.get("select_aggregates", []):
            if isinstance(agg, dict):
                agg_fn = agg.get("fn", "COUNT").upper()
                agg_expr = agg.get("expr")
                agg_alias = agg.get("alias")
                
                try:
                    agg_type = AggregationType[agg_fn]
                    select_aggregates.append(
                        AggregateField(
                            function=agg_type,
                            column=agg_expr,
                            alias=agg_alias
                        )
                    )
                except KeyError:
                    logger.warning(f"[PLAN-FIRST] Unknown aggregation function: {agg_fn}")
        
        # Convert where conditions - ROBUST: handle dict format with new right_kind field
        where_conds = []
        for cond in plan_dict.get("where_conditions", []):
            # Handle case where LLM returns a string instead of object (e.g., "status = 'ACTIVE'")
            if isinstance(cond, str):
                # Skip malformed string conditions
                logger.debug(f"[PLAN-FIRST] Skipping string WHERE condition (not structured): {cond[:50]}")
                continue
            elif isinstance(cond, dict):
                # Support both old and new field names for backward compatibility
                left = cond.get("left") or cond.get("column", "")
                operator = cond.get("operator", "=")
                right = cond.get("right") or cond.get("value")
                
                # NEW: Parse right_kind (defaults to LITERAL for backward compatibility)
                right_kind_str = cond.get("right_kind", "literal").lower()
                try:
                    right_kind = RightKind(right_kind_str)
                except (ValueError, KeyError):
                    logger.debug(f"[PLAN-FIRST] Unknown right_kind'{right_kind_str}', defaulting to literal")
                    right_kind = RightKind.LITERAL
                
                where_conds.append(
                    WhereCondition(
                        left=left,
                        operator=operator,
                        right=right,
                        right_kind=right_kind,
                        is_literal=(right_kind == RightKind.LITERAL)  # For backward compatibility
                    )
                )
        
        # Convert joins - ROBUST: handle both LLM output formats
        join_clauses = []
        for idx, join in enumerate(plan_dict.get("joins", [])):
            if not isinstance(join, dict):
                logger.debug(f"[PLAN-FIRST] Skipping malformed JOIN (not dict): {join}")
                continue
            
            # âœ… HANDLE BOTH FORMATS:
            # Format 1 (structured): {"right_table": "table_b", "on_conditions": [...]}
            # Format 2 (LLM actual): {"table": "table_b", "on_expression": "table_b.fk_id = ..."}
            
            right_table = join.get("right_table") or join.get("table", "")
            join_type = join.get("join_type", "INNER").upper()
            logger.debug(f"[PLAN-FIRST] Processing JOIN #{idx+1}: right_table='{right_table}', type='{join_type}', keys={list(join.keys())}")
            
            conditions = []
            
            # Try structured on_conditions first (Format 1)
            structured_conds = join.get("on_conditions", [])
            if structured_conds:
                for cond in structured_conds:
                    if not isinstance(cond, dict):
                        logger.debug(f"[PLAN-FIRST] Skipping string JOIN condition: {cond}")
                        continue
                    conditions.append(
                        JoinCondition(
                            left_col=cond.get("left_column", ""),
                            right_col=cond.get("right_column", ""),
                            operator=cond.get("operator", "="),
                        )
                    )
            else:
                # Try on_expression (Format 2 - LLM actual output)
                on_expr = join.get("on_expression", "")
                if on_expr:
                    # Parse simple expression like "table_b.fk_id = table_a.id"
                    # This is a string, we'll store it as a raw condition for now
                    # The renderer handles ON clause generation differently
                    logger.debug(f"[PLAN-FIRST] JOIN #{idx+1} has on_expression (string): {on_expr}")
                    # TODO: Parse this more intelligently if needed
            
            if not right_table:
                logger.warning(f"[PLAN-FIRST] âš ï¸ JOIN #{idx+1} has empty right_table! Full join object: {join}")
            
            # Parse join_type enum
            try:
                join_type_enum = JoinType[join_type]
            except KeyError:
                logger.warning(f"[PLAN-FIRST] Unknown join type '{join_type}', defaulting to INNER")
                join_type_enum = JoinType.INNER
            
            join_clauses.append(
                JoinClause(
                    join_type=join_type_enum,
                    right_table=right_table,
                    right_schema=None,
                    conditions=conditions,
                )
            )
        
        # âœ… NEW: Parse group_by (list of column names)
        group_by = plan_dict.get("group_by", [])
        if not isinstance(group_by, list):
            group_by = []
        
        # âœ… NEW: Parse having_conditions (same structure as where_conditions)
        having_conds = []
        for cond in plan_dict.get("having_conditions", []):
            if isinstance(cond, dict):
                left = cond.get("left") or cond.get("column", "")
                operator = cond.get("operator", "=")
                right = cond.get("right") or cond.get("value")
                
                right_kind_str = cond.get("right_kind", "literal").lower()
                try:
                    right_kind = RightKind(right_kind_str)
                except (ValueError, KeyError):
                    right_kind = RightKind.LITERAL
                
                having_conds.append(
                    WhereCondition(
                        left=left,
                        operator=operator,
                        right=right,
                        right_kind=right_kind,
                        is_literal=(right_kind == RightKind.LITERAL)
                    )
                )
        
        # âœ… NEW: Parse order_by (list of {column, direction} objects)
        order_by = []
        for field in plan_dict.get("order_by", []):
            if isinstance(field, dict):
                col = field.get("column", "")
                direction_str = field.get("direction", "ASC").upper()
                try:
                    direction = OrderDirection[direction_str]
                except KeyError:
                    direction = OrderDirection.ASC
                
                if col:
                    order_by.append(OrderByField(column=col, direction=direction))
        
        # âœ… NEW: Parse limit and offset
        limit = plan_dict.get("limit")
        if limit is not None:
            try:
                limit = int(limit) if limit else None
            except (ValueError, TypeError):
                limit = None
        
        offset = plan_dict.get("offset")
        if offset is not None:
            try:
                offset = int(offset) if offset else None
            except (ValueError, TypeError):
                offset = None
        
        # âœ… NEW: Parse distinct
        distinct = plan_dict.get("distinct", False)
        if not isinstance(distinct, bool):
            distinct = False
        
        # âœ… NEW: Parse clarification fields
        clarification_needed = plan_dict.get("clarification_needed", False)
        clarification_question = plan_dict.get("clarification_question")
        clarification_options = plan_dict.get("clarification_options", [])
        
        # Parse CTEs from LLM output
        ctes = plan_dict.get("ctes", [])
        if not isinstance(ctes, list):
            ctes = []

        # Parse set operation from LLM output
        set_op = plan_dict.get("set_op")
        if set_op is not None and not isinstance(set_op, dict):
            set_op = None

        plan = GeneratedQueryPlan(
            select_expressions=select_exprs,
            select_aggregates=select_aggregates,
            from_table=from_table,
            from_schema=plan_dict.get("from_schema"),
            where_conditions=where_conds,
            joins=join_clauses,
            group_by=group_by,
            having_conditions=having_conds,
            order_by=order_by,
            limit=limit,
            offset=offset,
            distinct=distinct,
            ctes=ctes,
            set_op=set_op,
            clarification_needed=clarification_needed,
            clarification_question=clarification_question,
            clarification_options=clarification_options,
            intent="select",
            confidence=0.85,  # LLM-generated plans have good confidence
        )

        return plan
    
    def _create_fallback_plan(
        self,
        tables: List[str],
        columns_per_table: Dict[str, List[str]],
        user_query: str,
        schema_context: str,
        table_metadata: Optional[Dict[str, Any]] = None,
    ) -> GeneratedQueryPlan:
        """Smarter fallback heuristic: uses ALL tables and creates JOINs when possible.
        
        Args:
            tables: List of table names
            columns_per_table: Dict mapping table names to column lists
            user_query: Original user query for context
            schema_context: Schema description text
            table_metadata: Optional dict with table/column metadata including sample_values
        """
        
        # Use first table as primary
        from_table = tables[0] if tables else "unknown"
        primary_cols = columns_per_table.get(from_table, [])[:5] if columns_per_table else []
        select_exprs = primary_cols if primary_cols else ["*"]
        
        # âœ… IMPROVEMENT: Create JOINs if multiple tables available
        joins = []
        if len(tables) > 1:
            # Find the actual primary key of from_table (not hardcoded "id")
            from_cols = columns_per_table.get(from_table, [])
            from_pk = None
            
            # Try patterns: table_id, tableid, id (in order of likelihood)
            from_table_singular = from_table.rstrip('s')
            pk_patterns = [
                f"{from_table_singular}_id",
                f"{from_table}_id",
                f"{from_table}id",
                "id",
                f"{from_table_singular}id"
            ]
            for pattern in pk_patterns:
                if pattern in from_cols:
                    from_pk = pattern
                    break
            
            # If no PK found, use first column
            if not from_pk and from_cols:
                from_pk = from_cols[0]
            
            if not from_pk:
                from_pk = "id"  # Fallback
            
            # Try to create basic JOINs for remaining tables
            # This is a heuristic: assume tables can be joined via common patterns
            # e.g., table_a â†’ table_b via shared foreign key patterns
            
            for i in range(1, min(len(tables), 4)):  # Max 3 tables (limit complexity)
                right_table = tables[i]
                # Heuristic join condition: look for foreign key pattern
                # Most common: table_name_id (e.g., entity_id for entity table)
                join_condition_col = None
                
                # Try common patterns: {from_table_singular}_id
                target_fk = f"{from_table_singular}_id"
                
                # Check if target table has this FK
                target_cols = columns_per_table.get(right_table, [])
                if target_fk in target_cols or f"{from_table}_id" in target_cols:
                    join_condition_col = target_fk if target_fk in target_cols else f"{from_table}_id"
                    
                    joins.append(JoinClause(
                        join_type=JoinType.LEFT,
                        right_table=right_table,
                        conditions=[
                            JoinCondition(
                                left_col=f"{from_table}.{from_pk}",
                                right_col=f"{right_table}.{join_condition_col}",
                                operator="="
                            )
                        ]
                    ))
                    logger.info(f"[FALLBACK] Auto-creating JOIN: {from_table}.{from_pk} â†’ {right_table}.{join_condition_col}")
        
        # âœ… PURE DYNAMIC APPROACH: NO hardcoded filter logic
        # Filters should come from LLM analysis of user query, not pattern matching
        # The fallback plan should NOT add any WHERE conditions
        # All filtering logic is LLM-driven (query_plan_generator LLM call above)
        where_conditions = []
        
        plan = GeneratedQueryPlan(
            select_expressions=select_exprs,
            from_table=from_table,
            from_schema=None,
            where_conditions=where_conditions,
            joins=joins,
            intent="select",
            confidence=0.3,  # Lower confidence for fallback
        )
        
        return plan





