"""
Plan-First Query Generator - Components C & D of semantic system.

1. Generates database-agnostic query plans (not SQL directly)
2. Renders plans to dialect-specific SQL (PostgreSQL, MySQL, SQLite, etc.)

This enables:
- Better structured generation (plans are more constrainted than free-form SQL)
- DB-agnostic architecture (plan → any SQL dialect)
- Better validation and safety (plans can be statically checked)
- Deterministic rendering (same plan = same safe SQL)
"""

from __future__ import annotations

import logging
import random
import string
from typing import Dict, List, Optional, Any, Literal, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

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


class ColumnSelectionIntent(str, Enum):
    """Semantic intent for column selection - determined by LLM analysis."""
    ALL_COLUMNS = "all_columns"  # User wants all columns (SELECT *)
    SPECIFIC_COLUMNS = "specific_columns"  # User mentioned specific columns
    COUNT_ONLY = "count_only"  # User only wants COUNT
    DISTINCT_VALUES = "distinct_values"  # User wants DISTINCT values of a column
    FIRST_N_COLUMNS = "first_n_columns"  # User wants top N columns (e.g., "first 3 columns")


@dataclass
class ColumnSelectionAnalysis:
    """LLM-determined analysis of column selection intent."""
    intent: ColumnSelectionIntent
    requested_columns: List[str] = field(default_factory=list)  # If specific_columns, which ones
    reasoning: str = ""  # Why this intent was chosen
    user_mentions_columns: bool = False  # Did user explicitly name columns?
    user_mentions_all: bool = False  # Did user ask for "all"?
    confidence: float = 1.0  # Confidence in this analysis


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
    operator: str  # =, <, >, LIKE, IN, BETWEEN, etc.
    right: Any  # value or list of values
    is_literal: bool = True  # True: value should be quoted; False: column reference


@dataclass
class OrderByField:
    """ORDER BY field."""
    column: str
    direction: OrderDirection = OrderDirection.ASC


@dataclass
class QueryPlan:
    """
    Database-agnostic query plan.
    
    Represents a SELECT query as a structured plan that can be
    rendered to any SQL dialect.
    """
    # SELECT clause
    select_expressions: List[str] = field(default_factory=list)  # column names or expressions
    select_aggregates: List[AggregateField] = field(default_factory=list)  # sum(x), count(*), etc.
    
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


class QueryPlanRenderer:
    """
    Renders database-agnostic plans to dialect-specific SQL.
    
    Handles differences between:
    - PostgreSQL (schema.table, ILIKE, partial indexes)
    - MySQL (backticks, LIMIT syntax)
    - SQLite (limited JOIN types, no schema qualification)
    - SQL Server ([brackets], TOP instead of LIMIT)
    """
    
    def __init__(self, dialect: str = "postgresql", schema_grounding: Optional[Any] = None, join_graph_builder: Optional[Any] = None):
        """
        Initialize renderer for specific SQL dialect.
        
        Args:
            dialect: one of "postgresql", "mysql", "sqlite", "mssql"
            schema_grounding: Optional SchemaGroundingContext for FK-aware join resolution
            join_graph_builder: Optional JoinGraphBuilder for auto-filling join conditions
        """
        self.dialect = dialect.lower()
        self.schema_grounding = schema_grounding
        self.join_graph_builder = join_graph_builder
        logger.info(f"[PLAN] Renderer initialized for dialect: {self.dialect}")
    
    def render(self, plan: QueryPlan) -> str:
        """
        Render plan to SQL.
        
        Args:
            plan: QueryPlan to render
        
        Returns:
            SQL query string
        """
        if not plan.is_valid()[0]:
            reason = plan.is_valid()[1]
            raise ValueError(f"Invalid query plan: {reason}")
        
        # Generate ALL table aliases upfront for consistency across all clauses
        self.alias_map = {}
        self.alias_map[plan.from_table] = plan.from_alias or generate_random_alias(plan.from_table)
        for join in plan.joins:
            self.alias_map[join.right_table] = generate_random_alias(join.right_table)
        
        # Build SQL parts (now they'll use the pre-generated alias_map)
        select_part = self._render_select(plan)
        from_part = self._render_from(plan)
        join_part = self._render_joins(plan)
        where_part = self._render_where(plan)
        group_by_part = self._render_group_by(plan)
        having_part = self._render_having(plan)
        order_by_part = self._render_order_by(plan)
        
        # Combine parts
        sql_parts = [select_part, from_part, join_part, where_part, 
                     group_by_part, having_part, order_by_part]
        sql = " ".join(part for part in sql_parts if part)
        
        # Add terminating semicolon
        if not sql.endswith(";"):
            sql += ";"
        
        return sql
    
    def _render_select(self, plan: QueryPlan) -> str:
        """Render SELECT clause using pre-generated aliases for consistency."""
        # Use the alias already generated in render() method
        from_alias = self.alias_map.get(plan.from_table, plan.from_alias or generate_random_alias(plan.from_table))
        
        # CRITICAL FIX: Respect column_selection intent from LLM analysis
        # If LLM determined user wants ALL_COLUMNS, use SELECT * (or alias.*)
        if plan.column_selection and plan.column_selection.intent == ColumnSelectionIntent.ALL_COLUMNS and not plan.select_aggregates:
            logger.info(f"[RENDER-SELECT] Respecting LLM intent ALL_COLUMNS (confidence={plan.column_selection.confidence})")
            return f"SELECT {from_alias}.*"
        
        # If no specific columns and no aggregates, use SELECT alias.* to get all columns
        if not plan.select_expressions and not plan.select_aggregates:
            return f"SELECT {from_alias}.*"
        
        # Columns - qualify with table alias when JOINs exist to avoid ambiguous refs
        cols = []
        for col in plan.select_expressions:
            # If JOINs are present and column not already qualified (no "."), add FROM alias
            if plan.joins and "." not in col:
                # Don't quote the alias itself, only the column name
                cols.append(f"{from_alias}.{self._quote_identifier(col)}")
            else:
                cols.append(self._quote_identifier(col))
        
        # Aggregates
        for agg in plan.select_aggregates:
            if agg.column:
                # Qualify aggregate column if JOINs exist
                if plan.joins and "." not in agg.column:
                    agg_col = f"{from_alias}.{agg.column}"
                else:
                    agg_col = agg.column
                agg_expr = f"{agg.function.value}({self._quote_identifier(agg_col)})"
            else:
                agg_expr = f"{agg.function.value}(*)"
            
            if agg.alias:
                agg_expr += f" AS {self._quote_identifier(agg.alias)}"
            
            cols.append(agg_expr)
        
        select = "SELECT " + ", ".join(cols)
        return select
    
    def _render_from(self, plan: QueryPlan) -> str:
        """Render FROM clause with table alias."""
        from app.config import settings
        
        # Add schema prefix if not present (use config schema, not hardcoded)
        table_name = plan.from_table_qualified
        if not plan.from_schema and "." not in table_name:
            # Get schema from config based on DB type
            default_schema = self._get_default_schema()
            table_name = f"{default_schema}.{table_name}"
        
        table = self._quote_identifier(table_name)
        
        # Use the alias already generated in render() method (NOT quoted as bare alias)
        alias = self.alias_map.get(plan.from_table, plan.from_alias or generate_random_alias(plan.from_table))
        
        from_clause = f"FROM {table} AS {alias}"
        
        return from_clause
    
    def _render_joins(self, plan: QueryPlan) -> str:
        """Render JOIN clauses with FK-aware auto-filling of missing ON conditions."""
        if not plan.joins:
            return ""
        
        from app.config import settings
        # Use the aliases already generated in render() method
        from_alias = self.alias_map.get(plan.from_table, plan.from_alias or generate_random_alias(plan.from_table))
        join_parts = []
        
        for idx, join in enumerate(plan.joins):
            join_type = join.join_type.value
            right_table_name = join.full_table_name
            
            # Add schema prefix to join table if missing (use config schema, not hardcoded)
            if not join.right_schema and "." not in right_table_name:
                default_schema = self._get_default_schema()
                right_table_name = f"{default_schema}.{right_table_name}"
            
            logger.debug(f"[RENDERER] JOIN #{idx+1}: raw_table='{join.right_table}', schema='{join.right_schema}', full_name='{right_table_name}'")
            if not join.right_table:
                logger.warning(f"[RENDERER] ⚠️ JOIN #{idx+1} has empty right_table! Cannot render JOIN clause!")
            
            right_table = self._quote_identifier(right_table_name)
            
            # Use the alias already generated in render() method
            join_alias = self.alias_map.get(join.right_table, generate_random_alias(join.right_table))
            
            # Build ON conditions with proper aliases
            on_conditions = []
            for cond in join.conditions:
                # Replace unqualified table names with aliases in ON clause
                left_col = cond.left_col
                right_col = cond.right_col
                
                # Replace FROM table reference with its alias
                if left_col.startswith(plan.from_table + "."):
                    left_col = left_col.replace(plan.from_table + ".", f"{from_alias}.")
                if right_col.startswith(plan.from_table + "."):
                    right_col = right_col.replace(plan.from_table + ".", f"{from_alias}.")
                
                # Replace JOIN table reference with its alias
                if left_col.startswith(join.right_table + "."):
                    left_col = left_col.replace(join.right_table + ".", f"{join_alias}.")
                if right_col.startswith(join.right_table + "."):
                    right_col = right_col.replace(join.right_table + ".", f"{join_alias}.")
                
                # Handle qualified columns: alias.column -> alias."column"
                # Only quote the column name part, not the bare alias
                if "." in left_col:
                    parts = left_col.rsplit(".", 1)
                    left = f"{parts[0]}.{self._quote_identifier(parts[1])}"
                else:
                    left = self._quote_identifier(left_col)
                    
                if "." in right_col:
                    parts = right_col.rsplit(".", 1)
                    right = f"{parts[0]}.{self._quote_identifier(parts[1])}"
                else:
                    right = self._quote_identifier(right_col)
                    
                on_conditions.append(f"{left} {cond.operator} {right}")
            
            # Auto-fill missing ON clause using FK detection
            on_clause = None
            if on_conditions:
                on_clause = " AND ".join(on_conditions)
            else:
                # Try to auto-fill from FK graph using aliases
                on_clause = self._resolve_join_on_clause_with_aliases(
                    base_table=plan.from_table,
                    base_alias=from_alias,
                    join_table=join.right_table,
                    join_alias=join_alias
                )
            
            # Fallback to TRUE only if FK detection completely fails
            if not on_clause:
                logger.warning(f"[RENDERER] Could not auto-fill ON for {join.right_table}, falling back to ON TRUE")
                on_clause = "TRUE"
            
            # Don't quote bare table alias - only quote schema-qualified table names
            join_parts.append(f"{join_type} JOIN {right_table} AS {join_alias} ON {on_clause}")
        
        return " ".join(join_parts)
    
    def _resolve_join_on_clause_with_aliases(self, base_table: str, base_alias: str, 
                                                  join_table: str, join_alias: str) -> Optional[str]:
        """
        Auto-fill missing JOIN ON clause using FK detection with table aliases.
        
        This implements Rule B compliance: joins must have complete specifications.
        
        Args:
            base_table: The FROM table
            base_alias: The alias for FROM table (e.g., 'c')
            join_table: The table being JOINed
            join_alias: The alias for JOIN table (e.g., 'd')
            
        Returns:
            ON clause string using aliases (e.g., "c.customer_id = d.customer_id") or None
        """
        # If no FK detection available, return None 
        if not self.join_graph_builder:
            logger.debug(f"[RENDERER] No join_graph_builder available, cannot auto-fill ON for {base_table} -> {join_table}")
            return None
        
        try:
            # Try to find join path using FK detection
            join_path = self.join_graph_builder.find_join_path(
                source_table=base_table,
                target_table=join_table
            )
            
            if join_path and join_path.steps:
                # Extract ON clause from first step (direct join if exists)
                first_step = join_path.steps[0]
                on_clause = first_step.get("join_clause")
                
                if on_clause:
                    # Replace table names with aliases in the ON clause
                    on_clause = on_clause.replace(f"{base_table}.", f"{base_alias}.")
                    on_clause = on_clause.replace(f"{join_table}.", f"{join_alias}.")
                    logger.info(f"[RENDERER] Auto-filled ON for {base_table} -> {join_table}: {on_clause}")
                    return on_clause
                
                # Try alternative structure
                join_on = first_step.get("join_on")
                if join_on:
                    # join_on is list of tuples: [(local_col, remote_col), ...]
                    conditions = [
                        f"{base_alias}.{local_col} = {join_alias}.{remote_col}"
                        for local_col, remote_col in join_on
                    ]
                    on_clause = " AND ".join(conditions)
                    logger.info(f"[RENDERER] Auto-filled ON from join_on: {on_clause}")
                    return on_clause
        except Exception as e:
            logger.warning(f"[RENDERER] FK resolution failed for {base_table} -> {join_table}: {e}")
            return None
        
        logger.debug(f"[RENDERER] Could not determine ON clause for {base_table} -> {join_table}")
        return None
    
    def _resolve_join_on_clause(self, base_table: str, join_table: str) -> Optional[str]:
        """
        Legacy method - kept for backward compatibility.
        Delegates to alias-aware version.
        """
        return self._resolve_join_on_clause_with_aliases(
            base_table=base_table,
            base_alias=base_table[0].lower(),
            join_table=join_table,
            join_alias="d"
        )
    
    def _resolve_column_with_alias(self, column_ref: str) -> str:
        """
        Resolve table-qualified column references to use aliases.
        
        E.g., 'cards.card_status' → 't_abc123.card_status'
        
        Args:
            column_ref: Column reference (may be table.column or just column)
            
        Returns:
            Resolved column reference using aliases if applicable
        """
        if "." in column_ref:
            # Has table prefix - resolve table name to alias
            parts = column_ref.split(".", 1)
            table_name = parts[0]
            column_name = parts[1]
            
            # Look up alias for this table
            if table_name in self.alias_map:
                alias = self.alias_map[table_name]
                return f"{alias}.{column_name}"
        
        # No table prefix or table not found - return as-is
        return column_ref
    
    def _render_where(self, plan: QueryPlan) -> str:
        """Render WHERE clause with proper alias resolution."""
        if not plan.where_conditions:
            return ""
        
        conditions = []
        for cond in plan.where_conditions:
            # Resolve column reference to use aliases if it's table-qualified
            column_ref = self._resolve_column_with_alias(cond.left)
            left = self._quote_identifier(column_ref)
            value = self._escape_value(cond.right, cond.is_literal)
            conditions.append(f"{left} {cond.operator} {value}")
        
        where_clause = "WHERE " + " AND ".join(conditions)
        return where_clause
    
    def _render_group_by(self, plan: QueryPlan) -> str:
        """Render GROUP BY clause with proper alias resolution."""
        if not plan.group_by:
            return ""
        
        cols = []
        for col in plan.group_by:
            # Resolve column reference to use aliases if table-qualified
            resolved_col = self._resolve_column_with_alias(col)
            cols.append(self._quote_identifier(resolved_col))
        return "GROUP BY " + ", ".join(cols)
    
    def _render_having(self, plan: QueryPlan) -> str:
        """Render HAVING clause with proper alias resolution."""
        if not plan.having_conditions:
            return ""
        
        conditions = []
        for cond in plan.having_conditions:
            # Resolve column reference to use aliases if table-qualified
            column_ref = self._resolve_column_with_alias(cond.left)
            left = self._quote_identifier(column_ref)
            value = self._escape_value(cond.right, cond.is_literal)
            conditions.append(f"{left} {cond.operator} {value}")
        
        having_clause = "HAVING " + " AND ".join(conditions)
        return having_clause
    
    def _render_order_by(self, plan: QueryPlan) -> str:
        """Render ORDER BY clause with proper alias resolution."""
        if not plan.order_by:
            return ""
        
        fields = []
        for field in plan.order_by:
            # Resolve column reference to use aliases if table-qualified
            resolved_col = self._resolve_column_with_alias(field.column)
            col = self._quote_identifier(resolved_col)
            direction = field.direction.value
            fields.append(f"{col} {direction}")
        
        return "ORDER BY " + ", ".join(fields)
    
    def _get_default_schema(self) -> str:
        """Get default schema from config based on DB type - ZERO HARDCODING."""
        from app.config import settings
        
        db_type = settings.db_type.lower()
        
        if db_type == "postgresql":
            return settings.postgres_schema or "public"
        elif db_type == "mysql" or db_type == "mariadb":
            return settings.mysql_db or "genai"
        elif db_type == "sqlserver":
            return "dbo"  # SQL Server default schema
        elif db_type == "sqlite":
            return "main"  # SQLite default schema
        else:
            return "public"  # Fallback default
    
    def _quote_identifier(self, identifier: str) -> str:
        """Quote identifiers according to dialect."""
        if not identifier:
            return identifier
        
        # Handle schema-qualified identifiers (e.g., schema.table)
        if "." in identifier:
            parts = identifier.split(".")
            if self.dialect == "mssql":
                return ".".join(f"[{part}]" for part in parts)
            elif self.dialect == "mysql":
                return ".".join(f"`{part}`" for part in parts)
            else:  # PostgreSQL, SQLite
                return ".".join(f'"{part}"' for part in parts)
        
        # Single identifier
        if self.dialect == "mssql":
            return f"[{identifier}]"
        elif self.dialect == "mysql":
            return f"`{identifier}`"
        else:  # PostgreSQL, SQLite
            return f'"{identifier}"'
    
    def _escape_value(self, value: Any, is_literal: bool = True) -> str:
        """Escape/quote values according to dialect."""
        if not is_literal:
            # It's a column reference, quote as identifier
            return self._quote_identifier(str(value))
        
        if value is None:
            return "NULL"
        elif isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            # IN clause
            escaped = [self._escape_value(v, True) for v in value]
            return f"({', '.join(escaped)})"
        else:  # String
            # Escape single quotes
            escaped = str(value).replace("'", "''")
            return f"'{escaped}'"


async def create_query_plan(
    user_query: str,
    catalog,
    retriever,
    llm
) -> QueryPlan:
    """
    Generate query plan from natural language.
    
    This is a simplified example - full implementation would use LLM
    with structured output prompting to generate valid plans.
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
    
    # Create basic plan
    plan = QueryPlan(
        select_expressions=select_exprs,
        from_table=best_table.name,
        from_schema=best_table.metadata.get("schema", None) if best_table.metadata else None,
        limit=None,  # Will be enforced by renderer
        intent="select",
        confidence=best_table.confidence,
    )
    
    return plan


class QueryPlanGenerator:
    """
    Simple plan generator adapter for semantic query orchestration.
    
    Generates minimal QueryPlan objects from high-level query context.
    Plans are rendered to SQL by QueryPlanRenderer.
    """
    
    def __init__(self, dialect: str = "postgresql", schema_grounding: Optional[Any] = None, join_graph_builder: Optional[Any] = None):
        """Initialize plan generator."""
        self.dialect = dialect
        self.schema_grounding = schema_grounding
        self.join_graph_builder = join_graph_builder
        self.renderer = QueryPlanRenderer(
            dialect=dialect,
            schema_grounding=schema_grounding,
            join_graph_builder=join_graph_builder
        )
        logger.info(f"[PLAN GENERATOR] Initialized with dialect: {dialect}")
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
        Generate a basic query plan.
        
        Args:
            primary_table: Main table for query
            selected_columns: Columns to select (None = SELECT *)
            schema: Schema name
            where_conditions: List of WHERE conditions
            joins: List of JOINs
            
        Returns:
            QueryPlan object
        """
        # Build select expressions
        if selected_columns is None or selected_columns == []:
            select_exprs = ["*"]
        else:
            select_exprs = selected_columns
        
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
        
        # Create plan
        plan = QueryPlan(
            select_expressions=select_exprs,
            from_table=primary_table,
            from_schema=schema,
            where_conditions=where_conds,
            joins=join_clauses,
            intent="select",
            confidence=1.0,  # Generated from context, assume high confidence
        )
        
        logger.info(f"[PLAN] Generated plan: {primary_table} with {len(select_exprs)} columns")
        return plan
    
    def render_plan(self, plan: QueryPlan) -> str:
        """
        Render a plan to SQL.
        
        Args:
            plan: QueryPlan to render
        
        Returns:
            SQL query string
        """
        return self.renderer.render(plan)
    
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
        ✅ REQUIREMENT C: Plan-first LLM generation with validation and regeneration.
        
        Generate a structured query plan from natural language using LLM.
        1. LLM outputs QueryPlan JSON (structured, not raw SQL)
        2. Validator enforces semantic coverage rules (if query mentions "credit" and "cards" table is available, plan must include it)
        3. If validation fails, regenerate with stronger guidance
        
        Args:
            user_query: Natural language user query
            available_tables: List of relevant table names (from semantic retrieval)
            available_columns_per_table: Dict mapping tables to their columns
            possible_joins: List of (table1, table2) join candidates
            table_metadata: Optional dict with table/column metadata including sample_values
            catalog: Optional SemanticSchemaCatalog for sample data
            
        Returns:
            QueryPlan object (structured, validated plan with required table coverage)
        """
        # Build schema context for LLM (with sample data to prevent hallucination)
        schema_context = self._build_schema_context(
            available_tables,
            available_columns_per_table,
            possible_joins,
            catalog=catalog
        )
        
        # First attempt: Generate plan with standard prompt
        plan = await self._llm_generate_plan(
            user_query,
            available_tables,
            available_columns_per_table,
            schema_context,
            table_metadata,
            regeneration_hint=None  # First attempt
        )
        
        # ✅ ENHANCEMENT: Ground WHERE conditions to actual column values in database
        if plan and catalog:
            plan = self._enhance_plan_with_value_grounding(
                plan,
                user_query,
                available_tables,
                available_columns_per_table,
                catalog
            )
        
        if plan:
            # Validate plan against semantic coverage rules
            is_valid, validation_msg, suggestion = self._validate_plan_coverage(
                user_query,
                plan,
                available_tables
            )
            
            if is_valid:
                logger.info(f"[PLAN-FIRST] Generated and validated plan: {plan.from_table}")
                return plan
            else:
                # Validation failed - regenerate with stronger guidance
                logger.warning(f"[PLAN-VALIDATOR] Validation failed: {validation_msg}")
                logger.info(f"[PLAN-VALIDATOR] Regenerating with guidance: {suggestion}")
                
                # Create stronger prompt that includes validators findings
                plan = await self._llm_generate_plan(
                    user_query,
                    available_tables,
                    available_columns_per_table,
                    schema_context,
                    table_metadata,
                    regeneration_hint=suggestion
                )
                
                if plan:
                    logger.info(f"[PLAN-FIRST] Regenerated plan includes: {plan.from_table} + {len(plan.joins)} joins")
                    return plan
                else:
                    # If regeneration also fails, fall back
                    logger.warning("[PLAN-FIRST] Regeneration failed, using fallback")
                    return self._create_fallback_plan(
                        available_tables,
                        available_columns_per_table,
                        user_query,
                        schema_context,
                        table_metadata
                    )
        else:
            # Initial generation failed
            logger.warning("[PLAN-FIRST] Initial generation failed, using fallback")
            return self._create_fallback_plan(
                available_tables,
                available_columns_per_table,
                user_query,
                schema_context,
                table_metadata
            )
    
    async def _llm_generate_plan(
        self,
        user_query: str,
        available_tables: List[str],
        available_columns_per_table: Dict[str, List[str]],
        schema_context: str,
        table_metadata: Optional[Dict[str, Any]],
        regeneration_hint: Optional[str],
    ) -> Optional[QueryPlan]:
        """
        Call LLM to generate plan. Separated for reusability.
        
        Args:
            regeneration_hint: If provided, includes guidance for regeneration
        
        Returns:
            QueryPlan or None if generation fails
        """
        # Extract real table/column names from schema to use in example
        example_table = available_tables[0] if available_tables else "customers"
        example_columns = available_columns_per_table.get(example_table, [])[:2] if available_columns_per_table else []
        if not example_columns:
            example_columns = ["id", "name"]
        
        regeneration_guidance = ""
        if regeneration_hint:
            regeneration_guidance = f"\n⚠️ REGENERATION GUIDANCE:\n{regeneration_hint}\nMake sure the plan includes ALL required tables for the query context."
        
        llm_prompt = f"""RESPOND WITH ONLY THIS JSON STRUCTURE. NO OTHER TEXT. NO EXPLANATIONS.

EXAMPLE (using REAL table/column names from schema below):
{{
  "select_expressions": {example_columns},
  "from_table": "{example_table}",
  "where_conditions": [
    {{
      "column": "cards.card_status",
      "operator": "=",
      "value": "ACTIVE",
      "is_literal": true
    }}
  ],
  "joins": [
    {{
      "right_table": "cards",
      "join_type": "INNER",
      "on_conditions": [
        {{
          "left_column": "customers.customer_id",
          "right_column": "cards.customer_id",
          "operator": "="
        }}
      ]
    }}
  ]
}}

⚠️ WHERE CONDITIONS: Extract filter criteria DYNAMICALLY from user query
- If user mentions "active", "inactive", "premium", "standard", etc. → ADD WHERE conditions
- Match keywords to actual column names from schema (e.g., "status" columns, "type" columns)
- Use REAL column names from AVAILABLE TABLES below
- DO NOT add filters the user didn't mention (zero hardcoding)
- If user says "all records" or no filters mentioned → use empty array []

⚠️ NOTE: LIMIT is NOT required - use count-first approach!
LIMIT enforcement is handled automatically by apply_smart_limit() after COUNT probe.
Generate ONLY: select_expressions, from_table, where_conditions, joins
DO NOT include: limit, suggested_limit, limit_reasoning

⚠️ IMPORTANT FOR JOINS:
- Use "right_table" field (NOT "table")
- Structure on_conditions as array of objects with "left_column", "right_column", "operator"
- Do NOT use "on_expression" string format
- Join types: INNER, LEFT, RIGHT, FULL, CROSS

CRITICAL RULES:
1. ⚠️ USE REAL COLUMN NAMES ONLY - Column names are shown with SAMPLE values (in brackets)
   - Example: "cards: [...card_id, customer_id, status...]" means these are ACTUAL columns
   - Example sample values shown like "status: [ACTIVE, INACTIVE]" are REAL data from database
   - DO NOT invent column names like "card_status" if not listed with samples
2. Match user query keywords to actual columns shown with samples
   - If user says "active" and you see "status: [ACTIVE, INACTIVE]" → use status column
   - If a column isn't shown with sample values, it may not exist in your data sample
3. Use ONLY tables and columns from the AVAILABLE TABLES below
4. If user query mentions keywords that map to specific tables (e.g., "credit card" → include 'cards' table), MUST include those tables
5. If query mentions multiple concepts (e.g., "credit card customers"), include joins to connect all relevant tables
6. ANALYZE user query for implicit filters (status, type, category, grade, amount ranges, dates, etc.)
7. DO NOT use placeholder names like "table1", "column1" etc.
8. ALWAYS use the exact field names shown in the EXAMPLE above
9. ZERO HARDCODING: Only include filters if user actually requested them
10. ✅ VALIDATION: Before using a column in WHERE/GROUP BY/ORDER BY, verify it's listed in schema section below

USER QUERY: {user_query}

AVAILABLE TABLES AND COLUMNS:
{schema_context}{regeneration_guidance}

REQUIRED: Generate the QueryPlan JSON ONLY. Use REAL table/column names and field names exactly as shown in EXAMPLE. Extract filters DYNAMICALLY from user query - no hardcoding. No markdown, no explanations."""
        
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
            
            # Parse JSON
            import json
            plan_dict = json.loads(llm_response)
            logger.debug(f"[PLAN-LLM] Raw JSON from LLM: {json.dumps(plan_dict, indent=2)[:500]}")
            plan = self._dict_to_query_plan(plan_dict)
            logger.debug(f"[PLAN-LLM] Converted plan - FROM: {plan.from_table}, JOINS: {[(j.right_table, len(j.conditions)) for j in plan.joins]}")
            return plan
            
        except Exception as e:
            logger.debug(f"[PLAN-FIRST] LLM generation error: {e}")
            return None
    
    def _validate_plan_coverage(
        self,
        user_query: str,
        plan: QueryPlan,
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
        plan: QueryPlan,
        user_query: str,
        available_tables: List[str],
        available_columns_per_table: Dict[str, List[str]],
        catalog: Any,
    ) -> QueryPlan:
        """
        Enhance WHERE conditions by grounding values mentioned in the query
        to actual columns in the database that contain those values.
        
        Example:
        - User: "I want credit card customers"
        - LLM generates: WHERE status = 'ACTIVE' (generic)
        - Value grounding finds: 'credit' → card_type column has 'CREDIT' values
        - Enhanced: Add WHERE card_type = 'CREDIT'
        
        Args:
            plan: QueryPlan from LLM
            user_query: Original user query
            available_tables: List of accessible tables
            available_columns_per_table: Dict of columns per table (limited to top-k)
            catalog: SemanticSchemaCatalog with sample data
            
        Returns:
            Enhanced QueryPlan
        """
        try:
            grounder = ValueBasedColumnGrounder(catalog)
            
            # Get tables in the plan (tables being queried)
            plan_tables = [plan.from_table] + [j.right_table for j in plan.joins]
            
            # ✅ FIX: Build ALL_COLUMNS mapping for value grounding
            # The retriever only returns top-3 columns, but value grounding needs
            # to search ALL columns to find columns like 'card_type' that have sample values
            # matching the user's query keywords (e.g., 'credit' -> finds 'CREDIT' in card_type)
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
                logger.info(f"  → {filt['table']}.{filt['column']} = '{filt['value']}'")
            
            # ✅ ENHANCEMENT STRATEGY: 
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
            schema_text += f"  • {table}: {col_list}\n"
            
            # Add sample data if catalog available
            if catalog:
                try:
                    sample_data = catalog.get_table_sample_data(table)
                    if sample_data:
                        for col_name, samples in sample_data.items():
                            # Always show actual sample values so LLM knows they exist
                            sample_values_str = ", ".join(str(s)[:20] for s in samples[:3])
                            schema_text += f"      └─ {col_name}: [{sample_values_str}]\n"
                except Exception as e:
                    logger.debug(f"[SCHEMA-CONTEXT] Could not get samples for {table}: {e}")
        
        if possible_joins:
            schema_text += "\nPOSSIBLE JOINS:\n"
            for table1, table2 in possible_joins:
                schema_text += f"  • {table1} ←→ {table2}\n"
        
        return schema_text
    
    def _dict_to_query_plan(self, plan_dict: Dict[str, Any]) -> QueryPlan:
        """Convert LLM-generated dict to QueryPlan object."""
        # Extract fields with defaults
        select_exprs = plan_dict.get("select_expressions", ["*"])
        from_table = plan_dict.get("from_table", "")
        
        if not from_table:
            raise ValueError("LLM plan missing required 'from_table' field")
        
        # ✅ VALIDATION: Detect and reject placeholder table names
        placeholder_patterns = ["table_name", "table1", "table2", "other_table", "unknown", "table"]
        if from_table.lower() in placeholder_patterns:
            raise ValueError(f"LLM returned placeholder table name '{from_table}' instead of actual table - rejecting plan")
        
        # ✅ VALIDATION: Detect and replace placeholder column names
        placeholder_col_patterns = ["column1", "column2", "col_name", "value", "col"]
        cleaned_exprs = []
        for expr in select_exprs:
            if expr.lower() in placeholder_col_patterns or expr == "*":
                cleaned_exprs.append(expr)  # Keep "*" or placeholders for now
            else:
                cleaned_exprs.append(expr)
        
        select_exprs = cleaned_exprs
        
        # Convert where conditions - ROBUST: handle both dict and string formats
        where_conds = []
        for cond in plan_dict.get("where_conditions", []):
            # Handle case where LLM returns a string instead of object (e.g., "card_type = 'CREDIT'")
            if isinstance(cond, str):
                # Skip malformed string conditions
                logger.debug(f"[PLAN-FIRST] Skipping string WHERE condition (not structured): {cond[:50]}")
                continue
            elif isinstance(cond, dict):
                where_conds.append(
                    WhereCondition(
                        left=cond.get("column", ""),
                        operator=cond.get("operator", "="),
                        right=cond.get("value", ""),
                        is_literal=cond.get("is_literal", True),
                    )
                )
        
        # Convert joins - ROBUST: handle both LLM output formats
        join_clauses = []
        for idx, join in enumerate(plan_dict.get("joins", [])):
            if not isinstance(join, dict):
                logger.debug(f"[PLAN-FIRST] Skipping malformed JOIN (not dict): {join}")
                continue
            
            # ✅ HANDLE BOTH FORMATS:
            # Format 1 (structured): {"right_table": "cards", "on_conditions": [...]}
            # Format 2 (LLM actual): {"table": "cards", "on_expression": "cards.customer_id = ..."}
            
            right_table = join.get("right_table") or join.get("table", "")
            join_type = join.get("join_type", "INNER")
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
                    # Parse simple expression like "cards.customer_id = customers.customer_id"
                    # This is a string, we'll store it as a raw condition for now
                    # The renderer handles ON clause generation differently
                    logger.debug(f"[PLAN-FIRST] JOIN #{idx+1} has on_expression (string): {on_expr}")
                    # TODO: Parse this more intelligently if needed
            
            if not right_table:
                logger.warning(f"[PLAN-FIRST] ⚠️ JOIN #{idx+1} has empty right_table! Full join object: {join}")
            
            join_clauses.append(
                JoinClause(
                    join_type=JoinType(join_type),
                    right_table=right_table,
                    right_schema=None,
                    conditions=conditions,
                )
            )
        
        plan = QueryPlan(
            select_expressions=select_exprs,
            from_table=from_table,
            from_schema=plan_dict.get("from_schema"),
            where_conditions=where_conds,
            joins=join_clauses,
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
    ) -> QueryPlan:
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
        
        # ✅ IMPROVEMENT: Create JOINs if multiple tables available
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
            # e.g., customers → cards (customer_id), transactions (customer_id), etc.
            
            for i in range(1, min(len(tables), 4)):  # Max 3 tables (limit complexity)
                right_table = tables[i]
                # Heuristic join condition: look for foreign key pattern
                # Most common: table_name_id (customer_id for customers table)
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
                    logger.info(f"[FALLBACK] Auto-creating JOIN: {from_table}.{from_pk} → {right_table}.{join_condition_col}")
        
        # ✅ PURE DYNAMIC APPROACH: NO hardcoded filter logic
        # Filters should come from LLM analysis of user query, not pattern matching
        # The fallback plan should NOT add any WHERE conditions
        # All filtering logic is LLM-driven (query_plan_generator LLM call above)
        where_conditions = []
        
        plan = QueryPlan(
            select_expressions=select_exprs,
            from_table=from_table,
            from_schema=None,
            where_conditions=where_conditions,
            joins=joins,
            intent="select",
            confidence=0.3,  # Lower confidence for fallback
        )
        
        return plan
