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


class RightKind(str, Enum):
    """Type of right-hand side in a condition."""
    LITERAL = "literal"        # "AP", 100, True
    LIST = "list"              # ["AP","TS"]
    RANGE = "range"            # ["2024-01-01","2024-12-31"]
    COLUMN = "column"          # other_col reference
    SUBQUERY = "subquery"      # "SELECT ..."
    RAW = "raw"                # "NOW()", "DATE_TRUNC(...)"


class SetOpType(str, Enum):
    """Set operation types."""
    UNION = "UNION"
    UNION_ALL = "UNION ALL"
    INTERSECT = "INTERSECT"
    EXCEPT = "EXCEPT"


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
class QueryPlan:
    """
    Database-agnostic query plan.
    
    Represents a SELECT query as a structured plan that can be
    rendered to any SQL dialect.
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
        limit_offset_part = self._render_limit_offset(plan)
        
        # Combine parts
        sql_parts = [select_part, from_part, join_part, where_part, 
                     group_by_part, having_part, order_by_part, limit_offset_part]
        sql = " ".join(part for part in sql_parts if part)
        
        # Add terminating semicolon
        if not sql.endswith(";"):
            sql += ";"
        
        return sql
    
    def _render_select(self, plan: QueryPlan) -> str:
        """Render SELECT clause using pre-generated aliases for consistency."""
        # Use the alias already generated in render() method
        from_alias = self.alias_map.get(plan.from_table, plan.from_alias or generate_random_alias(plan.from_table))
        
        # Build SELECT keyword with optional DISTINCT
        select_keyword = "SELECT DISTINCT" if plan.distinct else "SELECT"
        
        # CRITICAL FIX: Respect column_selection intent from LLM analysis
        # If LLM determined user wants ALL_COLUMNS, use SELECT * (or alias.*)
        if plan.column_selection and plan.column_selection.intent == ColumnSelectionIntent.ALL_COLUMNS and not plan.select_aggregates:
            logger.info(f"[RENDER-SELECT] Respecting LLM intent ALL_COLUMNS (confidence={plan.column_selection.confidence})")
            return f"{select_keyword} {from_alias}.*"
        
        # If no specific columns and no aggregates, use SELECT alias.* to get all columns
        if not plan.select_expressions and not plan.select_aggregates:
            return f"{select_keyword} {from_alias}.*"
        
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
                func_val = agg.function.value if hasattr(agg.function, 'value') else str(agg.function)
                agg_expr = f"{func_val}({self._quote_identifier(agg_col)})"
            else:
                func_val = agg.function.value if hasattr(agg.function, 'value') else str(agg.function)
                agg_expr = f"{func_val}(*)"
            
            if agg.alias:
                agg_expr += f" AS {self._quote_identifier(agg.alias)}"
            
            cols.append(agg_expr)
        
        select = f"{select_keyword} " + ", ".join(cols)
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
            join_type_val = join.join_type.value if hasattr(join.join_type, 'value') else str(join.join_type)
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
            join_parts.append(f"{join_type_val} JOIN {right_table} AS {join_alias} ON {on_clause}")
        
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
            ON clause string using aliases (e.g., "a.entity_id = b.entity_id") or None
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
        
        E.g., 'table_name.column_name' → 't_abc123.column_name'
        
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
        """
        Render WHERE clause with support for advanced operators.
        Handles BETWEEN, EXISTS, IN-subquery, IS NULL, LIKE, etc. dynamically.
        """
        if not plan.where_conditions:
            return ""
        
        conditions = []
        for cond in plan.where_conditions:
            rendered = self._render_condition(cond, plan)
            if rendered:
                conditions.append(rendered)
        
        if not conditions:
            return ""
        
        where_clause = "WHERE " + " AND ".join(conditions)
        return where_clause
    
    def _render_condition(self, cond: WhereCondition, plan: QueryPlan) -> str:
        """
        Dynamically render a single condition based on operator and right_kind.
        This is the core dynamic rendering logic (completely composable).
        
        Args:
            cond: WhereCondition to render
            plan: QueryPlan (for alias resolution)
            
        Returns:
            Rendered condition string
        """
        left = self._resolve_column_with_alias(cond.left)
        left = self._quote_identifier(left)
        op = cond.operator.upper()
        right_kind = cond.right_kind
        
        # CASE 1: IS NULL / IS NOT NULL (no right operand needed)
        if op in ("IS NULL", "IS NOT NULL"):
            return f"{left} {op}"
        
        # CASE 2: BETWEEN with range
        if op == "BETWEEN" and right_kind == RightKind.RANGE:
            if isinstance(cond.right, (list, tuple)) and len(cond.right) >= 2:
                a = self._escape_value(cond.right[0], True)
                b = self._escape_value(cond.right[1], True)
                return f"{left} BETWEEN {a} AND {b}"
            else:
                logger.warning(f"BETWEEN operator missing range values: {cond.right}")
                return ""
        
        # CASE 3: EXISTS with subquery
        if op == "EXISTS":
            subquery = str(cond.right).strip()
            # Subquery should be wrapped in parentheses
            if not subquery.startswith("("):
                subquery = f"({subquery})"
            return f"EXISTS {subquery}"
        
        # CASE 4: IN with list of literals
        if op == "IN" and right_kind == RightKind.LIST:
            if isinstance(cond.right, (list, tuple)):
                escaped_values = [self._escape_value(v, True) for v in cond.right]
                value_list = "(" + ", ".join(escaped_values) + ")"
                return f"{left} IN {value_list}"
            else:
                logger.warning(f"IN operator missing list: {cond.right}")
                return ""
        
        # CASE 5: IN with subquery
        if op == "IN" and right_kind == RightKind.SUBQUERY:
            subquery = str(cond.right).strip()
            if not subquery.startswith("("):
                subquery = f"({subquery})"
            return f"{left} IN {subquery}"
        
        # CASE 6: NOT IN with list
        if op == "NOT IN" and right_kind == RightKind.LIST:
            if isinstance(cond.right, (list, tuple)):
                escaped_values = [self._escape_value(v, True) for v in cond.right]
                value_list = "(" + ", ".join(escaped_values) + ")"
                return f"{left} NOT IN {value_list}"
            else:
                return ""
        
        # CASE 7: NOT IN with subquery
        if op == "NOT IN" and right_kind == RightKind.SUBQUERY:
            subquery = str(cond.right).strip()
            if not subquery.startswith("("):
                subquery = f"({subquery})"
            return f"{left} NOT IN {subquery}"
        
        # CASE 8: Column reference
        if right_kind == RightKind.COLUMN:
            right_col = self._resolve_column_with_alias(str(cond.right))
            right = self._quote_identifier(right_col)
            return f"{left} {op} {right}"
        
        # CASE 9: Raw expression (function calls, etc.)
        if right_kind == RightKind.RAW:
            return f"{left} {op} {cond.right}"
        
        # DEFAULT: Literal value
        value = self._escape_value(cond.right, True)
        return f"{left} {op} {value}"
    
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
        """Render HAVING clause with proper alias resolution and dynamic condition rendering."""
        if not plan.having_conditions:
            return ""
        
        conditions = []
        for cond in plan.having_conditions:
            # For HAVING clauses with aggregate functions, handle specially
            rendered = self._render_having_condition(cond, plan)
            if rendered:
                conditions.append(rendered)
        
        if not conditions:
            return ""
        
        having_clause = "HAVING " + " AND ".join(conditions)
        return having_clause
    
    def _render_having_condition(self, cond: WhereCondition, plan: QueryPlan) -> str:
        """
        Render HAVING condition with special handling for aggregate functions.
        
        For expressions like COUNT(table_name.id) > 10, we need to:
        1. Resolve table aliases in the column reference
        2. NOT quote the entire aggregate function call
        """
        left = cond.left.strip()
        op = cond.operator.upper()
        right_kind = cond.right_kind
        
        # Check if left side is an aggregate function (COUNT, SUM, AVG, MIN, MAX)
        is_aggregate = any(func in left.upper() for func in ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX(", "STDDEV("])
        
        if is_aggregate:
            # For aggregates, resolve column aliases within the function call
            # Example: COUNT(table_name.id) → COUNT(t_abc.id)
            left = self._resolve_column_with_alias(left)
            # Don't quote the entire function - instead quote only column references inside
            # This is handled by _resolve_column_with_alias already
        else:
            # For non-aggregates, use normal column quoting
            left = self._resolve_column_with_alias(left)
            left = self._quote_identifier(left)
        
        # CASE 1: IS NULL / IS NOT NULL (no right operand needed)
        if op in ("IS NULL", "IS NOT NULL"):
            return f"{left} {op}"
        
        # CASE 2: BETWEEN with range
        if op == "BETWEEN" and right_kind == RightKind.RANGE:
            if isinstance(cond.right, (list, tuple)) and len(cond.right) >= 2:
                a = self._escape_value(cond.right[0], True)
                b = self._escape_value(cond.right[1], True)
                return f"{left} BETWEEN {a} AND {b}"
            else:
                logger.warning(f"BETWEEN operator missing range values: {cond.right}")
                return ""
        
        # CASE 3: EXISTS with subquery
        if op == "EXISTS":
            subquery = str(cond.right).strip()
            if not subquery.startswith("("):
                subquery = f"({subquery})"
            return f"EXISTS {subquery}"
        
        # CASE 4: IN with list of literals
        if op == "IN" and right_kind == RightKind.LIST:
            if isinstance(cond.right, (list, tuple)):
                escaped_values = [self._escape_value(v, True) for v in cond.right]
                value_list = "(" + ", ".join(escaped_values) + ")"
                return f"{left} IN {value_list}"
            else:
                logger.warning(f"IN operator missing list: {cond.right}")
                return ""
        
        # DEFAULT: Literal value comparison (most common for HAVING)
        value = self._escape_value(cond.right, True)
        return f"{left} {op} {value}"
    
    def _render_order_by(self, plan: QueryPlan) -> str:
        """Render ORDER BY clause with proper alias resolution."""
        if not plan.order_by:
            return ""
        
        fields = []
        for field in plan.order_by:
            # Resolve column reference to use aliases if table-qualified
            resolved_col = self._resolve_column_with_alias(field.column)
            col = self._quote_identifier(resolved_col)
            direction_val = field.direction.value if hasattr(field.direction, 'value') else str(field.direction)
            fields.append(f"{col} {direction_val}")
        
        return "ORDER BY " + ", ".join(fields)
    
    def _render_limit_offset(self, plan: QueryPlan) -> str:
        """
        Render LIMIT and OFFSET clauses.
        Dialect-aware: handles different SQL dialects (PostgreSQL, MySQL, SQL Server, SQLite).
        """
        parts = []
        
        if plan.limit is not None:
            limit_val = int(plan.limit)
            if self.dialect == "mssql":
                # SQL Server uses OFFSET ... ROWS FETCH NEXT ... ROWS ONLY
                # This gets handled specially in the main render() if both limit and offset exist
                if plan.offset is not None:
                    # Will be handled as special case below
                    pass
                else:
                    # Just LIMIT (TOP in SQL Server) - but OFFSET/FETCH is the modern way
                    parts.append(f"OFFSET 0 ROWS FETCH NEXT {limit_val} ROWS ONLY")
            else:
                # PostgreSQL, MySQL, SQLite use LIMIT syntax
                parts.append(f"LIMIT {limit_val}")
        
        if plan.offset is not None:
            offset_val = int(plan.offset)
            if self.dialect == "mssql":
                # SQL Server: OFFSET ... ROWS FETCH NEXT ... ROWS ONLY
                if plan.limit is not None:
                    # Replace the LIMIT clause we just added
                    parts = []
                    parts.append(f"OFFSET {offset_val} ROWS FETCH NEXT {int(plan.limit)} ROWS ONLY")
                else:
                    # Just OFFSET
                    parts = []
                    parts.append(f"OFFSET {offset_val} ROWS")
            else:
                # PostgreSQL, MySQL, SQLite
                parts.append(f"OFFSET {offset_val}")
        
        return " ".join(parts)
    
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
        2. Validator enforces semantic coverage rules (if query explicitly references a table/entity and it is available, plan must include it)
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
        example_table = available_tables[0] if available_tables else "table_name"
        example_columns = available_columns_per_table.get(example_table, [])[:2] if available_columns_per_table else []
        if not example_columns:
            example_columns = ["id", "name"]
        example_metric_column = example_columns[0] if example_columns else "id"
        
        regeneration_guidance = ""
        if regeneration_hint:
            regeneration_guidance = f"\n⚠️ REGENERATION GUIDANCE:\n{regeneration_hint}\nMake sure the plan includes ALL required tables for the query context."
        
        llm_prompt = f"""RESPOND WITH ONLY THIS JSON STRUCTURE. NO OTHER TEXT. NO EXPLANATIONS.

You are generating a DATABASE-AGNOSTIC QUERY PLAN that supports:
- Single-table queries: DISTINCT, filters, aggregations, pagination, sorting
- Multi-table queries: JOINs (INNER/LEFT/RIGHT/FULL), aggregations across joins, EXISTS/NOT EXISTS
- Complex operators: BETWEEN, IN (literal/subquery), LIKE/ILIKE, IS NULL, EXISTS

⚠️ CRITICAL: ONLY include query features that the USER explicitly requests:
- If user says "get all X" → NO joins, NO aggregations, NO filters. Use SELECT * 
- If user says "count X" → use COUNT aggregate
- If user mentions "where/filter/specific" → add WHERE conditions
- If user asks for multiple tables → use JOINs
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
⚠️ SUPERLATIVE RULE: Words like "highest", "maximum", "top", "greatest", "largest" → ORDER BY column DESC LIMIT 1
⚠️ SUPERLATIVE RULE: Words like "lowest", "minimum", "bottom", "smallest", "least" → ORDER BY column ASC LIMIT 1
⚠️ If user asks "which X has highest Y" → MUST use ORDER BY Y DESC LIMIT 1

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
✅ Your query plan is a COMPOSITION of these OPTIONAL clauses:
   - SELECT [DISTINCT] columns [+ aggregates]
   - FROM table1
   - [JOIN table2 ON conditions] (0 or more joins)
   - [WHERE conditions] (0 or more conditions ANDed together)
   - [GROUP BY columns] [HAVING conditions]
   - [ORDER BY columns ASC|DESC]
   - [LIMIT n] [OFFSET n]

✅ OPERATORS & RIGHT_KIND MAPPING (How to structure conditions):
    - Literal value: {{"operator": "=", "right": "VALUE", "right_kind": "literal"}}
    - List of values: {{"operator": "IN", "right": ["VALUE1", "VALUE2"], "right_kind": "list"}}
   - Range (BETWEEN): {{"operator": "BETWEEN", "right": ["2024-01-01", "2024-12-31"], "right_kind": "range"}}
   - Column reference: {{"operator": "=", "right": "other_table.other_column", "right_kind": "column"}}
   - Subquery: {{"operator": "IN", "right": "SELECT id FROM...", "right_kind": "subquery"}}
   - Raw expression: {{"operator": "=", "right": "NOW()", "right_kind": "raw"}}

✅ SUPPORTED OPERATORS (will be rendered dynamically regardless of dialect):
   Comparison: =, <>, <, >, <=, >=, <>, !=
   Pattern: LIKE, ILIKE, NOT LIKE
   Null: IS NULL, IS NOT NULL
   Range: BETWEEN
   List: IN, NOT IN
   Existence: EXISTS, NOT EXISTS

⚠️ WHERE CONDITIONS: Extract filter criteria DYNAMICALLY from user query
- If user mentions "active", "inactive", "premium", "standard", etc. → ADD WHERE conditions
- Match keywords to actual column names from schema (e.g., "status" columns, "type" columns)
- Use REAL column names from AVAILABLE TABLES below
- DO NOT add filters the user didn't mention (zero hardcoding)
- If user says "all records" or no filters mentioned → use empty array []

⚠️ CLARIFICATION FIELDS (LLM-driven):
If you need user input BEFORE generating a full query, set:
  "clarification_needed": true,
  "clarification_question": "Which metric would you like to see: record count or total amount?",
  "clarification_options": ["Record Count", "Total Amount", "Both"]

This immediately returns the question to the user (ChatGPT-style), and the query generation retries after user responds.

⚠️ AGGREGATES STRUCTURE:
Use "fn" (function name), "expr" (column to aggregate), "alias" (result name):
  [
    {{"fn": "COUNT", "expr": "table.id", "alias": "record_count"}},
    {{"fn": "SUM", "expr": "table.amount", "alias": "total"}},
    {{"fn": "AVG", "expr": "table.value", "alias": "avg_value"}}
  ]

Functions: COUNT, SUM, AVG, MIN, MAX, STDDEV

⚠️ NOTE: LIMIT is NOT required - use count-first approach!
LIMIT enforcement is handled automatically by apply_smart_limit() after COUNT probe.

⚠️ RESPONSE CONSTRUCTION RULES:
1. Always include these base fields: distinct, select_expressions, from_table, limit, offset, clarification_needed
2. Use empty arrays/null for optional features NOT requested:
   - select_aggregates: [] (empty if no COUNT/SUM/AVG mentioned)
   - joins: [] (empty if only single table query)
   - where_conditions: [] (empty if no filters mentioned)
   - group_by: [] (empty if no grouping mentioned)
   - having_conditions: [] (empty if no group filtering mentioned)
   - order_by: [] (empty if no sorting mentioned)
3. DO NOT add fields or features beyond what the user requests
4. If "get all X" or "list X" → select_expressions=["*"], no joins, no aggregates
5. If user mentions "count" or "how many" → use select_aggregates with COUNT
6. If user mentions specific columns → list them instead of ["*"]

⚠️ IMPORTANT FOR JOINS:
- Use "right_table" field (NOT "table")
- Structure on_conditions as array of objects with "left_column", "right_column", "operator"
- Do NOT use "on_expression" string format
- Join types: INNER, LEFT, RIGHT, FULL, CROSS

CRITICAL RULES:
1. ⚠️ PREVENT HALLUCINATION: Do NOT invent features
    - If user says "get all {example_table}" → DO NOT add where conditions, joins, or aggregates
   - Only include what user explicitly asks for
   - Empty arrays [] for unused features (not null, not ignored)
   
2. ⚠️ USE REAL COLUMN NAMES ONLY - Column names are shown with SAMPLE values (in brackets)
    - Example: "some_table: [...id, related_id, status...]" means these are ACTUAL columns
   - Example sample values shown like "status: [ACTIVE, INACTIVE]" are REAL data from database
    - DO NOT invent column names like "status_flag" if not listed with samples
   
3. Match user query keywords to actual columns shown with samples
   - If user says "active" and you see "status: [ACTIVE, INACTIVE]" → use status column
   - If a column isn't shown with sample values, it may not exist in your data sample
   
4. Use ONLY tables and columns from the AVAILABLE TABLES below
5. If the user explicitly names an entity that matches a table in AVAILABLE TABLES, include that table
6. If the query mentions multiple entities that correspond to different tables, include joins to connect them
7. ANALYZE user query for explicit filters ONLY (status, type, category, grade, amount ranges, dates, etc.)
8. DO NOT use placeholder names like "table1", "column1" etc.
9. ALWAYS use the exact field names shown in the EXAMPLE above
10. ✅ VALIDATION: Before using a column in WHERE/GROUP BY/ORDER BY, verify it's listed in schema section below
11. ⚠️ SUPERLATIVES: If user asks for "highest/top/maximum/greatest" → ORDER BY column DESC LIMIT 1
    If user asks for "lowest/bottom/minimum/smallest/least" → ORDER BY column ASC LIMIT 1
    Match the superlative word to the relevant column (e.g., "highest amount" → ORDER BY amount DESC LIMIT 1)

USER QUERY: {user_query}

AVAILABLE TABLES AND COLUMNS:
{schema_context}{regeneration_guidance}

REQUIRED: Generate the complete QueryPlan JSON ONLY. Use REAL table/column names. Only include fields for features the user requests. Empty arrays [] for unused features. No markdown, no explanations."""
        
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
        - User: "Show records with category premium"
        - LLM generates: WHERE status = 'ACTIVE' (generic)
        - Value grounding finds: 'premium' → category column has 'PREMIUM' values
        - Enhanced: Add WHERE category = 'PREMIUM'
        
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
        """
        Convert LLM-generated dict to QueryPlan object.
        Handles all fields including new ones: distinct, limit, offset, group_by, having,
        order_by, and clarification fields.
        """
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
        
        # ✅ NEW: Parse select_aggregates with dynamic "fn", "expr", "alias" fields
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
            
            # ✅ HANDLE BOTH FORMATS:
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
                logger.warning(f"[PLAN-FIRST] ⚠️ JOIN #{idx+1} has empty right_table! Full join object: {join}")
            
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
        
        # ✅ NEW: Parse group_by (list of column names)
        group_by = plan_dict.get("group_by", [])
        if not isinstance(group_by, list):
            group_by = []
        
        # ✅ NEW: Parse having_conditions (same structure as where_conditions)
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
        
        # ✅ NEW: Parse order_by (list of {column, direction} objects)
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
        
        # ✅ NEW: Parse limit and offset
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
        
        # ✅ NEW: Parse distinct
        distinct = plan_dict.get("distinct", False)
        if not isinstance(distinct, bool):
            distinct = False
        
        # ✅ NEW: Parse clarification fields
        clarification_needed = plan_dict.get("clarification_needed", False)
        clarification_question = plan_dict.get("clarification_question")
        clarification_options = plan_dict.get("clarification_options", [])
        
        plan = QueryPlan(
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
            # e.g., table_a → table_b via shared foreign key patterns
            
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
