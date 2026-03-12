"""
Query Plan Unifier - Single Canonical SQL Pipeline

This module ensures ALL SQL generation routes through ONE canonical pipeline:
    NL -> GeneratedQueryPlan (internal) -> convert -> QueryPlan (canonical) -> Compiler -> SQL

The CANONICAL QueryPlan model is defined in query_plan.py. This is the ONLY
public-facing query plan type.

INTERNAL plan models exist in:
- query_plan_generator.GeneratedQueryPlan: Used for LLM output parsing

These internal models are CONVERTED to the canonical QueryPlan before SQL
compilation. External code should NEVER depend on internal models.

Architecture:
- ONE canonical QueryPlan (from query_plan.py) - the PUBLIC API
- Internal GeneratedQueryPlan for LLM output parsing (implementation detail)
- Converters to transform internal -> canonical
- All SQL goes through query_plan_compiler.py
"""

import logging
from typing import Dict, List, Optional, Any, Union, Literal
from dataclasses import dataclass

# ============================================================================
# CANONICAL IMPORTS - The ONE true QueryPlan model (PUBLIC API)
# ============================================================================
from .query_plan import (
    QueryPlan as CanonicalQueryPlan,
    SelectClause,
    FromClause,
    JoinClause,
    JoinCondition,
    GroupByClause,
    HavingClause,
    OrderByClause,
    OrderByField,
    BinaryCondition,
    LogicalCondition,
    NotCondition,
    Condition,
    ColumnRef,
    Literal as LiteralValue,
    SubqueryValue,
    QueryPlanValidationError,
    CTEClause,
    SetOperation,
)

from .query_plan_compiler import compile_query_plan, get_dialect_compiler

logger = logging.getLogger(__name__)


# ============================================================================
# Converter: query_plan_generator.GeneratedQueryPlan -> Canonical QueryPlan
# ============================================================================

def convert_from_generator_plan(
    generator_plan: Any,  # query_plan_generator.GeneratedQueryPlan
    dialect: str = "postgresql"
) -> CanonicalQueryPlan:
    """
    Convert query_plan_generator.GeneratedQueryPlan to canonical QueryPlan.
    
    Maps the flat structure:
        select_expressions, from_table, where_conditions, joins, etc.
    
    To the AST structure:
        SelectClause, FromClause, List[JoinClause], List[Condition], etc.
    """
    logger.info("[UNIFIER] Converting query_plan_generator.GeneratedQueryPlan to canonical")
    
    # Build SELECT clause
    select_fields = list(generator_plan.select_expressions) if generator_plan.select_expressions else []
    
    # Handle aggregates
    for agg in (generator_plan.select_aggregates or []):
        func = agg.function.value if hasattr(agg.function, 'value') else str(agg.function)
        col = agg.column or "*"
        expr = f"{func}({col})"
        if agg.alias:
            expr += f" AS {agg.alias}"
        select_fields.append(expr)
    
    select_clause = SelectClause(
        fields=select_fields if select_fields else ["*"],
        distinct=generator_plan.distinct
    )
    
    # Build FROM clause
    from_clause = FromClause(
        table=generator_plan.from_table,
        alias=generator_plan.from_alias
    )
    
    # Build JOIN clauses
    joins = []
    for gen_join in (generator_plan.joins or []):
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
        if gen_join.on_left and gen_join.on_right:
            on_conditions.append(JoinCondition(
                left=gen_join.on_left,
                op="=",
                right=gen_join.on_right
            ))
        
        joins.append(JoinClause(
            type=join_type,
            table=gen_join.right_table,
            alias=None,  # Generator model doesn't track aliases well
            on=on_conditions if on_conditions else None
        ))
    
    # Build WHERE conditions
    where_conditions = []
    for where_cond in (generator_plan.where_conditions or []):
        condition = _convert_where_condition(where_cond)
        if condition:
            where_conditions.append(condition)
    
    # Build GROUP BY
    group_by = None
    if generator_plan.group_by:
        group_by = GroupByClause(fields=list(generator_plan.group_by))
    
    # Build HAVING conditions
    having = None
    if generator_plan.having_conditions:
        having_conds = []
        for hav_cond in generator_plan.having_conditions:
            condition = _convert_where_condition(hav_cond)
            if condition:
                having_conds.append(condition)
        if having_conds:
            having = HavingClause(conditions=having_conds)
    
    # Build ORDER BY
    order_by = None
    if generator_plan.order_by:
        order_fields = []
        for ob in generator_plan.order_by:
            direction = "asc"
            if hasattr(ob.direction, 'value'):
                direction = "desc" if ob.direction.value.lower() == "desc" else "asc"
            order_fields.append(OrderByField(expr=ob.column, direction=direction))
        order_by = OrderByClause(fields=order_fields)
    
    # Window expressions — append to SELECT fields (already present as raw strings)
    for wexpr in (getattr(generator_plan, "window_expressions", None) or []):
        if wexpr and wexpr not in select_clause.fields:
            select_clause.fields.append(wexpr)

    # CTEs
    ctes = None
    raw_ctes = getattr(generator_plan, "ctes", None)
    if raw_ctes:
        ctes = []
        for c in raw_ctes:
            if isinstance(c, dict):
                ctes.append(CTEClause(
                    name=c.get("name", "cte"),
                    raw_sql=c.get("sql", c.get("raw_sql", "")),
                    recursive=c.get("recursive", False),
                ))

    # Set operation
    set_op = None
    raw_set_op = getattr(generator_plan, "set_op", None)
    if raw_set_op and isinstance(raw_set_op, dict):
        _op_map = {
            "UNION": "union", "UNION ALL": "union_all",
            "INTERSECT": "intersect", "EXCEPT": "except",
        }
        _op = _op_map.get(str(raw_set_op.get("type", "UNION")).upper(), "union")
        _right_sql = raw_set_op.get("right_sql", raw_set_op.get("raw_sql", ""))
        if _right_sql:
            set_op = SetOperation(op=_op, raw_sql=_right_sql)

    return CanonicalQueryPlan(
        intent="data_query",
        select=select_clause,
        from_=from_clause,
        joins=joins if joins else None,
        where=where_conditions if where_conditions else None,
        group_by=group_by,
        having=having,
        order_by=order_by,
        limit=generator_plan.limit,
        offset=generator_plan.offset,
        ctes=ctes,
        set_op=set_op,
        metadata={
            "source": "query_plan_generator",
            "confidence": generator_plan.confidence
        }
    )


def _convert_where_condition(where_cond: Any) -> Optional[Condition]:
    """Convert a generator WhereCondition to canonical Condition."""
    try:
        # Handle different operator formats
        operator = where_cond.operator
        if hasattr(operator, 'value'):
            operator = operator.value
        operator = str(operator).upper()
        
        # Map to canonical operators
        op_map = {
            "EQ": "=", "=": "=", "EQUALS": "=",
            "NE": "!=", "!=": "!=", "NOT_EQUALS": "!=", "<>": "!=",
            "LT": "<", "<": "<", "LESS_THAN": "<",
            "LE": "<=", "<=": "<=", "LESS_THAN_OR_EQUALS": "<=",
            "GT": ">", ">": ">", "GREATER_THAN": ">",
            "GE": ">=", ">=": ">=", "GREATER_THAN_OR_EQUALS": ">=",
            "LIKE": "LIKE", "ILIKE": "ILIKE",
            "IN": "IN", "NOT_IN": "NOT IN",
            "IS_NULL": "IS NULL", "IS_NOT_NULL": "IS NOT NULL",
            "BETWEEN": "BETWEEN",
        }
        canonical_op = op_map.get(operator, operator)
        
        # Build the condition
        left = ColumnRef(
            table=where_cond.table if hasattr(where_cond, 'table') else None,
            column=where_cond.column
        )
        
        # Handle different value types
        if canonical_op == "IN":
            # IN operator with multiple values
            values = where_cond.values if hasattr(where_cond, 'values') else [where_cond.value]
            right = [_python_to_literal(v) for v in values]
            return BinaryCondition(left=left, op=canonical_op, right=right)
        elif canonical_op in ("IS NULL", "IS NOT NULL"):
            return BinaryCondition(left=left, op=canonical_op, right=None)
        else:
            right = _python_to_literal(where_cond.value)
            return BinaryCondition(left=left, op=canonical_op, right=right)
            
    except Exception as e:
        logger.warning(f"[UNIFIER] Failed to convert where condition: {e}")
        return None


def _python_to_literal(value: Any) -> LiteralValue:
    """Convert Python value to canonical Literal."""
    if isinstance(value, bool):
        return LiteralValue(type="boolean", value=value)
    elif isinstance(value, int):
        return LiteralValue(type="integer", value=value)
    elif isinstance(value, float):
        return LiteralValue(type="number", value=value)
    elif value is None:
        return LiteralValue(type="null", value=None)
    else:
        return LiteralValue(type="string", value=str(value))


# ============================================================================
# Converter: Legacy plan_first_sql_generator format -> Canonical QueryPlan
# NOTE: plan_first_sql_generator.py now produces canonical QueryPlan directly.
# This converter handles LEGACY dictionary-based formats for backward compatibility.
# ============================================================================

def convert_from_plan_first(
    plan_first_plan: Any,
    dialect: str = "postgresql"
) -> CanonicalQueryPlan:
    """
    Convert legacy plan_first_sql_generator format to canonical QueryPlan.
    
    NOTE: The current plan_first_sql_generator.py now produces canonical QueryPlan
    directly, so this converter is only needed for legacy code that might
    still use the old dictionary-based plan format.
    
    If the input is already a canonical QueryPlan, it's returned as-is.
    """
    # Check if already canonical
    if isinstance(plan_first_plan, CanonicalQueryPlan):
        return plan_first_plan
    
    # Check if it has canonical QueryPlan structure (select, from_, etc)
    if hasattr(plan_first_plan, 'select') and hasattr(plan_first_plan, 'from_'):
        # Already canonical structure
        return plan_first_plan
    
    logger.info("[UNIFIER] Converting legacy plan_first_sql_generator plan to canonical")
    
    # Build SELECT clause
    select_fields = []
    for clause in (plan_first_plan.select_clauses or []):
        if clause.get("type") == "aggregate":
            func = clause.get("function", "COUNT")
            col = clause.get("column", "*")
            select_fields.append(f"{func}({col})")
        elif clause.get("type") == "wildcard":
            select_fields.append("*")
        else:
            select_fields.append(clause.get("column", "*"))
    
    select_clause = SelectClause(
        fields=select_fields if select_fields else ["*"],
        distinct=False
    )
    
    # Build FROM clause
    from_clause = FromClause(
        table=plan_first_plan.primary_table,
        alias=None
    )
    
    # Build WHERE conditions
    where_conditions = []
    for cond in (plan_first_plan.where_conditions or []):
        # Handle temporal expressions specially
        if cond.get("type") == "temporal":
            # Temporal conditions have pre-built expressions
            # We store them as raw SQL in metadata for now
            # TODO: Parse temporal expressions into proper AST
            where_conditions.append(BinaryCondition(
                left=ColumnRef(column=cond.get("column", "unknown")),
                op="RAW",  # Special marker for raw SQL
                right=LiteralValue(type="string", value=cond.get("expression", ""))
            ))
        else:
            operator = cond.get("operator", "=")
            if operator == "IN":
                values = cond.get("values", [])
                where_conditions.append(BinaryCondition(
                    left=ColumnRef(column=cond.get("column")),
                    op="IN",
                    right=[_python_to_literal(v) for v in values]
                ))
            else:
                where_conditions.append(BinaryCondition(
                    left=ColumnRef(column=cond.get("column")),
                    op=operator,
                    right=_python_to_literal(cond.get("value"))
                ))
    
    # Build JOIN clauses  
    joins = []
    for join_def in (plan_first_plan.joins or []):
        joins.append(JoinClause(
            type=join_def.get("type", "inner"),
            table=join_def.get("table"),
            alias=join_def.get("alias"),
            on=None  # Plan-first doesn't track ON conditions well
        ))
    
    # Map intent
    intent_val = plan_first_plan.intent
    if hasattr(intent_val, 'value'):
        intent_val = intent_val.value
    
    return CanonicalQueryPlan(
        intent="data_query",
        select=select_clause,
        from_=from_clause,
        joins=joins if joins else None,
        where=where_conditions if where_conditions else None,
        group_by=GroupByClause(fields=plan_first_plan.group_by) if plan_first_plan.group_by else None,
        having=None,
        order_by=None,  # Plan-first doesn't track ORDER BY consistently
        limit=plan_first_plan.limit,
        offset=None,
        metadata={
            "source": "plan_first_sql_generator",
            "semantic_intent": intent_val,
            "confidence": plan_first_plan.confidence
        }
    )


# ============================================================================
# Unified SQL Generation Entry Point
# ============================================================================

async def generate_sql_from_canonical_plan(
    plan: CanonicalQueryPlan,
    dialect: str = "postgresql",
    schema_name: Optional[str] = None,
) -> str:
    """
    Generate SQL from canonical QueryPlan using the unified compiler.
    
    This is the SINGLE entry point for SQL generation.
    All other paths should convert to CanonicalQueryPlan first, then call this.
    
    Args:
        plan: Canonical QueryPlan from query_plan.py
        dialect: SQL dialect (postgresql, mysql, sqlite, mssql)
        schema_name: Database schema for table qualification
    
    Returns:
        SQL string
    """
    from app.config import settings as _settings
    schema_name = schema_name or _settings.postgres_schema

    logger.info(f"[UNIFIER] Generating SQL via canonical compiler (dialect={dialect})")

    # Get the dialect-specific compiler
    compiler = get_dialect_compiler(dialect)

    # Compile the plan
    sql = compiler.generate(plan)

    # Add schema qualification if not present
    if schema_name and f"{schema_name}." not in sql:
        # Simple schema injection for FROM clause
        sql = _inject_schema_prefix(sql, schema_name, plan)
    
    logger.info(f"[UNIFIER] Generated SQL: {sql}")
    return sql


def _inject_schema_prefix(sql: str, schema_name: str, plan: CanonicalQueryPlan) -> str:
    """Inject schema prefix into table references."""
    import re
    
    if not plan.from_:
        return sql
    
    table = plan.from_.table
    # Replace unqualified table references
    pattern = rf'\bFROM\s+{re.escape(table)}\b'
    sql = re.sub(pattern, f'FROM {schema_name}.{table}', sql, flags=re.IGNORECASE)
    
    # Also handle JOINs
    if plan.joins:
        for join in plan.joins:
            pattern = rf'\bJOIN\s+{re.escape(join.table)}\b'
            sql = re.sub(pattern, f'JOIN {schema_name}.{join.table}', sql, flags=re.IGNORECASE)
    
    return sql


# ============================================================================
# Deprecation Warnings for Direct SQL Paths
# ============================================================================

def warn_direct_sql_generation(caller: str):
    """
    Log deprecation warning when a caller bypasses the canonical pipeline.
    
    Call this from sql_generator.generate_sql() and other direct paths
    to track usage and encourage migration.
    """
    import warnings
    msg = (
        f"[DEPRECATION] {caller} is generating SQL directly without QueryPlan. "
        "This bypasses validation and should be migrated to use query_plan_unifier."
    )
    logger.warning(msg)
    warnings.warn(msg, DeprecationWarning, stacklevel=3)


# ============================================================================
# Convenience: Auto-detect and Convert
# ============================================================================

def convert_to_canonical(
    plan: Any,
    source_hint: Optional[str] = None
) -> CanonicalQueryPlan:
    """
    Auto-detect QueryPlan type and convert to canonical.
    
    Args:
        plan: QueryPlan from any source
        source_hint: Optional hint about source ("generator", "plan_first")
    
    Returns:
        Canonical QueryPlan
    
    Raises:
        ValueError: If plan type is unknown
    """
    # Already canonical
    if isinstance(plan, CanonicalQueryPlan):
        return plan
    
    # Detect by attribute presence
    if hasattr(plan, 'select_expressions') and hasattr(plan, 'from_table'):
        # query_plan_generator.QueryPlan
        return convert_from_generator_plan(plan)
    
    if hasattr(plan, 'primary_table') and hasattr(plan, 'select_clauses'):
        # plan_first_sql_generator.QueryPlan
        return convert_from_plan_first(plan)
    
    # Check source hint
    if source_hint == "generator":
        return convert_from_generator_plan(plan)
    elif source_hint == "plan_first":
        return convert_from_plan_first(plan)
    
    raise ValueError(f"Unknown QueryPlan type: {type(plan).__name__}")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Canonical model re-exports
    "CanonicalQueryPlan",
    "SelectClause",
    "FromClause", 
    "JoinClause",
    "JoinCondition",
    "GroupByClause",
    "HavingClause",
    "OrderByClause",
    "OrderByField",
    "BinaryCondition",
    "LogicalCondition",
    "NotCondition",
    "Condition",
    "ColumnRef",
    "LiteralValue",
    "SubqueryValue",
    # Converters
    "convert_from_generator_plan",
    "convert_from_plan_first",
    "convert_to_canonical",
    # Unified generation
    "generate_sql_from_canonical_plan",
    # Deprecation helper
    "warn_direct_sql_generation",
]

