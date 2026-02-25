"""
QueryPlan AST (Abstract Syntax Tree) - Universal query representation.

This module defines a dialect-neutral intermediate representation (IR) for SQL queries.
A QueryPlan represents a fully-structured, validated query that can be:
1. Validated against schema metadata (no hallucinations)
2. Compiled to any SQL dialect (PostgreSQL, MySQL, SQLite, SQL Server, Oracle)
3. Introspected for follow-up queries (persisted structure)
4. Executed safely (SELECT-only, checked before execution)

The design is fully composable - every clause (WHERE, JOIN, GROUP BY, etc.) is optional,
so a single AST handles all SQL query types automatically.

Example Usage:
    plan = QueryPlan(
        intent="data_query",
        select=SelectClause(fields=["t.*"]),
        from_=FromClause(table="transactions", alias="t"),
        joins=[JoinClause(
            type="inner",
            table="customers",
            alias="c",
            on=[BinaryCondition(left="t.customer_id", op="=", right="c.customer_id")]
        )],
        where=[BinaryCondition(left="c.kyc_verified", op="=", right=Literal(value=True, type="bool"))],
        order_by=[OrderByField(expr="t.txn_time", direction="desc")],
        limit=10
    )
    
    validator = QueryPlanValidator(schema_catalog)
    validated_plan = await validator.validate(plan)
    
    compiler = DialectCompiler(dialect="postgresql")
    sql = compiler.compile(validated_plan)
    
    result = await run_sql(session, sql)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union
from enum import Enum


# ============================================================================
# Value Representations
# ============================================================================

class ValueType(str, Enum):
    """SQL value types."""
    STRING = "string"
    NUMBER = "number"
    BOOL = "bool"
    DATE = "date"
    DATETIME = "datetime"
    NULL = "null"


@dataclass
class Literal:
    """A literal value in the query."""
    value: Any
    type: ValueType
    
    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "type": self.type.value}


@dataclass
class ColumnRef:
    """Reference to a column (table.column or alias.column)."""
    column: str  # "customer_id"
    table: Optional[str] = None  # "customers" or "c"
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"column": self.column}
        if self.table:
            result["table"] = self.table
        return result


@dataclass
class SubqueryValue:
    """A subquery used as a value in WHERE/HAVING."""
    query: QueryPlan
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "subquery", "query": self.query.to_dict()}


# Value can be: Literal, ColumnRef, or SubqueryValue
Value = Union[Literal, ColumnRef, SubqueryValue]


# ============================================================================
# Conditions (WHERE, HAVING, ON clauses)
# ============================================================================

@dataclass
class BinaryCondition:
    """A binary condition: left OP right (e.g., "t.amount > 1000")."""
    left: Union[ColumnRef, str]  # str is shorthand for ColumnRef(column=str)
    op: Literal["=", "!=", "<>", ">", "<", ">=", "<=", "in", "not in", "like", "is", "is not"]
    right: Union[Value, str]  # Can be Literal, ColumnRef, SubqueryValue, or string shorthand
    
    def to_dict(self) -> Dict[str, Any]:
        left_val = self.left.to_dict() if isinstance(self.left, (ColumnRef, Literal, SubqueryValue)) else self.left
        right_val = self.right.to_dict() if isinstance(self.right, (ColumnRef, Literal, SubqueryValue)) else self.right
        return {
            "type": "binary",
            "left": left_val,
            "op": self.op,
            "right": right_val
        }


@dataclass
class LogicalCondition:
    """Logical combination of conditions: AND/OR."""
    operator: Literal["and", "or"]
    conditions: List[Condition]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "logical",
            "operator": self.operator,
            "conditions": [c.to_dict() for c in self.conditions]
        }


@dataclass
class NotCondition:
    """Negation of a condition: NOT condition."""
    condition: Condition
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "not",
            "condition": self.condition.to_dict()
        }


# Condition can be any of these types
Condition = Union[BinaryCondition, LogicalCondition, NotCondition]


# ============================================================================
# SELECT Clause
# ============================================================================

@dataclass
class SelectClause:
    """SELECT clause specification."""
    fields: List[str]  # ["t.*", "c.customer_id", "SUM(t.amount) as total"]
    distinct: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields": self.fields,
            "distinct": self.distinct
        }


# ============================================================================
# FROM Clause
# ============================================================================

@dataclass
class FromClause:
    """FROM clause specification."""
    table: str  # "transactions"
    alias: Optional[str] = None  # "t"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "alias": self.alias
        }


# ============================================================================
# JOIN Clauses
# ============================================================================

@dataclass
class JoinCondition:
    """A single join condition (ON clause part)."""
    left: Union[ColumnRef, str]
    op: str  # "=", ">", etc.
    right: Union[ColumnRef, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "left": self.left.to_dict() if isinstance(self.left, ColumnRef) else self.left,
            "op": self.op,
            "right": self.right.to_dict() if isinstance(self.right, ColumnRef) else self.right
        }


@dataclass
class JoinClause:
    """A single JOIN specification."""
    type: Literal["inner", "left", "right", "full", "cross"]
    table: str  # "customers"
    alias: Optional[str] = None  # "c"
    on: Optional[List[JoinCondition]] = None  # CROSS JOINs don't have ON
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": self.type,
            "table": self.table
        }
        if self.alias:
            result["alias"] = self.alias
        if self.on:
            result["on"] = [cond.to_dict() for cond in self.on]
        return result


# ============================================================================
# GROUP BY, HAVING, ORDER BY, LIMIT
# ============================================================================

@dataclass
class GroupByClause:
    """GROUP BY clause."""
    fields: List[str]  # ["c.city", "t.txn_date"]
    
    def to_dict(self) -> Dict[str, Any]:
        return {"fields": self.fields}


@dataclass
class HavingClause:
    """HAVING clause (conditions on grouped results)."""
    conditions: List[Condition]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conditions": [c.to_dict() for c in self.conditions]
        }


@dataclass
class OrderByField:
    """A single field in ORDER BY."""
    expr: str  # "t.txn_time" or "SUM(t.amount)"
    direction: Literal["asc", "desc"] = "asc"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "expr": self.expr,
            "direction": self.direction
        }


@dataclass
class OrderByClause:
    """ORDER BY clause."""
    fields: List[OrderByField]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields": [f.to_dict() for f in self.fields]
        }


# ============================================================================
# Main QueryPlan AST
# ============================================================================

@dataclass
class QueryPlan:
    """
    Universal, composable query plan (AST) for SQL queries.
    
    Every clause is optional, making this suitable for:
    - Simple: SELECT ... FROM ...
    - With WHERE: ... WHERE ...
    - With JOINs: ... JOIN ...
    - With GROUP BY: ... GROUP BY ...
    - With subqueries: ... WHERE col IN (SELECT ...)
    - Complex: All of the above combined
    """
    
    # Core clauses (always present for data_query)
    intent: Literal["data_query", "file_query", "chat"]
    select: Optional[SelectClause] = None
    from_: Optional[FromClause] = None
    
    # Optional clauses (composable)
    joins: Optional[List[JoinClause]] = None
    where: Optional[List[Condition]] = None
    group_by: Optional[GroupByClause] = None
    having: Optional[HavingClause] = None
    order_by: Optional[OrderByClause] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    
    # Metadata (NOT part of validation, for introspection)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        result = {
            "intent": self.intent,
        }
        
        if self.select:
            result["select"] = self.select.to_dict()
        if self.from_:
            result["from"] = self.from_.to_dict()
        if self.joins:
            result["joins"] = [j.to_dict() for j in self.joins]
        if self.where:
            result["where"] = [c.to_dict() for c in self.where]
        if self.group_by:
            result["group_by"] = self.group_by.to_dict()
        if self.having:
            result["having"] = self.having.to_dict()
        if self.order_by:
            result["order_by"] = self.order_by.to_dict()
        if self.limit is not None:
            result["limit"] = self.limit
        if self.offset is not None:
            result["offset"] = self.offset
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QueryPlan:
        """Deserialize from dictionary."""
        # This is a simplified version - a production version would handle
        # all the nested object reconstruction
        return cls(
            intent=data.get("intent", "data_query"),
            metadata=data.get("metadata", {})
        )


# ============================================================================
# QueryPlan Comparison & Analysis
# ============================================================================

@dataclass
class QueryArtifacts:
    """Metadata extracted from a validated QueryPlan."""
    tables_used: List[str]  # ["transactions", "customers"]
    columns_used: List[str]  # ["t.txn_id", "c.customer_id", ...]
    joins_used: List[Dict[str, Any]]  # [{from: ..., to: ..., type: ...}]
    where_conditions: List[str]  # ["c.kyc_verified = true", ...]
    group_by_fields: List[str]
    having_conditions: List[str]
    order_by_fields: List[str]
    limit_value: Optional[int]
    has_subqueries: bool
    is_aggregated: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tables_used": self.tables_used,
            "columns_used": self.columns_used,
            "joins_used": self.joins_used,
            "where_conditions": self.where_conditions,
            "group_by_fields": self.group_by_fields,
            "having_conditions": self.having_conditions,
            "order_by_fields": self.order_by_fields,
            "limit_value": self.limit_value,
            "has_subqueries": self.has_subqueries,
            "is_aggregated": self.is_aggregated,
        }


# ============================================================================
# Validation Errors
# ============================================================================

class QueryPlanValidationError(Exception):
    """Raised when QueryPlan validation fails."""
    def __init__(self, message: str, error_type: str = "validation_error"):
        self.message = message
        self.error_type = error_type
        super().__init__(message)


class ColumnNotFoundError(QueryPlanValidationError):
    """Column referenced in plan doesn't exist in schema."""
    def __init__(self, column: str, table: str = None):
        msg = f"Column '{column}' not found"
        if table:
            msg += f" in table '{table}'"
        super().__init__(msg, "column_not_found")


class TableNotFoundError(QueryPlanValidationError):
    """Table referenced in plan doesn't exist in schema."""
    def __init__(self, table: str):
        super().__init__(f"Table '{table}' not found", "table_not_found")


class JoinPathNotFoundError(QueryPlanValidationError):
    """No valid join path exists between tables."""
    def __init__(self, table1: str, table2: str):
        super().__init__(
            f"No join path found between '{table1}' and '{table2}'",
            "join_path_not_found"
        )


class TypeMismatchError(QueryPlanValidationError):
    """Type mismatch in condition."""
    def __init__(self, column: str, expected_type: str, got_type: str):
        super().__init__(
            f"Type mismatch for '{column}': expected {expected_type}, got {got_type}",
            "type_mismatch"
        )


class InvalidSubqueryError(QueryPlanValidationError):
    """Subquery validation failed."""
    def __init__(self, reason: str):
        super().__init__(f"Invalid subquery: {reason}", "invalid_subquery")
