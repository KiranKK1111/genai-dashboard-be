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
        from_=FromClause(table="<primary_table>", alias="t"),
        joins=[JoinClause(
            type="inner",
            table="<related_table>",
            alias="r",
            on=[BinaryCondition(left="t.related_id", op="=", right="r.id")]
        )],
        where=[BinaryCondition(left="r.is_verified", op="=", right=Literal(value=True, type="bool"))],
        order_by=[OrderByField(expr="t.created_at", direction="desc")],
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


class ColumnSelectionIntent(str, Enum):
    """Semantic intent for column selection - determined by LLM analysis.
    
    This enum is part of the canonical query representation, used to capture
    user intent about what columns to return.
    """
    ALL_COLUMNS = "all_columns"  # User wants all columns (SELECT *)
    SPECIFIC_COLUMNS = "specific_columns"  # User mentioned specific columns
    COUNT_ONLY = "count_only"  # User only wants COUNT
    DISTINCT_VALUES = "distinct_values"  # User wants DISTINCT values of a column
    FIRST_N_COLUMNS = "first_n_columns"  # User wants top N columns (e.g., "first 3 columns")


@dataclass
class ColumnSelectionAnalysis:
    """LLM-determined analysis of column selection intent.
    
    This is part of the canonical query plan metadata, capturing semantic
    analysis of what columns the user wants.
    """
    intent: ColumnSelectionIntent
    requested_columns: List[str] = field(default_factory=list)  # If specific_columns, which ones
    reasoning: str = ""  # Why this intent was chosen
    user_mentions_columns: bool = False  # Did user explicitly name columns?
    user_mentions_all: bool = False  # Did user ask for "all"?
    confidence: float = 1.0  # Confidence in this analysis
    
    def to_dict(self) -> Dict[str, Any]:
        intent_val = self.intent.value if hasattr(self.intent, 'value') else str(self.intent)
        return {
            "intent": intent_val,
            "requested_columns": self.requested_columns,
            "reasoning": self.reasoning,
            "user_mentions_columns": self.user_mentions_columns,
            "user_mentions_all": self.user_mentions_all,
            "confidence": self.confidence,
        }


@dataclass
class Literal:
    """A literal value in the query."""
    value: Any
    type: ValueType
    
    def to_dict(self) -> Dict[str, Any]:
        type_val = self.type.value if hasattr(self.type, 'value') else str(self.type)
        return {"value": self.value, "type": type_val}


@dataclass
class ColumnRef:
    """Reference to a column (table.column or alias.column)."""
    column: str  # "column_name"
    table: Optional[str] = None  # "table_name" or "alias"
    
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
    fields: List[str]  # ["t.*", "a.entity_id", "SUM(t.amount) as total"]
    distinct: bool = False
    windows: Optional[List["WindowFunction"]] = None  # Window function expressions appended to field list

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "fields": self.fields,
            "distinct": self.distinct,
        }
        if self.windows:
            result["windows"] = [w.to_dict() for w in self.windows]
        return result


# ============================================================================
# FROM Clause
# ============================================================================

@dataclass
class FromClause:
    """FROM clause specification."""
    table: str  # "table_name"
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
    table: str  # "related_table"
    alias: Optional[str] = None  # "r"
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
# Window Functions
# ============================================================================

@dataclass
class WindowSpec:
    """OVER (...) specification for window functions."""
    partition_by: List[str] = field(default_factory=list)  # ["dept", "region"]
    order_by: List[OrderByField] = field(default_factory=list)  # [OrderByField("salary", "desc")]
    frame: Optional[str] = None  # "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "partition_by": self.partition_by,
            "order_by": [f.to_dict() for f in self.order_by],
            "frame": self.frame,
        }


@dataclass
class WindowFunction:
    """A window function expression added to the SELECT list.

    Examples::
        ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn
        SUM(amount) OVER (PARTITION BY region ORDER BY txn_date) AS running_total
    """
    func: str          # "ROW_NUMBER()", "RANK()", "SUM(amount)", etc.
    over: WindowSpec
    alias: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "func": self.func,
            "over": self.over.to_dict(),
            "alias": self.alias,
        }

    def to_sql_fragment(self) -> str:
        """Render as a SQL expression string (dialect-neutral)."""
        over_parts = []
        if self.over.partition_by:
            pb = ", ".join(self.over.partition_by)
            over_parts.append(f"PARTITION BY {pb}")
        if self.over.order_by:
            ob = ", ".join(f"{f.expr} {f.direction.upper()}" for f in self.over.order_by)
            over_parts.append(f"ORDER BY {ob}")
        if self.over.frame:
            over_parts.append(self.over.frame)
        over_clause = " ".join(over_parts)
        expr = f"{self.func} OVER ({over_clause})"
        if self.alias:
            expr += f" AS {self.alias}"
        return expr


# ============================================================================
# CTEs (Common Table Expressions)
# ============================================================================

@dataclass
class CTEClause:
    """A single CTE definition in a WITH clause.

    ``raw_sql`` holds the body SQL (e.g. ``SELECT … FROM …``).
    Set ``recursive=True`` to emit ``WITH RECURSIVE``.
    """
    name: str
    raw_sql: str
    recursive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "raw_sql": self.raw_sql, "recursive": self.recursive}


# ============================================================================
# Set Operations (UNION / INTERSECT / EXCEPT)
# ============================================================================

@dataclass
class SetOperation:
    """A set operation appended to the main SELECT.

    ``op`` is one of: ``union``, ``union_all``, ``intersect``, ``except``.
    ``raw_sql`` is the right-side SELECT statement.
    """
    op: str   # "union" | "union_all" | "intersect" | "except"
    raw_sql: str  # "SELECT … FROM …"

    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "raw_sql": self.raw_sql}


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
    
    LLM Interaction Fields:
    - clarification_needed: If True, LLM is asking for user clarification
    - clarification_question: The question to ask the user
    - clarification_options: Multiple choice options for the user
    - confidence: LLM's confidence score (0.0-1.0)
    
    These fields are IGNORED during SQL compilation but used for orchestration.
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

    # Advanced SQL constructs
    ctes: Optional[List[CTEClause]] = None          # WITH / WITH RECURSIVE clauses
    set_op: Optional[SetOperation] = None           # UNION / INTERSECT / EXCEPT

    # LLM Interaction Fields (ignored during SQL compilation)
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    clarification_options: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    # Metadata (NOT part of validation, for introspection)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    # =========================================================================
    # Compatibility Properties (for migration from GeneratedQueryPlan)
    # =========================================================================
    @property
    def from_table(self) -> str:
        """Convenience accessor: returns from_.table or empty string."""
        return self.from_.table if self.from_ else ""
    
    @property
    def from_schema(self) -> Optional[str]:
        """Convenience accessor: returns schema from metadata if set."""
        return self.metadata.get("from_schema") if self.metadata else None
    
    @property
    def where_conditions(self) -> List[Condition]:
        """Alias for `where` - for backward compatibility."""
        return self.where or []
    
    @property
    def select_expressions(self) -> List[str]:
        """Convenience accessor: returns select.fields or empty list."""
        return self.select.fields if self.select else []
    
    @property
    def select_aggregates(self) -> List[Any]:
        """Returns aggregates from metadata (if stored there)."""
        return self.metadata.get("select_aggregates", []) if self.metadata else []
    
    @property
    def column_selection(self) -> Optional[Any]:
        """Returns column_selection from metadata."""
        return self.metadata.get("column_selection") if self.metadata else None
    
    @column_selection.setter
    def column_selection(self, value: Any) -> None:
        """Sets column_selection in metadata."""
        if self.metadata is None:
            self.metadata = {}
        self.metadata["column_selection"] = value
    
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
        if self.ctes:
            result["ctes"] = [c.to_dict() for c in self.ctes]
        if self.set_op:
            result["set_op"] = self.set_op.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata

        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryPlan":
        """Deserialize a QueryPlan from the dictionary produced by ``to_dict()``.

        All clauses are fully reconstructed so that the round-trip
        ``QueryPlan.from_dict(plan.to_dict())`` produces an equivalent plan.
        """

        # ── helpers ──────────────────────────────────────────────────────────
        def _literal(d: Dict) -> Literal:
            return Literal(value=d["value"], type=d.get("type", "string"))

        def _col_ref(d: Dict) -> ColumnRef:
            return ColumnRef(column=d["column"], table=d.get("table"))

        def _value(d) -> Union["Literal", "ColumnRef", "SubqueryValue", str]:
            if isinstance(d, str):
                return d
            vtype = d.get("type")
            if vtype == "subquery":
                return SubqueryValue(query=cls.from_dict(d["query"]))
            if "column" in d:
                return _col_ref(d)
            return _literal(d)

        def _condition(d: Dict) -> Condition:
            ctype = d.get("type", "binary")
            if ctype == "logical":
                return LogicalCondition(
                    operator=d["operator"],
                    conditions=[_condition(c) for c in d.get("conditions", [])],
                )
            if ctype == "not":
                return NotCondition(condition=_condition(d["condition"]))
            # default: binary
            right_raw = d.get("right")
            right: Union[Value, str, list]
            if isinstance(right_raw, list):
                right = [_value(v) for v in right_raw]
            else:
                right = _value(right_raw) if right_raw is not None else ""
            return BinaryCondition(
                left=_value(d.get("left", "")) if isinstance(d.get("left"), dict) else d.get("left", ""),
                op=d.get("op", "="),
                right=right,
            )

        def _join_cond(d: Dict) -> JoinCondition:
            left = _col_ref(d["left"]) if isinstance(d["left"], dict) else d["left"]
            right = _col_ref(d["right"]) if isinstance(d["right"], dict) else d["right"]
            return JoinCondition(left=left, op=d.get("op", "="), right=right)

        # ── clauses ──────────────────────────────────────────────────────────
        select: Optional[SelectClause] = None
        if "select" in data and data["select"]:
            s = data["select"]
            windows = None
            if s.get("windows"):
                windows = []
                for w in s["windows"]:
                    over_data = w.get("over", {})
                    ob_fields = [OrderByField(expr=f["expr"], direction=f.get("direction", "asc")) for f in over_data.get("order_by", [])]
                    ws = WindowSpec(
                        partition_by=over_data.get("partition_by", []),
                        order_by=ob_fields,
                        frame=over_data.get("frame"),
                    )
                    windows.append(WindowFunction(func=w["func"], over=ws, alias=w.get("alias")))
            select = SelectClause(fields=s.get("fields", []), distinct=s.get("distinct", False), windows=windows)

        from_: Optional[FromClause] = None
        if "from" in data and data["from"]:
            f = data["from"]
            from_ = FromClause(table=f["table"], alias=f.get("alias"))

        joins: Optional[List[JoinClause]] = None
        if "joins" in data and data["joins"]:
            joins = []
            for j in data["joins"]:
                on = [_join_cond(c) for c in j["on"]] if j.get("on") else None
                joins.append(JoinClause(
                    type=j.get("type", "inner"),
                    table=j["table"],
                    alias=j.get("alias"),
                    on=on,
                ))

        where: Optional[List[Condition]] = None
        if "where" in data and data["where"]:
            where = [_condition(c) for c in data["where"]]

        group_by: Optional[GroupByClause] = None
        if "group_by" in data and data["group_by"]:
            group_by = GroupByClause(fields=data["group_by"].get("fields", []))

        having: Optional[HavingClause] = None
        if "having" in data and data["having"]:
            having = HavingClause(
                conditions=[_condition(c) for c in data["having"].get("conditions", [])]
            )

        order_by: Optional[OrderByClause] = None
        if "order_by" in data and data["order_by"]:
            order_by = OrderByClause(fields=[
                OrderByField(expr=f["expr"], direction=f.get("direction", "asc"))
                for f in data["order_by"].get("fields", [])
            ])

        ctes: Optional[List[CTEClause]] = None
        if data.get("ctes"):
            ctes = [CTEClause(name=c["name"], raw_sql=c["raw_sql"], recursive=c.get("recursive", False)) for c in data["ctes"]]

        set_op: Optional[SetOperation] = None
        if data.get("set_op"):
            sop = data["set_op"]
            set_op = SetOperation(op=sop.get("op", "union"), raw_sql=sop.get("raw_sql", ""))

        return cls(
            intent=data.get("intent", "data_query"),
            select=select,
            from_=from_,
            joins=joins,
            where=where,
            group_by=group_by,
            having=having,
            order_by=order_by,
            limit=data.get("limit"),
            offset=data.get("offset"),
            ctes=ctes,
            set_op=set_op,
            clarification_needed=data.get("clarification_needed", False),
            clarification_question=data.get("clarification_question"),
            clarification_options=data.get("clarification_options", []),
            confidence=float(data.get("confidence", 1.0)),
            metadata=data.get("metadata") or {},
        )


# ============================================================================
# QueryPlan Comparison & Analysis
# ============================================================================

@dataclass
class QueryArtifacts:
    """Metadata extracted from a validated QueryPlan."""
    tables_used: List[str]  # ["table1", "table2"]
    columns_used: List[str]  # ["t.event_id", "a.entity_id", ...]
    joins_used: List[Dict[str, Any]]  # [{from: ..., to: ..., type: ...}]
    where_conditions: List[str]  # ["a.is_verified = true", ...]
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
