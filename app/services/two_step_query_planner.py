"""
PRINCIPLE 4: Two-Step Generation (Plan → SQL)
==============================================
Instead of directly generating SQL, make LLM produce a structured PLAN first,
then convert plan to SQL deterministically.

Benefits:
1. LLM focuses on logic (what to select, filter) not SQL syntax
2. Plan is easier to validate (no SQL parsing)
3. Deterministic template → SQL (no hallucinations in final query)
4. Easier to regenerate if validation fails

Plan Schema:
{
    "select": ["table"],
    "columns": ["col", ...],
    "joins": [{"table": "...", "on": "..."}],
    "filters": [{"column": "...", "operator": "=", "value": "..."}],
    "having": [...],
    "order_by": [{"column": "...", "direction": "ASC"}],
    "limit": 100,
    "distinct": false
}
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class FilterOperator(str, Enum):
    """Valid SQL filter operators."""
    EQ = "="
    NEQ = "!="
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    IN = "IN"
    NOT_IN = "NOT IN"
    LIKE = "LIKE"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    BETWEEN = "BETWEEN"


@dataclass
class FilterStep:
    """A single filter predicate."""
    table: str
    column: str
    operator: FilterOperator
    value: str
    logical_op: str = "AND"  # AND / OR

    def to_sql(self) -> str:
        """Convert to SQL WHERE clause fragment."""
        col_ref = f"{self.table}.{self.column}"
        
        if self.operator == FilterOperator.IS_NULL:
            return f"{col_ref} IS NULL"
        elif self.operator == FilterOperator.IS_NOT_NULL:
            return f"{col_ref} IS NOT NULL"
        elif self.operator == FilterOperator.IN:
            # value should be comma-separated list
            return f"{col_ref} IN ({self.value})"
        elif self.operator == FilterOperator.BETWEEN:
            # value should be "start AND end"
            return f"{col_ref} BETWEEN {self.value}"
        else:
            # Simple operators
            return f"{col_ref} {self.operator.value} '{self.value}'"


@dataclass
class JoinStep:
    """A single join."""
    table: str
    alias: Optional[str] = None
    on_clause: str = ""
    join_type: str = "LEFT"  # LEFT, INNER, RIGHT, FULL

    def to_sql(self) -> str:
        """Convert to SQL JOIN clause."""
        table_ref = f"{self.table}" + (f" {self.alias}" if self.alias else "")
        return f"{self.join_type} JOIN {table_ref} ON {self.on_clause}"


@dataclass
class OrderByStep:
    """Order by specification."""
    table: str
    column: str
    direction: str = "ASC"  # ASC / DESC

    def to_sql(self) -> str:
        return f"{self.table}.{self.column} {self.direction}"


@dataclass
class QueryPlan:
    """Structured query plan (not SQL, more abstract)."""
    
    select_table: str
    select_columns: List[str]  # ["*"] or specific columns
    joins: List[JoinStep]
    filters: List[FilterStep]
    distinct: bool = False
    order_by: List[OrderByStep] = None
    limit: Optional[int] = None
    having: Optional[str] = None

    def __post_init__(self):
        if self.order_by is None:
            self.order_by = []

    def to_sql(self) -> str:
        """
        Convert plan to concrete SQL.
        This is deterministic; no hallucination risk.
        
        Returns:
            SQL query string
        """
        parts = []

        # SELECT clause
        distinct_kw = "DISTINCT " if self.distinct else ""
        cols_str = ", ".join(self.select_columns) if self.select_columns else "*"
        parts.append(f"SELECT {distinct_kw}{cols_str}")

        # FROM clause
        parts.append(f"FROM {self.select_table}")

        # JOIN clauses
        for join in self.joins:
            parts.append(join.to_sql())

        # WHERE clause
        if self.filters:
            where_parts = []
            for i, filt in enumerate(self.filters):
                where_parts.append(filt.to_sql())
            where_clause = "\n  ".join([" AND ".join(where_parts)])
            parts.append(f"WHERE {where_clause}")

        # HAVING clause
        if self.having:
            parts.append(f"HAVING {self.having}")

        # ORDER BY
        if self.order_by:
            order_str = ", ".join([ob.to_sql() for ob in self.order_by])
            parts.append(f"ORDER BY {order_str}")

        # LIMIT
        if self.limit:
            parts.append(f"LIMIT {self.limit}")

        sql = "\n".join(parts)
        logger.debug(f"[PLAN_TO_SQL] Generated:\n{sql}")
        return sql

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON for LLM."""
        return {
            "select_table": self.select_table,
            "select_columns": self.select_columns,
            "joins": [
                {
                    "table": j.table,
                    "alias": j.alias,
                    "on_clause": j.on_clause,
                    "join_type": j.join_type
                }
                for j in self.joins
            ],
            "filters": [
                {
                    "table": f.table,
                    "column": f.column,
                    "operator": f.operator.value,
                    "value": f.value,
                    "logical_op": f.logical_op
                }
                for f in self.filters
            ],
            "distinct": self.distinct,
            "order_by": [
                {"table": o.table, "column": o.column, "direction": o.direction}
                for o in self.order_by
            ],
            "limit": self.limit,
            "having": self.having
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "QueryPlan":
        """Deserialize from JSON (from LLM response)."""
        joins = [
            JoinStep(
                table=j["table"],
                alias=j.get("alias"),
                on_clause=j.get("on_clause", ""),
                join_type=j.get("join_type", "LEFT")
            )
            for j in data.get("joins", [])
        ]

        filters = [
            FilterStep(
                table=f["table"],
                column=f["column"],
                operator=FilterOperator(f["operator"]),
                value=f["value"],
                logical_op=f.get("logical_op", "AND")
            )
            for f in data.get("filters", [])
        ]

        order_by = [
            OrderByStep(
                table=o["table"] if "table" in o else "",
                column=o["column"],
                direction=o.get("direction", "ASC")
            )
            for o in data.get("order_by", [])
        ]

        return cls(
            select_table=data["select_table"],
            select_columns=data.get("select_columns", ["*"]),
            joins=joins,
            filters=filters,
            distinct=data.get("distinct", False),
            order_by=order_by,
            limit=data.get("limit"),
            having=data.get("having")
        )


class QueryPlanGenerator:
    """Generates two-step queries: LLM produces plan, then we render to SQL."""

    def __init__(self, schema_grounding, table_selector):
        """
        Args:
            schema_grounding: For schema lookup
            table_selector: For join recommendations
        """
        self.schema = schema_grounding
        self.selector = table_selector

    def generate_plan_prompt(
        self,
        user_query: str,
        table_selection: Dict
    ) -> str:
        """
        Generate prompt that asks LLM to produce a QueryPlan (JSON).
        
        Args:
            user_query: User's natural language question
            table_selection: From relationship_aware_table_selector
            
        Returns:
            Prompt text for LLM
        """
        prompt = f"""
You are a SQL query planner. Convert the user's question into a structured query plan (JSON).

USER QUESTION: {user_query}

KNOWN CONSTRAINTS:
- Primary table: {table_selection['primary_table']}
- Available joins: {table_selection['recommended_joins']}

OUTPUT ONLY valid JSON (no markdown, no code block):
{{
    "select_table": "customers",
    "select_columns": ["customer_id", "customer_name", "email"],
    "joins": [
        {{
            "table": "cards",
            "alias": null,
            "on_clause": "customers.customer_id = cards.customer_id",
            "join_type": "LEFT"
        }}
    ],
    "filters": [
        {{
            "table": "cards",
            "column": "card_type",
            "operator": "=",
            "value": "CREDIT",
            "logical_op": "AND"
        }}
    ],
    "distinct": false,
    "order_by": [],
    "limit": 100
}}

REQUIREMENTS:
1. Only use operators: =, !=, >, <, >=, <=, IN, NOT IN, LIKE, IS NULL, IS NOT NULL, BETWEEN
2. For IN operator, format value as: "'value1','value2','value3'"
3. Never invent tables or columns
4. Use join recommendations provided above
5. Return ONLY the JSON object
"""
        return prompt

    def render_plan_to_sql(self, plan: QueryPlan) -> str:
        """
        Deterministically render plan to SQL.
        
        This is guaranteed not to hallucinate because it's just template filling.
        """
        return plan.to_sql()

    def validate_plan_before_rendering(self, plan: QueryPlan) -> Tuple[bool, List[str]]:
        """
        Validate plan structure before rendering to SQL.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check select_table exists
        if plan.select_table not in self.schema.tables:
            errors.append(f"Unknown table: {plan.select_table}")

        # Check select_columns exist
        if plan.select_columns and "*" not in plan.select_columns:
            table_cols = self.schema.tables[plan.select_table]["columns"].keys()
            for col in plan.select_columns:
                if col not in table_cols:
                    errors.append(f"Unknown column: {plan.select_table}.{col}")

        # Check join tables and columns
        for join in plan.joins:
            if join.table not in self.schema.tables:
                errors.append(f"Unknown join table: {join.table}")

        # Check filter columns exist
        for filt in plan.filters:
            if filt.table not in self.schema.tables:
                errors.append(f"Unknown filter table: {filt.table}")
            elif filt.column not in self.schema.tables[filt.table]["columns"]:
                errors.append(f"Unknown filter column: {filt.table}.{filt.column}")

        return len(errors) == 0, errors
