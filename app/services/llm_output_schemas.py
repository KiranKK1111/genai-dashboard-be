"""
LLM Output Validation Schemas.

Pydantic models that validate raw LLM JSON before it enters
`_dict_to_query_plan()` in query_plan_generator.py.

Why this matters:
  - LLMs sometimes omit required keys, return null for list fields, or use
    wrong types (e.g. string "5" for an integer limit).
  - Without validation, these produce confusing KeyError / AttributeError
    crashes deep inside the compiler pipeline.
  - With validation, every field gets a safe default so partial LLM outputs
    always produce a workable (if degraded) plan.

Usage:
    raw = json.loads(llm_response)
    validated = LLMQueryPlanOutput.model_validate_lenient(raw)
    plan = _dict_to_query_plan(validated.to_dict())
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_VALID_JOIN_TYPES = {"INNER", "LEFT", "RIGHT", "FULL", "CROSS"}
_VALID_OPERATORS = {
    "=", "!=", "<>", "<", ">", "<=", ">=",
    "IN", "NOT IN", "LIKE", "ILIKE",
    "IS NULL", "IS NOT NULL", "BETWEEN", "NOT BETWEEN",
}
_VALID_DIRECTIONS = {"ASC", "DESC"}
_VALID_AGGS = {"COUNT", "SUM", "AVG", "MIN", "MAX", "STDDEV", "COUNT_DISTINCT"}


# ── Sub-models ────────────────────────────────────────────────────────────────

class LLMJoinConditionOutput(BaseModel):
    left_col: str = ""
    right_col: str = ""
    operator: str = "="

    @field_validator("operator")
    @classmethod
    def _operator(cls, v: str) -> str:
        return v.upper() if v and v.upper() in _VALID_OPERATORS else "="


class LLMJoinOutput(BaseModel):
    join_type: str = "INNER"
    right_table: str = ""
    right_schema: Optional[str] = None
    right_alias: Optional[str] = None
    conditions: List[LLMJoinConditionOutput] = Field(default_factory=list)

    @field_validator("join_type")
    @classmethod
    def _join_type(cls, v: str) -> str:
        upper = (v or "").upper()
        return upper if upper in _VALID_JOIN_TYPES else "INNER"

    @field_validator("conditions", mode="before")
    @classmethod
    def _conditions(cls, v: Any) -> List:
        if not isinstance(v, list):
            return []
        return v


class LLMWhereConditionOutput(BaseModel):
    left: str = ""
    operator: str = "="
    right: Any = None
    right_kind: str = "literal"   # literal | column | subquery | list

    @field_validator("operator")
    @classmethod
    def _operator(cls, v: str) -> str:
        upper = (v or "").upper()
        return upper if upper in _VALID_OPERATORS else "="


class LLMAggregateOutput(BaseModel):
    function: str = "COUNT"
    column: str = "*"
    alias: Optional[str] = None
    distinct: bool = False

    @field_validator("function")
    @classmethod
    def _function(cls, v: str) -> str:
        upper = (v or "").upper()
        return upper if upper in _VALID_AGGS else "COUNT"


class LLMOrderByOutput(BaseModel):
    column: str = ""
    direction: str = "ASC"

    @field_validator("direction")
    @classmethod
    def _direction(cls, v: str) -> str:
        upper = (v or "").upper()
        return upper if upper in _VALID_DIRECTIONS else "ASC"


class LLMHavingConditionOutput(BaseModel):
    left: str = ""
    operator: str = ">"
    right: Any = None


class LLMCTEOutput(BaseModel):
    name: str = "cte"
    sql: str = ""
    recursive: bool = False


class LLMSetOpOutput(BaseModel):
    type: str = "UNION"          # UNION | UNION ALL | INTERSECT | EXCEPT
    right_sql: str = ""


# ── Top-level model ───────────────────────────────────────────────────────────

class LLMQueryPlanOutput(BaseModel):
    """
    Validates raw LLM JSON output before conversion to GeneratedQueryPlan.

    All fields have safe defaults so partial LLM outputs never crash.
    """
    # FROM
    from_table: str = ""
    from_schema: Optional[str] = None
    from_alias: Optional[str] = None

    # SELECT
    select_expressions: List[str] = Field(default_factory=lambda: ["*"])
    select_aggregates: List[LLMAggregateOutput] = Field(default_factory=list)
    distinct: bool = False
    window_expressions: List[str] = Field(default_factory=list)

    # JOINs
    joins: List[LLMJoinOutput] = Field(default_factory=list)

    # WHERE
    where_conditions: List[LLMWhereConditionOutput] = Field(default_factory=list)
    where_operator: str = "AND"

    # GROUP BY / HAVING
    group_by: List[str] = Field(default_factory=list)
    having_conditions: List[LLMHavingConditionOutput] = Field(default_factory=list)

    # ORDER BY / LIMIT
    order_by: List[LLMOrderByOutput] = Field(default_factory=list)
    limit: Optional[int] = None
    offset: Optional[int] = None

    # Advanced SQL
    ctes: List[LLMCTEOutput] = Field(default_factory=list)
    set_op: Optional[LLMSetOpOutput] = None

    # Meta
    intent: str = "data_query"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    clarification_options: List[str] = Field(default_factory=list)

    # ── Field-level coercions ─────────────────────────────────────────────────

    @field_validator("from_table", mode="before")
    @classmethod
    def _from_table(cls, v: Any) -> str:
        return str(v).strip() if v else ""

    @field_validator("select_expressions", mode="before")
    @classmethod
    def _select_expr(cls, v: Any) -> List[str]:
        if not v:
            return ["*"]
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(x) for x in v if x] or ["*"]
        return ["*"]

    @field_validator("select_aggregates", mode="before")
    @classmethod
    def _select_aggs(cls, v: Any) -> List:
        return v if isinstance(v, list) else []

    @field_validator("joins", mode="before")
    @classmethod
    def _joins(cls, v: Any) -> List:
        return v if isinstance(v, list) else []

    @field_validator("where_conditions", mode="before")
    @classmethod
    def _where(cls, v: Any) -> List:
        return v if isinstance(v, list) else []

    @field_validator("group_by", mode="before")
    @classmethod
    def _group_by(cls, v: Any) -> List[str]:
        if not v:
            return []
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v if x]

    @field_validator("order_by", mode="before")
    @classmethod
    def _order_by(cls, v: Any) -> List:
        if not v:
            return []
        # Accept both {"column": ..., "direction": ...} and plain strings
        result = []
        for item in v if isinstance(v, list) else []:
            if isinstance(item, str):
                result.append({"column": item, "direction": "ASC"})
            elif isinstance(item, dict):
                result.append(item)
        return result

    @field_validator("limit", mode="before")
    @classmethod
    def _limit(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            val = int(v)
            return val if val > 0 else None
        except (TypeError, ValueError):
            return None

    @field_validator("offset", mode="before")
    @classmethod
    def _offset(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            val = int(v)
            return val if val >= 0 else None
        except (TypeError, ValueError):
            return None

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, v: Any) -> float:
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (TypeError, ValueError):
            return 0.5

    @field_validator("ctes", mode="before")
    @classmethod
    def _ctes(cls, v: Any) -> List:
        return v if isinstance(v, list) else []

    @field_validator("window_expressions", mode="before")
    @classmethod
    def _window_expr(cls, v: Any) -> List[str]:
        if not v:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x]
        return []

    # ── Model-level guard ─────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _warn_empty_from(self) -> "LLMQueryPlanOutput":
        if not self.from_table and not self.ctes:
            logger.warning(
                "[LLM OUTPUT VALIDATION] from_table is empty and no CTEs defined — "
                "plan may be incomplete"
            )
        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    @classmethod
    def model_validate_lenient(cls, raw: Any) -> "LLMQueryPlanOutput":
        """
        Validate raw LLM dict with full error logging but no crash.
        Returns a best-effort plan even if validation partially fails.
        """
        if not isinstance(raw, dict):
            logger.error(
                "[LLM OUTPUT VALIDATION] Expected dict, got %s — using empty plan",
                type(raw).__name__,
            )
            return cls()
        try:
            return cls.model_validate(raw)
        except Exception as exc:
            logger.warning(
                "[LLM OUTPUT VALIDATION] Validation error (partial plan used): %s", exc
            )
            # Strip unknown fields and retry with only recognised keys
            known_keys = set(cls.model_fields.keys())
            clean = {k: v for k, v in raw.items() if k in known_keys}
            try:
                return cls.model_validate(clean)
            except Exception:
                return cls()

    def to_raw_dict(self) -> Dict[str, Any]:
        """Dump back to the dict format _dict_to_query_plan() expects."""
        return self.model_dump(exclude_none=False)
