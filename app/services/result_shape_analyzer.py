"""
Result Shape Analyzer — Deterministic SQL result interpreter.

Classifies SQL results into semantic answer types and infers column roles
with zero LLM calls. Pure data inspection.

Architecture: ResponseGeneration.md § 8-9
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column role inference
# ---------------------------------------------------------------------------

_TIME_KEYWORDS = frozenset([
    "date", "time", "year", "month", "quarter", "week", "day",
    "period", "timestamp", "created_at", "updated_at", "dt",
    "created", "updated", "modified", "at",
])
_METRIC_KEYWORDS = frozenset([
    "count", "total", "sum", "avg", "average", "max", "min",
    "qty", "quantity", "score", "rank", "num", "number", "amount",
])
_CURRENCY_KEYWORDS = frozenset([
    "amount", "revenue", "price", "salary", "cost", "balance",
    "fee", "payment", "income", "spend", "budget",
])
_PERCENTAGE_KEYWORDS = frozenset([
    "rate", "pct", "percent", "ratio", "share", "proportion",
])
_IDENTIFIER_SUFFIXES = ("_id", "_key", "_no", "_num", "_code", "_ref", "_uuid", "id")
_CATEGORY_KEYWORDS = frozenset([
    "type", "category", "status", "state", "class", "group",
    "tier", "level", "segment", "label", "tag", "kind",
])
_TEXT_KEYWORDS = frozenset([
    "note", "comment", "message", "remark", "description",
    "summary", "detail", "text", "body", "content", "remarks",
])
_DIMENSION_KEYWORDS = frozenset([
    "name", "title", "city", "country", "region", "branch",
    "department", "product", "customer", "district", "location",
    "state", "zone",
])

_ISO_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}|^\d{2}/\d{2}/\d{4}|^\d{4}/\d{2}/\d{2}"
)


def infer_column_role(col_name: str, sample_value: Any = None) -> tuple[str, str, Optional[str]]:
    """
    Infer (datatype, semantic_role, format_hint) for a column.

    Returns:
        datatype:     "number" | "string" | "date" | "boolean"
        semantic_role: "metric"|"dimension"|"time"|"category"|"identifier"|"text"|"currency"|"percentage"
        format_hint:  "integer"|"decimal"|"currency"|"percentage"|"date"|"datetime"| None
    """
    name = col_name.lower().replace(" ", "_")
    tokens = set(re.split(r"[_\s\-]+", name))

    # Determine datatype from sample
    is_numeric = isinstance(sample_value, (int, float)) and not isinstance(sample_value, bool)
    is_bool = isinstance(sample_value, bool)
    is_str = isinstance(sample_value, str)
    is_date_str = is_str and bool(_ISO_DATE_RE.match(sample_value))

    if is_bool:
        return "boolean", "category", None
    if is_date_str:
        return "date", "time", "date"
    if is_numeric:
        datatype = "number"
    else:
        datatype = "date" if is_date_str else "string"

    # --- Role inference (priority order) ---

    # 1. Time
    if tokens & _TIME_KEYWORDS or any(t in name for t in ("date", "time", "month", "year", "quarter")):
        return "date", "time", "datetime" if "time" in name else "date"

    # 2. Identifier (name ends with id-style suffix, usually numeric/short string)
    if any(name.endswith(s) for s in _IDENTIFIER_SUFFIXES):
        if not (tokens & _METRIC_KEYWORDS):  # avoid "count_id" being an identifier
            return datatype, "identifier", None

    # 3. Currency (numeric + currency keyword)
    if is_numeric and tokens & _CURRENCY_KEYWORDS:
        return "number", "currency", "currency"

    # 4. Percentage
    if is_numeric and tokens & _PERCENTAGE_KEYWORDS:
        return "number", "percentage", "percentage"

    # 5. Metric
    if tokens & _METRIC_KEYWORDS or is_numeric:
        fmt = "integer" if isinstance(sample_value, int) else ("decimal" if is_numeric else None)
        return "number", "metric", fmt

    # 6. Category (short-ish string, limited vocabulary)
    if tokens & _CATEGORY_KEYWORDS:
        return "string", "category", None

    # 7. Text (long description)
    if tokens & _TEXT_KEYWORDS:
        return "string", "text", None

    # 8. Dimension (everything else that's a string)
    if tokens & _DIMENSION_KEYWORDS:
        return "string", "dimension", None

    # Fallback: if value is a short string → dimension, long string → text
    if is_str:
        if len(str(sample_value)) > 80:
            return "string", "text", None
        return "string", "dimension", None

    return datatype, "dimension", None


# ---------------------------------------------------------------------------
# Result shape dataclass
# ---------------------------------------------------------------------------

@dataclass
class ResultShape:
    """Output of ResultShapeAnalyzer.analyze()."""
    answer_type: str
    column_metas: list          # List[ColumnMeta dicts] — lazy import to avoid circular
    chart_recommended: bool
    chart_type_hint: Optional[str]   # "line" | "bar" | "bar_horizontal" | None
    has_single_metric: bool
    time_column: Optional[str]
    category_column: Optional[str]
    metric_columns: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

_AGG_FUNCS = re.compile(r"\b(COUNT|SUM|AVG|MAX|MIN)\s*\(", re.IGNORECASE)
_GROUP_BY = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
_ORDER_BY_DESC = re.compile(r"\bORDER\s+BY\b.+?\bDESC\b", re.IGNORECASE | re.DOTALL)
_LIMIT = re.compile(r"\bLIMIT\b", re.IGNORECASE)


class ResultShapeAnalyzer:
    """
    Deterministic, zero-LLM SQL result shape analyzer.

    Usage:
        shape = ResultShapeAnalyzer.analyze(sql=sql, rows=rows)
    """

    @staticmethod
    def analyze(sql: str, rows: List[Dict[str, Any]]) -> ResultShape:
        """Analyze SQL result and return a ResultShape."""
        from app.schemas import ColumnMeta  # lazy to avoid circular at module load

        if not rows:
            return ResultShape(
                answer_type="tabular_result",
                column_metas=[],
                chart_recommended=False,
                chart_type_hint=None,
                has_single_metric=False,
                time_column=None,
                category_column=None,
                metric_columns=[],
            )

        row_count = len(rows)
        col_names = list(rows[0].keys())
        first_row = rows[0]

        # Build ColumnMeta for each column
        column_metas = []
        time_col: Optional[str] = None
        category_col: Optional[str] = None
        metric_cols: List[str] = []
        identifier_cols: List[str] = []

        for col in col_names:
            sample = first_row.get(col)
            datatype, role, fmt = infer_column_role(col, sample)
            column_metas.append(ColumnMeta(
                name=col,
                label=col.replace("_", " ").title(),
                datatype=datatype,
                semantic_role=role,
                format_hint=fmt,
            ))
            if role == "time" and time_col is None:
                time_col = col
            elif role in ("category", "dimension") and category_col is None:
                category_col = col
            elif role in ("metric", "currency", "percentage"):
                metric_cols.append(col)
            elif role == "identifier":
                identifier_cols.append(col)

        has_agg = bool(_AGG_FUNCS.search(sql))
        has_group = bool(_GROUP_BY.search(sql))
        has_order_desc = bool(_ORDER_BY_DESC.search(sql))
        has_limit = bool(_LIMIT.search(sql))

        # Non-identifier, non-text cols determine "real" columns
        data_cols = [c for c in col_names if c not in identifier_cols]
        has_single_metric = (
            row_count == 1
            and len(metric_cols) >= 1
            and len(data_cols) <= 3
        )

        answer_type = ResultShapeAnalyzer._infer_answer_type(
            row_count=row_count,
            metric_cols=metric_cols,
            time_col=time_col,
            category_col=category_col,
            identifier_cols=identifier_cols,
            data_cols=data_cols,
            has_agg=has_agg,
            has_group=has_group,
            has_order_desc=has_order_desc,
            has_limit=has_limit,
            col_count=len(col_names),
        )

        chart_recommended, chart_type_hint = ResultShapeAnalyzer._chart_hint(
            answer_type, time_col, category_col, metric_cols, row_count
        )

        logger.debug(
            "[SHAPE_ANALYZER] rows=%d answer_type=%s chart=%s(%s) "
            "time=%s category=%s metrics=%s",
            row_count, answer_type, chart_recommended, chart_type_hint,
            time_col, category_col, metric_cols,
        )

        return ResultShape(
            answer_type=answer_type,
            column_metas=column_metas,
            chart_recommended=chart_recommended,
            chart_type_hint=chart_type_hint,
            has_single_metric=has_single_metric,
            time_column=time_col,
            category_column=category_col,
            metric_columns=metric_cols,
        )

    @staticmethod
    def _infer_answer_type(
        row_count: int,
        metric_cols: List[str],
        time_col: Optional[str],
        category_col: Optional[str],
        identifier_cols: List[str],
        data_cols: List[str],
        has_agg: bool,
        has_group: bool,
        has_order_desc: bool,
        has_limit: bool,
        col_count: int,
    ) -> str:
        from app.schemas import AnswerType

        # 1. Single row with metric(s) — KPI / COUNT result
        if row_count == 1 and metric_cols and len(data_cols) <= 3:
            return AnswerType.SINGLE_METRIC

        # 2. Time series (GROUP BY time + metric) — check before row-count rules
        if time_col and metric_cols and row_count > 1:
            return AnswerType.TREND

        # 3. Ranking (ordered by metric DESC with LIMIT)
        if category_col and metric_cols and has_order_desc and has_limit and not time_col:
            return AnswerType.RANKING

        # 4. Distribution (GROUP BY category + metric, no time) — check before METRIC_WITH_TABLE
        if category_col and metric_cols and (has_group or has_agg) and not time_col:
            return AnswerType.DISTRIBUTION

        # 5. Comparison (multiple metrics across categories)
        if metric_cols and len(metric_cols) >= 2 and category_col:
            return AnswerType.COMPARISON

        # 6. Few rows (2-5), at least one metric, no grouping — compact metric summary
        if row_count <= 5 and metric_cols and not time_col and not has_group:
            return AnswerType.METRIC_WITH_TABLE

        # 7. Detail records (raw row-level, no aggregation)
        if not has_agg and not has_group and row_count > 1:
            return AnswerType.DETAIL_RECORDS

        return AnswerType.TABULAR_RESULT

    @staticmethod
    def _chart_hint(
        answer_type: str,
        time_col: Optional[str],
        category_col: Optional[str],
        metric_cols: List[str],
        row_count: int,
    ) -> tuple[bool, Optional[str]]:
        """Return (chart_recommended, chart_type_hint)."""
        from app.schemas import AnswerType

        if answer_type == AnswerType.TREND:
            return True, "line" if row_count > 10 else "bar"
        if answer_type == AnswerType.DISTRIBUTION:
            return True, "bar"
        if answer_type == AnswerType.RANKING:
            return True, "bar_horizontal"
        if answer_type == AnswerType.COMPARISON:
            return True, "bar"
        return False, None
