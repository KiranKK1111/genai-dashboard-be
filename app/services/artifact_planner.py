"""
Artifact Planner — Maps a ResultShape to a list of renderable UI artifact dicts.

Deterministic, zero-LLM. Follows the ResponseGeneration.md artifact-centric design.

Frontend renders each artifact independently using its `type` field:
  stat_card, table, bar_chart, line_chart, bar_chart_horizontal
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .result_shape_analyzer import ResultShape

logger = logging.getLogger(__name__)

_DISPLAY_LIMIT = 500  # max rows sent inside table artifact


def _artifact(artifact_type: str, idx: int, title: str, config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": f"artifact_{idx}",
        "type": artifact_type,
        "title": title,
        "order": idx,
        **config,
    }


def _stat_card(idx: int, rows: List[Dict[str, Any]], shape: ResultShape) -> Optional[Dict[str, Any]]:
    """Build a stat card artifact from the first metric column of a single-row result."""
    if not rows or not shape.metric_columns:
        return None
    col = shape.metric_columns[0]
    val = rows[0].get(col)
    if val is None:
        return None

    # Build subtitle from any dimension/category column values in the row
    subtitle_parts = []
    for c, meta in zip(rows[0].keys(), shape.column_metas):
        if meta.semantic_role in ("category", "dimension") and rows[0].get(c) is not None:
            subtitle_parts.append(f"{meta.label or c} = {rows[0][c]}")

    label = col.replace("_", " ").title()
    return _artifact("stat_card", idx, label, {
        "value": val,
        "label": label,
        "subtitle": ", ".join(subtitle_parts) if subtitle_parts else None,
        "format_hint": next(
            (m.format_hint for m in shape.column_metas if m.name == col), None
        ),
    })


def _table(idx: int, rows: List[Dict[str, Any]], title: str, row_count: int) -> Dict[str, Any]:
    """Build a table artifact."""
    if not rows:
        return _artifact("table", idx, title, {"columns": [], "rows": [], "row_count": 0, "exportable": True})

    headers = list(rows[0].keys())
    display = rows[:_DISPLAY_LIMIT]
    row_arrays = [[r.get(h) for h in headers] for r in display]
    return _artifact("table", idx, title, {
        "columns": headers,
        "rows": row_arrays,
        "row_count": row_count,
        "truncated": row_count > _DISPLAY_LIMIT,
        "sortable": True,
        "filterable": True,
        "exportable": True,
    })


def _chart(
    artifact_type: str,
    idx: int,
    title: str,
    x_field: str,
    y_fields: List[str],
    stacked: bool = False,
) -> Dict[str, Any]:
    """Build a chart artifact."""
    return _artifact(artifact_type, idx, title, {
        "x_axis": {"field": x_field, "label": x_field.replace("_", " ").title()},
        "y_axis": {"field": y_fields[0], "label": y_fields[0].replace("_", " ").title()} if y_fields else None,
        "series": [{"field": f, "label": f.replace("_", " ").title()} for f in y_fields],
        "stacked": stacked,
        "data_ref": "response.data.rows",
    })


class ArtifactPlanner:
    """
    Maps a ResultShape to an ordered list of renderable artifact dicts.

    Usage:
        artifacts = ArtifactPlanner.plan(shape=shape, rows=rows, user_query=query)
    """

    @staticmethod
    def plan(
        shape: ResultShape,
        rows: List[Dict[str, Any]],
        user_query: str = "",
        row_count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        from app.schemas import AnswerType

        total = row_count if row_count is not None else len(rows)
        artifacts: List[Dict[str, Any]] = []
        idx = 0

        # Derive a query-aware title
        title_base = (
            user_query.strip().rstrip("?!.")[:60].capitalize()
            if user_query
            else "Query Results"
        )

        at = shape.answer_type

        # ── Case 1: Single metric (COUNT, SUM, AVG) ──────────────────────────
        if at == AnswerType.SINGLE_METRIC:
            stat = _stat_card(idx, rows, shape)
            if stat:
                artifacts.append(stat)
                idx += 1
            artifacts.append(_table(idx, rows, f"{title_base} — Detail", total))
            idx += 1

        # ── Case 2: Trend (time series) ───────────────────────────────────────
        elif at == AnswerType.TREND and shape.time_column and shape.metric_columns:
            artifacts.append(_table(idx, rows, title_base, total))
            idx += 1
            chart_type = "line_chart" if shape.chart_type_hint == "line" else "bar_chart"
            artifacts.append(_chart(
                chart_type, idx,
                f"{title_base} — Chart",
                x_field=shape.time_column,
                y_fields=shape.metric_columns[:2],
            ))
            idx += 1

        # ── Case 3: Distribution (category × metric) ─────────────────────────
        elif at == AnswerType.DISTRIBUTION and shape.category_column and shape.metric_columns:
            artifacts.append(_table(idx, rows, title_base, total))
            idx += 1
            artifacts.append(_chart(
                "bar_chart", idx,
                f"{title_base} — Chart",
                x_field=shape.category_column,
                y_fields=shape.metric_columns[:1],
            ))
            idx += 1

        # ── Case 4: Ranking ───────────────────────────────────────────────────
        elif at == AnswerType.RANKING and shape.category_column and shape.metric_columns:
            artifacts.append(_table(idx, rows, title_base, total))
            idx += 1
            artifacts.append(_chart(
                "bar_chart_horizontal", idx,
                f"{title_base} — Chart",
                x_field=shape.metric_columns[0],
                y_fields=[shape.category_column],
            ))
            idx += 1

        # ── Case 5: Comparison (multiple metrics) ─────────────────────────────
        elif at == AnswerType.COMPARISON:
            artifacts.append(_table(idx, rows, title_base, total))
            idx += 1
            if shape.category_column and shape.metric_columns:
                artifacts.append(_chart(
                    "bar_chart", idx,
                    f"{title_base} — Comparison",
                    x_field=shape.category_column,
                    y_fields=shape.metric_columns[:3],
                    stacked=len(shape.metric_columns) > 2,
                ))
                idx += 1

        # ── Default: table only ────────────────────────────────────────────────
        else:
            artifacts.append(_table(idx, rows, title_base, total))
            idx += 1

        logger.debug(
            "[ARTIFACT_PLANNER] answer_type=%s → %d artifacts: %s",
            at,
            len(artifacts),
            [a["type"] for a in artifacts],
        )
        return artifacts
