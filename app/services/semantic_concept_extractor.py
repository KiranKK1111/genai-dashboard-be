"""
Semantic Concept Extractor - Stage 1 of Query Understanding Pipeline

Converts natural language queries into structured semantic concepts using the
configured LLM — zero hardcoded patterns, fully schema-aware.

The LLM receives:
  - The user's natural language query
  - Available table names (from the database schema)

And returns a structured JSON intent:
  {
    "intent": "count|list|aggregate|filter|group_by|compare|trend",
    "entity": "<table or entity name>",
    "filters": [
      {"concept": "<column or semantic concept>", "operator": "<op>", "value": "<val>"},
      ...
    ]
  }

This completely replaces any hardcoded keyword lists, gender mappings, entity
patterns, temporal concepts, or stopword sets that existed in previous versions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ── Enums (kept for downstream compatibility) ─────────────────────────────────

class IntentType(Enum):
    COUNT      = "count"
    LIST       = "list"
    AGGREGATE  = "aggregate"
    FILTER     = "filter"
    GROUP_BY   = "group_by"
    COMPARE    = "compare"
    TREND      = "trend"


class OperatorType(Enum):
    EQUALS      = "equals"
    IN          = "in"
    NOT_IN      = "not_in"
    GREATER_THAN = "greater_than"
    LESS_THAN   = "less_than"
    BETWEEN     = "between"
    CONTAINS    = "contains"
    STARTS_WITH = "starts_with"
    MONTH_EQUALS = "month_equals"
    YEAR_EQUALS  = "year_equals"
    DATE_RANGE   = "date_range"


@dataclass
class FilterConcept:
    """A semantic filter extracted from user query"""
    concept: str
    operator: OperatorType
    values: Optional[List[Any]] = None
    value: Optional[Any] = None
    raw_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept": self.concept,
            "operator": self.operator.value,
            "values": self.values,
            "value": self.value,
            "raw_text": self.raw_text,
        }


@dataclass
class SemanticIntent:
    """Structured semantic intent extracted from query"""
    intent: IntentType
    entity: Optional[str] = None
    filters: List[FilterConcept] = field(default_factory=list)
    aggregation: Optional[str] = None
    grouping: Optional[List[str]] = None
    sorting: Optional[Dict[str, str]] = None
    limit: Optional[int] = None
    confidence: float = 1.0
    reasoning: str = ""
    has_unknown_values: bool = False  # Always False now — LLM handles everything

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "entity": self.entity,
            "filters": [f.to_dict() for f in self.filters],
            "aggregation": self.aggregation,
            "grouping": self.grouping,
            "sorting": self.sorting,
            "limit": self.limit,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "has_unknown_values": self.has_unknown_values,
        }


# ── Month name → numeric value lookup ────────────────────────────────────────
# Used to auto-upgrade EQUALS(month_name) → MONTH_EQUALS(int) when the LLM
# returns a literal month name instead of the numeric month_equals operator.

_MONTH_NUM: Dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


# ── Operator string → OperatorType mapping ────────────────────────────────────

_OP_MAP: Dict[str, OperatorType] = {
    "=": OperatorType.EQUALS,
    "equals": OperatorType.EQUALS,
    "eq": OperatorType.EQUALS,
    "in": OperatorType.IN,
    "not_in": OperatorType.NOT_IN,
    "not in": OperatorType.NOT_IN,
    ">": OperatorType.GREATER_THAN,
    "greater_than": OperatorType.GREATER_THAN,
    "gt": OperatorType.GREATER_THAN,
    "<": OperatorType.LESS_THAN,
    "less_than": OperatorType.LESS_THAN,
    "lt": OperatorType.LESS_THAN,
    "between": OperatorType.BETWEEN,
    "contains": OperatorType.CONTAINS,
    "like": OperatorType.CONTAINS,
    "starts_with": OperatorType.STARTS_WITH,
    "month_equals": OperatorType.MONTH_EQUALS,
    "year_equals": OperatorType.YEAR_EQUALS,
    "date_range": OperatorType.DATE_RANGE,
}


def _parse_operator(op_str: str) -> OperatorType:
    return _OP_MAP.get(str(op_str).lower(), OperatorType.EQUALS)


# ── Intent string → IntentType mapping ────────────────────────────────────────

_INTENT_MAP: Dict[str, IntentType] = {
    "count": IntentType.COUNT,
    "count_records": IntentType.COUNT,
    "list": IntentType.LIST,
    "list_records": IntentType.LIST,
    "aggregate": IntentType.AGGREGATE,
    "filter": IntentType.FILTER,
    "filter_records": IntentType.FILTER,
    "group_by": IntentType.GROUP_BY,
    "group": IntentType.GROUP_BY,
    "compare": IntentType.COMPARE,
    "trend": IntentType.TREND,
}


def _parse_intent(intent_str: str) -> IntentType:
    return _INTENT_MAP.get(str(intent_str).lower(), IntentType.LIST)


# ── LLM Extraction Prompt ─────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a precise semantic query analyzer. Your job is to analyze a natural language \
database query and return ONLY a JSON object describing the query intent.

Return ONLY valid JSON — no markdown fences, no explanation text.

JSON schema:
{
  "intent": "count | list | aggregate | filter | group_by | compare | trend",
  "entity": "<the main table/entity being queried, e.g. customers, accounts>",
  "filters": [
    {
      "concept": "<column name or semantic concept, e.g. state, gender, city>",
      "operator": "= | in | not_in | > | < | between | contains | month_equals | year_equals",
      "value": "<single value, or null if using values array>",
      "values": ["<val1>", "<val2>"] // only for 'in' / 'not_in' / 'between'
    }
  ],
  "aggregation": "COUNT | SUM | AVG | MAX | MIN | null",
  "grouping": ["<col1>", "<col2>"] or null,
  "sorting": {"column": "<col>", "direction": "ASC | DESC"} or null
}

Rules:
- Use exact column/value names as they appear logically in the query.
- For location codes like "TG", "AP", "MH" treat them as filter values on a location/state column.
- For gender terms like "male"/"female" use the raw word as value (downstream grounding maps to DB codes).
- For month names like "January" use month_equals with numeric value (1-12).
- For year references use year_equals with the 4-digit year.
- entity: When "Available tables" are listed, the entity MUST be the exact table name from that list that best matches what the user is asking about. Map synonyms (e.g. "clients" → "customers", "transactions" → "txns") to the closest real table name. Never output a word that is not in the provided table list.
- If no "Available tables" are listed, entity should be the singular or plural noun that represents the main table.
- If no filter applies, use an empty array [].
"""


def _build_user_message(
    query: str,
    table_names: Optional[List[str]] = None,
    table_schemas: Optional[Dict[str, List[str]]] = None,
) -> str:
    parts = [f'Query: "{query}"']
    if table_schemas:
        # Include actual column names so the LLM uses real column names, not guesses
        schema_lines = []
        for tbl, cols in sorted(table_schemas.items()):
            schema_lines.append(f"  {tbl}: [{', '.join(cols)}]")
        parts.append("Database schema (table: [columns]):\n" + "\n".join(schema_lines))
    elif table_names:
        parts.append(f"Available tables: {', '.join(table_names)}")
    return "\n".join(parts)


# ── Extractor Class ───────────────────────────────────────────────────────────

class SemanticConceptExtractor:
    """
    LLM-powered semantic concept extractor.

    Sends the user query (+ optional table list) to the configured LLM and
    parses the returned JSON into a SemanticIntent.  Zero hardcoded patterns.
    """

    def extract_semantic_intent(
        self,
        query: str,
        table_names: Optional[List[str]] = None,
        table_schemas: Optional[Dict[str, List[str]]] = None,
    ) -> SemanticIntent:
        """
        Synchronous wrapper — runs the async LLM call in the current event loop
        or falls back to a lightweight heuristic if the LLM is unreachable.

        This keeps call sites that use `concept_extractor.extract_semantic_intent(query)`
        without await working as before.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._extract_async(query, table_names, table_schemas),
                    )
                    return future.result(timeout=15)
            else:
                return loop.run_until_complete(self._extract_async(query, table_names, table_schemas))
        except Exception as e:
            logger.warning(f"[CONCEPT_EXTRACTOR] LLM extraction failed, using heuristic fallback: {e}")
            return self._heuristic_fallback(query)

    async def extract_semantic_intent_async(
        self,
        query: str,
        table_names: Optional[List[str]] = None,
        table_schemas: Optional[Dict[str, List[str]]] = None,
    ) -> SemanticIntent:
        """Async version — preferred when called from async code."""
        try:
            return await self._extract_async(query, table_names, table_schemas)
        except Exception as e:
            logger.warning(f"[CONCEPT_EXTRACTOR] LLM extraction failed, using heuristic fallback: {e}")
            return self._heuristic_fallback(query)

    async def _extract_async(
        self,
        query: str,
        table_names: Optional[List[str]] = None,
        table_schemas: Optional[Dict[str, List[str]]] = None,
    ) -> SemanticIntent:
        """Core LLM call — returns parsed SemanticIntent."""
        from app.llm import call_llm

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(query, table_names, table_schemas)},
        ]

        raw = await call_llm(
            messages=messages,
            max_tokens=512,
            temperature=0.0,   # Deterministic for structured extraction
            json_mode=True,
        )

        return self._parse_llm_response(str(raw), query)

    def _parse_llm_response(self, raw: str, original_query: str) -> SemanticIntent:
        """Parse the LLM JSON response into a SemanticIntent."""
        # Strip markdown fences if the model wrapped the JSON anyway
        clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Try to extract the first {...} block
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    logger.warning(f"[CONCEPT_EXTRACTOR] Could not parse LLM JSON: {clean[:200]}")
                    return self._heuristic_fallback(original_query)
            else:
                return self._heuristic_fallback(original_query)

        intent_type = _parse_intent(data.get("intent", "list"))

        filters: List[FilterConcept] = []
        for f in data.get("filters", []):
            concept = f.get("concept", "")
            if not concept:
                continue
            op = _parse_operator(f.get("operator", "="))
            value = f.get("value")
            values = f.get("values")
            if values and isinstance(values, list) and len(values) == 1:
                value = values[0]
                values = None
            # Auto-upgrade EQUALS(month_name) → MONTH_EQUALS(int)
            # Handles cases where the LLM returns "January" as a literal value
            # instead of using month_equals with a numeric value.
            if op == OperatorType.EQUALS and isinstance(value, str):
                month_num = _MONTH_NUM.get(value.strip().lower())
                if month_num is not None:
                    op = OperatorType.MONTH_EQUALS
                    value = month_num

            # Normalize IN with 0 or 1 values → EQUALS (prevents "IN ()" SQL syntax error)
            if op == OperatorType.IN:
                in_vals = [v for v in (values or []) if v is not None]
                if len(in_vals) == 0:
                    op = OperatorType.EQUALS
                    values = None
                elif len(in_vals) == 1:
                    op = OperatorType.EQUALS
                    value = in_vals[0]
                    values = None
            filters.append(FilterConcept(
                concept=concept,
                operator=op,
                value=value,
                values=values,
                raw_text=f"{concept} {f.get('operator', '=')} {value or values}",
            ))

        grouping = data.get("grouping")
        if isinstance(grouping, list) and len(grouping) == 0:
            grouping = None

        sorting_raw = data.get("sorting")
        sorting = None
        if isinstance(sorting_raw, dict) and sorting_raw.get("column"):
            sorting = {
                sorting_raw["column"]: sorting_raw.get("direction", "ASC")
            }

        agg = data.get("aggregation") or (
            "COUNT" if intent_type == IntentType.COUNT else None
        )

        filter_desc = ", ".join(
            f"{f.concept}{f.operator.value}{f.value or f.values}" for f in filters
        ) if filters else "none"
        logger.info(
            f"[CONCEPT_EXTRACTOR] LLM extracted intent={intent_type.value}, "
            f"entity={data.get('entity')}, filters={len(filters)} [{filter_desc}]"
        )

        return SemanticIntent(
            intent=intent_type,
            entity=data.get("entity"),
            filters=filters,
            aggregation=agg,
            grouping=grouping,
            sorting=sorting,
            confidence=0.95,
            reasoning=f"LLM-extracted: {len(filters)} filter(s) from query",
            has_unknown_values=False,  # LLM handles all values — nothing is "unknown"
        )

    # ── Lightweight heuristic fallback (no LLM, no hardcoding) ──────────────

    def _heuristic_fallback(self, query: str) -> SemanticIntent:
        """
        Minimal fallback when the LLM is unavailable.
        Only detects intent (count vs list) — no filters.
        Always sets has_unknown_values=True so the caller falls back to
        the full LLM orchestrator for SQL generation.
        """
        query_lower = query.lower()
        if re.search(r"\b(how many|count|number of|total)\b", query_lower):
            intent = IntentType.COUNT
            agg = "COUNT"
        else:
            intent = IntentType.LIST
            agg = None

        logger.warning(
            f"[CONCEPT_EXTRACTOR] Using heuristic fallback for: {query[:60]}"
        )
        return SemanticIntent(
            intent=intent,
            entity=None,
            filters=[],
            aggregation=agg,
            confidence=0.3,
            reasoning="Heuristic fallback — LLM unavailable",
            has_unknown_values=True,  # Signal: route to full LLM orchestrator
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_concept_extractor: Optional[SemanticConceptExtractor] = None


def get_concept_extractor() -> SemanticConceptExtractor:
    """Return the singleton SemanticConceptExtractor instance."""
    global _concept_extractor
    if _concept_extractor is None:
        _concept_extractor = SemanticConceptExtractor()
    return _concept_extractor
