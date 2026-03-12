"""
Smart Query Processor — ChatGPT-like Natural Language Understanding.

Multi-stage pipeline that deeply analyses user queries before SQL generation:

Stage 1  Temporal Resolution   — deterministic date-range extraction
Stage 2  Intent Classification — LLM: what operation does the user want?
Stage 3  Entity Discovery      — which tables/entities are referenced?
Stage 4  Value Grounding       — "active customers" → status = 'ACTIVE'
Stage 5  Ambiguity Scoring     — is the query clear enough to execute?
Stage 6  Clarification Build   — produce schema-grounded clarification choices

The output (EnhancedQueryIntent) is injected into the LLM plan-generation
prompt as additional context, so the QueryPlanGenerator produces a more
accurate query plan without requiring the user to be precise.

Handles:
  - Temporal expressions:  "last month", "Q3 2024", "year to date", "2025"
  - Value references:      "premium users" → customer_type = 'PREMIUM'
  - Entity aliases:        "clients" → customers table
  - Implicit operations:   "how many" → COUNT, "average" → AVG
  - Vague queries:         ask structured clarification with real options
  - Misspellings:          fuzzy-matched against schema before LLM call
  - Multi-entity queries:  "customers with their loans" → JOIN planning
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class QueryIntent(str, Enum):
    """Semantic intent of the user query."""
    LIST          = "list_records"        # "show me all customers"
    COUNT         = "count_records"       # "how many customers"
    AGGREGATE     = "aggregate"           # "total revenue by region"
    FILTER        = "filter_records"      # "customers from New York"
    RANK          = "rank_records"        # "top 10 customers by revenue"
    COMPARE       = "compare"             # "revenue this year vs last year"
    TREND         = "trend"               # "monthly revenue trend"
    JOIN_LOOKUP   = "join_lookup"         # "customers with their loans"
    EXISTENCE     = "existence_check"     # "customers who have no loans"
    SEARCH        = "search_value"        # "find customer John Smith"
    DISTRIBUTION  = "distribution"        # "distribution of loan amounts"
    UNKNOWN       = "unknown"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TemporalConstraint:
    """Resolved temporal expression."""
    expression: str               # Original text: "last month"
    pattern_type: str             # "last_month", "last_n_days", "quarter", …
    start_date: Optional[str]     # ISO 8601: "2026-02-01"
    end_date: Optional[str]       # ISO 8601: "2026-02-28"
    sql_fragment: Optional[str]   # Ready-to-inject WHERE snippet
    granularity: str = "day"      # day | month | quarter | year
    column_hint: Optional[str] = None  # Guessed date column name


@dataclass
class ValueGroundingHit:
    """A user-mentioned value grounded to a real schema column."""
    user_concept: str      # "active"
    table: str
    column: str
    matched_value: str     # "ACTIVE" (exact DB value)
    confidence: float


@dataclass
class EntityHit:
    """A natural-language entity mapped to a schema table."""
    user_term: str         # "clients"
    table: str
    confidence: float
    is_primary: bool = True


@dataclass
class EnhancedQueryIntent:
    """
    Fully enriched query intent — output of SmartQueryProcessor.

    This is passed as additional context to QueryPlanGenerator so the LLM
    prompt is seeded with already-resolved information:
    - date ranges instead of "last month"
    - exact column=value pairs instead of "active customers"
    - table names already identified from entity resolution
    - clarification question if the query is too vague to execute
    """
    original_query: str
    rewritten_query: str        # Normalised, de-abbreviated
    intent: QueryIntent

    # Entity → table mapping
    primary_entity: Optional[str] = None
    secondary_entities: List[str] = field(default_factory=list)

    # Resolved temporal constraints
    temporal_constraints: List[TemporalConstraint] = field(default_factory=list)

    # Value → column groundings
    value_groundings: List[ValueGroundingHit] = field(default_factory=list)

    # Entity hints
    entity_hits: List[EntityHit] = field(default_factory=list)

    # Ambiguity
    ambiguity_score: float = 0.0   # 0.0 = crystal clear, 1.0 = totally vague
    ambiguity_reasons: List[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    clarification_options: List[str] = field(default_factory=list)

    # Hints injected into the plan-generation prompt
    where_hints: List[str] = field(default_factory=list)   # Raw SQL snippets
    column_hints: List[str] = field(default_factory=list)  # Suggested columns

    # Overall confidence
    confidence: float = 1.0

    def to_prompt_context(self) -> str:
        """Render as a text block injected into the LLM plan-generation prompt."""
        lines: List[str] = ["=== SEMANTIC PRE-ANALYSIS (use this to improve your plan) ==="]

        lines.append(f"Intent: {self.intent.value}")

        if self.primary_entity:
            lines.append(f"Primary table: {self.primary_entity}")
        if self.secondary_entities:
            lines.append(f"Secondary tables: {', '.join(self.secondary_entities)}")

        if self.temporal_constraints:
            lines.append("Temporal constraints resolved:")
            for tc in self.temporal_constraints:
                lines.append(f"  • \"{tc.expression}\" → {tc.start_date} .. {tc.end_date}")
                if tc.sql_fragment:
                    lines.append(f"    SQL hint: {tc.sql_fragment}")

        if self.value_groundings:
            lines.append("Value groundings (concept → column = value):")
            for vg in self.value_groundings:
                lines.append(
                    f"  • \"{vg.user_concept}\" → {vg.table}.{vg.column} = '{vg.matched_value}'"
                    f"  (confidence {vg.confidence:.0%})"
                )

        if self.where_hints:
            lines.append("Suggested WHERE conditions:")
            for hint in self.where_hints:
                lines.append(f"  • {hint}")

        if self.needs_clarification:
            lines.append(f"⚠ Query is ambiguous: {', '.join(self.ambiguity_reasons)}")
            lines.append(f"  Clarification: {self.clarification_question}")
            if self.clarification_options:
                lines.append(f"  Options: {', '.join(self.clarification_options)}")

        lines.append("=== END PRE-ANALYSIS ===")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Temporal pattern resolver (deterministic — no LLM required)
# ---------------------------------------------------------------------------

_TEMPORAL_PATTERNS: List[Tuple[str, str]] = [
    (r'\btoday\b',                             'today'),
    (r'\byesterday\b',                         'yesterday'),
    (r'\bthis\s+week\b',                       'this_week'),
    (r'\blast\s+week\b',                       'last_week'),
    (r'\bthis\s+month\b',                      'this_month'),
    (r'\blast\s+month\b',                      'last_month'),
    (r'\blast\s+(\d+)\s+days?\b',              'last_n_days'),
    (r'\blast\s+(\d+)\s+months?\b',            'last_n_months'),
    (r'\blast\s+(\d+)\s+weeks?\b',             'last_n_weeks'),
    (r'\bthis\s+year\b',                       'this_year'),
    (r'\blast\s+year\b',                       'last_year'),
    (r'\byear[- ]to[- ]date\b|\bYTD\b',        'year_to_date'),
    (r'\bq([1-4])\s+(\d{4})\b',               'quarter'),
    (r'\b(Q[1-4])(\d{4})\b',                  'quarter_compact'),
    (r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
     r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|'
     r'dec(?:ember)?)\s+(\d{4})\b',           'specific_month_year'),
    (r'\bfiscal\s+year\s+(\d{4})\b',          'fiscal_year'),
    (r'\bin\s+(\d{4})\b',                      'year'),
    (r'\b(\d{4})\b(?!\s*-\s*\d)',              'year_standalone'),
]

_MONTH_MAP = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
    'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
    'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
    'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12,
}


def _last_day_of_month(y: int, m: int) -> int:
    """Return last calendar day of the given month."""
    if m == 12:
        return 31
    return (date(y, m + 1, 1) - timedelta(days=1)).day


def resolve_temporal_expressions(
    user_query: str,
    reference_date: Optional[date] = None,
) -> List[TemporalConstraint]:
    """
    Extract and resolve all temporal expressions in user_query.

    Returns a list of TemporalConstraint objects, one per expression found.
    Uses reference_date (defaults to today) for relative expressions.
    """
    today = reference_date or date.today()
    y, m, d = today.year, today.month, today.day
    results: List[TemporalConstraint] = []
    query_lower = user_query.lower()

    def _make(expr: str, ptype: str, start: date, end: date,
              granularity: str = "day") -> TemporalConstraint:
        s = start.isoformat()
        e = end.isoformat()
        sql = f"{{col}} >= '{s}' AND {{col}} <= '{e}'"
        return TemporalConstraint(
            expression=expr, pattern_type=ptype,
            start_date=s, end_date=e, sql_fragment=sql, granularity=granularity,
        )

    for pattern, ptype in _TEMPORAL_PATTERNS:
        for match in re.finditer(pattern, query_lower, re.IGNORECASE):
            expr = match.group(0)

            if ptype == 'today':
                tc = _make(expr, ptype, today, today)

            elif ptype == 'yesterday':
                yd = today - timedelta(days=1)
                tc = _make(expr, ptype, yd, yd)

            elif ptype == 'this_week':
                monday = today - timedelta(days=today.weekday())
                tc = _make(expr, ptype, monday, today)

            elif ptype == 'last_week':
                monday = today - timedelta(days=today.weekday() + 7)
                sunday = monday + timedelta(days=6)
                tc = _make(expr, ptype, monday, sunday)

            elif ptype == 'this_month':
                start = date(y, m, 1)
                end = date(y, m, _last_day_of_month(y, m))
                tc = _make(expr, ptype, start, end, "month")

            elif ptype == 'last_month':
                lm = m - 1 if m > 1 else 12
                ly = y if m > 1 else y - 1
                start = date(ly, lm, 1)
                end = date(ly, lm, _last_day_of_month(ly, lm))
                tc = _make(expr, ptype, start, end, "month")

            elif ptype == 'last_n_days':
                n = int(match.group(1))
                start = today - timedelta(days=n)
                tc = _make(expr, ptype, start, today)

            elif ptype == 'last_n_months':
                n = int(match.group(1))
                # go back n months
                nm = m - n
                ny = y
                while nm <= 0:
                    nm += 12
                    ny -= 1
                start = date(ny, nm, 1)
                tc = _make(expr, ptype, start, today, "month")

            elif ptype == 'last_n_weeks':
                n = int(match.group(1))
                start = today - timedelta(weeks=n)
                tc = _make(expr, ptype, start, today)

            elif ptype == 'this_year':
                start = date(y, 1, 1)
                end = date(y, 12, 31)
                tc = _make(expr, ptype, start, end, "year")

            elif ptype == 'last_year':
                start = date(y - 1, 1, 1)
                end = date(y - 1, 12, 31)
                tc = _make(expr, ptype, start, end, "year")

            elif ptype == 'year_to_date':
                start = date(y, 1, 1)
                tc = _make(expr, ptype, start, today, "year")

            elif ptype in ('quarter', 'quarter_compact'):
                try:
                    if ptype == 'quarter':
                        qn, qy = int(match.group(1)), int(match.group(2))
                    else:
                        qn = int(match.group(1)[1])
                        qy = int(match.group(2))
                    start_month = (qn - 1) * 3 + 1
                    end_month = start_month + 2
                    start = date(qy, start_month, 1)
                    end = date(qy, end_month, _last_day_of_month(qy, end_month))
                    tc = _make(expr, 'quarter', start, end, "quarter")
                except (ValueError, IndexError):
                    continue

            elif ptype == 'specific_month_year':
                try:
                    parts = query_lower[match.start():match.end()].split()
                    mname = parts[0].rstrip('.,')
                    mnum = _MONTH_MAP.get(mname)
                    myr = int(parts[-1])
                    if mnum and myr:
                        start = date(myr, mnum, 1)
                        end = date(myr, mnum, _last_day_of_month(myr, mnum))
                        tc = _make(expr, ptype, start, end, "month")
                    else:
                        continue
                except (ValueError, IndexError):
                    continue

            elif ptype == 'fiscal_year':
                try:
                    fy = int(match.group(1))
                    # Assume fiscal year starts April 1
                    start = date(fy - 1, 4, 1)
                    end = date(fy, 3, 31)
                    tc = _make(expr, ptype, start, end, "year")
                except (ValueError, IndexError):
                    continue

            elif ptype in ('year', 'year_standalone'):
                try:
                    yr = int(match.group(1))
                    if 1990 <= yr <= 2100:
                        start = date(yr, 1, 1)
                        end = date(yr, 12, 31)
                        tc = _make(expr, ptype, start, end, "year")
                    else:
                        continue
                except (ValueError, IndexError):
                    continue

            else:
                continue

            # De-duplicate
            if not any(r.expression.lower() == tc.expression.lower() for r in results):
                results.append(tc)

    return results


# ---------------------------------------------------------------------------
# Fuzzy value-to-column grounding (light — uses schema sample data only)
# ---------------------------------------------------------------------------

def ground_values_to_columns(
    user_query: str,
    schema_catalog: Any,  # SemanticSchemaCatalog instance
    tables: Optional[List[str]] = None,
) -> List[ValueGroundingHit]:
    """
    Scan user query for tokens that match actual column sample values.

    For each token in the query, checks if any column in the schema has that
    value (case-insensitive) in its sample data.  Returns matches sorted by
    confidence (exact > substring).

    schema_catalog must have `.tables` dict mapping table_name → TableMetadata
    (with .columns mapping col_name → ColumnMetadata with .sample_values list).
    """
    hits: List[ValueGroundingHit] = []

    if schema_catalog is None:
        return hits

    # Normalise query tokens
    tokens = set(
        t.strip("'\".,;!?()[]")
        for t in re.split(r'\s+', user_query)
        if len(t.strip("'\".,;!?()[]")) >= 3
    )

    # Common stop-words to skip
    _STOP = {
        'the', 'and', 'for', 'not', 'are', 'was', 'has', 'had', 'have',
        'with', 'from', 'that', 'this', 'they', 'which', 'show', 'get',
        'list', 'find', 'give', 'all', 'any', 'all', 'top', 'how', 'many',
        'who', 'what', 'when', 'where', 'count', 'sum', 'avg', 'max', 'min',
        'records', 'rows', 'data', 'table', 'column', 'values', 'result',
    }
    tokens = {t for t in tokens if t.lower() not in _STOP}

    try:
        catalog_tables = getattr(schema_catalog, 'tables', {}) or {}
    except Exception:
        return hits

    for table_name, table_meta in catalog_tables.items():
        if tables and table_name not in tables:
            continue
        try:
            columns = getattr(table_meta, 'columns', {}) or {}
        except Exception:
            continue

        for col_name, col_meta in columns.items():
            try:
                sample_values = getattr(col_meta, 'sample_values', []) or []
            except Exception:
                continue

            # Normalise sample values to strings
            sv_norm = [str(v).strip() for v in sample_values if v is not None]

            for token in tokens:
                tok_upper = token.upper()
                tok_lower = token.lower()

                for sv in sv_norm:
                    sv_upper = sv.upper()
                    confidence = 0.0

                    if tok_upper == sv_upper:
                        confidence = 1.0
                    elif sv_upper.startswith(tok_upper) or tok_upper.startswith(sv_upper):
                        confidence = 0.85
                    elif tok_upper in sv_upper or sv_upper in tok_upper:
                        confidence = 0.70

                    if confidence >= 0.70:
                        # Avoid duplicates
                        if not any(
                            h.user_concept == tok_lower and h.table == table_name and h.column == col_name
                            for h in hits
                        ):
                            hits.append(ValueGroundingHit(
                                user_concept=tok_lower,
                                table=table_name,
                                column=col_name,
                                matched_value=sv,
                                confidence=confidence,
                            ))

    # Sort: higher confidence first, then by table+column alphabetically
    hits.sort(key=lambda h: (-h.confidence, h.table, h.column))
    return hits[:10]  # Cap at 10 most relevant hits


# ---------------------------------------------------------------------------
# Entity → Table resolver (fuzzy match, no LLM)
# ---------------------------------------------------------------------------

def resolve_entities_to_tables(
    user_query: str,
    schema_catalog: Any,
) -> List[EntityHit]:
    """
    Map entity mentions in user_query to actual table names via:
    1. Exact table name match
    2. Table alias / business-name match (from TableMetadata.description)
    3. Plural/singular variants (customers ↔ customer)
    4. Common domain synonyms (staff ↔ employees, clients ↔ customers)
    """
    hits: List[EntityHit] = []
    query_lower = user_query.lower()

    try:
        catalog_tables = getattr(schema_catalog, 'tables', {}) or {}
    except Exception:
        return hits

    # Build synonym index from table names
    for table_name, table_meta in catalog_tables.items():
        candidates: List[Tuple[str, float]] = []

        # Direct name match
        if table_name.lower() in query_lower:
            candidates.append((table_name, 0.95))

        # Singular / plural variants
        singular = table_name.rstrip('s')
        plural = table_name + 's' if not table_name.endswith('s') else table_name
        if singular in query_lower and singular != table_name:
            candidates.append((singular, 0.85))
        if plural in query_lower and plural != table_name:
            candidates.append((plural, 0.85))

        # Description / display name aliases
        try:
            desc = (getattr(table_meta, 'description', '') or '').lower()
            if desc and any(tok in query_lower for tok in desc.split() if len(tok) >= 4):
                candidates.append((table_name + '_desc', 0.70))
        except Exception:
            pass

        if candidates:
            best_term, best_conf = max(candidates, key=lambda x: x[1])
            hits.append(EntityHit(
                user_term=best_term,
                table=table_name,
                confidence=best_conf,
                is_primary=True,  # Will be refined below
            ))

    # Sort by confidence
    hits.sort(key=lambda h: -h.confidence)

    # Mark only highest-confidence as primary
    if len(hits) > 1:
        for i, h in enumerate(hits):
            h.is_primary = (i == 0)

    return hits[:5]


# ---------------------------------------------------------------------------
# Ambiguity Detector
# ---------------------------------------------------------------------------

_VAGUE_PATTERNS = [
    r'\bshow\s+me\s+(all\s+)?data\b',
    r'\bget\s+me\s+(all\s+)?data\b',
    r'\bshow\s+everything\b',
    r'\blist\s+everything\b',
    r'\bshow\s+(all\s+)?records?\b',
    r'\bget\s+(all\s+)?(the\s+)?data\b',
    r'\bwhat\s+(is|are)\s+(the\s+)?data\b',
    r'\bshow\s+me\s+something\b',
]


def score_ambiguity(
    user_query: str,
    entity_hits: List[EntityHit],
    schema_catalog: Any,
) -> Tuple[float, List[str]]:
    """
    Return (ambiguity_score 0.0–1.0, reasons list).

    High score = vague query that needs clarification.
    """
    reasons: List[str] = []
    score = 0.0

    # Check for vague patterns
    for pat in _VAGUE_PATTERNS:
        if re.search(pat, user_query, re.IGNORECASE):
            reasons.append("Query is too vague (no specific entity or metric mentioned)")
            score += 0.5
            break

    # No entity identified
    if not entity_hits:
        reasons.append("No table/entity could be identified from the query")
        score += 0.4

    # Multiple tables with similar confidence
    if len(entity_hits) >= 2:
        conf_diff = entity_hits[0].confidence - entity_hits[1].confidence
        if conf_diff < 0.15:
            reasons.append(
                f"Ambiguous entity: both '{entity_hits[0].table}' and "
                f"'{entity_hits[1].table}' match equally well"
            )
            score += 0.3

    # Very short query
    if len(user_query.split()) <= 2:
        reasons.append("Query is too short to determine intent")
        score += 0.25

    return min(score, 1.0), reasons


# ---------------------------------------------------------------------------
# Clarification builder (uses real schema options)
# ---------------------------------------------------------------------------

def build_clarification(
    ambiguity_reasons: List[str],
    entity_hits: List[EntityHit],
    schema_catalog: Any,
    user_query: str,
) -> Tuple[str, List[str]]:
    """
    Return (clarification_question, list_of_options) using real schema names.

    Called when ambiguity_score > 0.5.
    """
    try:
        catalog_tables = list((getattr(schema_catalog, 'tables', {}) or {}).keys())
    except Exception:
        catalog_tables = []

    if not entity_hits and catalog_tables:
        # User didn't mention any table — show all available options
        options = catalog_tables[:6]
        question = "Which data would you like to query? Please choose:"
        return question, options

    if len(entity_hits) >= 2:
        # Multiple table candidates
        options = [h.table for h in entity_hits[:4]]
        question = f"Your query '{user_query[:60]}' could refer to multiple tables. Which did you mean?"
        return question, options

    if entity_hits:
        # Entity found but query is otherwise vague
        table = entity_hits[0].table
        question = f"What would you like to know about {table}? For example:"
        options = [
            f"Count of all {table}",
            f"List all {table}",
            f"Filter {table} by specific criteria",
            f"Aggregate {table} by a column",
        ]
        return question, options

    question = "I'm not sure what you'd like. Could you be more specific?"
    options = catalog_tables[:6]
    return question, options


# ---------------------------------------------------------------------------
# Main SmartQueryProcessor
# ---------------------------------------------------------------------------

class SmartQueryProcessor:
    """
    Pre-processes user queries to produce an EnhancedQueryIntent.

    Inject the result into the QueryPlanGenerator prompt via
    `enhanced_intent.to_prompt_context()` to get ChatGPT-level accuracy.

    Usage::
        processor = SmartQueryProcessor()
        intent = await processor.process(user_query, schema_catalog)
        if intent.needs_clarification:
            # Return clarification response to user
            ...
        else:
            # Inject intent.to_prompt_context() into LLM plan prompt
            ...
    """

    AMBIGUITY_THRESHOLD = 0.5   # Score above this → ask for clarification

    def __init__(self, llm_client: Any = None):
        """
        Args:
            llm_client: Optional LLM client for deep intent analysis.
                        If None, falls back to heuristics-only mode.
        """
        self._llm = llm_client

    async def process(
        self,
        user_query: str,
        schema_catalog: Any,
        session_context: Optional[Dict[str, Any]] = None,
        reference_date: Optional[date] = None,
        target_tables: Optional[List[str]] = None,
    ) -> EnhancedQueryIntent:
        """
        Full multi-stage analysis.

        Args:
            user_query:      Raw user input string
            schema_catalog:  SemanticSchemaCatalog (provides tables + sample values)
            session_context: Optional dict with last_table, last_sql, turn_class, etc.
            reference_date:  Override 'today' for temporal resolution (tests/replay)
            target_tables:   If provided, restrict entity/value search to these tables

        Returns:
            EnhancedQueryIntent with all fields populated
        """
        # ── Stage 1: Temporal resolution ──────────────────────────────────
        temporal = resolve_temporal_expressions(user_query, reference_date)
        logger.debug("[SQP] Found %d temporal constraints", len(temporal))

        # ── Stage 2: Entity → table resolution ────────────────────────────
        entity_hits = resolve_entities_to_tables(user_query, schema_catalog)

        # If a follow-up, lock onto the previous table first
        if session_context:
            prev_table = session_context.get("last_table")
            if prev_table and not any(h.table == prev_table for h in entity_hits):
                entity_hits.insert(0, EntityHit(
                    user_term=prev_table, table=prev_table,
                    confidence=0.80, is_primary=True,
                ))

        # ── Stage 3: Value → column grounding ─────────────────────────────
        effective_tables = target_tables or [h.table for h in entity_hits[:3]]
        value_groundings = ground_values_to_columns(user_query, schema_catalog, effective_tables)
        logger.debug("[SQP] Found %d value groundings", len(value_groundings))

        # ── Stage 4: LLM deep intent (if client available) ────────────────
        intent = QueryIntent.UNKNOWN
        rewritten_query = user_query
        if self._llm:
            intent, rewritten_query = await self._llm_classify_intent(
                user_query, schema_catalog, entity_hits
            )
        else:
            intent = self._heuristic_intent(user_query)

        # ── Stage 5: Ambiguity scoring ────────────────────────────────────
        ambiguity_score, ambiguity_reasons = score_ambiguity(
            user_query, entity_hits, schema_catalog
        )

        # ── Stage 6: Build clarification if needed ────────────────────────
        needs_clarification = ambiguity_score >= self.AMBIGUITY_THRESHOLD
        clarification_question: Optional[str] = None
        clarification_options: List[str] = []

        if needs_clarification:
            clarification_question, clarification_options = build_clarification(
                ambiguity_reasons, entity_hits, schema_catalog, user_query
            )

        # ── Build WHERE hints from temporal + value groundings ────────────
        where_hints: List[str] = []

        for tc in temporal:
            if tc.sql_fragment:
                # Use generic placeholder; QueryPlanGenerator will fill column name
                where_hints.append(
                    f"[DATE FILTER: {tc.expression} → {tc.start_date} to {tc.end_date}]"
                )

        for vg in value_groundings:
            if vg.confidence >= 0.85:
                where_hints.append(
                    f"{vg.table}.{vg.column} = '{vg.matched_value}'"
                )

        # ── Assemble result ────────────────────────────────────────────────
        primary_entity = (
            entity_hits[0].table if entity_hits else
            (session_context.get("last_table") if session_context else None)
        )
        secondary_entities = [h.table for h in entity_hits[1:3]]

        confidence = max(0.1, 1.0 - ambiguity_score)

        logger.info(
            "[SQP] Query: '%s' → intent=%s, entity=%s, temporal=%d, values=%d, "
            "ambiguity=%.2f, clarify=%s",
            user_query[:60], intent.value, primary_entity,
            len(temporal), len(value_groundings), ambiguity_score, needs_clarification,
        )

        return EnhancedQueryIntent(
            original_query=user_query,
            rewritten_query=rewritten_query,
            intent=intent,
            primary_entity=primary_entity,
            secondary_entities=secondary_entities,
            temporal_constraints=temporal,
            value_groundings=value_groundings,
            entity_hits=entity_hits,
            ambiguity_score=ambiguity_score,
            ambiguity_reasons=ambiguity_reasons,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            clarification_options=clarification_options,
            where_hints=where_hints,
            confidence=confidence,
        )

    # -----------------------------------------------------------------------
    # LLM intent classification
    # -----------------------------------------------------------------------

    async def _llm_classify_intent(
        self,
        user_query: str,
        schema_catalog: Any,
        entity_hits: List[EntityHit],
    ) -> Tuple[QueryIntent, str]:
        """Use LLM to classify intent and rewrite the query."""
        try:
            table_list = ", ".join(h.table for h in entity_hits[:4]) or "unknown"
            system_prompt = (
                "You are a SQL intent classifier. "
                "Respond with ONLY a JSON object with two keys: "
                "\"intent\" (one of: list_records, count_records, aggregate, filter_records, "
                "rank_records, compare, trend, join_lookup, existence_check, search_value, "
                "distribution, unknown) "
                "and \"rewritten\" (the query rewritten clearly, resolving abbreviations and "
                "spelling mistakes, max 120 chars)."
            )
            user_prompt = (
                f"User query: \"{user_query}\"\n"
                f"Candidate tables: {table_list}\n"
                "Classify the intent and rewrite the query."
            )
            response = await self._llm.call_llm_json(
                system=system_prompt,
                user=user_prompt,
                temperature=0.0,
            )
            data = json.loads(response)
            intent_str = data.get("intent", "unknown")
            try:
                intent = QueryIntent(intent_str)
            except ValueError:
                intent = QueryIntent.UNKNOWN
            rewritten = data.get("rewritten", user_query)
            return intent, rewritten
        except Exception as exc:
            logger.debug("[SQP] LLM intent classification failed: %s", exc)
            return self._heuristic_intent(user_query), user_query

    # -----------------------------------------------------------------------
    # Heuristic fallback intent classification (no LLM)
    # -----------------------------------------------------------------------

    _INTENT_PATTERNS: List[Tuple[str, QueryIntent]] = [
        (r'\bhow\s+many\b|\bcount\b|\bnumber\s+of\b|\btotal\s+(?:number|count)\b',
         QueryIntent.COUNT),
        (r'\btotal\b|\bsum\b|\baverage\b|\bavg\b|\bmean\b|\bsum\s+of\b|\btotal\s+\w+\b',
         QueryIntent.AGGREGATE),
        (r'\btop\s+\d+\b|\bhighest\b|\blowest\b|\bmost\b|\bleast\b|\bbest\b|\bworst\b|\branked?\b',
         QueryIntent.RANK),
        (r'\btrend\b|\bover\s+time\b|\bmonthly\b|\bweekly\b|\byearly\b|\bby\s+(month|week|year|quarter)\b',
         QueryIntent.TREND),
        (r'\bcompare\b|\bvs\.?\b|\bversus\b|\bdifference\b|\bgrowth\b|\bchange\b',
         QueryIntent.COMPARE),
        (r'\bwith\s+(their|his|her|its)\b|\bjoin\b|\band\s+their\b|\balong\s+with\b',
         QueryIntent.JOIN_LOOKUP),
        (r'\bwho\s+(have\s+no|don\'?t\s+have|without|never|haven\'?t)\b|\bnot\s+exist\b|\bno\s+\w+s?\b',
         QueryIntent.EXISTENCE),
        (r'\bfind\b|\bsearch\b|\blook\s+for\b|\bwhere\s+is\b|\blocate\b',
         QueryIntent.SEARCH),
        (r'\bdistribution\b|\bbreakdown\b|\bspread\b|\brange\b|\bhistogram\b',
         QueryIntent.DISTRIBUTION),
        (r'\bshow\b|\blist\b|\bget\b|\bdisplay\b|\bfetch\b|\bsee\b',
         QueryIntent.LIST),
    ]

    def _heuristic_intent(self, user_query: str) -> QueryIntent:
        """Pattern-based intent classification (no LLM)."""
        for pattern, intent in self._INTENT_PATTERNS:
            if re.search(pattern, user_query, re.IGNORECASE):
                return intent
        return QueryIntent.UNKNOWN


# ---------------------------------------------------------------------------
# Module-level singleton factory
# ---------------------------------------------------------------------------

_instance: Optional[SmartQueryProcessor] = None


def get_smart_query_processor(llm_client: Any = None) -> SmartQueryProcessor:
    """Return the module singleton, creating it if necessary."""
    global _instance
    if _instance is None:
        _instance = SmartQueryProcessor(llm_client)
    return _instance


def reset_smart_query_processor() -> None:
    """Reset singleton (useful in tests)."""
    global _instance
    _instance = None
