"""
Schema Intelligence Service - Semantic understanding of database schema.

This module provides intelligent schema reasoning through precomputed profiles
and semantic matching, improving table/column resolution accuracy.

Features:
    - Table profiles (business meaning, aliases, confidence keywords)
    - Column semantic profiles (type, meaning, synonyms, filterability)
    - Join graph (ranked relationships)
    - Query templates (common patterns)

Approach:
    - Precompute and cache schema intelligence
    - Use weighted scoring for disambiguation
    - Trigger clarification when confidence is low

Author: GitHub Copilot
Created: 2026-03-11
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TableRole(str, Enum):
    """Business role classification for tables."""
    MASTER_ENTITY = "master_entity"      # Main business entities (customers, employees)
    TRANSACTIONAL = "transactional"      # Event/transaction records (orders, payments)
    REFERENCE = "reference"              # Lookup tables (statuses, categories)
    JUNCTION = "junction"                # Many-to-many relationship tables
    SYSTEM = "system"                    # System/metadata tables
    UNKNOWN = "unknown"


@dataclass
class TableProfile:
    """Comprehensive profile of a database table."""
    table_name: str
    schema_name: Optional[str] = None
    
    # Business understanding
    business_meaning: str = ""
    role: TableRole = TableRole.UNKNOWN
    aliases: List[str] = field(default_factory=list)
    confidence_keywords: List[str] = field(default_factory=list)
    
    # Structure
    row_count_estimate: int = 0
    primary_key: Optional[str] = None
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)  # [{column, references_table, references_column}]
    
    # Important columns
    key_columns: List[str] = field(default_factory=list)
    display_columns: List[str] = field(default_factory=list)  # For human-readable output
    
    # Relationships
    join_targets: List[str] = field(default_factory=list)  # Tables this commonly joins with
    
    # Query patterns
    common_filters: List[str] = field(default_factory=list)  # Columns often filtered
    common_aggregations: List[str] = field(default_factory=list)  # Columns often aggregated
    
    # Metadata
    entity_type: Optional[str] = None  # personnel, financial, customer, etc.
    is_temporal: bool = False  # Has time-series data
    
    # Dynamic scoring features
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ColumnProfile:
    """Semantic profile of a database column."""
    column_name: str
    table_name: str
    
    # Data type
    data_type: str
    is_nullable: bool = True
    
    # Semantic understanding
    business_meaning: str = ""
    synonyms: List[str] = field(default_factory=list)
    
    # Query characteristics
    is_filterable: bool = True
    is_aggregatable: bool = False
    is_groupable: bool = False
    is_sortable: bool = True
    
    # Data characteristics
    is_high_cardinality: bool = False  # Many unique values
    is_categorical: bool = False
    is_numeric: bool = False
    is_temporal: bool = False
    
    # Visualization suitability
    suitable_for_charts: bool = False
    preferred_chart_types: List[str] = field(default_factory=list)
    
    # Sample values (for value grounding)
    sample_values: List[Any] = field(default_factory=list)
    value_distribution: Dict[str, int] = field(default_factory=dict)  # For categorical columns
    
    # Metadata
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JoinSpecification:
    """Specification of a table join with confidence score."""
    from_table: str
    to_table: str
    from_column: str
    to_column: str
    join_type: str = "INNER"  # INNER, LEFT, RIGHT, FULL
    
    # Business semantics
    relationship_type: str = "one_to_many"  # one_to_one, one_to_many, many_to_many
    is_mandatory: bool = False
    confidence: float = 1.0
    
    # Usage metadatafrequency_score: float = 0.5  # How often this join is used


@dataclass
class TableResolutionResult:
    """Result of table name resolution with confidence scoring."""
    resolved_table: str
    confidence: float
    reasoning: str
    alternatives: List[Tuple[str, float]] = field(default_factory=list)  # [(table, score)]
    needs_clarification: bool = False


class SchemaIntelligenceService:
    """
    Intelligent schema understanding and resolution.   
    Zero hardcoding approach:
    - Profiles are generated dynamically from database
    - Scoring weights are configurable
    - Rules are data-driven
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize schema intelligence service.
        
        Args:
            config: Optional configuration with weights, thresholds, patterns
        """
        self.config = config or self._default_config()
        self.table_profiles: Dict[str, TableProfile] = {}
        self.column_profiles: Dict[Tuple[str, str], ColumnProfile] = {}  # (table, column)
        self.join_graph: Dict[Tuple[str, str], JoinSpecification] = {}  # (from_table, to_table)
        
        self.initialized = False
        logger.info("[SCHEMA INTELLIGENCE] Service created")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration (tunable)."""
        return {
            # Resolution thresholds
            "high_confidence_threshold": 0.85,
            "clarification_threshold": 0.60,
            "ambiguity_gap_threshold": 0.15,  # If top 2 scores are within this, ask
            
            # Scoring weights for table resolution
            "table_resolution_weights": {
                "exact_name_match": 0.40,
                "alias_match": 0.35,
                "keyword_match": 0.15,
                "role_match": 0.10,
            },
            
            # Default table role keywords (can be learned from data)
            "role_keywords": {
                TableRole.MASTER_ENTITY: ["customer", "employee", "user", "client", "staff", "product"],
                TableRole.TRANSACTIONAL: ["order", "transaction", "payment", "sale", "purchase"],
                TableRole.REFERENCE: ["status", "type", "category", "lookup"],
            },
            
            # Join detection
            "max_join_depth": 2,  # Maximum number of joins for simple queries
        }
    
    async def initialize(self, db_session: Any, schema_name: Optional[str] = None) -> None:
        """
        Initialize by analyzing database schema via information_schema.

        Dynamically profiles all tables and columns — no hardcoding.
        Detects FKs, row counts, temporal columns, and generates aliases
        from naming conventions.
        """
        if self.initialized:
            logger.info("[SCHEMA INTELLIGENCE] Already initialized")
            return

        if schema_name is None:
            from app.config import settings as _settings
            schema_name = _settings.postgres_schema

        logger.info("[SCHEMA INTELLIGENCE] Starting schema profiling (DB introspection)...")
        try:
            await self._introspect_schema(db_session, schema_name)
        except Exception as exc:
            logger.warning(
                "[SCHEMA INTELLIGENCE] DB introspection failed (%s) — service will use fallback resolution",
                exc,
            )
        self.initialized = True
        logger.info(
            "[SCHEMA INTELLIGENCE] Initialization complete: %d tables profiled",
            len(self.table_profiles),
        )

    # ------------------------------------------------------------------
    # Internal: real DB introspection
    # ------------------------------------------------------------------

    async def _introspect_schema(self, db_session: Any, schema_name: str) -> None:
        """Query information_schema and pg_constraint to build table profiles."""
        from sqlalchemy import text

        # ── 1. List all base tables ────────────────────────────────────
        tables_result = await db_session.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = :schema
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            ),
            {"schema": schema_name},
        )
        table_names = [row[0] for row in tables_result.fetchall()]
        logger.info("[SCHEMA INTELLIGENCE] Found %d tables in schema '%s'", len(table_names), schema_name)

        # ── 2. Fetch all columns in one query ─────────────────────────
        cols_result = await db_session.execute(
            text(
                """
                SELECT table_name, column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = :schema
                ORDER BY table_name, ordinal_position
                """
            ),
            {"schema": schema_name},
        )
        cols_by_table: Dict[str, List[Dict]] = {}
        for row in cols_result.fetchall():
            tbl, col, dtype, nullable = row
            cols_by_table.setdefault(tbl, []).append(
                {"name": col, "type": dtype, "nullable": nullable == "YES"}
            )

        # ── 3. Fetch primary keys ──────────────────────────────────────
        pk_result = await db_session.execute(
            text(
                """
                SELECT tc.table_name, kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema    = kcu.table_schema
                WHERE tc.table_schema = :schema
                  AND tc.constraint_type = 'PRIMARY KEY'
                """
            ),
            {"schema": schema_name},
        )
        pk_by_table: Dict[str, str] = {}
        for row in pk_result.fetchall():
            tbl, col = row
            pk_by_table.setdefault(tbl, col)  # keep first PK column

        # ── 4. Fetch foreign keys ──────────────────────────────────────
        fk_result = await db_session.execute(
            text(
                """
                SELECT
                    kcu.table_name,
                    kcu.column_name,
                    ccu.table_name  AS foreign_table,
                    ccu.column_name AS foreign_column
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema    = kcu.table_schema
                JOIN information_schema.constraint_column_usage ccu
                  ON ccu.constraint_name = tc.constraint_name
                 AND ccu.table_schema    = tc.table_schema
                WHERE tc.table_schema = :schema
                  AND tc.constraint_type = 'FOREIGN KEY'
                """
            ),
            {"schema": schema_name},
        )
        fk_by_table: Dict[str, List[Dict]] = {}
        join_targets_by_table: Dict[str, List[str]] = {}
        for row in fk_result.fetchall():
            tbl, col, ftbl, fcol = row
            fk_by_table.setdefault(tbl, []).append(
                {"column": col, "references_table": ftbl, "references_column": fcol}
            )
            join_targets_by_table.setdefault(tbl, [])
            if ftbl not in join_targets_by_table[tbl]:
                join_targets_by_table[tbl].append(ftbl)
            # reverse direction (the referenced table can also join back)
            join_targets_by_table.setdefault(ftbl, [])
            if tbl not in join_targets_by_table[ftbl]:
                join_targets_by_table[ftbl].append(tbl)

        # ── 5. Row counts (best-effort) ────────────────────────────────
        row_counts: Dict[str, int] = {}
        for tbl in table_names:
            try:
                cnt = await db_session.execute(
                    text(f"SELECT COUNT(*) FROM {schema_name}.{tbl}")  # noqa: S608
                )
                row_counts[tbl] = cnt.scalar() or 0
            except Exception:
                row_counts[tbl] = 0

        # ── 6. Build TableProfile for each table ──────────────────────
        role_kw = self.config.get("role_keywords", {})
        for tbl in table_names:
            cols = cols_by_table.get(tbl, [])
            col_names = [c["name"] for c in cols]

            # Classify role from naming conventions
            role = self._infer_role(tbl, role_kw)

            # Identify temporal columns
            temporal_types = {"date", "timestamp", "timestamp without time zone", "timestamp with time zone"}
            is_temporal = any(c["type"] in temporal_types for c in cols)

            # Key / display columns heuristic
            pk = pk_by_table.get(tbl)
            key_cols = ([pk] if pk else []) + [
                c["name"] for c in cols
                if c["name"] != pk and any(
                    suffix in c["name"].lower()
                    for suffix in ("_id", "_code", "name", "number")
                )
            ]
            display_cols = [
                c["name"] for c in cols
                if any(
                    token in c["name"].lower()
                    for token in ("name", "title", "label", "description", "city", "status", "type")
                )
            ][:6]

            # Common filters: low-cardinality likely candidates
            common_filters = [
                c["name"] for c in cols
                if any(
                    token in c["name"].lower()
                    for token in ("status", "type", "gender", "city", "region", "branch",
                                  "category", "level", "flag", "active")
                )
            ][:8]

            # Common aggregations: numeric columns
            numeric_types = {
                "integer", "bigint", "smallint", "numeric", "decimal",
                "real", "double precision", "money",
            }
            common_aggs = [
                c["name"] for c in cols if c["type"] in numeric_types
            ][:6]

            # Aliases from naming conventions
            aliases = self._generate_aliases(tbl)
            keywords = self._generate_keywords(tbl, col_names)

            profile = TableProfile(
                table_name=tbl,
                schema_name=schema_name,
                business_meaning=self._infer_business_meaning(tbl, role),
                role=role,
                aliases=aliases,
                confidence_keywords=keywords,
                row_count_estimate=row_counts.get(tbl, 0),
                primary_key=pk,
                foreign_keys=fk_by_table.get(tbl, []),
                key_columns=list(dict.fromkeys(key_cols))[:8],  # deduplicate, preserve order
                display_columns=display_cols,
                join_targets=join_targets_by_table.get(tbl, []),
                common_filters=common_filters,
                common_aggregations=common_aggs,
                is_temporal=is_temporal,
            )
            self.table_profiles[tbl] = profile
            logger.debug("[SCHEMA INTELLIGENCE] Profiled table '%s' (%d cols)", tbl, len(cols))

    # ------------------------------------------------------------------
    # Helpers for dynamic profiling
    # ------------------------------------------------------------------

    def _infer_role(
        self,
        table_name: str,
        role_kw: Dict,
    ) -> TableRole:
        """Infer table role from name + columns."""
        tbl_lower = table_name.lower()
        # Junction tables: name contains two nouns or ends in _map/_rel
        if re.search(r"_(map|rel|link|xref|assoc|junction)$", tbl_lower):
            return TableRole.JUNCTION
        # Reference/lookup tables: small, name has type/status/category
        if re.search(r"(type|status|category|lookup|ref|code|enum|master)s?$", tbl_lower):
            return TableRole.REFERENCE
        # Transactional: verbs/events
        if re.search(r"(order|transaction|payment|sale|purchase|log|event|audit|history|record)s?$", tbl_lower):
            return TableRole.TRANSACTIONAL
        # Master entity: nouns
        entity_kws = role_kw.get(TableRole.MASTER_ENTITY, ["customer", "employee", "user", "product"])
        if any(kw in tbl_lower for kw in entity_kws):
            return TableRole.MASTER_ENTITY
        # System
        if tbl_lower.startswith(("sys_", "cfg_", "config_", "setting", "migration")):
            return TableRole.SYSTEM
        return TableRole.UNKNOWN

    def _generate_aliases(self, table_name: str) -> List[str]:
        """Generate plural/singular/common aliases from table name."""
        aliases: List[str] = [table_name]
        # snake_case parts
        parts = table_name.split("_")
        noun = parts[-1]  # last part is usually the noun
        aliases.append(noun)
        # singular / plural
        if noun.endswith("ies"):
            aliases.append(noun[:-3] + "y")
        elif noun.endswith("ses"):
            aliases.append(noun[:-2])
        elif noun.endswith("s"):
            aliases.append(noun[:-1])
        else:
            aliases.append(noun + "s")
        # space-separated variant
        aliases.append(table_name.replace("_", " "))
        return list(dict.fromkeys(a for a in aliases if a))

    def _generate_keywords(self, table_name: str, col_names: List[str]) -> List[str]:
        """Generate confidence keywords from table and column names."""
        keywords: List[str] = []
        # Table name parts
        keywords.extend(p for p in table_name.split("_") if len(p) > 2)
        # Prominent column name parts (id/code stripped)
        for col in col_names[:10]:
            for part in col.split("_"):
                if len(part) > 3 and part not in ("null", "none", "true", "false"):
                    keywords.append(part)
        return list(dict.fromkeys(keywords))[:12]

    def _infer_business_meaning(self, table_name: str, role: TableRole) -> str:
        """Generate a human-readable business meaning from table name."""
        readable = table_name.replace("_", " ").title()
        role_suffix = {
            TableRole.MASTER_ENTITY: "records",
            TableRole.TRANSACTIONAL: "transactions",
            TableRole.REFERENCE: "reference data",
            TableRole.JUNCTION: "associations",
            TableRole.SYSTEM: "system configuration",
        }.get(role, "data")
        return f"{readable} {role_suffix}"
    
    async def resolve_table(
        self,
        query_text: str,
        intent: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TableResolutionResult:
        """
        Resolve table name from natural language query.
        
        Uses weighted scoring to match query against table profiles.
        
        Args:
            query_text: User query text
            intent: Query intent (list, count, etc.) if known
            context: Additional context (conversation state, etc.)
        
        Returns:
            TableResolutionResult with resolved table and confidence
        """
        if not self.initialized:
            logger.warning("[SCHEMA INTELLIGENCE] Not initialized, using fallback resolution")
            return self._fallback_resolution(query_text)
        
        logger.info(f"[SCHEMA INTELLIGENCE] Resolving table for query: '{query_text}'")
        
        query_lower = query_text.lower()
        weights = self.config["table_resolution_weights"]
        
        # Score each table
        scores: List[Tuple[str, float, Dict[str, float]]] = []
        
        for table_name, profile in self.table_profiles.items():
            score = 0.0
            score_breakdown = {}
            
            # Feature 1: Exact table name match
            if table_name in query_lower:
                score += weights["exact_name_match"]
                score_breakdown["exact_name"] = weights["exact_name_match"]
            
            # Feature 2: Alias match
            alias_match_score = 0.0
            for alias in profile.aliases:
                if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
                    alias_match_score = weights["alias_match"]
                    break
            score += alias_match_score
            score_breakdown["alias"] = alias_match_score
            
            # Feature 3: Keyword match
            keyword_matches = sum(
                1 for kw in profile.confidence_keywords
                if re.search(r'\b' + re.escape(kw) + r'\b', query_lower)
            )
            keyword_score = min(keyword_matches * 0.05, weights["keyword_match"])
            score += keyword_score
            score_breakdown["keywords"] = keyword_score
            
            # Feature 4: Role match (based on intent)
            role_score = 0.0
            if intent == "list" and profile.role == TableRole.MASTER_ENTITY:
                role_score = weights["role_match"]
            score += role_score
            score_breakdown["role"] = role_score
            
            scores.append((table_name, score, score_breakdown))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if not scores or scores[0][1] == 0:
            return TableResolutionResult(
                resolved_table="",
                confidence=0.0,
                reasoning="No matching tables found",
                needs_clarification=True,
            )
        
        top_table, top_score, top_breakdown = scores[0]
        thresholds = self.config
        
        # Check if we need clarification
        needs_clarification = False
        
        # Case 1: Low confidence
        if top_score < thresholds["clarification_threshold"]:
            needs_clarification = True
        
        # Case 2: Ambiguous (multiple tables with similar scores)
        if len(scores) > 1:
            second_score = scores[1][1]
            gap = top_score - second_score
            if gap < thresholds["ambiguity_gap_threshold"]:
                needs_clarification = True
        
        # Build result
        alternatives = [(table, score) for table, score, _ in scores[1:4]]
        
        reasoning = f"Matched '{top_table}' with score {top_score:.2f}"
        if top_breakdown:
            breakdown_str = ", ".join(f"{k}={v:.2f}" for k, v in top_breakdown.items() if v > 0)
            reasoning += f" ({breakdown_str})"
        
        result = TableResolutionResult(
            resolved_table=top_table,
            confidence=top_score,
            reasoning=reasoning,
            alternatives=alternatives,
            needs_clarification=needs_clarification,
        )
        
        logger.info(
            f"[SCHEMA INTELLIGENCE] Resolved to '{top_table}' "
            f"(confidence={top_score:.2f}, clarify={needs_clarification})"
        )
        
        return result
    
    def _fallback_resolution(self, query_text: str = "") -> TableResolutionResult:  # noqa: ARG002
        """Fallback resolution when not initialized — no hardcoded table names."""
        # When not initialized, surface low confidence so callers can fall through
        # to the schema catalog or ask the user for clarification.
        return TableResolutionResult(
            resolved_table="",
            confidence=0.0,
            reasoning="Schema intelligence not yet initialized — cannot resolve table",
            needs_clarification=True,
        )
    
    def get_table_profile(self, table_name: str) -> Optional[TableProfile]:
        """Get profile for a specific table."""
        return self.table_profiles.get(table_name)
    
    def suggest_joins(
        self,
        from_table: str,
        required_columns: Optional[List[str]] = None,
    ) -> List[JoinSpecification]:
        """
        Suggest appropriate joins for a table.
        
        Returns ranked list of suggested joins based on:
        - FK relationships
        - Common usage patterns
        - Required columns
        """
        suggestions = []
        
        profile = self.get_table_profile(from_table)
        if not profile:
            return suggestions
        
        # Use join targets from profile
        for to_table in profile.join_targets:
            # Find FK relationship
            fk_spec = None
            for fk in profile.foreign_keys:
                if fk.get("references_table") == to_table:
                    fk_spec = fk
                    break
            
            if fk_spec:
                join_spec = JoinSpecification(
                    from_table=from_table,
                    to_table=to_table,
                    from_column=fk_spec["column"],
                    to_column=fk_spec["references_column"],
                    join_type="LEFT",  # Default to LEFT to preserve main table rows
                    confidence=0.9,
                )
                suggestions.append(join_spec)
        
        logger.info(f"[SCHEMA INTELLIGENCE] Suggested {len(suggestions)} joins for '{from_table}'")
        return suggestions


# Singleton instance
_schema_intelligence: Optional[SchemaIntelligenceService] = None


def get_schema_intelligence(config: Optional[Dict[str, Any]] = None) -> SchemaIntelligenceService:
    """Get or create singleton schema intelligence service instance."""
    global _schema_intelligence
    if _schema_intelligence is None:
        _schema_intelligence = SchemaIntelligenceService(config)
    return _schema_intelligence
