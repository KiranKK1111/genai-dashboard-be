"""
Semantic Concept Discovery - Maps semantic user concepts to database schema.

This module provides dynamic discovery of concept-to-schema mappings through:
1. Column name pattern inference (birth, gender, date patterns)
2. Schema introspection (actual column types and names)
3. LLM-assisted semantic similarity (optional enhancement)

The goal is ZERO HARDCODING - all mappings are discovered from schema at runtime.
Hand-authored heuristics are clearly isolated here for maintainability.

ARCHITECTURE:
    User Concept → SemanticConceptDiscovery → Column/Table Mappings → QueryPlan

This keeps the SQL generation layer (plan_first_sql_generator, query_plan_compiler)
free from semantic inference logic.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)


class ConceptCategory(str, Enum):
    """Categories of semantic concepts."""
    TEMPORAL = "temporal"      # Dates, times, periods
    IDENTITY = "identity"      # Gender, age, names
    ENTITY = "entity"          # Tables/business objects
    MEASURE = "measure"        # Amounts, counts, quantities
    STATUS = "status"          # Active/inactive, approved/pending
    RELATIONSHIP = "relationship"  # Foreign keys, parent-child


@dataclass
class DiscoveredConcept:
    """A discovered concept-to-schema mapping."""
    concept: str              # User concept (e.g., "birth_month")
    table: str                # Target table
    column: str               # Target column
    category: ConceptCategory  # What type of concept
    confidence: float = 1.0    # Discovery confidence (0.0-1.0)
    transform: Optional[str] = None  # SQL transformation (e.g., "EXTRACT(MONTH FROM {col})")
    source: str = "schema"     # How it was discovered: "schema", "heuristic", "llm"


@dataclass
class ConceptDiscoveryCache:
    """Cache for discovered concepts per schema."""
    schema_name: str
    concepts: Dict[str, List[DiscoveredConcept]] = field(default_factory=dict)
    tables: List[str] = field(default_factory=list)
    discovered: bool = False


class SemanticConceptDiscovery:
    """
    Discovers semantic concept mappings from database schema.
    
    This class encapsulates all heuristic and pattern-based logic for
    mapping user concepts to actual database columns/tables.
    
    Heuristics are CLEARLY DOCUMENTED and ISOLATED here, not scattered
    throughout the SQL generation code.
    """
    
    # =========================================================================
    # HEURISTIC PATTERNS (documented, isolated, maintainable)
    # =========================================================================
    
    # Temporal column patterns - maps concept to column name patterns
    TEMPORAL_PATTERNS = {
        "birth_month": (r"dob|birth.*date|birthdate|date.*birth", "EXTRACT(MONTH FROM {col})"),
        "birth_year": (r"dob|birth.*date|birthdate|date.*birth", "EXTRACT(YEAR FROM {col})"),
        "birth_date": (r"dob|birth.*date|birthdate|date.*birth", None),
        "created_date": (r"created|created_at|create_time|creation", None),
        "updated_date": (r"updated|updated_at|update_time|modified", None),
        "transaction_date": (r"txn.*date|transaction.*date|date", None),
    }
    
    # Identity column patterns
    IDENTITY_PATTERNS = {
        "gender": (r"gender|sex", None),
        "age": (r"age|birth", "EXTRACT(YEAR FROM AGE({col}))"),  # If birth date column
        "name": (r"name|full_name|first_name|last_name", None),
    }
    
    # Status column patterns
    STATUS_PATTERNS = {
        "active": (r"active|is_active|status", None),
        "verified": (r"verified|is_verified|verification", None),
        "approved": (r"approved|is_approved|approval", None),
    }

    # Runtime cache: entity → resolved table name (populated by LLM on first miss).
    # Cleared on process restart; no hardcoded entries.
    _entity_resolution_cache: Dict[str, Optional[str]] = {}
    
    def __init__(self, db: AsyncSession, schema_name: Optional[str] = None):
        """Initialize concept discovery.

        Args:
            db: Database session for schema introspection
            schema_name: Schema to discover from (defaults to settings.postgres_schema)
        """
        from app.config import settings as _settings
        self.db = db
        self.schema_name = schema_name or _settings.postgres_schema
        self._cache = ConceptDiscoveryCache(schema_name=self.schema_name)
        
        logger.info(f"[CONCEPT_DISCOVERY] Initialized for schema: {schema_name}")
    
    async def discover_all(self) -> Dict[str, List[DiscoveredConcept]]:
        """Discover all concept mappings from schema.
        
        Returns:
            Dict mapping concept names to list of possible DiscoveredConcept matches
        """
        if self._cache.discovered:
            return self._cache.concepts
        
        # Get all tables
        tables = await self._get_tables()
        self._cache.tables = tables
        
        concepts: Dict[str, List[DiscoveredConcept]] = {}
        
        # Discover entity concepts (table mappings)
        entity_concepts = await self._discover_entity_concepts(tables)
        concepts.update(entity_concepts)
        
        # Discover column concepts for each table
        for table in tables:
            columns = await self._get_columns(table)
            
            # Temporal concepts
            temporal = self._discover_temporal_concepts(table, columns)
            for concept, matches in temporal.items():
                if concept not in concepts:
                    concepts[concept] = []
                concepts[concept].extend(matches)
            
            # Identity concepts
            identity = self._discover_identity_concepts(table, columns)
            for concept, matches in identity.items():
                if concept not in concepts:
                    concepts[concept] = []
                concepts[concept].extend(matches)
            
            # Status concepts
            status = self._discover_status_concepts(table, columns)
            for concept, matches in status.items():
                if concept not in concepts:
                    concepts[concept] = []
                concepts[concept].extend(matches)
        
        self._cache.concepts = concepts
        self._cache.discovered = True
        
        logger.info(f"[CONCEPT_DISCOVERY] Discovered {len(concepts)} concepts from schema")
        return concepts
    
    async def get_concept_mapping(self, concept: str) -> Optional[DiscoveredConcept]:
        """Get the best mapping for a specific concept.
        
        Args:
            concept: Semantic concept to look up
            
        Returns:
            Best matching DiscoveredConcept or None
        """
        all_concepts = await self.discover_all()
        
        matches = all_concepts.get(concept.lower(), [])
        if not matches:
            # Try fuzzy matching
            for key, values in all_concepts.items():
                if concept.lower() in key or key in concept.lower():
                    matches.extend(values)
        
        if not matches:
            return None
        
        # Return highest confidence match
        return max(matches, key=lambda x: x.confidence)
    
    async def get_table_for_entity(self, entity: str) -> Optional[str]:
        """Get the table name for a semantic entity.

        Resolution order:
        1. Direct match against actual table names
        2. Stem/plural fuzzy match (customers ↔ customer)
        3. LLM semantic match — asks the LLM which table best corresponds to
           the entity given the real list of tables.  Result is cached in-process
           so the LLM is only called once per unique unknown entity per run.

        Args:
            entity: Entity concept (e.g., "customers", "clients")

        Returns:
            Table name or None
        """
        await self.discover_all()

        entity_lower = entity.lower()
        entity_stem = entity_lower.rstrip('s')  # Normalize: customers -> customer

        # 1. Direct match
        if entity in self._cache.tables:
            return entity
        if entity_lower in self._cache.tables:
            return entity_lower

        # 2. Stem/plural fuzzy match
        for table in self._cache.tables:
            if entity_stem == table.rstrip('s').lower():
                return table

        # 3. LLM-based semantic resolution (no hardcoded synonyms)
        cache_key = entity_lower
        if cache_key in SemanticConceptDiscovery._entity_resolution_cache:
            return SemanticConceptDiscovery._entity_resolution_cache[cache_key]

        resolved = await self._llm_resolve_entity(entity, self._cache.tables)
        SemanticConceptDiscovery._entity_resolution_cache[cache_key] = resolved
        if resolved:
            logger.info(
                "[CONCEPT_DISCOVERY] LLM resolved entity '%s' → table '%s'",
                entity, resolved,
            )
        return resolved

    async def _llm_resolve_entity(self, entity: str, tables: List[str]) -> Optional[str]:
        """Ask the LLM which table best matches a user entity term.

        The LLM must return ONLY a table name from the provided list, or the
        string "none" if no table matches.  The response is validated against
        the actual table list so hallucinated names are rejected.
        """
        from app.llm import call_llm

        if not tables:
            return None

        table_list = ", ".join(sorted(tables))
        prompt = (
            f"You are a database schema expert. A user referred to '{entity}' in a query.\n"
            f"Available tables: {table_list}\n\n"
            f"Which single table does '{entity}' most likely refer to?\n"
            f"Reply with ONLY the exact table name from the list above, or 'none' if no table fits.\n"
            f"Do not explain. Do not add punctuation."
        )

        try:
            raw = await call_llm(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0,
            )
            answer = str(raw).strip().lower().strip("'\".,;* ")
            logger.info("[CONCEPT_DISCOVERY] LLM returned '%s' for entity '%s'", answer, entity)
            if answer in tables:
                return answer
            # Case-insensitive match
            for t in tables:
                if t.lower() == answer:
                    return t
            # Partial match: table name contained in answer (e.g. "customers table" → "customers")
            for t in tables:
                if t.lower() in answer:
                    return t
            logger.warning(
                "[CONCEPT_DISCOVERY] LLM answer '%s' not in tables %s", answer, tables
            )
        except Exception as exc:
            logger.warning("[CONCEPT_DISCOVERY] LLM entity resolution failed: %s", exc)

        return None
    
    # =========================================================================
    # PRIVATE DISCOVERY METHODS
    # =========================================================================
    
    async def _get_tables(self) -> List[str]:
        """Get all tables in schema."""
        result = await self.db.execute(
            text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = :schema_name
                ORDER BY table_name
            """),
            {"schema_name": self.schema_name},
        )
        return [row[0] for row in result]

    async def _get_columns(self, table: str) -> Dict[str, str]:
        """Get columns and their types for a table."""
        result = await self.db.execute(
            text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = :schema_name AND table_name = :table_name
                ORDER BY column_name
            """),
            {"schema_name": self.schema_name, "table_name": table},
        )
        return {row[0]: row[1] for row in result}
    
    async def _discover_entity_concepts(self, tables: List[str]) -> Dict[str, List[DiscoveredConcept]]:
        """Discover entity (table) concepts."""
        concepts: Dict[str, List[DiscoveredConcept]] = {}
        
        for table in tables:
            # Add table as its own concept
            concepts[table.lower()] = [DiscoveredConcept(
                concept=table.lower(),
                table=table,
                column="*",
                category=ConceptCategory.ENTITY,
                confidence=1.0,
                source="schema"
            )]
            
            # Add singular form
            singular = table.lower().rstrip('s')
            if singular != table.lower():
                if singular not in concepts:
                    concepts[singular] = []
                concepts[singular].append(DiscoveredConcept(
                    concept=singular,
                    table=table,
                    column="*",
                    category=ConceptCategory.ENTITY,
                    confidence=0.9,
                    source="heuristic"
                ))
        
        return concepts
    
    def _discover_temporal_concepts(
        self, table: str, columns: Dict[str, str]
    ) -> Dict[str, List[DiscoveredConcept]]:
        """Discover temporal concepts from column patterns."""
        concepts: Dict[str, List[DiscoveredConcept]] = {}
        
        for concept, (pattern, transform) in self.TEMPORAL_PATTERNS.items():
            for col_name, col_type in columns.items():
                if re.search(pattern, col_name, re.IGNORECASE):
                    if concept not in concepts:
                        concepts[concept] = []
                    concepts[concept].append(DiscoveredConcept(
                        concept=concept,
                        table=table,
                        column=col_name,
                        category=ConceptCategory.TEMPORAL,
                        confidence=0.85,
                        transform=transform,
                        source="heuristic"
                    ))
        
        return concepts
    
    def _discover_identity_concepts(
        self, table: str, columns: Dict[str, str]
    ) -> Dict[str, List[DiscoveredConcept]]:
        """Discover identity concepts from column patterns."""
        concepts: Dict[str, List[DiscoveredConcept]] = {}
        
        for concept, (pattern, transform) in self.IDENTITY_PATTERNS.items():
            for col_name, col_type in columns.items():
                if re.search(pattern, col_name, re.IGNORECASE):
                    if concept not in concepts:
                        concepts[concept] = []
                    concepts[concept].append(DiscoveredConcept(
                        concept=concept,
                        table=table,
                        column=col_name,
                        category=ConceptCategory.IDENTITY,
                        confidence=0.85,
                        transform=transform,
                        source="heuristic"
                    ))
        
        return concepts
    
    def _discover_status_concepts(
        self, table: str, columns: Dict[str, str]
    ) -> Dict[str, List[DiscoveredConcept]]:
        """Discover status concepts from column patterns."""
        concepts: Dict[str, List[DiscoveredConcept]] = {}
        
        for concept, (pattern, transform) in self.STATUS_PATTERNS.items():
            for col_name, col_type in columns.items():
                if re.search(pattern, col_name, re.IGNORECASE):
                    if concept not in concepts:
                        concepts[concept] = []
                    concepts[concept].append(DiscoveredConcept(
                        concept=concept,
                        table=table,
                        column=col_name,
                        category=ConceptCategory.STATUS,
                        confidence=0.80,
                        transform=transform,
                        source="heuristic"
                    ))
        
        return concepts


# ============================================================================
# Convenience functions
# ============================================================================

_discovery_cache: Dict[str, SemanticConceptDiscovery] = {}


async def get_concept_discovery(
    db: AsyncSession,
    schema_name: Optional[str] = None,
) -> SemanticConceptDiscovery:
    """Get or create concept discovery instance.

    Cached per schema for efficiency.
    """
    from app.config import settings as _settings
    resolved = schema_name or _settings.postgres_schema
    cache_key = resolved

    if cache_key not in _discovery_cache:
        _discovery_cache[cache_key] = SemanticConceptDiscovery(db, resolved)

    return _discovery_cache[cache_key]


def clear_discovery_cache():
    """Clear the discovery cache (useful for testing or schema changes)."""
    global _discovery_cache
    _discovery_cache = {}
