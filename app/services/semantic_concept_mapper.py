"""
Semantic Concept Mapper - Maps semantic concepts to actual columns WITHOUT hardcoding.

Replaces hardcoded concept lists like approval keywords with LLM-based mapping.
Works for any concept: approval, status, verification, engagement, etc.

Enhanced with Business Glossary integration for persistent term learning.

Used to find columns representing semantic concepts without hardcoding.
"""

from __future__ import annotations

import logging
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .. import llm
from .business_glossary import get_business_glossary, DomainType

logger = logging.getLogger(__name__)


@dataclass
class ConceptMatch:
    """A column matching a semantic concept."""
    table_name: str
    column_name: str
    reason: str
    confidence: float


class SemanticConceptMapper:
    """
    Maps semantic concepts to actual columns WITHOUT hardcoding.
    
    Enhanced with Business Glossary integration:
    - Checks glossary for known term mappings first
    - Falls back to LLM-based semantic analysis
    - Learns from successful matches to improve over time
    """
    
    def __init__(self, cache_size: int = 50, domain: DomainType = DomainType.GENERIC):
        self.cache_size = cache_size
        self.domain = domain
        self._concept_cache: Dict[str, List[ConceptMatch]] = {}
        self._glossary = get_business_glossary(domain=domain)
        
        logger.info(f"[CONCEPT_MAPPER] Initialized with domain: {domain.value}")
    
    async def find_columns_for_concept(
        self,
        concept: str,
        schema_context: Dict[str, List[str]],
        boolean_columns: Optional[Dict[str, List[str]]] = None,
        enum_columns: Optional[Dict[str, List[Tuple[str, List[str]]]]] = None
    ) -> List[ConceptMatch]:
        """
        Find ALL columns matching a semantic concept.
        
        Process:
        1. Check Business Glossary for known mappings (FAST)
        2. If not found, use LLM-based semantic analysis (COMPREHENSIVE)
        3. Learn from successful matches for future queries
        
        Examples:
        - concept="approval" → 
          Returns: [
            ConceptMatch("table1", "is_verified", "verification status", 0.95),
            ConceptMatch("table2", "is_approved", "approval indicator", 0.92),
            ConceptMatch("table3", "compliance_check", "compliance verification", 0.88)
          ]
        
        - concept="status" →
          Returns: [
            ConceptMatch("records", "record_status", "record state", 0.93),
            ConceptMatch("entities", "entity_status", "entity condition", 0.91),
            ConceptMatch("items", "item_status", "item state", 0.90)
          ]
        
        Args:
            concept: Semantic concept to search for (approval, status, verification, etc.)
            schema_context: Dict of available tables and columns
            boolean_columns: Optional dict of boolean columns per table (helps with approval detection)
            enum_columns: Optional dict of enum columns per table (helps with status detection)
        
        Returns:
            List of ConceptMatch ordered by confidence (highest first)
        """
        
        cache_key = concept.lower()
        if cache_key in self._concept_cache:
            logger.debug(f"[CONCEPT_MAPPER] Cache hit for concept: {concept}")
            return self._concept_cache[cache_key]
        
        logger.info(f"[CONCEPT_MAPPER] Finding columns for concept: {concept}")
        
        # STEP 1: Check Business Glossary first (fast path)
        glossary_matches = self._check_glossary_for_concept(
            concept, schema_context
        )
        
        if glossary_matches:
            logger.info(
                f"[CONCEPT_MAPPER] Found {len(glossary_matches)} matches in glossary for '{concept}'"
            )
            # Still cache and return
            if len(self._concept_cache) >= self.cache_size:
                self._concept_cache.pop(next(iter(self._concept_cache)))
            self._concept_cache[cache_key] = glossary_matches
            return glossary_matches
        
        # STEP 2: Fall back to LLM-based semantic analysis
        logger.debug(f"[CONCEPT_MAPPER] No glossary match, using LLM for: {concept}")
        
        # Build context string for LLM
        schema_desc = json.dumps(schema_context, indent=2)
        
        bool_cols_desc = ""
        if boolean_columns:
            bool_cols_desc = f"\nBoolean columns (likely approval/status/verification): {json.dumps(boolean_columns, indent=2)}"
        
        enum_cols_desc = ""
        if enum_columns:
            enum_cols_desc = f"\nEnum columns with values:\n"
            for table, cols in enum_columns.items():
                for col, values in cols:
                    enum_cols_desc += f"  {table}.{col}: {values}\n"
        
        prompt = f"""
You are a database schema analyzer. Find ALL columns that represent this semantic concept:

Concept: "{concept}"

Available schema:
{schema_desc}
{bool_cols_desc}
{enum_cols_desc}

Find columns that represent this concept. Use semantic understanding, not just keywords.

Guidelines:
1. Look at column NAMES semantically
2. Use TYPES information (boolean columns are good for yes/no concepts, enum columns for status)
3. Think about what each column REPRESENTS, not just its name
4. Be comprehensive - find ALL relevant columns
5. Order by confidence (highest first)

Examples for concept="approval":
- is_verified (boolean) → strong match, explicit verification
- is_approved (boolean) → strong match, explicit approval
- compliance_check (boolean) → strong match, compliance verification
- verification_status (enum) → very strong match
- approval_date (date) → weak match (indicates something was approved, but not direct approval status)

Respond with ONLY valid JSON (no markdown):
{{
  "matches": [
    {{
      "table": "table_name",
      "column": "column_name",
      "reason": "Why this column represents the concept",
      "confidence": 0.95
    }},
    {{...}}
  ],
  "reasoning": "Overall analysis of how concept appears in this schema"
}}
"""
        
        try:
            response = await llm.call_llm([
                {
                    "role": "system",
                    "content": "You are a database schema analyzer. Respond with ONLY valid JSON. Be comprehensive."
                },
                {"role": "user", "content": prompt}
            ], max_tokens=2000, temperature=0.3)
            
            response_text = str(response)
            result = json.loads(response_text)
            
            matches = []
            for match_data in result.get("matches", []):
                match = ConceptMatch(
                    table_name=match_data.get("table", ""),
                    column_name=match_data.get("column", ""),
                    reason=match_data.get("reason", ""),
                    confidence=float(match_data.get("confidence", 0.5))
                )
                if match.table_name and match.column_name:
                    matches.append(match)
            
            # Sort by confidence (highest first)
            matches.sort(key=lambda m: m.confidence, reverse=True)
            
            # STEP 3: Learn from high-confidence matches for future queries
            for match in matches:
                if match.confidence >= 0.85:
                    self._glossary.learn_from_query(
                        user_term=concept,
                        resolved_table=match.table_name,
                        resolved_column=match.column_name,
                        confidence=match.confidence
                    )
            
            # Cache result
            if len(self._concept_cache) >= self.cache_size:
                # Remove oldest entry (naive approach)
                self._concept_cache.pop(next(iter(self._concept_cache)))
            
            self._concept_cache[cache_key] = matches
            
            logger.info(
                f"[CONCEPT_MAPPER] Found {len(matches)} columns for concept '{concept}': "
                f"{[(m.table_name, m.column_name, m.confidence) for m in matches[:3]]}"
            )
            
            return matches
        
        except json.JSONDecodeError as e:
            logger.error(f"[CONCEPT_MAPPER] Failed to parse LLM response for concept '{concept}': {e}")
            return []
        
        except Exception as e:
            logger.error(f"[CONCEPT_MAPPER] Error finding columns for concept '{concept}': {e}", exc_info=True)
            return []
    
    def _check_glossary_for_concept(
        self,
        concept: str,
        schema_context: Dict[str, List[str]]
    ) -> List[ConceptMatch]:
        """
        Check Business Glossary for known mappings.
        
        Returns matches if found, empty list otherwise.
        """
        # Get list of tables in query context
        tables_in_query = list(schema_context.keys())
        
        # Try to resolve term using glossary
        mapping = self._glossary.resolve_term(
            user_term=concept,
            tables_in_query=tables_in_query
        )
        
        if not mapping:
            return []
        
        # Convert glossary mapping to ConceptMatch
        matches = []
        
        # Primary column mapping
        if mapping.primary_table and mapping.primary_column:
            # Check if this table/column exists in schema context
            if (mapping.primary_table in schema_context and 
                mapping.primary_column in schema_context[mapping.primary_table]):
                matches.append(ConceptMatch(
                    table_name=mapping.primary_table,
                    column_name=mapping.primary_column,
                    reason=f"Business glossary mapping (domain: {self.domain.value})",
                    confidence=mapping.confidence
                ))
        
        # Related columns
        for related_col in mapping.related_columns:
            # Try to find which table this column belongs to
            for table, columns in schema_context.items():
                if related_col in columns:
                    # Avoid duplicates
                    if not any(m.table_name == table and m.column_name == related_col for m in matches):
                        matches.append(ConceptMatch(
                            table_name=table,
                            column_name=related_col,
                            reason=f"Business glossary related mapping",
                            confidence=mapping.confidence * 0.9  # Slightly lower confidence
                        ))
        
        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        return matches
    
    def learn_successful_mapping(
        self,
        concept: str,
        table_name: str,
        column_name: str,
        confidence: float = 0.9
    ):
        """
        Learn from a successful concept→column mapping.
        
        Call this when a query successfully resolves a concept to a specific column.
        The mapping will be stored in the Business Glossary for future use.
        
        Args:
            concept: The semantic concept that was searched for
            table_name: The table containing the resolved column
            column_name: The column that represented the concept
            confidence: Confidence in this mapping (0-1)
        """
        self._glossary.learn_from_query(
            user_term=concept,
            resolved_table=table_name,
            resolved_column=column_name,
            confidence=confidence
        )
        
        logger.info(
            f"[CONCEPT_MAPPER] Learned mapping: {concept} → {table_name}.{column_name}"
        )
    
    async def find_columns_for_concept_in_table(
        self,
        concept: str,
        table_name: str,
        columns: List[str],
        boolean_columns: Optional[List[str]] = None,
        enum_columns: Optional[List[Tuple[str, List[str]]]] = None
    ) -> List[ConceptMatch]:
        """
        Find columns in a SPECIFIC table matching a semantic concept.
        
        Faster than find_columns_for_concept when you know which table to search in.
        
        Example:
        find_columns_for_concept_in_table(
            "approval",
            "users",
            ["id", "name", "is_verified", "is_approved", "status"]
        )
        Returns: [
            ConceptMatch("users", "is_verified", "...", 0.95),
            ConceptMatch("users", "is_approved", "...", 0.92)
        ]
        """
        
        logger.debug(f"Finding '{concept}' columns in {table_name}")
        
        # Build schema context just for this table
        schema_context = {table_name: columns}
        
        # Build optional column type info
        bool_cols = {table_name: boolean_columns} if boolean_columns else None
        enum_cols = {table_name: enum_columns} if enum_columns else None
        
        all_matches = await self.find_columns_for_concept(
            concept,
            schema_context,
            bool_cols,
            enum_cols
        )
        
        # Filter to only this table
        table_matches = [m for m in all_matches if m.table_name == table_name]
        return table_matches
    
    async def evaluate_column_for_concept(
        self,
        concept: str,
        table_name: str,
        column_name: str,
        column_type: Optional[str] = None
    ) -> Tuple[bool, float, str]:
        """
        Evaluate if a specific column represents a concept.
        
        Args:
            concept: Semantic concept (approval, status, etc.)
            table_name: Table containing the column
            column_name: Column name to evaluate
            column_type: Optional column data type (boolean, enum, varchar, etc.)
        
        Returns:
            (is_match, confidence, reasoning)
        """
        
        prompt = f"""
Does this column represent the semantic concept "{concept}"?

Table: {table_name}
Column: {column_name}
Type: {column_type or 'unknown'}

Consider:
1. Column name semantics
2. If it's a boolean column, it likely represents a yes/no concept
3. If it's an enum column, it represents enumerated values (often status-like)
4. The column might represent the concept indirectly

Respond with ONLY valid JSON:
{{
  "is_match": true|false,
  "confidence": 0.0-1.0,
  "reasoning": "Why or why not this column matches the concept"
}}
"""
        
        try:
            response = await llm.call_llm([
                {
                    "role": "system",
                    "content": "You are a database column analyzer. Respond with ONLY valid JSON."
                },
                {"role": "user", "content": prompt}
            ], max_tokens=300, temperature=0.2)
            
            response_text = str(response)
            result = json.loads(response_text)
            
            is_match = result.get("is_match", False)
            confidence = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "")
            
            logger.debug(
                f"{table_name}.{column_name} for concept '{concept}': "
                f"match={is_match}, confidence={confidence}"
            )
            
            return (is_match, confidence, reasoning)
        
        except Exception as e:
            logger.error(f"Error evaluating column for concept: {e}")
            return (False, 0.0, str(e))
    
    def clear_cache(self) -> None:
        """Clear concept cache."""
        self._concept_cache.clear()
        logger.debug("Cleared semantic concept cache")
