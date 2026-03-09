"""
Schema Normalizer - Maps domain concepts to database tables/columns.

This service:
1. Maintains high-level domain entity types (for classification only)
2. Uses LLM-based semantic matching for table/column discovery
3. Provides generic heuristics as a fallback
4. Relies on hybrid_matcher's LLM scoring for semantic understanding

Makes the system database-agnostic by decoupling domain logic from DB schema.
NO HARDCODED SYNONYMS - all semantic understanding comes from LLM.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from ..llm import call_llm

logger = logging.getLogger(__name__)


class DomainEntity(Enum):
    """High-level domain entities the system understands (classification only)."""
    RECORD = "record"
    USER = "user"
    ACCOUNT = "account"
    ORDER = "order"
    PRODUCT = "product"
    INVOICE = "invoice"
    PAYMENT = "payment"
    DOCUMENT = "document"
    LEDGER = "ledger"
    EVENT = "event"
    AUDIT = "audit"


@dataclass
class ConceptMapping:
    """Maps a domain concept - now LLM-driven instead of hardcoded."""
    entity: DomainEntity
    # Removed hardcoded synonyms - LLM handles semantic matching


class SchemaNormalizer:
    """
    Maps actual database schema to canonical domain concepts.
    
    Uses LLM-based semantic matching instead of hardcoded synonyms.
    All semantic understanding is delegated to the LLM.
    """
    
    def __init__(self):
        # No hardcoded mappings - use LLM for semantic matching
        self.concepts: Dict[DomainEntity, ConceptMapping] = {
            entity: ConceptMapping(entity=entity)
            for entity in DomainEntity
        }
    
    def get_synonyms_for_entity(self, entity: DomainEntity) -> Dict[str, List[str]]:
        """Get synonyms for an entity - now returns empty (LLM handles semantics)."""
        return {
            "instance": [],
            "id": [],
            "timestamp": [],
            "amount": [],
        }
    
    async def guess_entity_from_term_llm(self, term: str, available_tables: List[str]) -> Optional[DomainEntity]:
        """Use LLM to guess which domain entity a user term refers to."""
        term_lower = term.lower().strip()
        
        # Direct match with entity names
        for entity in DomainEntity:
            if entity.value.lower() == term_lower:
                return entity
        
        # Use LLM for semantic matching
        entity_names = [e.value for e in DomainEntity]
        prompt = f"""Given the user term "{term}", determine which domain entity it refers to.

Available domain entities: {', '.join(entity_names)}
Available database tables: {', '.join(available_tables[:20])}

Rules:
- Match semantically based on the available domain entities and database tables
- If no clear match, return "none"

Return ONLY the entity name (lowercase) or "none". No explanation."""

        try:
            response = await call_llm([
                {"role": "system", "content": "You are a semantic matching assistant. Return only the entity name."},
                {"role": "user", "content": prompt}
            ], max_tokens=50, temperature=0.0)
            
            result = str(response).strip().lower()
            
            # Find matching entity
            for entity in DomainEntity:
                if entity.value.lower() == result:
                    return entity
        except Exception as e:
            logger.warning(f"LLM entity matching failed for '{term}': {e}")
        
        return None
    
    def guess_entity_from_term(self, term: str) -> Optional[DomainEntity]:
        """
        Guess entity from term - sync fallback with direct matching only.
        For semantic matching, use guess_entity_from_term_llm() async version.
        """
        term_lower = term.lower().strip()
        
        # Direct match only - no hardcoded synonyms
        for entity in DomainEntity:
            if entity.value.lower() == term_lower:
                return entity
            # Simple substring match
            if entity.value.lower() in term_lower or term_lower in entity.value.lower():
                return entity
        
        return None
    
    def find_candidate_tables(
        self,
        entity: DomainEntity,
        all_tables: Dict[str, any]
    ) -> List[Tuple[str, float]]:
        """
        Find candidate tables for an entity using generic heuristics.
        
        NO HARDCODED SYNONYMS - uses direct name matching only.
        LLM scoring in hybrid_matcher handles semantic understanding.
        
        Returns:
            List of (table_name, confidence) tuples, sorted by confidence descending.
        """
        entity_name = entity.value.lower()
        candidates: Dict[str, float] = {}
        
        for table_name in all_tables.keys():
            table_lower = table_name.lower()
            score = 0.0
            
            # Direct name matching (no synonyms)
            if table_lower == entity_name:
                score = max(score, 1.0)
            elif entity_name in table_lower:
                score = max(score, 0.8)
            elif table_lower in entity_name:
                score = max(score, 0.6)
            # Plural/singular variations
            elif table_lower == entity_name + "s" or table_lower + "s" == entity_name:
                score = max(score, 0.9)
            
            # Generic column-based scoring (no hardcoded signature columns)
            table_obj = all_tables.get(table_name)
            if table_obj and hasattr(table_obj, 'columns'):
                col_names = [c.lower() for c in table_obj.columns.keys()]
                
                # Check if entity name appears in column names (indicates relation)
                entity_related_cols = sum(1 for c in col_names if entity_name in c)
                if entity_related_cols > 0:
                    score = max(score, 0.4 + min(0.2, entity_related_cols * 0.05))
            
            if score > 0:
                candidates[table_name] = min(score, 1.0)
        
        # Return sorted by confidence descending
        return sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    
    async def find_candidate_tables_llm(
        self,
        entity: DomainEntity,
        all_tables: Dict[str, any]
    ) -> List[Tuple[str, float]]:
        """
        Find candidate tables using LLM semantic matching.
        
        Use this for full semantic understanding when async is available.
        """
        entity_name = entity.value
        table_names = list(all_tables.keys())[:30]  # Limit for prompt size
        
        prompt = f"""Given the domain entity "{entity_name}", find which database tables likely store this data.

Available tables: {', '.join(table_names)}

Rules:
- Consider semantic relationships and synonyms naturally
- Consider plural/singular variations
- Return tables with confidence scores 0.0-1.0

Return JSON array: [{{"table": "name", "score": 0.9}}, ...]
Only return tables with score > 0.3. Return empty array [] if no matches."""

        try:
            response = await call_llm([
                {"role": "system", "content": "You are a database schema analyzer. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ], max_tokens=300, temperature=0.0)
            
            response_text = str(response).strip()
            # Extract JSON from response
            if "[" in response_text:
                json_start = response_text.index("[")
                json_end = response_text.rindex("]") + 1
                json_text = response_text[json_start:json_end]
                matches = json.loads(json_text)
                
                result = []
                for match in matches:
                    if match.get("table") in all_tables:
                        result.append((match["table"], float(match.get("score", 0.5))))
                
                return sorted(result, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.warning(f"LLM table matching failed for '{entity_name}': {e}")
        
        # Fallback to heuristic matching
        return self.find_candidate_tables(entity, all_tables)
    
    def find_candidate_id_columns(
        self,
        entity: DomainEntity,
        table_name: str,
        column_names: Set[str]
    ) -> List[Tuple[str, float]]:
        """
        Find candidate ID columns using generic heuristics.
        
        NO HARDCODED SYNONYMS - uses pattern matching only.
        
        Returns:
            List of (column_name, confidence) tuples.
        """
        entity_name = entity.value.lower()
        candidates: Dict[str, float] = {}
        
        for column_name in column_names:
            col_lower = column_name.lower()
            score = 0.0
            
            # Generic ID patterns (no hardcoded synonyms)
            # Pattern: {entity}_id or {entity}id
            if col_lower == f"{entity_name}_id" or col_lower == f"{entity_name}id":
                score = max(score, 1.0)
            # Pattern: id (standalone)
            elif col_lower == "id":
                score = max(score, 0.8)
            # Pattern: contains entity name + id
            elif entity_name in col_lower and "id" in col_lower:
                score = max(score, 0.9)
            # Generic ID patterns
            elif col_lower.endswith("_id") or col_lower.endswith("id"):
                score = max(score, 0.5)
            # Code/number patterns
            elif "code" in col_lower or "number" in col_lower or "num" in col_lower:
                score = max(score, 0.4)
            
            if score > 0:
                candidates[column_name] = score
        
        return sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    
    def is_numeric_type(self, data_type: str) -> bool:
        """Check if a data type is numeric."""
        numeric_types = [
            "integer", "bigint", "smallint", "numeric",
            "decimal", "double", "float", "real", "int"
        ]
        return any(t in data_type.lower() for t in numeric_types)
    
    def is_datetime_type(self, data_type: str) -> bool:
        """Check if a data type is datetime-like."""
        datetime_types = [
            "timestamp", "date", "time", "datetime"
        ]
        return any(t in data_type.lower() for t in datetime_types)
