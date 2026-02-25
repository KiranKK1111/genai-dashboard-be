"""
Hybrid Matcher - Combines LLM scoring with deterministic ranking.

Strategy:
1. Candidate generation (deterministic) - filter by name/column signatures
2. Candidate ranking (semantic) - use LLM + embeddings to score
3. Final selection - pick best or ask clarification if ambiguous

This creates a robust, "thinking" system that:
- Handles synonyms (txn, transaction, transact, etc.)
- Uses semantic understanding (LLM scoring)
- Has fallback clarification questions
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from ..services.schema_discovery import SchemaCatalog, TableInfo
from .schema_normalizer import SchemaNormalizer, DomainEntity

logger = logging.getLogger(__name__)


@dataclass
class CandidateTable:
    """A candidate table for a user query."""
    table_name: str
    schema: str
    row_count: int
    confidence: float
    reasons: List[str]
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CandidateColumn:
    """A candidate column for an entity ID."""
    column_name: str
    data_type: str
    is_primary_key: bool
    confidence: float
    reasons: List[str]
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MatchResult:
    """Result of hybrid matching."""
    entity: DomainEntity
    table_name: str
    id_column: str
    confidence: float
    needs_clarification: bool
    clarifying_question: Optional[str] = None
    candidates: Optional[List[CandidateTable]] = None
    
    def to_dict(self) -> dict:
        return {
            "entity": self.entity.value,
            "table_name": self.table_name,
            "id_column": self.id_column,
            "confidence": self.confidence,
            "needs_clarification": self.needs_clarification,
            "clarifying_question": self.clarifying_question,
            "candidates": [c.to_dict() for c in (self.candidates or [])],
        }


class HybridMatcher:
    """
    Matches user intent to database table and column.
    
    Combines:
    - Deterministic ranking (synonym matching, signature columns)
    - LLM semantic scoring
    - Confidence thresholds (ask clarification if low)
    """
    
    def __init__(self, catalog: SchemaCatalog, normalizer: SchemaNormalizer):
        self.catalog = catalog
        self.normalizer = normalizer
        self.confidence_threshold = 0.65  # Below this, ask clarification
    
    async def match_entity_to_table(
        self,
        entity: DomainEntity,
        llm_scorer: Optional[callable] = None,
    ) -> List[CandidateTable]:
        """
        Find candidate tables for an entity.
        
        Uses:
        1. Name matching (deterministic)
        2. Signature column matching (deterministic)
        3. LLM scoring (semantic)
        4. Returns ranked list
        """
        all_tables = self.catalog.get_all_tables()
        
        # Step 1: Deterministic candidate generation
        base_candidates = self.normalizer.find_candidate_tables(
            entity, all_tables
        )
        
        candidates: List[CandidateTable] = []
        
        for table_name, base_score in base_candidates:
            table_info = all_tables[table_name]
            reasons = []
            confidence = base_score
            
            # Check for signature columns
            col_names = set(table_info.columns.keys())
            mapping = self.normalizer.concepts.get(entity)
            
            if mapping:
                sig_matches = [
                    col for col in mapping.signature_columns
                    if any(col.lower() in cn.lower() for cn in col_names)
                ]
                if sig_matches:
                    reasons.append(f"Contains signature columns: {', '.join(sig_matches[:3])}")
                    confidence += 0.1
            
            # Step 2: LLM semantic scoring (if scorer provided)
            if llm_scorer:
                try:
                    llm_score = await llm_scorer(
                        entity=entity,
                        table_name=table_name,
                        table_info=table_info
                    )
                    # Weight LLM score into final confidence
                    confidence = (confidence * 0.4) + (llm_score * 0.6)
                    reasons.append(f"LLM semantic match: {llm_score:.2f}")
                except Exception as e:
                    logger.warning(f"LLM scoring failed for {table_name}: {e}")
            
            confidence = min(confidence, 1.0)
            
            candidates.append(CandidateTable(
                table_name=table_name,
                schema=table_info.schema,
                row_count=table_info.row_count,
                confidence=confidence,
                reasons=reasons
            ))
        
        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        return candidates
    
    async def match_id_column(
        self,
        entity: DomainEntity,
        table_name: str,
        llm_scorer: Optional[callable] = None,
    ) -> List[CandidateColumn]:
        """
        Find candidate ID columns for an entity in a table.
        """
        table_info = self.catalog.get_table(table_name)
        if not table_info:
            return []
        
        col_names = set(table_info.columns.keys())
        
        # Step 1: Deterministic matching
        base_candidates = self.normalizer.find_candidate_id_columns(
            entity, table_name, col_names
        )
        
        candidates: List[CandidateColumn] = []
        
        for col_name, base_score in base_candidates:
            col_info = table_info.columns[col_name]
            reasons = []
            confidence = base_score
            
            # Primary key bonus
            if col_info.is_primary_key:
                confidence += 0.2
                reasons.append("Is primary key")
            
            # Step 2: LLM semantic scoring
            if llm_scorer:
                try:
                    llm_score = await llm_scorer(
                        entity=entity,
                        table_name=table_name,
                        column_name=col_name,
                        column_info=col_info
                    )
                    confidence = (confidence * 0.4) + (llm_score * 0.6)
                    reasons.append(f"LLM match: {llm_score:.2f}")
                except:
                    pass
            
            confidence = min(confidence, 1.0)
            
            candidates.append(CandidateColumn(
                column_name=col_name,
                data_type=col_info.data_type,
                is_primary_key=col_info.is_primary_key,
                confidence=confidence,
                reasons=reasons
            ))
        
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates
    
    async def match_and_decide(
        self,
        entity: DomainEntity,
        llm_scorer: Optional[callable] = None,
    ) -> MatchResult:
        """
        Complete matching pipeline: find table, find ID column, decide if clarification needed.
        
        Returns:
            MatchResult with:
            - Best table and ID column (if confidence high)
            - Clarifying question (if confidence low or multiple good matches)
            - List of candidates
        """
        # Find candidate tables
        table_candidates = await self.match_entity_to_table(entity, llm_scorer)
        
        if not table_candidates:
            return MatchResult(
                entity=entity,
                table_name="",
                id_column="",
                confidence=0.0,
                needs_clarification=True,
                clarifying_question=f"I couldn't find a table for {entity.value}. " +
                                   "Could you specify the table name?"
            )
        
        best_table = table_candidates[0]
        
        # Find ID columns for best table
        id_candidates = await self.match_id_column(
            entity, best_table.table_name, llm_scorer
        )
        
        if not id_candidates:
            return MatchResult(
                entity=entity,
                table_name=best_table.table_name,
                id_column="",
                confidence=0.0,
                needs_clarification=True,
                clarifying_question=f"Found table {best_table.table_name}, but couldn't " +
                                   "identify the ID column. Could you specify it?"
            )
        
        best_id_column = id_candidates[0]
        
        # Decision logic
        combined_confidence = (best_table.confidence + best_id_column.confidence) / 2
        
        # Check for ambiguity
        needs_clarification = False
        clarifying_question = None
        
        # Case 1: Low confidence on table
        if best_table.confidence < self.confidence_threshold:
            needs_clarification = True
            options = [f"{t.table_name} ({t.confidence:.0%})" for t in table_candidates[:3]]
            clarifying_question = (
                f"I found multiple potential tables. Which one should I use?\n" +
                "\n".join(f"• {opt}" for opt in options)
            )
        # Case 2: Multiple good table matches (collision)
        elif len(table_candidates) > 1 and table_candidates[1].confidence > 0.7:
            if abs(best_table.confidence - table_candidates[1].confidence) < 0.15:
                needs_clarification = True
                options = [f"{t.table_name}" for t in table_candidates[:3]]
                clarifying_question = (
                    f"I found multiple strong matches for {entity.value}. Which table should I use?\n" +
                    "\n".join(f"• {opt}" for opt in options)
                )
        
        return MatchResult(
            entity=entity,
            table_name=best_table.table_name,
            id_column=best_id_column.column_name,
            confidence=combined_confidence,
            needs_clarification=needs_clarification,
            clarifying_question=clarifying_question,
            candidates=table_candidates[:3]  # Include top 3 for context
        )


async def create_hybrid_matcher(
    catalog: SchemaCatalog,
    normalizer: SchemaNormalizer
) -> HybridMatcher:
    """Factory function to create a hybrid matcher."""
    return HybridMatcher(catalog, normalizer)
