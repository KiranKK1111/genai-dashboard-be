"""Follow-up Query Management Service - Context Extraction Only

REFACTORED: This module now provides ONLY context extraction utilities.
All follow-up DECISIONS are made by DecisionArbiter.

This module provides:
1. Extracting context from previous queries (table, filters, intent)
2. Preserving conversation context across multiple turns
3. Data classes for follow-up context (for backward compatibility)

The analyze() method is DEPRECATED - use DecisionArbiter for routing decisions.
"""

from __future__ import annotations

import warnings
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any

# Import unified extraction utilities
from .query_context_extractor import (
    QueryContextExtractor,
    get_query_context_extractor,
    PreviousQueryContext,  # Re-export for backward compatibility
)

logger = logging.getLogger(__name__)

# Re-export PreviousQueryContext for backward compatibility
__all__ = ['FollowUpType', 'FollowUpContext', 'PreviousQueryContext', 'FollowUpAnalyzer', 'get_followup_analyzer']


class FollowUpType(Enum):
    """Classification of follow-up query types.
    
    NOTE: Follow-up type DECISIONS are now made by DecisionArbiter.
    This enum is kept for backward compatibility and as a signal type.
    """
    
    NEW = "new"  # Unrelated to previous query
    REFINEMENT = "refinement"  # Add more filters to same table
    EXPANSION = "expansion"  # Widen scope (remove filters, get more rows)
    CLARIFICATION = "clarification"  # Ask about specific row from previous result
    PIVOT = "pivot"  # Related table, using previous results as context
    CONTINUATION = "continuation"  # Continue previous query in different way


@dataclass
class FollowUpContext:
    """Complete follow-up analysis context.
    
    NOTE: This is now a CONTEXT container, not a decision.
    The `is_followup` and `followup_type` fields are SIGNALS for DecisionArbiter,
    not final routing decisions.
    """
    
    is_followup: bool  # Signal: likely a follow-up (not a final decision)
    followup_type: FollowUpType  # Signal: suggested type (not a final decision)
    confidence: float  # 0.0 to 1.0 confidence in the signal
    previous_context: Optional[PreviousQueryContext]  # Context from previous query
    reasoning: str  # Explanation of the signal
    
    def to_prompt_section(self) -> str:
        """Format for inclusion in LLM prompt."""
        if not self.is_followup or not self.previous_context:
            return ""
        
        ctx = self.previous_context
        followup_type_val = self.followup_type.value if hasattr(self.followup_type, 'value') else str(self.followup_type)
        section = f"""PREVIOUS QUERY CONTEXT:
Type: {followup_type_val.upper()}
Confidence: {self.confidence:.0%}

Previous user asked: {ctx.query_text}
Generated SQL: {ctx.generated_sql}
Results: {ctx.result_count} rows from table '{ctx.table_name}'

Previous filters applied:
"""
        for f in ctx.filters:
            section += f"  - {f['column']} {f['operator']} {f['value']}\n"
        
        if self.followup_type == FollowUpType.REFINEMENT:
            section += "\nThis is a REFINEMENT - add more filters, keep the same table"
        elif self.followup_type == FollowUpType.EXPANSION:
            section += "\nThis is an EXPANSION - relax filters, get broader results"
        elif self.followup_type == FollowUpType.CLARIFICATION:
            section += "\nThis is a CLARIFICATION - focus on specific row(s) from previous result"
        elif self.followup_type == FollowUpType.PIVOT:
            section += "\nThis is a PIVOT - different table but related to previous data"
        
        return section
    
    def to_debug_log(self) -> str:
        """Format for debug logging output."""
        if not self.is_followup:
            return f"[FOLLOWUP] No follow-up detected (confidence: {self.confidence:.0%})"
        
        followup_type_val = self.followup_type.value if hasattr(self.followup_type, 'value') else str(self.followup_type)
        msg = f"[FOLLOWUP] Type: {followup_type_val.upper()}, Confidence: {self.confidence:.0%}\n"
        msg += f"[FOLLOWUP] Reasoning: {self.reasoning}\n"
        
        if self.previous_context:
            msg += f"[FOLLOWUP] Previous table: {self.previous_context.table_name}, Rows: {self.previous_context.result_count}\n"
            
            if self.previous_context.filters:
                msg += f"[FOLLOWUP] Previous filters: "
                filters_str = ", ".join([
                    f"{f['column']} {f['operator']} {f['value']}" 
                    for f in self.previous_context.filters
                ])
                msg += filters_str
        
        # Add type-specific guidance
        if self.followup_type == FollowUpType.REFINEMENT:
            msg += "\n[FOLLOWUP] → Action: ADD MORE FILTERS to same table"
        elif self.followup_type == FollowUpType.EXPANSION:
            msg += "\n[FOLLOWUP] → Action: RELAX FILTERS for broader results"
        elif self.followup_type == FollowUpType.CLARIFICATION:
            msg += "\n[FOLLOWUP] → Action: FOCUS on specific rows from previous result"
        elif self.followup_type == FollowUpType.PIVOT:
            msg += "\n[FOLLOWUP] → Action: SWITCH to related table using previous data"
        elif self.followup_type == FollowUpType.CONTINUATION:
            msg += "\n[FOLLOWUP] → Action: CONTINUE previous query in different way"
        
        return msg


class FollowUpAnalyzer:
    """Provides follow-up context extraction.
    
    REFACTORED: The analyze() method is DEPRECATED.
    Follow-up DECISIONS are now made by DecisionArbiter.
    This class now only provides context extraction utilities.
    """
    
    def __init__(self):
        """Initialize the analyzer with shared extractor."""
        self._extractor = get_query_context_extractor()
    
    async def analyze(
        self,
        current_query: str,
        conversation_history: str,
        previous_sql: Optional[str] = None,
        previous_result_count: int = 0,
    ) -> FollowUpContext:
        """
        DEPRECATED: Use DecisionArbiter for follow-up routing decisions.
        
        This method now only extracts context - it does NOT make routing decisions.
        The is_followup and followup_type returned are SIGNALS for the arbiter,
        not final decisions.
        
        For backward compatibility, this method still returns a FollowUpContext,
        but the decision logic has been moved to DecisionArbiter.
        """
        warnings.warn(
            "FollowUpAnalyzer.analyze() is deprecated. "
            "Use DecisionArbiter for follow-up routing decisions. "
            "This method now only provides context extraction.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Check if there's previous conversation context
        has_previous = bool(conversation_history and conversation_history.strip())
        
        if not has_previous:
            return FollowUpContext(
                is_followup=False,
                followup_type=FollowUpType.NEW,
                confidence=0.0,
                previous_context=None,
                reasoning="No previous conversation context available"
            )
        
        # Extract previous query using shared extractor
        previous_query = self._extractor.extract_previous_query(conversation_history)
        
        if not previous_query:
            return FollowUpContext(
                is_followup=False,
                followup_type=FollowUpType.NEW,
                confidence=0.0,
                previous_context=None,
                reasoning="No previous user query found in conversation"
            )
        
        # Extract context from previous SQL (if available)
        previous_context = None
        if previous_sql:
            previous_context = self._extractor.extract_context_from_sql(
                query_text=previous_query,
                sql=previous_sql,
                result_count=previous_result_count
            )
            logger.info(f"[FOLLOWUP] Extracted context: table={previous_context.table_name}, "
                       f"filters={len(previous_context.filters)}, result_count={previous_result_count}")
        
        # Return context with minimal signal (decisions are made by arbiter)
        # We signal "potential follow-up" if there's previous SQL context
        return FollowUpContext(
            is_followup=bool(previous_sql),  # Simple signal based on context existence
            followup_type=FollowUpType.REFINEMENT if previous_sql else FollowUpType.NEW,
            confidence=0.5 if previous_sql else 0.0,  # Moderate confidence - arbiter decides
            previous_context=previous_context,
            reasoning="Context extracted - final decision by DecisionArbiter"
        )
    
    def extract_context_only(
        self,
        previous_query: str,
        previous_sql: str,
        previous_result_count: int = 0,
    ) -> Optional[PreviousQueryContext]:
        """
        Extract context from previous query without making any decisions.
        
        This is the RECOMMENDED method for getting previous context.
        Decisions about follow-up type should be made by DecisionArbiter.
        
        Args:
            previous_query: The previous user query text
            previous_sql: The SQL generated for the previous query
            previous_result_count: Number of rows from previous query
            
        Returns:
            PreviousQueryContext with extracted table, columns, filters
        """
        if not previous_sql:
            return None
        
        return self._extractor.extract_context_from_sql(
            query_text=previous_query,
            sql=previous_sql,
            result_count=previous_result_count
        )
    
    def extract_previous_query(self, conversation_history: str) -> Optional[str]:
        """
        Extract the previous user query from conversation history.
        
        Delegates to QueryContextExtractor.
        """
        return self._extractor.extract_previous_query(conversation_history)
    
    def extract_context_from_history(
        self,
        previous_query: str,
        conversation_history: str,
        result_count: int
    ) -> Optional[PreviousQueryContext]:
        """
        Extract context from conversation history when SQL is not available.
        
        Delegates to QueryContextExtractor.
        """
        return self._extractor.extract_context_from_history(
            previous_query=previous_query,
            conversation_history=conversation_history,
            result_count=result_count
        )


# Singleton instance
_analyzer = None


async def get_followup_analyzer() -> FollowUpAnalyzer:
    """Get or create the follow-up analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = FollowUpAnalyzer()
    return _analyzer
