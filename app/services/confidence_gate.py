"""
Confidence Gate & Clarification System - Gating low-confidence matches.

When the system is uncertain about table/column selection, it asks targeted
clarifying questions instead of guessing blindly:

Confidence Score Interpretation:
- 0.85+: High confidence, proceed with SQL generation
- 0.65-0.84: Medium confidence, might ask for clarification on edge cases
- <0.65: Low confidence, ask blocking clarification question

Clarification Types:
1. Multi-choice: "I found these tables... which one?"
2. Confirm: "Did you mean state = Andhra Pradesh?"
3. Filter: "Hmm, AP could mean... which did you mean?"
4. Ambiguity: "I found multiple columns that could mean 'state'. Which one?"
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ClarificationType(Enum):
    """Type of clarification needed."""
    TABLE_SELECTION = "table_select"        # Which table?
    COLUMN_SELECTION = "column_select"      # Which column?
    VALUE_DISAMBIGUATION = "value_disamb"   # Which value form?
    FILTER_CONFIRMATION = "filter_confirm"  # Confirm filter interpretation
    INTENT_CLARIFICATION = "intent_clarify" # What do you want? (count vs list, etc)
    CONTEXT_MISSING = "context_missing"     # Need more context


@dataclass
class ClarifyingQuestion:
    """A clarifying question to ask the user."""
    clarification_type: ClarificationType
    question_text: str
    options: List[str]            # Multiple choice options
    confidence_score: float       # How uncertain we were
    explanation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.clarification_type.value,
            "question": self.question_text,
            "options": self.options,
            "confidence": round(self.confidence_score, 2),
            "explanation": self.explanation,
        }


class ConfidenceGate:
    """
    Gates query processing based on confidence scores.
    
    Responsibilities:
    - Evaluate overall confidence in table/column selection
    - Determine if clarification is needed
    - Generate targeted clarifying questions
    - Track clarifications in conversation history
    """
    
    def __init__(
        self,
        table_confidence_threshold: float = 0.70,
        column_confidence_threshold: float = 0.65,
        value_confidence_threshold: float = 0.75,
    ):
        """
        Initialize confidence gate.
        
        Args:
            table_confidence_threshold: Min confidence to skip table clarification
            column_confidence_threshold: Min confidence to skip column clarification
            value_confidence_threshold: Min confidence to skip value clarification
        """
        self.table_threshold = table_confidence_threshold
        self.column_threshold = column_confidence_threshold
        self.value_threshold = value_confidence_threshold
    
    async def evaluate_match(
        self,
        match_result: Dict[str, Any],
    ) -> Tuple[bool, Optional[ClarifyingQuestion]]:
        """
        Evaluate if a match result is confident enough to proceed.
        
        Args:
            match_result: Result from SemanticSchemaMatcher.match_query()
            
        Returns:
            (can_proceed, clarifying_question_or_none)
        """
        
        overall_conf = match_result.get("overall_confidence", 0.0)
        
        if overall_conf >= self.table_threshold:
            # Check filters individually
            for filter_match in match_result.get("filters", []):
                filter_conf = filter_match.get("confidence", 0.0)
                if filter_conf < self.column_threshold:
                    question = self._generate_filter_clarification(
                        filter_match,
                        filter_conf
                    )
                    return False, question
            
            # All good
            return True, None
        
        # Overall confidence too low
        question = await self._generate_table_clarification(match_result)
        return False, question
    
    def _generate_filter_clarification(
        self,
        filter_match: Dict[str, Any],
        confidence: float,
    ) -> ClarifyingQuestion:
        """Generate clarification for a specific filter."""
        
        filter_type = filter_match.get("filter_type", "value")
        user_value = filter_match.get("user_value", "")
        normalized = filter_match.get("normalized_value", "")
        
        if confidence < 0.5:
            # Might be wrong value mapping
            question_text = (
                f"I think '{user_value}' maps to '{normalized}' for the "
                f"'{filter_type}' filter. Is that correct?"
            )
            options = [f"Yes, {normalized}", "No, let me specify", "Not sure"]
        else:
            # Probably right, just confirming
            question_text = f"Filtering by {filter_type} = {normalized}. Sound right?"
            options = ["Yes, that's correct", "No, I meant something else"]
        
        return ClarifyingQuestion(
            clarification_type=ClarificationType.FILTER_CONFIRMATION,
            question_text=question_text,
            options=options,
            confidence_score=confidence,
            explanation=f"Low confidence ({confidence:.1%}) in value interpretation",
        )
    
    async def _generate_table_clarification(
        self,
        match_result: Dict[str, Any],
    ) -> ClarifyingQuestion:
        """Generate table selection clarification."""
        
        tables = match_result.get("table", "unknown")
        entity = match_result.get("entity", "unknown")
        
        question_text = (
            f"I'm looking for data about '{entity}', but I'm not confident "
            f"about the table name. Could you tell me which table has this data?"
        )
        
        # In a real system, we'd have candidates to offer
        options = [
            "I'm not sure, let me check",
            "Show me available tables",
            "Let me write the table name",
        ]
        
        confidence = match_result.get("overall_confidence", 0.0)
        
        return ClarifyingQuestion(
            clarification_type=ClarificationType.TABLE_SELECTION,
            question_text=question_text,
            options=options,
            confidence_score=confidence,
            explanation=f"Low confidence ({confidence:.1%}) in table identification",
        )
    
    @staticmethod
    def generate_multichoice_clarification(
        clarification_type: ClarificationType,
        options: List[Dict[str, Any]],  # List of {name, description, confidence}
        context: str,
    ) -> ClarifyingQuestion:
        """
        Generate a multi-choice clarification question.
        
        Args:
            clarification_type: Type of clarification
            options: List of {name, description, confidence}
            context: What we're trying to determine ("table", "column", etc)
            
        Returns:
            ClarifyingQuestion
        """
        
        # Build question text
        if clarification_type == ClarificationType.TABLE_SELECTION:
            question_text = f"Which table contains the {context} data?"
        elif clarification_type == ClarificationType.COLUMN_SELECTION:
            question_text = f"Which column represents the '{context}' filter?"
        elif clarification_type == ClarificationType.VALUE_DISAMBIGUATION:
            question_text = f"When you said '{context}', did you mean:"
        else:
            question_text = f"Could you clarify: {context}?"
        
        # Build options with descriptions
        option_texts = []
        for opt in options:
            name = opt.get("name", "")
            desc = opt.get("description", "")
            conf = opt.get("confidence", 0.0)
            
            if desc:
                text = f"{name} ({desc})"
            else:
                text = f"{name} ({conf:.0%} confident)"
            
            option_texts.append(text)
        
        # Find best confidence among options
        best_conf = max((opt.get("confidence", 0.0) for opt in options), default=0.0)
        
        return ClarifyingQuestion(
            clarification_type=clarification_type,
            question_text=question_text,
            options=option_texts,
            confidence_score=1.0 - best_conf,  # Invert: higher = less confident
        )


class ClarificationTracker:
    """
    Tracks clarifications asked in conversation.
    
    Prevents asking the same clarification twice and learns from
    user responses to improve future matching.
    """
    
    def __init__(self):
        self.asked_clarifications: List[ClarifyingQuestion] = []
        self.user_responses: Dict[str, str] = {}  # clarification_id -> response
        self.learned_mappings: Dict[str, str] = {}  # "AP" -> "Andhra Pradesh"
    
    def record_clarification(
        self,
        question: ClarifyingQuestion,
        question_id: str,
    ) -> None:
        """Record a clarification question asked."""
        self.asked_clarifications.append(question)
    
    def record_response(
        self,
        question_id: str,
        user_response: str,
    ) -> None:
        """Record user's response to a clarification."""
        self.user_responses[question_id] = user_response
    
    def record_learned_mapping(
        self,
        original_value: str,
        normalized_value: str,
    ) -> None:
        """
        Record a value mapping we learned from user confirmation.
        
        This helps with future queries:
        - User said "AP"
        - We asked if it means "Andhra Pradesh"
        - User confirmed
        - Now we know AP -> Andhra Pradesh for future queries
        """
        self.learned_mappings[original_value.lower()] = normalized_value
    
    def get_learned_mapping(self, value: str) -> Optional[str]:
        """Get a previously learned mapping."""
        return self.learned_mappings.get(value.lower())
    
    def should_ask_again(self, clarification_type: ClarificationType) -> bool:
        """Check if we should ask this type of clarification again."""
        # Don't repeat the same type twice in a row
        if self.asked_clarifications:
            last = self.asked_clarifications[-1]
            return last.clarification_type != clarification_type
        return True
    
    def get_conversation_clarity(self) -> float:
        """
        Get overall clarity from conversation.
        
        Returns 0.0-1.0 where:
        - 0.0 = very unclear
        - 1.0 = completely clear
        """
        if not self.asked_clarifications:
            return 1.0  # No clarifications needed = clear
        
        # Fewer clarifications = clearer
        clarity = max(0.0, 1.0 - (len(self.asked_clarifications) * 0.2))
        return clarity


class ConfidenceGateConfig:
    """Configuration for confidence gating behavior."""
    
    # Confidence thresholds
    table_confidence_high = 0.85
    table_confidence_medium = 0.65
    table_confidence_low = 0.40
    
    column_confidence_high = 0.80
    column_confidence_medium = 0.60
    column_confidence_low = 0.35
    
    value_confidence_high = 0.90
    value_confidence_medium = 0.75
    value_confidence_low = 0.50
    
    # Clarification policy
    ask_clarification_below_medium = True  # Ask below "medium" threshold
    max_clarifications_per_query = 3       # Don't ask more than 3 questions
    
    @staticmethod
    def get_confidence_label(score: float) -> str:
        """Convert confidence score to human label."""
        if score >= 0.85:
            return "Very High"
        elif score >= 0.70:
            return "High"
        elif score >= 0.55:
            return "Medium"
        elif score >= 0.35:
            return "Low"
        else:
            return "Very Low"
