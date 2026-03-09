"""
Enhanced Follow-up Analyzer - Multi-factor detection.

Replaces pure LLM-based classification with hybrid approach:
1. LLM semantic analysis (40%)
2. Keyword pattern matching (30%)
3. Semantic similarity (20%)
4. Reference resolution (5%)
5. Temporal proximity (5%)

This significantly improves follow-up detection accuracy and speed.
"""

from __future__ import annotations

import logging
import re
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FollowupAnalysis:
    """Multi-factor follow-up analysis result."""
    is_followup: bool
    followup_type: str
    confidence: float  # 0.0-1.0
    
    # Individual factor scores
    llm_score: float
    keyword_score: float
    semantic_score: float
    reference_score: float
    temporal_score: float
    
    # Debug info
    matched_patterns: List[str]
    reasoning: str


class FollowupPatternMatcher:
    """Multi-factor follow-up detection."""
    
    # Pattern-based follow-up indicators
    REFINEMENT_PATTERNS = [
        r"but only|with only|except|excluding|not\s+\w+",
        r"where.*(?!from)",  # Additional WHERE clauses
        r"filter|filtered|narrow",
        r"just|only|limit to",
    ]
    
    EXPANSION_PATTERNS = [
        r"also|include|plus|and their|with their|including",
        r"expand|broader|more|additional",
        r"show.*transactions|show.*orders|show.*history",
        r"add|append|along with",
    ]
    
    AGGREGATION_PATTERNS = [
        r"how many|total|sum|count|average|mean",
        r"group by|group.*by",
        r"total.*for|overall",
    ]
    
    CONTINUATION_PATTERNS = [
        r"show more|more results|next|proceed",
        r"continue|next page|other",
        r"rest|rest of|remaining",
    ]
    
    PIVOT_PATTERNS = [
        r"now show|what about|for each|by",
        r"instead|switch to|change to",
        r"versus|compared to",
    ]
    
    # Reference resolution patterns
    REFERENCE_PATTERNS = {
        "their": r"their\s+(\w+)",
        "those": r"those\s+(\w+)",
        "them": r"(?:show|list|display|get)\s+them",
        "it": r"(?:show|display|with)\s+it",
        "that": r"that\s+(\w+)?",
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize pattern matcher with custom weights."""
        self.weights = weights or {
            "llm": 0.40,
            "keyword": 0.30,
            "semantic": 0.20,
            "reference": 0.05,
            "temporal": 0.05,
        }
    
    async def analyze_followup(
        self,
        current_query: str,
        previous_query: Optional[str] = None,
        conversation_context: Optional[str] = None,
    ) -> FollowupAnalysis:
        """
        Multi-factor follow-up detection.
        
        Args:
            current_query: User's current natural language query
            previous_query: Previous query (if any)
            conversation_context: Full conversation history
        
        Returns:
            FollowupAnalysis with confidence and type
        """
        
        scores = {}
        matched_patterns = []
        
        # FACTOR 1: LLM Classification
        llm_score = await self._get_llm_score(
            current_query,
            previous_query,
            conversation_context
        )
        scores["llm"] = llm_score
        
        # FACTOR 2: Keyword Pattern Matching
        keyword_score, patterns = await self._get_keyword_score(current_query)
        scores["keyword"] = keyword_score
        matched_patterns.extend(patterns)
        
        # FACTOR 3: Semantic Similarity
        semantic_score = 0.0
        if previous_query:
            semantic_score = await self._get_semantic_score(
                current_query,
                previous_query
            )
        scores["semantic"] = semantic_score
        
        # FACTOR 4: Reference Resolution
        reference_score, refs = await self._get_reference_score(current_query)
        scores["reference"] = reference_score
        matched_patterns.extend(refs)
        
        # FACTOR 5: Temporal (proximity to previous)
        temporal_score = 0.95 if previous_query else 0.0
        scores["temporal"] = temporal_score
        
        # Weighted total
        total_score = sum(
            scores.get(factor, 0) * self.weights.get(factor, 0)
            for factor in self.weights.keys()
        )
        
        # Determine followup type
        followup_type = self._determine_followup_type(
            current_query,
            matched_patterns
        )
        
        reasoning = (
            f"LLM:{scores['llm']:.2f} + "
            f"Keyword:{scores['keyword']:.2f} + "
            f"Semantic:{scores['semantic']:.2f} + "
            f"Reference:{scores['reference']:.2f} + "
            f"Temporal:{scores['temporal']:.2f} = "
            f"Total: {total_score:.2f}"
        )
        
        logger.info(
            f"[FOLLOWUP-DETECT] {followup_type.upper()} "
            f"(score={total_score:.2f}, patterns={matched_patterns})"
        )
        
        return FollowupAnalysis(
            is_followup=total_score >= 0.5,
            followup_type=followup_type,
            confidence=total_score,
            llm_score=scores["llm"],
            keyword_score=scores["keyword"],
            semantic_score=scores["semantic"],
            reference_score=scores["reference"],
            temporal_score=scores["temporal"],
            matched_patterns=matched_patterns,
            reasoning=reasoning,
        )
    
    async def _get_keyword_score(self, query: str) -> Tuple[float, List[str]]:
        """
        Keyword-based follow-up detection.
        Score: 0.0 (no patterns) to 1.0 (strong patterns)
        """
        query_lower = query.lower()
        matched = []
        scores = []
        
        # Check each pattern category
        for pattern in self.REFINEMENT_PATTERNS:
            if re.search(pattern, query_lower):
                matched.append("refinement")
                scores.append(0.9)
                break
        
        for pattern in self.EXPANSION_PATTERNS:
            if re.search(pattern, query_lower):
                matched.append("expansion")
                scores.append(0.85)
                break
        
        for pattern in self.AGGREGATION_PATTERNS:
            if re.search(pattern, query_lower):
                matched.append("aggregation")
                scores.append(0.8)
                break
        
        for pattern in self.CONTINUATION_PATTERNS:
            if re.search(pattern, query_lower):
                matched.append("continuation")
                scores.append(0.75)
                break
        
        for pattern in self.PIVOT_PATTERNS:
            if re.search(pattern, query_lower):
                matched.append("pivot")
                scores.append(0.7)
                break
        
        # Average score
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return min(avg_score, 1.0), matched
    
    async def _get_semantic_score(
        self,
        current: str,
        previous: str
    ) -> float:
        """
        Semantic similarity using lightweight_rag BM25.
        """
        try:
            from .lightweight_rag import BM25Ranker
            
            ranker = BM25Ranker()
            ranker.build_index([previous])
            score = ranker.score(current, previous)
            
            # Normalize to 0-1
            return min(score / 100, 1.0)
        except Exception as e:
            logger.warning(f"[SEMANTIC-SCORE] Error: {e}")
            return 0.0
    
    async def _get_reference_score(
        self,
        query: str
    ) -> Tuple[float, List[str]]:
        """
        Score based on reference resolution ("their", "those", etc.)
        """
        query_lower = query.lower()
        refs = []
        
        for reference, pattern in self.REFERENCE_PATTERNS.items():
            if re.search(pattern, query_lower):
                refs.append(f"ref:{reference}")
        
        # Strong indicator of follow-up if references found
        score = 0.8 if refs else 0.0
        
        return score, refs
    
    def _determine_followup_type(
        self,
        query: str,
        patterns: List[str]
    ) -> str:
        """
        Determine which type of follow-up based on matched patterns.
        """
        if not patterns:
            return "continuation"
        
        # Prioritize by frequency
        pattern_counts = {}
        for p in patterns:
            base = p.split(":")[0] if ":" in p else p
            pattern_counts[base] = pattern_counts.get(base, 0) + 1
        
        most_common = max(
            pattern_counts.items(),
            key=lambda x: x[1]
        )[0]
        
        return most_common
    
    async def _get_llm_score(
        self,
        current: str,
        previous: Optional[str],
        context: Optional[str]
    ) -> float:
        """
        Use existing LLM-based classification.
        Falls back to 0.5 if not available.
        """
        try:
            from .followup_manager import get_followup_analyzer
            
            analyzer = await get_followup_analyzer()
            result = await analyzer.analyze(
                current_query=current,
                conversation_history=context or "",
                previous_sql=None,
                previous_result_count=0,
            )
            
            return result.confidence
        except Exception as e:
            logger.warning(f"[LLM-SCORE] Error: {e}, returning 0.5")
            return 0.5


# Singleton instance
_followup_analyzer_enhanced: Optional[FollowupPatternMatcher] = None


async def get_followup_analyzer_enhanced(
    weights: Optional[Dict[str, float]] = None
) -> FollowupPatternMatcher:
    """Get or create enhanced followup analyzer."""
    global _followup_analyzer_enhanced
    
    if _followup_analyzer_enhanced is None:
        _followup_analyzer_enhanced = FollowupPatternMatcher(weights=weights)
    
    return _followup_analyzer_enhanced
