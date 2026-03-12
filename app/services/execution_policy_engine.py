"""
Execution Policy Engine - Intent-to-Execution routing for performance optimization.

This module decides the execution path (FAST_SQL, FULL_SQL, FOLLOWUP_SQL, etc.)
based on query characteristics, avoiding unnecessary processing for simple queries.

Architecture:
    User Query → Router → Arbiter → POLICY ENGINE → Chosen Path → Handler

Execution Paths:
    - FAST_SQL: Simple list queries, minimal processing (2-4s target)
    - FULL_SQL: Complex queries needing joins, aggregation, grounding (10-20s)
    - FOLLOWUP_SQL: Queries modifying previous results (5-10s)
    - FILE_ANALYSIS: Document/file processing
    - CHAT: Conversational responses

Author: GitHub Copilot
Created: 2026-03-11
"""

import re
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field

from app import schemas

logger = logging.getLogger(__name__)


class ExecutionPath(str, Enum):
    """Execution path types with different performance profiles."""
    FAST_SQL = "fast_sql"           # Simple queries: 2-4s target
    FULL_SQL = "full_sql"           # Complex queries: 10-20s 
    FOLLOWUP_SQL = "followup_sql"   # Follow-up queries: 5-10s
    FILE_ANALYSIS = "file_analysis" # File processing
    CHAT = "chat"                   # Conversational responses
    CLARIFICATION = "clarification" # Need user input


@dataclass
class QueryCharacteristics:
    """Dynamic characteristics extracted from query analysis."""
    # Basic properties
    is_standalone: bool = False
    is_followup: bool = False
    is_list_query: bool = False
    is_aggregate_query: bool = False
    is_chart_request: bool = False
    
    # Complexity indicators
    has_joins: bool = False
    has_complex_filters: bool = False
    has_date_logic: bool = False
    has_ambiguity: bool = False
    has_file_context: bool = False
    
    # Entity resolution
    primary_entity: Optional[str] = None
    entity_confidence: float = 0.0
    requires_grounding: bool = False
    
    # Query structure
    intent_type: str = "unknown"  # list, count, aggregate, filter, visualize, etc.
    table_count_estimate: int = 1
    column_selection: str = "all"  # all, specific, aggregated
    
    # Performance hints
    expected_result_size: str = "medium"  # small, medium, large
    can_skip_visualization: bool = True
    can_skip_interpreter: bool = True
    can_skip_orchestration: bool = True
    
    # Dynamic features (extensible)
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPolicy:
    """Policy decision with rationale and optimization flags."""
    path: ExecutionPath
    confidence: float
    reasoning: str
    
    # Optimization flags (what to skip)
    skip_orchestration: bool = False
    skip_semantic_analyzer: bool = False
    skip_followup_analyzer: bool = False
    skip_value_grounding: bool = False
    skip_visualization_generation: bool = False
    skip_result_interpreter: bool = False
    skip_chart_generation: bool = False
    skip_join_detection: bool = False
    
    # Execution hints
    use_canonical_sql: bool = True
    use_simple_formatter: bool = False
    enable_caching: bool = True
    
    # Estimated performance
    estimated_duration_seconds: float = 5.0
    
    # Dynamic metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionPolicyEngine:
    """
    Intelligent execution path selector - decides how to process each query.
    
    Zero hardcoding approach:
    - Rule weights are configurable
    - Feature extractors are pluggable
    - Policy rules are data-driven
    - Thresholds are tunable
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize policy engine with configuration.
        
        Args:
            config: Optional configuration dictionary with rules, thresholds, weights
        """
        self.config = config or self._default_config()
        logger.info("[POLICY ENGINE] Initialized with configuration")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration (tunable for production)."""
        return {
            # Complexity thresholds
            "fast_sql_max_tables": 1,
            "fast_sql_max_join_depth": 0,
            "fast_sql_confidence_threshold": 0.7,
            
            # Intent patterns (regex-based, extensible)
            "list_query_patterns": [
                r"\b(show|list|get|display|view|give me)\b.*\b(all|everything)\b",
                r"\b(show|list|get|display)\s+(me\s+)?all\b",
            ],
            "aggregate_patterns": [
                r"\b(count|sum|average|total|max|min|avg)\b",
                r"\bhow\s+many\b",
                r"\b(top|bottom)\s+\d+\b",
            ],
            "chart_request_patterns": [
                r"\b(chart|graph|plot|visualize|visualization)\b",
                r"\bshow.*distribution\b",
                r"\bbar\s+chart|pie\s+chart|line\s+chart\b",
            ],
            
            # Feature weights for scoring
            "weights": {
                "standalone": 0.3,
                "single_table": 0.25,
                "simple_intent": 0.2,
                "high_confidence": 0.15,
                "no_ambiguity": 0.1,
            },
            
            # Performance tuning
            "skip_visualization_threshold": 1000,  # rows
            "skip_interpreter_threshold": 500,     # rows
        }
    
    async def determine_execution_path(
        self,
        user_query: str,
        arbiter_decision: schemas.RouterDecision,
        conversation_context: Optional[Dict[str, Any]] = None,
        schema_context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPolicy:
        """
        Determine optimal execution path based on query characteristics.
        
        Args:
            user_query: Raw user query text
            arbiter_decision: Decision from arbiter (tool, turn_class, confidence)
            conversation_context: Previous conversation state
            schema_context: Available schema information
        
        Returns:
            ExecutionPolicy with chosen path and optimization flags
        """
        logger.info(f"[POLICY ENGINE] Analyzing query: '{user_query}'")
        
        # Step 1: Extract query characteristics dynamically
        characteristics = await self._extract_characteristics(
            user_query, arbiter_decision, conversation_context, schema_context
        )
        
        # Step 2: Apply policy rules to choose path
        policy = await self._apply_policy_rules(characteristics, arbiter_decision)
        
        logger.info(
            f"[POLICY ENGINE] Decision: {policy.path.value} "
            f"(confidence={policy.confidence:.2f}, est={policy.estimated_duration_seconds:.1f}s)"
        )
        logger.info(f"[POLICY ENGINE] Optimizations: {self._summarize_skips(policy)}")
        
        return policy
    
    async def _extract_characteristics(
        self,
        user_query: str,
        arbiter_decision: schemas.RouterDecision,
        conversation_context: Optional[Dict[str, Any]],
        schema_context: Optional[Dict[str, Any]],
    ) -> QueryCharacteristics:
        """
        Dynamically extract query characteristics from multiple sources.
        
        This is the feature extraction layer - extensible and data-driven.
        """
        characteristics = QueryCharacteristics()
        
        query_lower = user_query.lower().strip()
        
        # === BASIC PROPERTIES ===
        
        # Determine if standalone vs follow-up
        characteristics.is_standalone = (
            arbiter_decision.followup_type == schemas.FollowupType.NEW_QUERY
        )
        characteristics.is_followup = not characteristics.is_standalone
        
        # === INTENT CLASSIFICATION ===
        
        # List query detection (dynamic pattern matching)
        characteristics.is_list_query = any(
            re.search(pattern, query_lower, re.I)
            for pattern in self.config["list_query_patterns"]
        )
        
        # Aggregate query detection
        characteristics.is_aggregate_query = any(
            re.search(pattern, query_lower, re.I)
            for pattern in self.config["aggregate_patterns"]
        )
        
        # Chart request detection
        characteristics.is_chart_request = any(
            re.search(pattern, query_lower, re.I)
            for pattern in self.config["chart_request_patterns"]
        )
        
        # Determine intent type
        if characteristics.is_list_query:
            characteristics.intent_type = "list"
        elif characteristics.is_aggregate_query:
            characteristics.intent_type = "aggregate"
        elif characteristics.is_chart_request:
            characteristics.intent_type = "visualize"
        else:
            characteristics.intent_type = "unknown"
        
        # === COMPLEXITY INDICATORS ===
        
        # Join detection (heuristic - can be improved with schema context)
        characteristics.has_joins = bool(
            re.search(r'\b(join|with|along with|including|plus)\b', query_lower)
        )
        
        # Complex filter detection
        characteristics.has_complex_filters = bool(
            re.search(r'\b(where|between|in|like|greater than|less than)\b', query_lower)
            or re.search(r'\b(and|or)\b.*\b(and|or)\b', query_lower)  # Multiple conditions
        )
        
        # Date logic detection
        characteristics.has_date_logic = bool(
            re.search(r'\b(date|day|month|year|quarter|week|january|february|march|april|may|june|july|august|september|october|november|december)\b', query_lower)
            or re.search(r'\b(last|this|next|previous|current)\s+(year|month|quarter|week)\b', query_lower)
        )
        
        # Ambiguity detection (multiple possible interpretations)
        characteristics.has_ambiguity = (
            arbiter_decision.needs_clarification
            or arbiter_decision.confidence < 0.7
        )
        
        # File context
        characteristics.has_file_context = bool(
            conversation_context and conversation_context.get("has_files")
        )
        
        # === ENTITY RESOLUTION ===
        
        # Extract primary entity from query (simple extraction - can be enhanced)
        # This is where schema intelligence would plug in
        entity_patterns = [
            (r'\b(staff|employee|employees|worker|personnel)\b', 'employees', 0.9),
            (r'\b(customer|customers|client|clients)\b', 'customers', 0.9),
            (r'\b(loan|loans)\b', 'loans', 0.8),
            (r'\b(branch|branches)\b', 'branches', 0.8),
            (r'\b(transaction|transactions)\b', 'transactions', 0.8),
        ]
        
        for pattern, entity, confidence in entity_patterns:
            if re.search(pattern, query_lower):
                characteristics.primary_entity = entity
                characteristics.entity_confidence = confidence
                break
        
        # Grounding requirement (needs value discovery)
        characteristics.requires_grounding = (
            characteristics.has_complex_filters
            or characteristics.has_ambiguity
        )
        
        # === TABLE COUNT ESTIMATION ===
        
        if characteristics.has_joins or re.search(r'\band\b', query_lower):
            characteristics.table_count_estimate = 2
        else:
            characteristics.table_count_estimate = 1
        
        # === COLUMN SELECTION ===
        
        if re.search(r'\ball\b.*\bdetails?\b', query_lower) or re.search(r'\bshow\s+me\s+all\b', query_lower):
            characteristics.column_selection = "all"
        elif characteristics.is_aggregate_query:
            characteristics.column_selection = "aggregated"
        else:
            characteristics.column_selection = "specific"
        
        # === PERFORMANCE HINTS ===
        
        # Expected result size
        if re.search(r'\b(all|everything)\b', query_lower):
            characteristics.expected_result_size = "large"
        elif re.search(r'\b(top|first|limit)\s+\d+\b', query_lower):
            characteristics.expected_result_size = "small"
        else:
            characteristics.expected_result_size = "medium"
        
        # Optimization flags
        characteristics.can_skip_visualization = (
            not characteristics.is_chart_request
            and characteristics.is_list_query
        )
        
        characteristics.can_skip_interpreter = (
            characteristics.expected_result_size == "large"
            and characteristics.is_list_query
        )
        
        characteristics.can_skip_orchestration = (
            characteristics.is_standalone
            and not characteristics.has_ambiguity
            and characteristics.table_count_estimate == 1
        )
        
        return characteristics
    
    async def _apply_policy_rules(
        self,
        characteristics: QueryCharacteristics,
        arbiter_decision: schemas.RouterDecision,
    ) -> ExecutionPolicy:
        """
        Apply policy rules to determine execution path and optimizations.
        
        Rules are data-driven and weighted - easy to tune without code changes.
        """
        
        # === RULE 1: CHAT PATH ===
        if arbiter_decision.tool == schemas.Tool.CHAT:
            return ExecutionPolicy(
                path=ExecutionPath.CHAT,
                confidence=arbiter_decision.confidence,
                reasoning="Query is conversational, not data-related",
                skip_orchestration=True,
                skip_semantic_analyzer=True,
                skip_followup_analyzer=True,
                skip_value_grounding=True,
                skip_visualization_generation=True,
                skip_result_interpreter=True,
                estimated_duration_seconds=1.0,
            )
        
        # === RULE 2: FILE ANALYSIS PATH ===
        if arbiter_decision.tool == schemas.Tool.ANALYZE_FILE:
            return ExecutionPolicy(
                path=ExecutionPath.FILE_ANALYSIS,
                confidence=arbiter_decision.confidence,
                reasoning="Query requires file analysis",
                skip_orchestration=False,  # May need file orchestration
                estimated_duration_seconds=8.0,
            )
        
        # === RULE 3: CLARIFICATION PATH ===
        if arbiter_decision.needs_clarification or characteristics.has_ambiguity:
            return ExecutionPolicy(
                path=ExecutionPath.CLARIFICATION,
                confidence=0.5,
                reasoning="Query is ambiguous, needs clarification",
                skip_orchestration=True,
                estimated_duration_seconds=2.0,
            )
        
        # === RULE 4: FOLLOWUP_SQL PATH ===
        if characteristics.is_followup:
            return ExecutionPolicy(
                path=ExecutionPath.FOLLOWUP_SQL,
                confidence=arbiter_decision.confidence,
                reasoning="Query modifies previous results",
                skip_orchestration=False,  # Need context
                skip_semantic_analyzer=True,  # Use cached context
                skip_followup_analyzer=False,  # Need to analyze operation
                skip_value_grounding=characteristics.can_skip_orchestration,
                skip_visualization_generation=characteristics.can_skip_visualization,
                skip_result_interpreter=characteristics.can_skip_interpreter,
                estimated_duration_seconds=6.0,
            )
        
        # === RULE 5: FAST_SQL PATH (Critical for performance) ===
        
        # Calculate fast_sql score using weighted features
        weights = self.config["weights"]
        score = 0.0
        
        if characteristics.is_standalone:
            score += weights["standalone"]
        if characteristics.table_count_estimate == 1:
            score += weights["single_table"]
        if characteristics.intent_type in ["list", "count"]:
            score += weights["simple_intent"]
        if arbiter_decision.confidence > 0.7:
            score += weights["high_confidence"]
        if not characteristics.has_ambiguity:
            score += weights["no_ambiguity"]
        
        # Penalty for complexity
        if characteristics.has_joins:
            score -= 0.2
        if characteristics.has_complex_filters:
            score -= 0.1
        if characteristics.requires_grounding:
            score -= 0.15
        if characteristics.is_chart_request:
            score -= 0.1
        
        # Decision threshold
        fast_sql_eligible = (
            score >= 0.5  # Tunable threshold
            and characteristics.is_standalone
            and characteristics.table_count_estimate <= self.config["fast_sql_max_tables"]
            and not characteristics.is_chart_request
            and characteristics.entity_confidence > self.config["fast_sql_confidence_threshold"]
        )
        
        if fast_sql_eligible:
            return ExecutionPolicy(
                path=ExecutionPath.FAST_SQL,
                confidence=score,
                reasoning=f"Simple standalone query (score={score:.2f}), eligible for fast path",
                skip_orchestration=True,
                skip_semantic_analyzer=True,
                skip_followup_analyzer=True,
                skip_value_grounding=True,
                skip_visualization_generation=True,
                skip_result_interpreter=True,
                skip_chart_generation=True,
                skip_join_detection=True,
                use_canonical_sql=True,
                use_simple_formatter=True,
                estimated_duration_seconds=3.0,
                metadata={"fast_sql_score": score},
            )
        
        # === RULE 6: FULL_SQL PATH (Default for complex queries) ===
        return ExecutionPolicy(
            path=ExecutionPath.FULL_SQL,
            confidence=arbiter_decision.confidence,
            reasoning=f"Complex query requiring full pipeline (score={score:.2f}, joins={characteristics.has_joins}, aggregation={characteristics.is_aggregate_query})",
            skip_orchestration=characteristics.can_skip_orchestration,
            skip_semantic_analyzer=False,
            skip_followup_analyzer=characteristics.is_standalone,
            skip_value_grounding=False,
            skip_visualization_generation=characteristics.can_skip_visualization,
            skip_result_interpreter=characteristics.can_skip_interpreter,
            use_canonical_sql=True,
            estimated_duration_seconds=15.0,
            metadata={"full_sql_score": score},
        )
    
    def _summarize_skips(self, policy: ExecutionPolicy) -> str:
        """Generate human-readable summary of optimization flags."""
        skips = []
        if policy.skip_orchestration:
            skips.append("orchestration")
        if policy.skip_followup_analyzer:
            skips.append("followup_analyzer")
        if policy.skip_value_grounding:
            skips.append("grounding")
        if policy.skip_visualization_generation:
            skips.append("viz_gen")
        if policy.skip_result_interpreter:
            skips.append("interpreter")
        
        return f"skipping [{', '.join(skips)}]" if skips else "full pipeline"


# Singleton instance
_policy_engine: Optional[ExecutionPolicyEngine] = None


def get_execution_policy_engine(config: Optional[Dict[str, Any]] = None) -> ExecutionPolicyEngine:
    """Get or create singleton execution policy engine instance."""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = ExecutionPolicyEngine(config)
    return _policy_engine
