"""
Decision Arbiter - Single Authoritative Decision Pipeline

This module is the SOLE AUTHORITY for making routing and follow-up decisions.
All downstream code MUST consume only the final decision object from this arbiter.

The arbiter combines:
1. Router decision (fast pattern + intent + LLM)
2. Follow-up classifier output 
3. Last turn state (from TurnStateManager)
4. Hard signals (files, DB connection, etc.)
5. Query embeddings / semantic match

Output (ArbiterDecision):
- final_tool: The tool to use (RUN_SQL, CHAT, ANALYZE_FILE, MIXED)
- final_turn_class: NEW_QUERY | FOLLOW_UP | AMBIGUOUS  
- final_followup_subtype: FILTER_ADD | GROUP_CHANGE | SORT_CHANGE | etc.
- should_merge_state: Whether to merge with previous state
- should_reset_state: Whether to reset state completely
- can_skip_orchestration: Whether heavy orchestration can be skipped
- confidence: Final confidence score
- reasoning: Explanation of the decision

This is the FIX for the "split-brain" routing design problem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from .. import schemas, models

logger = logging.getLogger(__name__)


class TurnClass(Enum):
    """Classification of turn relationship to previous context."""
    NEW_QUERY = "new_query"
    FOLLOW_UP = "follow_up" 
    AMBIGUOUS = "ambiguous"


class FollowUpSubtype(Enum):
    """Specific type of follow-up modification."""
    # Filter modifications
    FILTER_ADD = "filter_add"
    FILTER_REMOVE = "filter_remove"
    FILTER_CHANGE = "filter_change"
    
    # Grouping/Aggregation
    GROUP_ADD = "group_add"
    GROUP_CHANGE = "group_change"
    GROUP_REMOVE = "group_remove"
    
    # Sorting
    SORT_ADD = "sort_add"
    SORT_CHANGE = "sort_change"
    SORT_REMOVE = "sort_remove"
    
    # Limit/Pagination
    LIMIT_CHANGE = "limit_change"
    
    # Column selection
    COLUMN_ADD = "column_add"
    COLUMN_REMOVE = "column_remove"
    
    # Data scope
    EXPAND_SCOPE = "expand_scope"
    NARROW_SCOPE = "narrow_scope"
    
    # Context-based
    DRILL_DOWN = "drill_down"
    PIVOT_TABLE = "pivot_table"
    CONTINUATION = "continuation"
    
    # None (for new queries)
    NONE = "none"


@dataclass
class ArbiterDecision:
    """
    The FINAL authoritative decision from the arbiter.
    
    All downstream code MUST use this and ONLY this for routing decisions.
    """
    # Core decision
    final_tool: schemas.Tool
    final_turn_class: TurnClass
    final_followup_subtype: FollowUpSubtype
    
    # State management
    should_merge_state: bool
    should_reset_state: bool
    
    # Performance optimization
    can_skip_orchestration: bool
    
    # Confidence and reasoning
    confidence: float
    reasoning: str
    
    # Signals used (for debugging)
    signals_used: Dict[str, Any] = field(default_factory=dict)
    
    # Original decisions (for debugging)
    router_decision: Optional[schemas.RouterDecision] = None
    followup_classifier_result: Optional[Dict[str, Any]] = None
    last_turn_state: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "final_tool": self.final_tool.value,
            "final_turn_class": self.final_turn_class.value,
            "final_followup_subtype": self.final_followup_subtype.value,
            "should_merge_state": self.should_merge_state,
            "should_reset_state": self.should_reset_state,
            "can_skip_orchestration": self.can_skip_orchestration,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "signals_used": self.signals_used,
        }


class DecisionArbiter:
    """
    The single authoritative decision maker for routing and follow-up classification.
    
    This replaces the fragmented decision-making across:
    - efficient_query_router.py
    - semantic_intent_router.py
    - session_query_handler.py
    - followup_manager.py
    
    Instead of each component making conflicting decisions, this arbiter
    collects ALL signals and makes ONE final decision.
    """
    
    # Configurable thresholds (can be tuned based on production data)
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    MEDIUM_CONFIDENCE_THRESHOLD = 0.60
    LOW_CONFIDENCE_THRESHOLD = 0.40
    
    # Follow-up signal patterns - dynamically extensible
    REFERENTIAL_INDICATORS = [
        "those", "them", "these", "that", "this", "it", "they",
        "the ones", "that data", "those results", "same", "above",
        "previous", "earlier", "before"
    ]
    
    MODIFICATION_INDICATORS = [
        "only", "just", "but", "except", "excluding", "including",
        "also", "additionally", "now", "instead"
    ]
    
    # Standalone query patterns - these indicate complete, self-contained queries
    # that should be NEW_QUERY even if entity matches previous query
    STANDALONE_QUERY_PATTERNS = [
        # Imperative patterns
        r"^(show|list|get|give|display|fetch)\s+(me\s+)?(all|the)\s+\w+",
        r"^(show|list|get|give|display)\s+\w+\s+details",
        r"^(how many|count)\s+\w+",
        
        # Question patterns
        r"^(what|which|who)\s+(are|is)\s+(the|all)\s+\w+",
        r"^(what|which)\s+\w+\s+(do|does|are|is)",
        
        # Complete descriptive patterns
        r"^(all|the)\s+\w+\s+(details|information|data|records)",
    ]
    
    # Entity keywords that map to database tables
    # This is used to detect explicit entity changes between queries
    ENTITY_KEYWORDS = {
        # Staff/Employees
        "staff": "employees",
        "employee": "employees",
        "employees": "employees",
        "worker": "employees",
        "workers": "employees",
        "team": "employees",
        "personnel": "employees",
        
        # Customers/Clients
        "customer": "customers",
        "customers": "customers",
        "client": "customers",
        "clients": "customers",
        
        # Loans
        "loan": "loans",
        "loans": "loans",
        "lending": "loans",
        
        # Accounts
        "account": "accounts",
        "accounts": "accounts",
        
        # Transactions
        "transaction": "transactions",
        "transactions": "transactions",
        "payment": "transactions",
        "payments": "transactions",
        
        # Cards
        "card": "cards",
        "cards": "cards",
        
        # Branches
        "branch": "branches",
        "branches": "branches",
        "office": "branches",
        "offices": "branches",
        "location": "branches",
        "locations": "branches",
    }
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._followup_patterns_cache: Optional[Dict[str, float]] = None
    
    async def arbitrate(
        self,
        user_query: str,
        router_decision: schemas.RouterDecision,
        followup_context: Optional[Dict[str, Any]] = None,
        last_turn_state: Optional[Dict[str, Any]] = None,
        hard_signals: Optional[schemas.RouterSignals] = None,
        session_id: str = "",
        user_id: str = "",
        is_new_session: bool = False,
        previous_messages_count: int = 0,
    ) -> ArbiterDecision:
        """
        Make the FINAL authoritative decision by combining all signals.
        
        This is the ONLY place where the final routing decision should be made.
        All previous "decisions" from router, followup analyzer, etc. are
        treated as PROVISIONAL INPUTS, not final decisions.
        
        Args:
            user_query: Current user query
            router_decision: Decision from efficient/semantic router (PROVISIONAL)
            followup_context: Result from followup analyzer (PROVISIONAL)
            last_turn_state: Last turn state from TurnStateManager
            hard_signals: Hard signals (files, DB, etc.)
            session_id: Session ID
            user_id: User ID
            is_new_session: Whether this is a brand new session
            previous_messages_count: Number of previous messages
            
        Returns:
            ArbiterDecision - the FINAL authoritative decision
        """
        
        signals_collected = {
            "router_tool": router_decision.tool.value if router_decision else None,
            "router_confidence": router_decision.confidence if router_decision else 0.0,
            "router_followup_type": router_decision.followup_type.value if router_decision else None,
            "is_new_session": is_new_session,
            "previous_messages_count": previous_messages_count,
            "has_last_turn_state": last_turn_state is not None,
        }
        
        # RULE 1: New session = ALWAYS new query, regardless of what router says
        if is_new_session or previous_messages_count == 0:
            logger.info("[ARBITER] New session detected → forcing NEW_QUERY")
            return self._create_new_query_decision(
                user_query=user_query,
                router_decision=router_decision,
                hard_signals=hard_signals,
                signals_collected=signals_collected,
                reasoning="New session - no previous context exists"
            )
        
        # RULE 2: Analyze query for referential/continuation signals
        query_signals = self._analyze_query_signals(user_query)
        signals_collected["query_signals"] = query_signals
        
        # RULE 3: Check follow-up classifier result (if available)
        followup_is_followup = False
        followup_type_str = "NEW"
        followup_confidence = 0.0
        
        if followup_context:
            followup_is_followup = followup_context.get("is_followup", False)
            followup_type_str = followup_context.get("followup_type", "NEW")
            followup_confidence = followup_context.get("confidence", 0.0)
            signals_collected["followup_is_followup"] = followup_is_followup
            signals_collected["followup_type"] = followup_type_str
            signals_collected["followup_confidence"] = followup_confidence
        
        # RULE 4: Check last turn state for SQL context
        has_previous_sql = False
        previous_table = None
        if last_turn_state:
            artifacts = last_turn_state.get("artifacts", {})
            has_previous_sql = bool(artifacts.get("sql"))
            previous_table = artifacts.get("tables", [None])[0] if artifacts.get("tables") else None
            signals_collected["has_previous_sql"] = has_previous_sql
            signals_collected["previous_table"] = previous_table
        
        # RULE 5: Determine final turn class by combining signals
        final_turn_class, turn_class_reasoning = self._determine_turn_class(
            router_decision=router_decision,
            followup_is_followup=followup_is_followup,
            followup_confidence=followup_confidence,
            query_signals=query_signals,
            has_previous_sql=has_previous_sql,
            previous_messages_count=previous_messages_count,
            previous_table=previous_table,  # NEW: Pass previous table for entity change detection
        )
        
        signals_collected["turn_class_reasoning"] = turn_class_reasoning
        
        # RULE 6: Determine final tool (may override router if context dictates)
        final_tool = self._determine_final_tool(
            router_decision=router_decision,
            final_turn_class=final_turn_class,
            hard_signals=hard_signals,
            has_previous_sql=has_previous_sql,
        )
        
        # RULE 7: Determine follow-up subtype
        final_followup_subtype = self._determine_followup_subtype(
            user_query=user_query,
            final_turn_class=final_turn_class,
            followup_type_str=followup_type_str,
            query_signals=query_signals,
        )
        
        # RULE 8: Determine state management (merge vs reset)
        should_merge_state = final_turn_class == TurnClass.FOLLOW_UP
        should_reset_state = final_turn_class == TurnClass.NEW_QUERY
        
        # RULE 9: Determine if orchestration can be skipped
        can_skip_orchestration = self._can_skip_orchestration(
            final_turn_class=final_turn_class,
            final_tool=final_tool,
            router_confidence=router_decision.confidence if router_decision else 0.0,
            has_previous_sql=has_previous_sql,
            user_query=user_query,
        )
        
        # RULE 10: Calculate final confidence
        final_confidence = self._calculate_final_confidence(
            router_confidence=router_decision.confidence if router_decision else 0.0,
            followup_confidence=followup_confidence,
            query_signals=query_signals,
            final_turn_class=final_turn_class,
        )
        
        # Build final decision
        final_reasoning = self._build_reasoning(
            final_turn_class=final_turn_class,
            turn_class_reasoning=turn_class_reasoning,
            final_tool=final_tool,
            final_followup_subtype=final_followup_subtype,
            signals_collected=signals_collected,
        )
        
        decision = ArbiterDecision(
            final_tool=final_tool,
            final_turn_class=final_turn_class,
            final_followup_subtype=final_followup_subtype,
            should_merge_state=should_merge_state,
            should_reset_state=should_reset_state,
            can_skip_orchestration=can_skip_orchestration,
            confidence=final_confidence,
            reasoning=final_reasoning,
            signals_used=signals_collected,
            router_decision=router_decision,
            followup_classifier_result=followup_context,
            last_turn_state=last_turn_state,
        )
        
        logger.info(
            f"[ARBITER] Final decision: tool={final_tool.value}, "
            f"turn_class={final_turn_class.value}, "
            f"subtype={final_followup_subtype.value}, "
            f"confidence={final_confidence:.2f}, "
            f"merge={should_merge_state}, reset={should_reset_state}"
        )
        
        return decision
    
    def _analyze_query_signals(self, query: str) -> Dict[str, Any]:
        """
        Analyze query text for referential and modification signals.
        
        This is a semantic analysis that looks for patterns indicating
        the query references previous context.
        """
        import re
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Check for referential indicators
        referential_count = sum(
            1 for indicator in self.REFERENTIAL_INDICATORS
            if indicator in query_lower
        )
        has_referential = referential_count > 0
        
        # Check for modification indicators
        modification_count = sum(
            1 for indicator in self.MODIFICATION_INDICATORS
            if indicator in query_lower
        )
        has_modification = modification_count > 0
        
        # NEW: Detect standalone/complete query patterns
        is_standalone = any(
            re.match(pattern, query_lower.strip())
            for pattern in self.STANDALONE_QUERY_PATTERNS
        )
        
        # NEW: Detect explicit entity mentions
        detected_entity = None
        for keyword, table_name in self.ENTITY_KEYWORDS.items():
            if keyword in query_words or keyword in query_lower:
                detected_entity = table_name
                break  # Use first match
        
        # Check for ellipsis patterns (incomplete queries that need context)
        is_elliptical = self._is_elliptical_query(query)
        
        # Calculate follow-up probability based on signals
        followup_probability = 0.0
        if has_referential:
            followup_probability += 0.4
        if has_modification:
            followup_probability += 0.3
        if is_elliptical:
            followup_probability += 0.3
        
        # CRITICAL: Referential markers override standalone patterns
        # Example: "list all those clients" has "those" → FOLLOW_UP
        # Example: "show me all staff details" has no "those" → NEW_QUERY (standalone)
        if has_referential or has_modification:
            # Has explicit reference/modification words → likely follow-up
            is_standalone = False  # Override standalone detection
        
        # Standalone queries should NOT be follow-ups (unless overridden above)
        if is_standalone:
            followup_probability = 0.0  # Force to zero
        # Otherwise, reduce probability if explicit entity detected without referential markers
        elif detected_entity and not has_referential:
            followup_probability *= 0.2  # Strongly reduce follow-up probability
        
        return {
            "has_referential": has_referential,
            "referential_count": referential_count,
            "has_modification": has_modification,
            "modification_count": modification_count,
            "is_elliptical": is_elliptical,
            "followup_probability": min(1.0, followup_probability),
            "detected_entity": detected_entity,  # NEW: Include detected entity
            "is_standalone": is_standalone,  # NEW: Include standalone flag (after override)
        }
    
    def _is_elliptical_query(self, query: str) -> bool:
        """
        Check if query is elliptical (incomplete, needs context).
        
        Examples:
        - "only approved ones" → elliptical
        - "in Q1 2024" → elliptical  
        - "pending ones" → elliptical
        - "that list" → elliptical
        - "show me all customers" → complete
        """
        query_lower = query.lower().strip()
        
        # Patterns that indicate elliptical queries
        elliptical_patterns = [
            # Starts with modifier
            query_lower.startswith("only "),
            query_lower.startswith("just "),
            query_lower.startswith("but "),
            query_lower.startswith("except "),
            query_lower.startswith("in "),
            query_lower.startswith("for "),
            query_lower.startswith("from "),
            
            # Status/state modifiers as standalone ("approved ones", "pending ones")
            any(
                query_lower.endswith(suffix)
                for suffix in [" ones", " one", " list", " results", " data", " records"]
            ) and len(query.split()) <= 3,
            
            # "that list", "the results", "those records"
            any(
                query_lower.startswith(prefix)
                for prefix in ["that ", "the ", "those ", "these "]
            ) and len(query.split()) <= 3,
            
            # Very short queries with modifiers
            len(query.split()) <= 4 and any(
                mod in query_lower 
                for mod in ["only", "just", "also", "same", "too", "approved", "pending", "active", "inactive"]
            ),
            
            # Time/filter modifiers without subject
            any(
                query_lower.startswith(prefix)
                for prefix in ["last ", "this ", "next ", "top ", "bottom "]
            ) and "show" not in query_lower and "get" not in query_lower,
        ]
        
        return any(elliptical_patterns)
    
    def _determine_turn_class(
        self,
        router_decision: schemas.RouterDecision,
        followup_is_followup: bool,
        followup_confidence: float,
        query_signals: Dict[str, Any],
        has_previous_sql: bool,
        previous_messages_count: int,
        previous_table: Optional[str] = None,
    ) -> tuple[TurnClass, str]:
        """
        Determine the final turn class by combining all signals.
        
        This is the CORE arbitration logic that resolves conflicts between
        different components.
        """
        reasoning_parts = []
        
        # Signal 1: Router says follow-up?
        router_says_followup = (
            router_decision.followup_type == schemas.FollowupType.RUN_SQL_FOLLOW_UP
            if router_decision else False
        )
        
        # Signal 2: LLM classifier says follow-up with high confidence?
        llm_says_followup_confident = (
            followup_is_followup and followup_confidence >= self.MEDIUM_CONFIDENCE_THRESHOLD
        )
        
        # Signal 3: Query signals indicate follow-up?
        query_signals_followup = query_signals.get("followup_probability", 0.0) >= 0.5
        
        # Signal 4: Has context to follow up on?
        # Check both last_turn_state SQL AND RAG-detected follow-up context.
        # After logout/re-login, last_turn_state may be empty but RAG still
        # finds relevant previous queries from the session history.
        has_followup_context = (
            (has_previous_sql and previous_messages_count > 0)
            or (followup_is_followup and followup_confidence > 0 and previous_messages_count > 0)
        )
        
        # NEW Signal 5: Entity change detection
        detected_entity = query_signals.get("detected_entity")
        has_referential = query_signals.get("has_referential", False)
        is_standalone = query_signals.get("is_standalone", False)
        entity_changed = detected_entity and previous_table and detected_entity != previous_table
        
        # CRITICAL OVERRIDE 1: Standalone complete queries = NEW_QUERY
        # Example: "show me all staff details" is complete and self-contained
        # Even if it's about the same entity as before, it's a fresh request
        if is_standalone:
            reasoning_parts.append("Query is standalone/complete (doesn't need prior context)")
            reasoning_parts.append(f"Pattern matched: self-contained query structure")
            return TurnClass.NEW_QUERY, " | ".join(reasoning_parts)
        
        # CRITICAL OVERRIDE 2: Explicit entity change without referential markers = NEW_QUERY
        # Example: "show me all staff details" after loans query → NEW_QUERY
        # Example: "list all those clients" after customers query → FOLLOW_UP (has referential)
        if entity_changed and not has_referential:
            reasoning_parts.append(f"Entity changed ({previous_table} → {detected_entity}) without referential markers")
            reasoning_parts.append("Treating as NEW_QUERY (explicit entity switch)")
            return TurnClass.NEW_QUERY, " | ".join(reasoning_parts)
        
        # Count signals in favor of follow-up
        followup_votes = sum([
            router_says_followup,
            llm_says_followup_confident,
            query_signals_followup,
        ])
        
        # Decision logic (requires BOTH signals AND context)
        if not has_followup_context:
            # No context = cannot be follow-up
            reasoning_parts.append("No previous SQL context exists")
            return TurnClass.NEW_QUERY, " | ".join(reasoning_parts)
        
        if followup_votes >= 2:
            # Strong follow-up signal (2+ sources agree)
            reasoning_parts.append(f"Strong follow-up signal ({followup_votes}/3 sources)")
            if router_says_followup:
                reasoning_parts.append("Router indicated follow-up")
            if llm_says_followup_confident:
                reasoning_parts.append(f"LLM classifier confident ({followup_confidence:.0%})")
            if query_signals_followup:
                reasoning_parts.append("Query contains referential/modification signals")
            return TurnClass.FOLLOW_UP, " | ".join(reasoning_parts)
        
        elif followup_votes == 1:
            # Weak/ambiguous signal - use LLM as tiebreaker
            if llm_says_followup_confident:
                reasoning_parts.append("LLM classifier indicates follow-up with confidence")
                return TurnClass.FOLLOW_UP, " | ".join(reasoning_parts)
            elif followup_is_followup and followup_confidence >= self.LOW_CONFIDENCE_THRESHOLD:
                reasoning_parts.append(f"LLM suggests follow-up (moderate confidence: {followup_confidence:.0%})")
                return TurnClass.AMBIGUOUS, " | ".join(reasoning_parts)
            else:
                reasoning_parts.append("Only one weak signal, treating as new query")
                return TurnClass.NEW_QUERY, " | ".join(reasoning_parts)
        
        else:
            # No follow-up signals
            reasoning_parts.append("No follow-up signals detected")
            return TurnClass.NEW_QUERY, " | ".join(reasoning_parts)
    
    def _determine_final_tool(
        self,
        router_decision: schemas.RouterDecision,
        final_turn_class: TurnClass,
        hard_signals: Optional[schemas.RouterSignals],
        has_previous_sql: bool,
    ) -> schemas.Tool:
        """
        Determine the final tool to use.
        
        Generally trusts router decision, but applies safety corrections.
        """
        if not router_decision:
            return schemas.Tool.CHAT
        
        tool = router_decision.tool
        
        # Safety: ANALYZE_FILE requires files
        if tool == schemas.Tool.ANALYZE_FILE:
            if hard_signals and not hard_signals.has_uploaded_files:
                logger.info("[ARBITER] ANALYZE_FILE without files → CHAT")
                return schemas.Tool.CHAT
        
        # Safety: RUN_SQL follow-up requires previous SQL
        if (tool == schemas.Tool.RUN_SQL and 
            final_turn_class == TurnClass.FOLLOW_UP and 
            not has_previous_sql):
            logger.info("[ARBITER] SQL follow-up without previous SQL → NEW_QUERY treatment")
            # Tool stays RUN_SQL, but turn_class should be NEW_QUERY
            # (handled by caller based on final_turn_class)
        
        return tool
    
    def _determine_followup_subtype(
        self,
        user_query: str,
        final_turn_class: TurnClass,
        followup_type_str: str,
        query_signals: Dict[str, Any],
    ) -> FollowUpSubtype:
        """
        Determine the specific type of follow-up modification.
        """
        if final_turn_class == TurnClass.NEW_QUERY:
            return FollowUpSubtype.NONE
        
        query_lower = user_query.lower()
        
        # Map LLM followup types to subtypes
        type_mapping = {
            "REFINEMENT": FollowUpSubtype.FILTER_ADD,
            "EXPANSION": FollowUpSubtype.EXPAND_SCOPE,
            "CLARIFICATION": FollowUpSubtype.DRILL_DOWN,
            "PIVOT": FollowUpSubtype.PIVOT_TABLE,
            "CONTINUATION": FollowUpSubtype.CONTINUATION,
        }
        
        if followup_type_str.upper() in type_mapping:
            return type_mapping[followup_type_str.upper()]
        
        # Heuristic detection based on query keywords
        if any(kw in query_lower for kw in ["group by", "break down", "by month", "by year"]):
            return FollowUpSubtype.GROUP_ADD
        elif any(kw in query_lower for kw in ["sort", "order by", "ascending", "descending"]):
            return FollowUpSubtype.SORT_ADD
        elif any(kw in query_lower for kw in ["top", "bottom", "limit", "first", "last"]):
            return FollowUpSubtype.LIMIT_CHANGE
        elif any(kw in query_lower for kw in ["only", "just", "where", "filter"]):
            return FollowUpSubtype.FILTER_ADD
        elif any(kw in query_lower for kw in ["all", "everything", "expand", "more"]):
            return FollowUpSubtype.EXPAND_SCOPE
        
        # Default to continuation
        return FollowUpSubtype.CONTINUATION
    
    def _can_skip_orchestration(
        self,
        final_turn_class: TurnClass,
        final_tool: schemas.Tool,
        router_confidence: float,
        has_previous_sql: bool,
        user_query: str = "",
    ) -> bool:
        """
        Determine if heavy orchestration can be skipped.
        
        THIS IS THE FIX for "orchestration skip too early" bug.
        Now we only skip AFTER the final turn class is determined.
        
        PERFORMANCE FIX: Skip orchestration for simple standalone queries.
        These are simple list queries that don't need multi-table analysis.
        """
        # Never skip for ambiguous cases
        if final_turn_class == TurnClass.AMBIGUOUS:
            return False
        
        # PERFORMANCE FIX: Skip orchestration for standalone query patterns
        # These are simple queries like "show me all staff details" that:
        # - Are NEW_QUERY (not follow-ups)
        # - Match standalone patterns (show/list/get all X)
        # - Don't need complex multi-table orchestration
        if final_turn_class == TurnClass.NEW_QUERY and final_tool == schemas.Tool.RUN_SQL:
            import re
            query_lower = user_query.lower().strip()
            for pattern in self.STANDALONE_QUERY_PATTERNS:
                if re.search(pattern, query_lower):
                    logger.info(f"[ARBITER] ⚡ PERFORMANCE: Skipping orchestration for standalone query pattern")
                    return True
        
        # Skip for high-confidence new SQL queries
        if (final_turn_class == TurnClass.NEW_QUERY and 
            final_tool == schemas.Tool.RUN_SQL and 
            router_confidence >= self.HIGH_CONFIDENCE_THRESHOLD):
            return True
        
        # Skip for chat queries
        if final_tool == schemas.Tool.CHAT:
            return True
        
        # For follow-ups, only skip if we have good previous context
        if (final_turn_class == TurnClass.FOLLOW_UP and
            has_previous_sql and
            router_confidence >= self.HIGH_CONFIDENCE_THRESHOLD):
            return True
        
        return False
    
    def _calculate_final_confidence(
        self,
        router_confidence: float,
        followup_confidence: float,
        query_signals: Dict[str, Any],
        final_turn_class: TurnClass,
    ) -> float:
        """
        Calculate final confidence score by combining multiple sources.
        """
        # Start with router confidence
        confidence = router_confidence * 0.5
        
        # Add followup classifier confidence (if relevant)
        if final_turn_class == TurnClass.FOLLOW_UP:
            confidence += followup_confidence * 0.3
        else:
            confidence += (1.0 - followup_confidence) * 0.3
        
        # Add query signal confidence
        query_prob = query_signals.get("followup_probability", 0.0)
        if final_turn_class == TurnClass.FOLLOW_UP:
            confidence += query_prob * 0.2
        else:
            confidence += (1.0 - query_prob) * 0.2
        
        # Ambiguous cases get lower confidence
        if final_turn_class == TurnClass.AMBIGUOUS:
            confidence *= 0.7
        
        return min(1.0, max(0.0, confidence))
    
    def _build_reasoning(
        self,
        final_turn_class: TurnClass,
        turn_class_reasoning: str,
        final_tool: schemas.Tool,
        final_followup_subtype: FollowUpSubtype,
        signals_collected: Dict[str, Any],
    ) -> str:
        """Build human-readable reasoning for the decision."""
        parts = [
            f"Turn class: {final_turn_class.value} ({turn_class_reasoning})",
            f"Tool: {final_tool.value}",
        ]
        
        if final_turn_class != TurnClass.NEW_QUERY:
            parts.append(f"Subtype: {final_followup_subtype.value}")
        
        return " | ".join(parts)
    
    def _create_new_query_decision(
        self,
        user_query: str,
        router_decision: schemas.RouterDecision,
        hard_signals: Optional[schemas.RouterSignals],
        signals_collected: Dict[str, Any],
        reasoning: str,
    ) -> ArbiterDecision:
        """Create a new query decision (helper for early returns)."""
        tool = router_decision.tool if router_decision else schemas.Tool.CHAT
        
        # Safety: ANALYZE_FILE requires files
        if tool == schemas.Tool.ANALYZE_FILE:
            if hard_signals and not hard_signals.has_uploaded_files:
                tool = schemas.Tool.CHAT
        
        return ArbiterDecision(
            final_tool=tool,
            final_turn_class=TurnClass.NEW_QUERY,
            final_followup_subtype=FollowUpSubtype.NONE,
            should_merge_state=False,
            should_reset_state=True,
            can_skip_orchestration=(
                router_decision.confidence >= self.HIGH_CONFIDENCE_THRESHOLD
                if router_decision else False
            ),
            confidence=router_decision.confidence if router_decision else 0.5,
            reasoning=reasoning,
            signals_used=signals_collected,
            router_decision=router_decision,
            followup_classifier_result=None,
            last_turn_state=None,
        )


# Singleton instance
_arbiter_instance: Optional[DecisionArbiter] = None


async def get_decision_arbiter(db: AsyncSession) -> DecisionArbiter:
    """Get or create the decision arbiter singleton."""
    global _arbiter_instance
    if _arbiter_instance is None:
        _arbiter_instance = DecisionArbiter(db)
    else:
        # Update DB session
        _arbiter_instance.db = db
    return _arbiter_instance
