"""ChatGPT-Style Session State Management

This module provides comprehensive session memory management with:
1. Structured query state persistence (domain, filters, joins, aggregations, etc.)
2. Tool call & execution result tracking (SQL queries, file lookups, etc.)
3. Intelligent follow-up classification (7 types)
4. Selective history retrieval (tiered, not full history)
5. Confidence-based clarification triggering

This is the CORE of ChatGPT-like follow-up behavior.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# PART 1: Query State - What was actually asked/computed last time
# ============================================================================

class QueryDomain(Enum):
    """Domain of the query."""
    DATABASE = "database"
    FILES = "files"
    GENERAL = "general"


@dataclass
class FilterCondition:
    """Normalized filter condition."""
    column: str
    operator: str  # =, !=, >, <, >=, <=, IN, LIKE, BETWEEN
    value: Any
    is_uncertain: bool = False  # True if schema grounding had low confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregationSpec:
    """Aggregation/grouping specification."""
    group_by: List[str] = field(default_factory=list)  # Columns to group by
    metrics: Dict[str, str] = field(default_factory=dict)  # {"column": "operation"}
    having_filters: List[FilterCondition] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_by": self.group_by,
            "metrics": self.metrics,
            "having_filters": [f.to_dict() for f in self.having_filters]
        }


@dataclass
class JoinSpecification:
    """How tables are joined."""
    left_table: str
    right_table: str
    join_type: str  # INNER, LEFT, RIGHT, FULL, CROSS
    on_conditions: List[Tuple[str, str]] = field(default_factory=list)  # [(left_col, right_col), ...]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "left_table": self.left_table,
            "right_table": self.right_table,
            "join_type": self.join_type,
            "on_conditions": self.on_conditions
        }


@dataclass
class QueryState:
    """
    Structured representation of the last query's state.
    
    This is what allows deterministic follow-ups like:
    - "Now only for 2024"
    - "Break it month-wise"
    - "Show top 10"
    - "Only KYC approved"
    
    WITHOUT this, the model will try to merge/hallucinate constraints.
    
    IMPROVEMENT: Added last_result_schema to support deterministic follow-up column resolution.
    """
    
    # ===== WHAT WAS ASKED ABOUT =====
    domain: QueryDomain = QueryDomain.DATABASE
    selected_entities: List[str] = field(default_factory=list)  # Tables/views used ("customers", "transactions")
    
    # ===== HOW THEY CONNECT =====
    joins: List[JoinSpecification] = field(default_factory=list)
    
    # ===== WHAT CONSTRAINTS WERE APPLIED =====
    filters: List[FilterCondition] = field(default_factory=list)
    time_range: Optional[Tuple[str, str]] = None  # (start_date, end_date) ISO format
    
    # ===== HOW RESULTS WERE ORGANIZED =====
    aggregation: Optional[AggregationSpec] = None
    sort_spec: Optional[Dict[str, str]] = None  # {"column": "ASC|DESC"}
    limit: Optional[int] = None
    offset: Optional[int] = None
    
    # ===== HOW TO PRESENT IT =====
    visualization_hint: Optional[str] = None  # "bar_chart", "table", "line_chart", etc.
    
    # ===== CONFIDENCE & ASSUMPTIONS =====
    confidence: float = 1.0  # 0.0-1.0
    assumptions: List[str] = field(default_factory=list)  # Unclear mappings like "AP -> Andhra Pradesh"
    
    # ===== DETERMINISTIC FOLLOW-UP RESOLUTION (NEW - IMPROVEMENT 4) =====
    last_result_schema: Optional[List[str]] = None  # Column names from previous result
    selected_columns: Optional[List[str]] = None  # Explicitly selected columns (subset of schema)
    
    # ===== METADATA =====
    created_at: datetime = field(default_factory=datetime.utcnow)
    user_query: str = ""  # The natural language query that led to this state
    generated_sql: Optional[str] = None  # The SQL that was executed
    result_count: int = 0  # Rows returned
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value,
            "selected_entities": self.selected_entities,
            "joins": [j.to_dict() for j in self.joins],
            "filters": [f.to_dict() for f in self.filters],
            "time_range": self.time_range,
            "aggregation": self.aggregation.to_dict() if self.aggregation else None,
            "sort_spec": self.sort_spec,
            "limit": self.limit,
            "offset": self.offset,
            "visualization_hint": self.visualization_hint,
            "confidence": self.confidence,
            "assumptions": self.assumptions,
            "last_result_schema": self.last_result_schema,
            "selected_columns": self.selected_columns,
            "created_at": self.created_at.isoformat(),
            "user_query": self.user_query,
            "generated_sql": self.generated_sql,
            "result_count": self.result_count,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> QueryState:
        """Reconstruct from dictionary."""
        if not data:
            return QueryState()
        
        state = QueryState()
        state.domain = QueryDomain(data.get("domain", "database"))
        state.selected_entities = data.get("selected_entities", [])
        state.time_range = data.get("time_range")
        state.sort_spec = data.get("sort_spec")
        state.limit = data.get("limit")
        state.offset = data.get("offset")
        state.visualization_hint = data.get("visualization_hint")
        state.confidence = data.get("confidence", 1.0)
        state.assumptions = data.get("assumptions", [])
        state.user_query = data.get("user_query", "")
        state.generated_sql = data.get("generated_sql")
        state.result_count = data.get("result_count", 0)
        state.last_result_schema = data.get("last_result_schema")
        state.selected_columns = data.get("selected_columns")
        
        # Reconstruct filters
        for f in data.get("filters", []):
            state.filters.append(FilterCondition(**f))
        
        # Reconstruct aggregation
        agg_data = data.get("aggregation")
        if agg_data:
            state.aggregation = AggregationSpec(
                group_by=agg_data.get("group_by", []),
                metrics=agg_data.get("metrics", {}),
                having_filters=[FilterCondition(**f) for f in agg_data.get("having_filters", [])]
            )
        
        # Reconstruct joins
        for j in data.get("joins", []):
            state.joins.append(JoinSpecification(**j))
        
        return state
    
    def create_followup_state(self, followup_type: 'FollowUpType', new_user_query: str) -> 'QueryState':
        """
        Create a deterministic follow-up state that preserves prior context.
        
        IMPROVEMENT 4: Ensures follow-up queries maintain schema and filter context.
        
        Args:
            followup_type: Type of follow-up (REFINE, TRANSFORM, DRILL_DOWN, etc.)
            new_user_query: The new user query for this follow-up
            
        Returns:
            New QueryState ready for LLM-based SQL generation with context preserved
        """
        new_state = QueryState()
        
        # Preserve these across all follow-up types
        new_state.domain = self.domain
        new_state.selected_entities = self.selected_entities.copy()
        new_state.joins = [JoinSpecification(
            left_table=j.left_table,
            right_table=j.right_table,
            join_type=j.join_type,
            on_conditions=j.on_conditions.copy()
        ) for j in self.joins]
        
        # CRITICAL: Preserve result schema for deterministic column resolution
        new_state.last_result_schema = self.last_result_schema.copy() if self.last_result_schema else None
        new_state.selected_columns = self.selected_columns.copy() if self.selected_columns else None
        
        # Handle follow-up type-specific merging
        if followup_type == FollowUpType.REFINE:
            # REFINEMENT: Keep existing filters, add new ones
            new_state.filters = [FilterCondition(
                column=f.column,
                operator=f.operator,
                value=f.value,
                is_uncertain=f.is_uncertain
            ) for f in self.filters]
            new_state.time_range = self.time_range
            new_state.aggregation = self.aggregation
            new_state.sort_spec = self.sort_spec
            new_state.limit = self.limit
            
        elif followup_type == FollowUpType.TRANSFORM:
            # TRANSFORMATION: Keep filters, change aggregation/sorting/limit
            new_state.filters = [FilterCondition(
                column=f.column,
                operator=f.operator,
                value=f.value,
                is_uncertain=f.is_uncertain
            ) for f in self.filters]
            new_state.time_range = self.time_range
            # aggregation, sort_spec, limit will be re-determined by LLM
            
        elif followup_type == FollowUpType.DRILL_DOWN:
            # DRILL-DOWN: Keep everything, prepare to filter to specific values
            new_state.filters = [FilterCondition(
                column=f.column,
                operator=f.operator,
                value=f.value,
                is_uncertain=f.is_uncertain
            ) for f in self.filters]
            new_state.time_range = self.time_range
            new_state.aggregation = self.aggregation
            new_state.sort_spec = self.sort_spec
            new_state.limit = self.limit
            
        elif followup_type == FollowUpType.EXPAND:
            # EXPANSION: Relax some filters (keep domain/entities)
            new_state.filters = [FilterCondition(
                column=f.column,
                operator=f.operator,
                value=f.value,
                is_uncertain=f.is_uncertain
            ) for f in self.filters]
            new_state.time_range = self.time_range
            new_state.aggregation = self.aggregation
            
        elif followup_type == FollowUpType.COMPARE:
            # COMPARISON: Preserve structure, prepare for multi-branch queries
            new_state.filters = [FilterCondition(
                column=f.column,
                operator=f.operator,
                value=f.value,
                is_uncertain=f.is_uncertain
            ) for f in self.filters]
            new_state.time_range = self.time_range
            new_state.aggregation = self.aggregation
            
        else:  # NEW_REQUEST, EXPLAIN, RESET
            # Start fresh but keep entities for context
            pass
        
        new_state.user_query = new_user_query
        new_state.confidence = 0.8  # Follow-ups have slightly lower initial confidence
        
        logger.info(f"[DETERMINISTIC FOLLOW-UP] Type: {followup_type.value}, "
                   f"Schema preserved: {bool(new_state.last_result_schema)}, "
                   f"Entities: {new_state.selected_entities}")
        
        return new_state


# ============================================================================
# PART 2: Tool Execution Tracking
# ============================================================================

class ToolType(Enum):
    """Types of tools executed."""
    SQL_QUERY = "sql_query"
    FILE_LOOKUP = "file_lookup"
    SCHEMA_DISCOVERY = "schema_discovery"
    ENTITY_EXTRACTION = "entity_extraction"


@dataclass
class ToolCall:
    """
    Record of a tool call and its result.
    
    Persisted so we know:
    - What SQL was executed and what columns came back
    - What files were looked up
    - What entities were extracted
    """
    
    id: str  # Unique ID
    tool_type: ToolType
    input_json: Dict[str, Any]  # What was passed to the tool
    output_json: Dict[str, Any]  # What the tool returned
    success: bool
    error_message: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    def duration_ms(self) -> int:
        """Duration in milliseconds."""
        end = self.end_time or datetime.utcnow()
        return int((end - self.start_time).total_seconds() * 1000)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tool_type": self.tool_type.value,
            "input": self.input_json,
            "output": self.output_json,
            "success": self.success,
            "error": self.error_message,
            "duration_ms": self.duration_ms(),
        }


# ============================================================================
# PART 3: Follow-up Classification (7 Types)
# ============================================================================

class FollowUpType(Enum):
    """Classification of follow-up query types."""
    
    NEW_REQUEST = "new_request"  # Unrelated to previous query
    REFINE = "refine"  # Add/modify filters, time range, limit to same query
    TRANSFORM = "transform"  # Change aggregation/visualization (group_by, sort, etc.)
    DRILL_DOWN = "drill_down"  # Focus on one item from previous results
    EXPAND = "expand"  # Widen scope (remove/relax filters, get more rows)
    COMPARE = "compare"  # Compare A vs B
    EXPLAIN = "explain"  # No SQL, just explanation of previous result
    RESET = "reset"  # Start over


# ============================================================================
# PART 4: Selective History Retrieval (Tiered)
# ============================================================================

@dataclass
class SelectiveHistory:
    """Retrieved history, tiered for efficient processing."""
    
    system_rules: str = ""  # System prompt / schema rules
    schema_summary: str = ""  # Brief schema overview
    last_query_state: Optional[QueryState] = None  # Last QueryState
    recent_messages: List[Dict[str, str]] = field(default_factory=list)  # Last N turns (messages)
    semantic_results: List[Dict[str, Any]] = field(default_factory=list)  # Older relevant messages (semantic search)
    tool_calls: List[ToolCall] = field(default_factory=list)  # Recent tool calls
    
    def to_prompt_context(self) -> str:
        """Format for inclusion in LLM prompt."""
        context_parts = []
        
        context_parts.append("=" * 80)
        context_parts.append("RETRIEVAL CONTEXT")
        context_parts.append("=" * 80)
        
        # System rules
        if self.system_rules:
            context_parts.append("\n[SYSTEM RULES]")
            context_parts.append(self.system_rules)
        
        # Schema summary
        if self.schema_summary:
            context_parts.append("\n[SCHEMA SUMMARY]")
            context_parts.append(self.schema_summary)
        
        # Last query state
        if self.last_query_state:
            context_parts.append("\n[LAST QUERY STATE]")
            context_parts.append("Domain: " + self.last_query_state.domain.value)
            context_parts.append("Tables: " + ", ".join(self.last_query_state.selected_entities))
            if self.last_query_state.filters:
                context_parts.append("Previous filters:")
                for f in self.last_query_state.filters:
                    context_parts.append(f"  - {f.column} {f.operator} {f.value}")
            context_parts.append("Previous query: " + self.last_query_state.user_query)
        
        # Recent messages
        if self.recent_messages:
            context_parts.append("\n[RECENT CONVERSATION]")
            for msg in self.recent_messages:
                context_parts.append(f"{msg['role'].upper()}: {msg['content'][:200]}")
        
        # Tool calls
        if self.tool_calls:
            context_parts.append("\n[RECENT TOOL CALLS]")
            for tc in self.tool_calls[-3:]:  # Last 3 tool calls
                context_parts.append(f"- {tc.tool_type.value}: {tc.input_json.get('query', tc.input_json)[:100]}")
        
        # Semantic results
        if self.semantic_results:
            context_parts.append("\n[SEMANTIC SEARCH RESULTS (if relevant)]")
            for result in self.semantic_results[:2]:  # Top 2
                context_parts.append(f"- {result.get('query', '')[:100]}")
        
        return "\n".join(context_parts)


class SelectiveRetriever:
    """
    Retrieve history TIERED (not full chat history).
    
    Tier 1: Always include:
      - System rules + schema summary
      - Last query state
    Tier 2: Include last N turns (e.g., N=6-12 messages)
    Tier 3: Optionally: semantic retrieval from older messages if needed
    """
    
    def __init__(self, system_rules: str = "", schema_summary: str = ""):
        self.system_rules = system_rules
        self.schema_summary = schema_summary
    
    async def retrieve(
        self,
        session_id: str,
        messages: List[Dict[str, str]],  # All messages from database
        tool_calls: List[ToolCall],  # All tool calls from database
        last_query_state: Optional[QueryState] = None,
        num_recent_turns: int = 6,
    ) -> SelectiveHistory:
        """
        Retrieve tiered history.
        
        Args:
            session_id: Session ID
            messages: All messages in session
            tool_calls: All tool calls in session
            last_query_state: Last QueryState
            num_recent_turns: How many recent turns to include
        
        Returns:
            SelectiveHistory with Tier 1, 2, and optional 3
        """
        
        # Tier 1: Always included
        history = SelectiveHistory(
            system_rules=self.system_rules,
            schema_summary=self.schema_summary,
            last_query_state=last_query_state,
            recent_messages=messages[-num_recent_turns:] if messages else [],
            tool_calls=tool_calls[-5:] if tool_calls else [],
        )
        
        # Tier 3: Semantic search (TODO: implement when vector DB is ready)
        # For now, skip this
        
        return history


# ============================================================================
# PART 5: SessionStateManager - Main Orchestrator
# ============================================================================

class SessionStateManager:
    """
    Orchestrates session memory with:
    1. Query state persistence
    2. Tool call tracking
    3. Follow-up classification
    4. Selective history retrieval
    
    This is the CORE that enables ChatGPT-like follow-ups.
    """
    
    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        
        # State
        self.last_query_state: Optional[QueryState] = None
        self.tool_calls: List[ToolCall] = []
        self.messages: List[Dict[str, str]] = []
    
    def record_query_execution(
        self,
        user_query: str,
        domain: QueryDomain,
        selected_entities: List[str],
        filters: List[FilterCondition],
        joins: List[JoinSpecification] = None,
        aggregation: AggregationSpec = None,
        sort_spec: Dict[str, str] = None,
        limit: Optional[int] = None,
        time_range: Optional[Tuple[str, str]] = None,
        generated_sql: Optional[str] = None,
        result_count: int = 0,
        visualization_hint: Optional[str] = None,
        confidence: float = 1.0,
        assumptions: List[str] = None,
    ) -> QueryState:
        """
        Record the execution of a query and its resulting state.
        
        This MUST be called after every SQL query, file lookup, or major operation.
        It's what makes follow-ups deterministic.
        """
        
        state = QueryState(
            domain=domain,
            selected_entities=selected_entities,
            joins=joins or [],
            filters=filters,
            time_range=time_range,
            aggregation=aggregation,
            sort_spec=sort_spec,
            limit=limit,
            visualization_hint=visualization_hint,
            confidence=confidence,
            assumptions=assumptions or [],
            user_query=user_query,
            generated_sql=generated_sql,
            result_count=result_count,
        )
        
        self.last_query_state = state
        return state
    
    def record_tool_call(
        self,
        tool_type: ToolType,
        input_json: Dict[str, Any],
        output_json: Dict[str, Any],
        success: bool,
        error_message: Optional[str] = None,
    ) -> ToolCall:
        """Record a tool call (SQL, file lookup, etc.)."""
        import uuid
        
        call = ToolCall(
            id=str(uuid.uuid4()),
            tool_type=tool_type,
            input_json=input_json,
            output_json=output_json,
            success=success,
            error_message=error_message,
            end_time=datetime.utcnow(),
        )
        
        self.tool_calls.append(call)
        return call
    
    def reset_state(self) -> None:
        """
        Reset the query state for new requests or session reset.
        
        Clears the query state while preserving message history for context.
        Called when follow-up type is NEW_REQUEST or RESET.
        """
        self.last_query_state = None
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    async def get_selective_history(
        self,
        num_recent_turns: int = 6,
    ) -> SelectiveHistory:
        """Get tiered history for prompt inclusion."""
        return await self.retriever.retrieve(
            session_id=self.session_id,
            messages=self.messages,
            tool_calls=self.tool_calls,
            last_query_state=self.last_query_state,
            num_recent_turns=num_recent_turns,
        )
    
    def to_session_dict(self) -> Dict[str, Any]:
        """Export session state for database persistence."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "last_query_state": self.last_query_state.to_dict() if self.last_query_state else None,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "message_count": len(self.messages),
            "state_updated_at": datetime.utcnow().isoformat(),
        }
    
    @classmethod
    def from_session_dict(cls, data: Dict[str, Any], session_id: str = None, user_id: str = None) -> SessionStateManager:
        """Reconstruct from database persistence.
        
        Args:
            cls: Class reference (auto-provided by @classmethod)
            data: Dictionary from session.session_state (may be empty on first query)
            session_id: Current session ID (passed separately, not stored in state dict)
            user_id: Current user ID (passed separately, not stored in state dict)
        """
        # Use passed-in session_id/user_id, or fall back to dict values (if upgrading from old format)
        manager = cls(
            session_id=session_id or data.get("session_id"),
            user_id=user_id or data.get("user_id"),
        )
        
        if data.get("last_query_state"):
            manager.last_query_state = QueryState.from_dict(data["last_query_state"])
        
        return manager
