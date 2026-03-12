"""
Semantic Conversation State Graph - Structured state tracking for intelligent follow-ups.

This module replaces text-based conversation history with structured state graphs,
enabling precise follow-up query handling similar to ChatGPT.

Instead of parsing SQL text, we maintain structured state:
  - Entities (tables)
  - Filters (conditions)
  - Projections (columns)
  - Aggregations
  - Groupings
  - Sort order
  - Result metadata

Follow-up queries modify structured state, not regenerate from scratch.

Author: GitHub Copilot
Created: 2026-03-11
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class FollowUpOperation(str, Enum):
    """Explicit follow-up operation types."""
    NEW_QUERY = "new_query"                # Fresh query, no state inheritance
    FILTER_ADD = "filter_add"              # Add new filter condition
    FILTER_REMOVE = "filter_remove"        # Remove specific filter
    FILTER_REPLACE = "filter_replace"      # Replace all filters
    PROJECTION_EXPAND = "projection_expand"    # Add more columns
    PROJECTION_REDUCE = "projection_reduce"    # Show fewer columns
    AGGREGATE_ADD = "aggregate_add"        # Add aggregation
    AGGREGATE_REMOVE = "aggregate_remove"  # Remove aggregation
    GROUPING_CHANGE = "grouping_change"    # Change GROUP BY
    SORT_CHANGE = "sort_change"            # Change ORDER BY
    LIMIT_CHANGE = "limit_change"          # Change row limit
    VISUALIZATION_REQUEST = "visualization_request"  # Add/change chart
    DRILL_DOWN = "drill_down"              # Expand aggregate to details
    ROLL_UP = "roll_up"                    # Summarize details to aggregate
    ENTITY_SWITCH = "entity_switch"        # Change primary table
    JOIN_ADD = "join_add"                  # Add table join
    COMPARE = "compare"                    # Compare with previous result
    EXPORT_REQUEST = "export_request"      # Export results


@dataclass
class FilterCondition:
    """Individual filter condition in structured form."""
    column: str
    operator: str  # =, !=, >, <, >=, <=, LIKE, IN, BETWEEN, etc.
    value: Any
    negated: bool = False
    
    def to_sql_fragment(self) -> str:
        """Generate SQL WHERE clause fragment."""
        neg = "NOT " if self.negated else ""
        if self.operator == "IN":
            values = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in self.value)
            return f"{neg}{self.column} IN ({values})"
        elif self.operator == "BETWEEN":
            return f"{neg}{self.column} BETWEEN {self.value[0]} AND {self.value[1]}"
        elif isinstance(self.value, str):
            return f"{neg}{self.column} {self.operator} '{self.value}'"
        else:
            return f"{neg}{self.column} {self.operator} {self.value}"


@dataclass
class Aggregation:
    """Aggregation specification."""
    function: str  # COUNT, SUM, AVG, MAX, MIN, etc.
    column: Optional[str] = None  # None for COUNT(*)
    alias: Optional[str] = None
    
    def to_sql_fragment(self) -> str:
        """Generate SQL aggregation fragment."""
        if self.column:
            expr = f"{self.function}({self.column})"
        else:
            expr = f"{self.function}(*)"
        
        if self.alias:
            expr += f" AS {self.alias}"
        
        return expr


@dataclass
class SemanticQueryState:
    """
    Structured state of a SQL query turn.
    
    This replaces raw SQL text parsing with explicit structure.
    """
    # Turn identification
    turn_id: int
    session_id: str
    timestamp: str
    
    # Query classification
    tool: str  # RUN_SQL, ANALYZE_FILE, CHAT
    turn_class: str  # new_query, follow_up, continuation
    operation: FollowUpOperation = FollowUpOperation.NEW_QUERY
    
    # Intent understanding
    query_intent: str = "unknown"  # list, count, aggregate, filter, visualize, etc.
    user_query_text: str = ""
    
    # Entity resolution
    primary_entity: Optional[str] = None  # Main table
    resolved_tables: List[str] = field(default_factory=list)  # All tables involved
    
    # Query structure
    selected_columns: List[str] = field(default_factory=list)  # ["*"] or specific columns
    filters: List[FilterCondition] = field(default_factory=list)
    aggregations: List[Aggregation] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    order_by: List[Dict[str, str]] = field(default_factory=list)  # [{"column": "x", "direction": "ASC"}]
    limit: Optional[int] = None
    offset: Optional[int] = None
    
    # Joins
    joins: List[Dict[str, Any]] = field(default_factory=list)  # Join specifications
    
    # Execution results
    sql_executed: Optional[str] = None
    result_count: int = 0
    result_shape: str = "tabular"  # tabular, aggregated, scalar
    execution_time_seconds: float = 0.0
    
    # Semantic metadata
    followup_capabilities: List[str] = field(default_factory=list)  # What user can do next
    ambiguous_terms: List[str] = field(default_factory=list)  # Terms that needed resolution
    confidence: float = 1.0
    
    # Visualization
    visualization_generated: bool = False
    chart_type: Optional[str] = None
    
    # State inheritance (for follow-ups)
    parent_turn_id: Optional[int] = None  # Which turn this modifies
    state_diff: Dict[str, Any] = field(default_factory=dict)  # What changed from parent
    
    # Extensions (for custom metadata)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticQueryState":
        """Reconstruct from dictionary."""
        # Reconstruct nested objects
        if "filters" in data:
            data["filters"] = [
                FilterCondition(**f) if isinstance(f, dict) else f
                for f in data["filters"]
            ]
        if "aggregations" in data:
            data["aggregations"] = [
                Aggregation(**a) if isinstance(a, dict) else a
                for a in data["aggregations"]
            ]
        if "operation" in data and isinstance(data["operation"], str):
            data["operation"] = FollowUpOperation(data["operation"])
        
        return cls(**data)


@dataclass
class StateGraphNode:
    """Node in the conversation state graph."""
    state: SemanticQueryState
    children: List[int] = field(default_factory=list)  # Turn IDs of follow-ups
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "state": self.state.to_dict(),
            "children": self.children,
        }


class SemanticStateGraph:
    """
    Manages conversation state graph for a session.
    
    This is the core of intelligent follow-up handling - maintains structured
    state instead of relying on text-based parsing.
    """
    
    def __init__(self, session_id: str):
        """Initialize state graph for a session."""
        self.session_id = session_id
        self.nodes: Dict[int, StateGraphNode] = {}  # turn_id -> node
        self.current_turn_id: Optional[int] = None
        logger.info(f"[STATE GRAPH] Initialized for session {session_id}")
    
    def add_turn(self, state: SemanticQueryState) -> None:
        """Add new turn to graph."""
        turn_id = state.turn_id
        
        # Create node
        node = StateGraphNode(state=state)
        self.nodes[turn_id] = node
        
        # Link to parent if this is a follow-up
        if state.parent_turn_id is not None and state.parent_turn_id in self.nodes:
            parent_node = self.nodes[state.parent_turn_id]
            parent_node.children.append(turn_id)
            logger.info(f"[STATE GRAPH] Linked turn {turn_id} to parent {state.parent_turn_id}")
        
        self.current_turn_id = turn_id
        logger.info(f"[STATE GRAPH] Added turn {turn_id}: {state.query_intent} on {state.primary_entity}")
    
    def get_current_state(self) -> Optional[SemanticQueryState]:
        """Get state of current turn."""
        if self.current_turn_id is None:
            return None
        
        node = self.nodes.get(self.current_turn_id)
        return node.state if node else None
    
    def get_state(self, turn_id: int) -> Optional[SemanticQueryState]:
        """Get state of specific turn."""
        node = self.nodes.get(turn_id)
        return node.state if node else None
    
    def get_last_sql_turn(self) -> Optional[SemanticQueryState]:
        """Get most recent SQL turn (ignore CHAT turns)."""
        # Walk backwards from current turn
        if self.current_turn_id is None:
            return None
        
        for turn_id in range(self.current_turn_id, 0, -1):
            state = self.get_state(turn_id)
            if state and state.tool == "RUN_SQL":
                return state
        
        return None
    
    def inherit_state(
        self,
        parent_turn_id: int,
        operation: FollowUpOperation,
    ) -> Optional[SemanticQueryState]:
        """
        Inherit state from parent turn for follow-up query.
        
        This is the key method - allows follow-ups to modify structured state
        instead of reparsing SQL text.
        """
        parent_state = self.get_state(parent_turn_id)
        if parent_state is None:
            logger.warning(f"[STATE GRAPH] Cannot inherit from turn {parent_turn_id}, not found")
            return None
        
        # Deep copy parent state
        inherited = SemanticQueryState(
            turn_id=self.current_turn_id + 1 if self.current_turn_id else 1,
            session_id=self.session_id,
            timestamp=datetime.utcnow().isoformat(),
            tool=parent_state.tool,
            turn_class="follow_up",
            operation=operation,
            query_intent=parent_state.query_intent,
            primary_entity=parent_state.primary_entity,
            resolved_tables=parent_state.resolved_tables.copy(),
            selected_columns=parent_state.selected_columns.copy(),
            filters=parent_state.filters.copy(),
            aggregations=parent_state.aggregations.copy(),
            group_by=parent_state.group_by.copy(),
            order_by=parent_state.order_by.copy(),
            limit=parent_state.limit,
            offset=parent_state.offset,
            joins=parent_state.joins.copy(),
            parent_turn_id=parent_turn_id,
        )
        
        logger.info(
            f"[STATE GRAPH] Inherited state from turn {parent_turn_id} "
            f"for {operation.value} operation"
        )
        
        return inherited
    
    def apply_filter_add(
        self,
        state: SemanticQueryState,
        filter_condition: FilterCondition,
    ) -> SemanticQueryState:
        """Apply FILTER_ADD operation to state."""
        state.filters.append(filter_condition)
        state.state_diff["filters_added"] = [filter_condition.to_sql_fragment()]
        logger.info(f"[STATE GRAPH] Added filter: {filter_condition.to_sql_fragment()}")
        return state
    
    def apply_projection_expand(
        self,
        state: SemanticQueryState,
        columns: List[str],
    ) -> SemanticQueryState:
        """Apply PROJECTION_EXPAND operation to state."""
        if "*" in state.selected_columns:
            # Already showing all, no expansion needed
            return state
        
        # Add new columns
        for col in columns:
            if col not in state.selected_columns:
                state.selected_columns.append(col)
        
        state.state_diff["columns_added"] = columns
        logger.info(f"[STATE GRAPH] Expanded projection with: {columns}")
        return state
    
    def apply_grouping_change(
        self,
        state: SemanticQueryState,
        group_by: List[str],
    ) -> SemanticQueryState:
        """Apply GROUPING_CHANGE operation to state."""
        old_group_by = state.group_by
        state.group_by = group_by
        state.query_intent = "aggregate"
        state.result_shape = "aggregated"
        state.state_diff["group_by_changed"] = {
            "from": old_group_by,
            "to": group_by,
        }
        logger.info(f"[STATE GRAPH] Changed grouping to: {group_by}")
        return state
    
    def apply_sort_change(
        self,
        state: SemanticQueryState,
        order_by: List[Dict[str, str]],
    ) -> SemanticQueryState:
        """Apply SORT_CHANGE operation to state."""
        state.order_by = order_by
        state.state_diff["order_by_changed"] = order_by
        logger.info(f"[STATE GRAPH] Changed sort order to: {order_by}")
        return state
    
    def apply_limit_change(
        self,
        state: SemanticQueryState,
        limit: Optional[int],
        offset: Optional[int] = None,
    ) -> SemanticQueryState:
        """Apply LIMIT_CHANGE operation to state."""
        old_limit = state.limit
        state.limit = limit
        state.offset = offset
        state.state_diff["limit_changed"] = {
            "from": old_limit,
            "to": limit,
        }
        logger.info(f"[STATE GRAPH] Changed limit from {old_limit} to {limit}")
        return state
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary for persistence."""
        return {
            "session_id": self.session_id,
            "current_turn_id": self.current_turn_id,
            "nodes": {
                turn_id: node.to_dict()
                for turn_id, node in self.nodes.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticStateGraph":
        """Deserialize graph from dictionary."""
        graph = cls(session_id=data["session_id"])
        graph.current_turn_id = data.get("current_turn_id")
        
        # Reconstruct nodes
        for turn_id_str, node_data in data.get("nodes", {}).items():
            turn_id = int(turn_id_str)
            state = SemanticQueryState.from_dict(node_data["state"])
            node = StateGraphNode(
                state=state,
                children=node_data.get("children", []),
            )
            graph.nodes[turn_id] = node
        
        return graph


# In-memory cache (replace with Redis/DB in production)
_state_graphs: Dict[str, SemanticStateGraph] = {}


def get_state_graph(session_id: str) -> SemanticStateGraph:
    """Get or create state graph for session."""
    if session_id not in _state_graphs:
        _state_graphs[session_id] = SemanticStateGraph(session_id)
    
    return _state_graphs[session_id]


def clear_state_graph(session_id: str) -> None:
    """Clear state graph for session (e.g., on reset)."""
    if session_id in _state_graphs:
        del _state_graphs[session_id]
        logger.info(f"[STATE GRAPH] Cleared graph for session {session_id}")
