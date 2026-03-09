"""
Context Chain Manager - Track conversation as DAG of queries.

Instead of only tracking the previous query, track entire chain:
Q1: "Get records from NY" → root of chain
Q2: "Show related records" → child of Q1, inherits NY filter
Q3: "Sort by date" → child of Q2, further refines
Q4: "Show top 10" → child of Q3

Benefits:
- Multi-turn refinements work correctly
- Contextual awareness across entire session
- Intelligent filter inheritance
- Pivot detection (when user asks unrelated question)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


@dataclass
class QueryNode:
    """Single query in conversation chain."""
    
    query_id: str  # Unique ID
    user_query: str  # Original NL query
    generated_sql: str  # Generated SQL
    result_count: int  # Rows returned
    table_name: str  # Primary table
    filters: List[Dict] = field(default_factory=list)  # WHERE conditions
    columns_selected: List[str] = field(default_factory=list)  # SELECT columns
    timestamp: datetime = field(default_factory=datetime.now)
    
    parent_id: Optional[str] = None  # Link to previous query
    followup_type: Optional[str] = None  # refinement, expansion, etc.
    followup_confidence: float = 0.0
    
    # Results for potential reuse
    result_data: Optional[List[Dict]] = None


class ContextChainManager:
    """
    Manage conversation as linked chain of queries.
    """
    
    def __init__(self, max_chain_depth: int = 10):
        """Initialize context chain manager."""
        self.max_chain_depth = max_chain_depth
        self._chains: Dict[str, List[QueryNode]] = {}  # session_id → chain of nodes
    
    async def add_query_to_chain(
        self,
        session_id: str,
        user_query: str,
        generated_sql: str,
        result_count: int,
        table_name: str,
        filters: Optional[List[Dict]] = None,
        columns: Optional[List[str]] = None,
        result_data: Optional[List[Dict]] = None,
        followup_type: Optional[str] = None,
        followup_confidence: float = 0.0,
    ) -> QueryNode:
        """
        Add new query to session's conversation chain.
        Automatically links to previous query.
        """
        
        if session_id not in self._chains:
            self._chains[session_id] = []
        
        chain = self._chains[session_id]
        
        # Generate unique query ID
        query_id = f"q_{uuid.uuid4().hex[:8]}"
        
        # Link to parent (previous query in chain)
        parent_id = chain[-1].query_id if chain else None
        
        # Create new node
        node = QueryNode(
            query_id=query_id,
            user_query=user_query,
            generated_sql=generated_sql,
            result_count=result_count,
            table_name=table_name,
            filters=filters or [],
            columns_selected=columns or [],
            timestamp=datetime.now(),
            parent_id=parent_id,
            followup_type=followup_type,
            followup_confidence=followup_confidence,
            result_data=result_data,
        )
        
        # Enforce max depth (keep recent queries)
        if len(chain) >= self.max_chain_depth:
            removed = chain.pop(0)
            logger.debug(f"[CHAIN-DEPTH] Removed oldest query: {removed.user_query[:40]}")
        
        chain.append(node)
        
        logger.info(
            f"[CONTEXT-CHAIN] Added query '{user_query[:40]}...' "
            f"(depth={len(chain)}, parent={parent_id[:8] if parent_id else 'NONE'}, "
            f"followup={followup_type})"
        )
        
        return node
    
    async def get_chain_for_session(self, session_id: str) -> List[QueryNode]:
        """Get entire query chain for session."""
        return self._chains.get(session_id, [])
    
    async def get_current_query(self, session_id: str) -> Optional[QueryNode]:
        """Get most recent query in chain."""
        chain = self._chains.get(session_id, [])
        return chain[-1] if chain else None
    
    async def get_relevant_ancestors(
        self,
        session_id: str,
        current_followup_type: Optional[str] = None,
        max_ancestors: int = 3,
    ) -> List[QueryNode]:
        """
        Get most relevant ancestor queries for current follow-up.
        
        REFINEMENT: immediate parent (to add more filters)
        EXPANSION: parent (to remove filters)
        PIVOT: root + parent (to understand context switch)
        CONTINUATION: immediate parent
        AGGREGATION: parent (to apply aggregation on same scope)
        """
        
        chain = self._chains.get(session_id, [])
        if len(chain) < 2:
            return []
        
        # Default to immediate parent
        if not current_followup_type:
            current_followup_type = "continuation"
        
        # Rules for which ancestors to include
        if current_followup_type in ["refinement", "continuation"]:
            # Get immediate parent + grandparent
            ancestors = chain[-2::-1][:max_ancestors]
        
        elif current_followup_type == "expansion":
            # Get parent (to understand what was filtered)
            ancestors = [chain[-1]] if chain else []
        
        elif current_followup_type == "pivot":
            # Get root query + immediate parent
            # Root shows original data model, parent shows recent work
            ancestors = []
            if chain:
                ancestors.append(chain[0])  # Root
                if len(chain) > 1:
                    ancestors.append(chain[-1])  # Parent
        
        elif current_followup_type == "aggregation":
            # Get parent (to apply aggregation on its scope)
            ancestors = [chain[-1]] if chain else []
        
        else:
            # Default: immediate parent
            ancestors = [chain[-1]] if chain else []
        
        logger.info(
            f"[ANCESTOR-SELECTION] {current_followup_type.upper()}: "
            f"returning {len(ancestors)} ancestors from chain of {len(chain)}"
        )
        
        return ancestors
    
    async def apply_transitive_filters(
        self,
        session_id: str,
        current_followup_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Apply filters transitively from ancestor queries.
        
        Example flow:
        Q1: "records from NY" → filters: [{city='NY'}]
        Q2 (expansion): "show orders" → inherit: [{city='NY'}]
        Q3 (refinement): "active only" → inherit: [{city='NY'}, {status='active'}]
        """
        
        ancestors = await self.get_relevant_ancestors(
            session_id,
            current_followup_type,
            max_ancestors=5
        )
        
        if not ancestors:
            return []
        
        inherited_filters = []
        
        for ancestor in ancestors:
            # Inherit filters based on followup type
            if current_followup_type in ["expansion", "continuation", "refinement"]:
                # Carry forward all filters from ancestor
                inherited_filters.extend(ancestor.filters)
            
            elif current_followup_type == "pivot":
                # For PIVOT to different table, might not apply same filters
                # But carry forward for reference
                inherited_filters.extend(ancestor.filters)
            
            elif current_followup_type == "aggregation":
                # Apply ancestor filters to aggregation scope
                inherited_filters.extend(ancestor.filters)
        
        # De-duplicate filters
        unique_filters = []
        seen = set()
        for f in inherited_filters:
            key = f"{f['column']}_{f['operator']}_{f['value']}"
            if key not in seen:
                unique_filters.append(f)
                seen.add(key)
        
        logger.info(
            f"[TRANSITIVE-FILTERS] {current_followup_type.upper()}: "
            f"inherited {len(unique_filters)} filters from {len(ancestors)} ancestors"
        )
        
        return unique_filters
    
    async def get_chain_summary(
        self,
        session_id: str,
    ) -> str:
        """
        Get human-readable summary of query chain.
        Useful for debugging and LLM prompts.
        """
        
        chain = self._chains.get(session_id, [])
        
        if not chain:
            return "No queries in chain"
        
        summary = f"Conversation Chain ({len(chain)} queries):\n"
        for i, node in enumerate(chain):
            indent = "  " * i
            followup_indicator = f" [{node.followup_type}]" if node.followup_type else ""
            summary += (
                f"{indent}Q{i+1}: {node.user_query[:60]}...{followup_indicator}\n"
                f"{indent}   Table: {node.table_name}, "
                f"Rows: {node.result_count}, "
                f"Filters: {len(node.filters)}\n"
            )
        
        return summary
    
    async def get_chain_context_for_llm(
        self,
        session_id: str,
    ) -> str:
        """
        Format chain as context for LLM.
        Used when passing to SQL generation.
        """
        
        chain = self._chains.get(session_id, [])
        
        if not chain:
            return "No conversation history"
        
        context = "CONVERSATION HISTORY:\n"
        
        for i, node in enumerate(chain):
            context += f"\n[Query {i+1}]\n"
            context += f"User: {node.user_query}\n"
            context += f"Generated SQL: {node.generated_sql[:150]}...\n"
            context += f"Table: {node.table_name}\n"
            context += f"Result rows: {node.result_count}\n"
            
            if node.filters:
                context += f"Filters: {node.filters}\n"
            
            if node.followup_type:
                context += f"Follow-up type: {node.followup_type}\n"
        
        return context
    
    async def get_pivot_detection_context(
        self,
        session_id: str,
    ) -> Dict:
        """
        Get context for detecting PIVOT follow-ups.
        Pivot = Significant shift in query scope.
        """
        
        chain = self._chains.get(session_id, [])
        
        if len(chain) < 2:
            return {}
        
        current = chain[-1]
        previous = chain[-2]
        
        # Detect if it's a pivot
        tables_different = current.table_name != previous.table_name
        filters_cleared = len(current.filters) < len(previous.filters)
        
        return {
            "is_potential_pivot": tables_different or filters_cleared,
            "previous_table": previous.table_name,
            "current_table": current.table_name,
            "previous_filters": previous.filters,
            "current_filters": current.filters,
        }
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear conversation chain for session (e.g., new conversation)."""
        if session_id in self._chains:
            removed_count = len(self._chains[session_id])
            del self._chains[session_id]
            logger.info(f"[CHAIN-CLEAR] Cleared {removed_count} queries from session {session_id}")
            return True
        return False


# Singleton instance
_context_chain_manager: Optional[ContextChainManager] = None


async def get_context_chain_manager(
    max_depth: int = 10
) -> ContextChainManager:
    """Get or create context chain manager."""
    global _context_chain_manager
    
    if _context_chain_manager is None:
        _context_chain_manager = ContextChainManager(max_chain_depth=max_depth)
    
    return _context_chain_manager
