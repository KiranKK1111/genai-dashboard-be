"""
Efficient Context Management for Followup Queries - ChatGPT Style

This module provides intelligent context retrieval and caching for followup queries,
enabling efficient chaining of operations like ChatGPT does.

Key Features:
- Filters message history by response_type (modal_response, followup_response, confirmation_response)
- Caches last usable context for efficient retrieval
- Enables chaining: Q1 → Q2 uses Q1, Q2 → Q3 uses Q2, etc.
- Tracks context lineage for audit/debugging
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ContextSnapshot:
    """Represents a point in conversation that can be used for followups.
    
    ChatGPT-like context management: each response becomes a potential
    foundation for future queries.
    """
    response_id: str  # Message ID
    response_type: str  # modal_response, followup_response, confirmation_response
    user_query: str  # What the user asked
    query_type: str  # data_query, file_query, etc.
    sql: Optional[str] = None  # The SQL that was executed
    data: Optional[List[Dict[str, Any]]] = None  # Result data
    metadata: Optional[Dict[str, Any]] = None  # Query metadata
    created_at: Optional[datetime] = None
    can_be_used_for_followup: bool = True  # Is this usable as a context for next query?
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage or transmission."""
        return {
            "response_id": self.response_id,
            "response_type": self.response_type,
            "user_query": self.user_query,
            "query_type": self.query_type,
            "sql": self.sql,
            "data_count": len(self.data) if self.data else 0,
            "metadata_keys": list(self.metadata.keys()) if self.metadata else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class FollowupContext:
    """ChatGPT-like context state for efficient followup processing.
    
    Tracks the conversation flow and what response can be built upon.
    """
    session_id: str
    user_id: str
    
    # Current context (what the last response was)
    current_context: Optional[ContextSnapshot] = None
    
    # Context history (all usable contexts in order)
    context_history: List[ContextSnapshot] = field(default_factory=list)
    
    # Context lineage (which response was used as foundation for which followup)
    lineage: List[Tuple[str, str]] = field(default_factory=list)  # (parent_id, child_id)
    
    def add_context(self, snapshot: ContextSnapshot) -> None:
        """Add a new context snapshot to the history."""
        self.context_history.append(snapshot)
        # Update current context
        if snapshot.can_be_used_for_followup:
            self.current_context = snapshot
    
    def mark_lineage(self, parent_response_id: str, child_response_id: str) -> None:
        """Mark that a followup was built on a previous response."""
        self.lineage.append((parent_response_id, child_response_id))
    
    def get_last_usable_context(self) -> Optional[ContextSnapshot]:
        """Get the most recent context that can be used for followups."""
        # Find the last context that can be used
        for snapshot in reversed(self.context_history):
            if snapshot.can_be_used_for_followup:
                return snapshot
        return None
    
    def get_context_for_followup(self, 
                                  allowed_response_types: Optional[List[str]] = None,
                                  response_types: Optional[List[str]] = None) -> Optional[ContextSnapshot]:
        """Get context for a followup query, with optional type filtering.
        
        Args:
            allowed_response_types: List of response types to consider (for backward compat).
                                  Default: ['modal_response', 'followup_response', 'confirmation_response']
            response_types: List of response types to consider (preferred parameter name).
                           Takes precedence over allowed_response_types if both provided.
        
        Returns:
            ContextSnapshot if available, None otherwise
        """
        # Support both parameter names for flexibility
        types_to_use = response_types or allowed_response_types or \
                      ['modal_response', 'followup_response', 'confirmation_response']
        
        # Find the last context matching the allowed types
        for snapshot in reversed(self.context_history):
            if (snapshot.can_be_used_for_followup and 
                snapshot.response_type in types_to_use):
                return snapshot
        return None
    
    def get_full_lineage_chain(self) -> List[Union[str, Tuple[str, str]]]:
        """Get the full execution lineage as a chain.
        
        Example:
            ["query1", ("query1", "query2"), "query2", ...]
        """
        chain = []
        if not self.context_history:
            return chain
        
        # Add first context
        if self.context_history:
            chain.append(self.context_history[0].response_id)
        
        # Add lineage relationships
        for parent_id, child_id in self.lineage:
            chain.append((parent_id, child_id))
            chain.append(child_id)
        
        return chain


class FollowupContextManager:
    """Efficient context manager for ChatGPT-like followup handling.
    
    This manager coordinates:
    1. Context extraction from message history (filtered by type)
    2. Context caching for efficient retrieval
    3. Followup eligibility determination
    4. Lineage tracking for audit trails
    """
    
    # Response types that can be used as foundation for followups
    USABLE_RESPONSE_TYPES = {'modal_response', 'followup_response', 'confirmation_response'}
    
    # Response types that should NOT be used as context
    SKIP_RESPONSE_TYPES = {'conversational_response', 'query', 'error', 'clarifying_question'}
    
    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self._context_cache: Dict[str, ContextSnapshot] = {}  # ID -> snapshot
        self._lineage_map: Dict[str, List[str]] = {}  # parent_id -> [child_id1, child_id2, ...]
    
    def extract_context_from_message(self, message: Any) -> Optional[ContextSnapshot]:
        """Extract context snapshot from a Message database model.
        
        Args:
            message: Message model from database
        
        Returns:
            ContextSnapshot if this message can be used for followups, None otherwise
        """
        try:
            # Skip if response_type is not usable for followups
            if not hasattr(message, 'response_type'):
                return None
            
            response_type = message.response_type
            if response_type not in self.USABLE_RESPONSE_TYPES:
                return None
            
            # Skip if no response data
            if not hasattr(message, 'response') or not message.response:
                return None
            
            # Parse response data - handle multiple formats
            response_data = message.response
            if isinstance(response_data, str):
                try:
                    response_data = json.loads(response_data)
                except json.JSONDecodeError:
                    # If JSON parsing fails, can't extract context
                    return None
            
            if not isinstance(response_data, dict):
                return None
            
            # Extract SQL and data
            sql = response_data.get('sql')
            data = response_data.get('data', [])
            metadata = response_data.get('metadata', {})
            
            # Only create snapshot if we have SQL
            if not sql:
                return None
            
            # Create the snapshot
            snapshot = ContextSnapshot(
                response_id=str(message.id) if hasattr(message, 'id') else "unknown",
                response_type=response_type,
                user_query=message.query if hasattr(message, 'query') else "",
                query_type=metadata.get('query_type', 'unknown') if metadata else 'unknown',
                sql=sql,
                data=data,
                metadata=metadata,
                created_at=message.created_at if hasattr(message, 'created_at') else None,
                can_be_used_for_followup=True
            )
            
            # Cache it
            self._context_cache[snapshot.response_id] = snapshot
            
            return snapshot
        
        except Exception as e:
            print(f"[CONTEXT ERROR] Failed to extract context from message: {str(e)}")
            return None
    
    def build_followup_context(self, messages: List[Any]) -> FollowupContext:
        """Build a FollowupContext from message history.
        
        Args:
            messages: List of Message models from database, ordered chronologically
        
        Returns:
            FollowupContext with full history and current state
        """
        context = FollowupContext(
            session_id=self.session_id,
            user_id=self.user_id
        )
        
        # Extract snapshots from usable messages
        snapshots_by_query = {}  # user_query -> snapshot (to detect chain)
        skipped_count = 0
        
        for message in messages:
            # Process all messages (don't filter by role - all from database should be valid)
            snapshot = self.extract_context_from_message(message)
            if snapshot:
                context.add_context(snapshot)
                snapshots_by_query[snapshot.user_query] = snapshot
                print(f"[CONTEXT] ✓ Added {snapshot.response_type} to context history")
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            print(f"[CONTEXT] Skipped {skipped_count} messages (no SQL or incompatible type)")
        
        return context
    
    def get_last_usable_context(self, 
                               response_types: Optional[List[str]] = None) -> Optional[ContextSnapshot]:
        """Get the most recent usable context snapshot.
        
        Args:
            response_types: Filter by response types. Default: all usable types
        
        Returns:
            ContextSnapshot or None
        """
        if not response_types:
            response_types = list(self.USABLE_RESPONSE_TYPES)
        
        # Search backwards through cache by creation time
        sorted_snapshots = sorted(
            (s for s in self._context_cache.values() 
             if s.response_type in response_types and s.can_be_used_for_followup),
            key=lambda s: s.created_at or datetime.min,
            reverse=True
        )
        
        return sorted_snapshots[0] if sorted_snapshots else None
    
    def can_use_as_followup_context(self, message: Any) -> bool:
        """Check if a message can be used as context for a followup.
        
        Args:
            message: Message model to check
        
        Returns:
            True if usable, False otherwise
        """
        if not hasattr(message, 'response_type'):
            return False
        
        # Check response type
        if message.response_type not in self.USABLE_RESPONSE_TYPES:
            return False
        
        # Check if has response data
        if not hasattr(message, 'response') or not message.response:
            return False
        
        # Check if has SQL in response
        try:
            response_data = message.response
            if isinstance(response_data, str):
                response_data = json.loads(response_data)
            return 'sql' in response_data and response_data.get('sql') is not None
        except:
            return False
    
    def get_followup_chain_summary(self, context: FollowupContext) -> Dict[str, Any]:
        """Get a summary of the followup chain for debugging/audit.
        
        Args:
            context: FollowupContext to summarize
        
        Returns:
            Dictionary with chain information
        """
        return {
            "session_id": context.session_id,
            "user_id": context.user_id,
            "context_count": len(context.context_history),
            "current_context": context.current_context.to_dict() if context.current_context else None,
            "lineage_depth": len(context.lineage),
            "full_chain": context.get_full_lineage_chain(),
            "snapshot_timeline": [s.to_dict() for s in context.context_history],
        }

    def get_intelligent_context_for_join(self, 
                                         context: FollowupContext,
                                         join_type: str) -> Optional[ContextSnapshot]:
        """Intelligently select context based on JOIN type requirements.
        
        For JOIN operations:
        - CROSS JOIN requires NO ON clause
        - Others (LEFT, RIGHT, INNER, FULL) require ON clause
        
        If current context doesn't have what's needed, searches back to find
        a prior JOIN that does.
        
        Args:
            context: The followup context with history
            join_type: The requested JOIN type (CROSS, LEFT, RIGHT, etc)
        
        Returns:
            ContextSnapshot that's suitable for the requested join_type
        """
        if not context.context_history:
            return None
        
        # Get current context
        current = context.get_last_usable_context()
        if not current or not current.sql:
            return None
        
        # Determine requirements
        has_on_clause = ' ON ' in current.sql
        needs_on_clause = join_type.upper() not in ['CROSS']
        
        # If current context satisfies requirements, use it
        if has_on_clause or not needs_on_clause:
            return current
        
        # Current context doesn't have ON clause but we need it
        # Search backwards for a prior JOIN WITH ON clause
        print(f"[CONTEXT] Searching for JOIN with ON clause (current has no ON but {join_type} needs it)...")
        
        for snapshot in reversed(context.context_history):
            if snapshot.can_be_used_for_followup and snapshot.sql:
                # Check if this has both JOIN and ON clause
                if 'JOIN' in snapshot.sql and ' ON ' in snapshot.sql:
                    print(f"[CONTEXT] Found suitable context: {snapshot.response_type} at {snapshot.created_at}")
                    return snapshot
        
        # Fallback: return current even if not ideal
        return current

