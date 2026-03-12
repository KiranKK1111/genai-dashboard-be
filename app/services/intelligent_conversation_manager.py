"""
Intelligent Conversation Manager - ChatGPT-Style Context Tracking

Advanced conversation state management with multi-turn context understanding,
dynamic memory management, and intelligent context synthesis.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ContextImportance(Enum):
    """Context importance levels for intelligent memory management"""
    CRITICAL = "critical"      # Core business entities, key results
    HIGH = "high"             # Important filters, recent successful queries
    MEDIUM = "medium"         # Supporting context, moderate relevance
    LOW = "low"              # Background info, older context
    EPHEMERAL = "ephemeral"   # Temporary context, can be discarded


@dataclass
class ConversationTurn:
    """Rich representation of a conversation turn"""
    
    # Core turn data
    timestamp: datetime
    user_query: str
    system_response: str
    
    # Query execution context
    sql_generated: Optional[str] = None
    results_count: Optional[int] = None
    execution_status: str = "unknown"
    
    # Semantic analysis
    extracted_entities: List[str] = field(default_factory=list)
    identified_intent: str = "unknown"
    confidence_score: float = 0.0
    
    # Context importance
    importance_level: ContextImportance = ContextImportance.MEDIUM
    context_tags: List[str] = field(default_factory=list)
    
    # Relationship tracking
    references_previous: bool = False
    referenced_by_later: List[int] = field(default_factory=list)  # Turn indices
    
    # Memory management
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationMemory:
    """Intelligent conversation memory with context synthesis"""
    
    # Session info
    session_id: str
    started_at: datetime
    last_active: datetime
    
    # Conversation turns
    turns: List[ConversationTurn] = field(default_factory=list)
    
    # Synthesized context
    dominant_topics: List[str] = field(default_factory=list)
    key_entities: Dict[str, int] = field(default_factory=dict)  # entity -> frequency
    conversation_flow: str = "exploratory"  # exploratory, analytical, targeted
    
    # Working context (most relevant for current turn)
    active_context: Dict[str, Any] = field(default_factory=dict)
    context_summary: str = ""
    
    # Memory optimization
    total_tokens: int = 0
    compressed_turns: List[str] = field(default_factory=list)


class IntelligentConversationManager:
    """
    ChatGPT-Style Conversation Management
    
    Features:
    - Multi-turn context understanding
    - Intelligent memory compression
    - Dynamic context synthesis  
    - Conversation flow analysis
    - Importance-based retention
    """
    
    def __init__(self, max_turns: int = 50, max_tokens: int = 8000):
        """Initialize with memory management parameters"""
        
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        
        # Session memory storage
        self.conversations: Dict[str, ConversationMemory] = {}
        
        # Context synthesis cache
        self.context_cache = {}
        
        logger.info(f"🧠 Intelligent Conversation Manager initialized (max_turns={max_turns}, max_tokens={max_tokens})")
    
    def get_conversation_context(
        self, 
        session_id: str, 
        query: str,
        include_full_history: bool = False
    ) -> Dict[str, Any]:
        """
        Get intelligent conversation context for current query
        
        Returns synthesized context optimized for current query relevance
        """
        
        if session_id not in self.conversations:
            return {
                "is_new_conversation": True,
                "context_summary": "",
                "relevant_history": [],
                "active_entities": [],
                "conversation_flow": "starting"
            }
        
        memory = self.conversations[session_id]
        
        # Analyze query for context requirements
        context_needs = self._analyze_query_context_needs(query, memory)
        
        # Get relevant turns based on query
        relevant_turns = self._select_relevant_turns(query, memory, context_needs)
        
        # Synthesize active context
        active_context = self._synthesize_active_context(relevant_turns, memory)
        
        # Update access patterns
        self._update_access_patterns(relevant_turns)
        
        return {
            "is_new_conversation": False,
            "context_summary": active_context["summary"],
            "relevant_history": relevant_turns,
            "active_entities": active_context["entities"],
            "conversation_flow": memory.conversation_flow,
            "dominant_topics": memory.dominant_topics,
            "full_history": memory.turns if include_full_history else None,
            "session_meta": {
                "turn_count": len(memory.turns),
                "session_duration": (datetime.now() - memory.started_at).total_seconds() / 60,
                "last_active": memory.last_active
            }
        }
    
    def add_conversation_turn(
        self,
        session_id: str,
        user_query: str,
        system_response: str,
        execution_metadata: Dict[str, Any] = None
    ) -> ConversationTurn:
        """Add new conversation turn with intelligent analysis"""
        
        # Initialize conversation if needed
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationMemory(
                session_id=session_id,
                started_at=datetime.now(),
                last_active=datetime.now()
            )
        
        memory = self.conversations[session_id]
        
        # Create turn with analysis
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_query=user_query,
            system_response=system_response
        )
        
        # Add execution metadata
        if execution_metadata:
            turn.sql_generated = execution_metadata.get("sql_query")
            turn.results_count = execution_metadata.get("results_count")
            turn.execution_status = execution_metadata.get("status", "unknown")
        
        # Analyze turn semantically
        self._analyze_turn_semantics(turn, memory)
        
        # Determine importance and relationships
        self._analyze_turn_importance(turn, memory)
        
        # Add to memory
        memory.turns.append(turn)
        memory.last_active = datetime.now()
        
        # Update conversation synthesis
        self._update_conversation_synthesis(memory)
        
        # Memory management
        self._manage_memory_limits(memory)
        
        logger.info(f"📝 Added turn {len(memory.turns)} to session {session_id} (importance: {turn.importance_level.value})")
        
        return turn
    
    def _analyze_query_context_needs(self, query: str, memory: ConversationMemory) -> Dict[str, Any]:
        """Analyze what context the current query needs"""
        
        query_lower = query.lower()
        
        needs = {
            "temporal": False,      # Needs time-based context
            "entity": False,        # Needs entity continuity
            "results": False,       # Needs previous results
            "filters": False,       # Needs filter context
            "comparison": False,    # Needs comparative context
            "procedural": False     # Needs step-by-step context
        }
        
        # Detect context needs from query language
        if any(word in query_lower for word in ["those", "them", "these", "that", "previous", "above"]):
            needs["entity"] = True
            needs["results"] = True
        
        if any(word in query_lower for word in ["also", "additionally", "and", "plus"]):
            needs["filters"] = True
            needs["entity"] = True
        
        if any(word in query_lower for word in ["compare", "vs", "versus", "against", "than"]):
            needs["comparison"] = True
            needs["temporal"] = True
        
        if any(word in query_lower for word in ["when", "since", "before", "after", "during"]):
            needs["temporal"] = True
        
        if any(word in query_lower for word in ["step", "next", "then", "after that"]):
            needs["procedural"] = True
        
        return needs
    
    def _select_relevant_turns(self, query: str, memory: ConversationMemory, context_needs: Dict) -> List[Dict]:
        """Select most relevant turns for current query context"""
        
        relevant_turns = []
        
        # Always include recent high-importance turns
        for turn in reversed(memory.turns[-5:]):  # Last 5 turns
            if turn.importance_level in [ContextImportance.CRITICAL, ContextImportance.HIGH]:
                relevant_turns.append({
                    "user_query": turn.user_query,
                    "timestamp": turn.timestamp,
                    "entities": turn.extracted_entities,
                    "sql": turn.sql_generated,
                    "results_count": turn.results_count,
                    "importance": turn.importance_level.value
                })
        
        # Add context-specific turns based on needs
        if context_needs.get("entity") or context_needs.get("results"):
            # Find turns with successful results and shared entities
            for turn in reversed(memory.turns[-10:]):
                if (turn.results_count and turn.results_count > 0 and 
                    any(entity in turn.extracted_entities for entity in self._extract_query_entities(query))):
                    
                    turn_data = {
                        "user_query": turn.user_query,
                        "sql": turn.sql_generated,
                        "results_count": turn.results_count,
                        "entities": turn.extracted_entities,
                        "relevance": "entity_match"
                    }
                    if turn_data not in relevant_turns:
                        relevant_turns.append(turn_data)
        
        if context_needs.get("temporal"):
            # Find turns with temporal elements
            for turn in reversed(memory.turns[-8:]):
                if any(temporal in turn.user_query.lower() for temporal in 
                      ["january", "february", "march", "2023", "2024", "last", "this", "month", "year"]):
                    turn_data = {
                        "user_query": turn.user_query,
                        "relevance": "temporal_context"
                    }
                    if turn_data not in relevant_turns:
                        relevant_turns.append(turn_data)
        
        # Limit to most relevant (avoid context overload)
        return relevant_turns[-7:]  # Max 7 relevant turns
    
    def _synthesize_active_context(self, relevant_turns: List[Dict], memory: ConversationMemory) -> Dict[str, Any]:
        """Synthesize active context from relevant turns"""
        
        # Extract active entities
        active_entities = []
        for turn_data in relevant_turns:
            entities = turn_data.get("entities", [])
            for entity in entities:
                if entity not in active_entities:
                    active_entities.append(entity)
        
        # Create context summary
        if relevant_turns:
            summary_parts = []
            
            # Most recent successful query
            for turn_data in relevant_turns:
                if turn_data.get("results_count", 0) > 0:
                    summary_parts.append(f"Previous query: {turn_data['user_query']}")
                    break
            
            # Key entities being discussed
            if active_entities:
                summary_parts.append(f"Active entities: {', '.join(active_entities[:5])}")
            
            # Conversation flow context
            summary_parts.append(f"Flow: {memory.conversation_flow}")
            
            context_summary = " | ".join(summary_parts)
        else:
            context_summary = "New conversation starting"
        
        return {
            "summary": context_summary,
            "entities": active_entities,
            "turn_count": len(relevant_turns)
        }
    
    def _analyze_turn_semantics(self, turn: ConversationTurn, memory: ConversationMemory):
        """Analyze turn for semantic content"""
        
        # Extract entities
        turn.extracted_entities = self._extract_query_entities(turn.user_query)
        
        # Detect intent
        turn.identified_intent = self._classify_query_intent(turn.user_query)
        
        # Check for references to previous turns
        turn.references_previous = self._check_previous_references(turn.user_query)
        
        # Add context tags
        turn.context_tags = self._generate_context_tags(turn)
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract business entities from query"""
        
        entities = []
        query_lower = query.lower()
        
        entity_patterns = {
            "customers": r'\b(customer|client|user|person|people)s?\b',
            "products": r'\b(product|item|service)s?\b', 
            "transactions": r'\b(transaction|payment|purchase|sale)s?\b',
            "accounts": r'\b(account|profile)s?\b',
            "employees": r'\b(employee|staff|worker)s?\b'
        }
        
        for entity_type, pattern in entity_patterns.items():
            if re.search(pattern, query_lower):
                entities.append(entity_type)
        
        return entities
    
    def _classify_query_intent(self, query: str) -> str:
        """Simple intent classification"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["list", "show", "display"]):
            return "retrieval"
        elif any(word in query_lower for word in ["count", "how many", "total"]):
            return "aggregation"
        elif any(word in query_lower for word in ["compare", "vs", "difference"]):
            return "comparison"
        elif any(word in query_lower for word in ["filter", "where", "only"]):
            return "filtering"
        else:
            return "exploration"
    
    def _check_previous_references(self, query: str) -> bool:
        """Check if query references previous conversation"""
        
        reference_indicators = [
            "those", "them", "these", "that", "previous", "above", 
            "earlier", "also", "additionally", "more"
        ]
        
        return any(indicator in query.lower() for indicator in reference_indicators)
    
    def _generate_context_tags(self, turn: ConversationTurn) -> List[str]:
        """Generate contextual tags for the turn"""
        
        tags = []
        
        # Add intent tag
        tags.append(f"intent:{turn.identified_intent}")
        
        # Add entity tags
        for entity in turn.extracted_entities:
            tags.append(f"entity:{entity}")
        
        # Add success/failure tag
        if turn.results_count and turn.results_count > 0:
            tags.append("successful_query")
        elif turn.execution_status == "error":
            tags.append("failed_query")
        
        # Add reference tag
        if turn.references_previous:
            tags.append("references_previous")
        
        return tags
    
    def _analyze_turn_importance(self, turn: ConversationTurn, memory: ConversationMemory):
        """Determine turn importance for memory management"""
        
        importance_score = 0.5  # Base score
        
        # Boost for successful queries with results
        if turn.results_count and turn.results_count > 0:
            importance_score += 0.3
        
        # Boost for entity-rich queries
        if len(turn.extracted_entities) > 2:
            importance_score += 0.2
        
        # Boost for query that references previous (likely important continuation)
        if turn.references_previous:
            importance_score += 0.2
        
        # Boost for queries with business-critical entities
        critical_entities = ["customers", "transactions", "accounts"]
        if any(entity in turn.extracted_entities for entity in critical_entities):
            importance_score += 0.2
        
        # Reduce for error queries
        if turn.execution_status == "error":
            importance_score -= 0.2
        
        # Map score to importance level
        if importance_score >= 0.8:
            turn.importance_level = ContextImportance.CRITICAL
        elif importance_score >= 0.6:
            turn.importance_level = ContextImportance.HIGH  
        elif importance_score >= 0.4:
            turn.importance_level = ContextImportance.MEDIUM
        elif importance_score >= 0.2:
            turn.importance_level = ContextImportance.LOW
        else:
            turn.importance_level = ContextImportance.EPHEMERAL
    
    def _update_conversation_synthesis(self, memory: ConversationMemory):
        """Update conversation-level synthesis"""
        
        # Update dominant topics
        entity_counts = {}
        for turn in memory.turns[-10:]:  # Recent turns
            for entity in turn.extracted_entities:
                entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        memory.key_entities = entity_counts
        memory.dominant_topics = sorted(entity_counts.keys(), key=entity_counts.get, reverse=True)[:5]
        
        # Update conversation flow
        recent_intents = [turn.identified_intent for turn in memory.turns[-5:]]
        if len(set(recent_intents)) == 1:
            memory.conversation_flow = "focused"
        elif "comparison" in recent_intents:
            memory.conversation_flow = "analytical"
        else:
            memory.conversation_flow = "exploratory"
    
    def _update_access_patterns(self, relevant_turns: List[Dict]):
        """Update access patterns for memory management"""
        
        for turn_data in relevant_turns:
            # This is simplified - in full implementation, 
            # we'd track which specific turns were accessed
            pass
    
    def _manage_memory_limits(self, memory: ConversationMemory):
        """Manage memory limits through intelligent compression"""
        
        # If we have too many turns, compress older ones
        if len(memory.turns) > self.max_turns:
            
            # Keep all CRITICAL and HIGH importance turns
            important_turns = [t for t in memory.turns if t.importance_level in [ContextImportance.CRITICAL, ContextImportance.HIGH]]
            
            # Keep recent turns regardless of importance
            recent_turns = memory.turns[-10:]
            
            # Compress the middle turns
            middle_turns = [t for t in memory.turns[:-10] if t not in important_turns]
            
            # Create compressed summaries for middle turns
            if middle_turns:
                compressed_summary = self._compress_turns_to_summary(middle_turns)
                memory.compressed_turns.append(compressed_summary)
            
            # Update turns list
            memory.turns = important_turns + recent_turns
            
            logger.info(f"🗜️ Compressed {len(middle_turns)} turns for session {memory.session_id}")
    
    def _compress_turns_to_summary(self, turns: List[ConversationTurn]) -> str:
        """Compress multiple turns into a summary"""
        
        # Extract key information
        entities = set()
        intents = []
        successful_queries = 0
        
        for turn in turns:
            entities.update(turn.extracted_entities)
            intents.append(turn.identified_intent)
            if turn.results_count and turn.results_count > 0:
                successful_queries += 1
        
        summary = (f"Compressed {len(turns)} turns: "
                  f"entities={list(entities)[:3]}, "
                  f"main_intents={list(set(intents))}, "
                  f"successful_queries={successful_queries}")
        
        return summary


# Global instance for session management
_conversation_manager: Optional[IntelligentConversationManager] = None

def get_conversation_manager() -> IntelligentConversationManager:
    """Get singleton conversation manager"""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = IntelligentConversationManager()
    return _conversation_manager