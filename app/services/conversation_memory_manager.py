"""
CONVERSATION MEMORY MANAGER (P2 - Zero Hardcoding)

Context-aware memory with rolling summarization:
- Token budgeting (context window aware)
- Rolling summarization for long chats
- Importance-weighted retention (recent + important)
- Zero hardcoded limits

Configuration drives all thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Message role in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ConversationMessage:
    """Single message in conversation."""
    role: MessageRole
    content: str
    created_at: datetime
    token_count: Optional[int] = None  # Cached token count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "token_count": self.token_count,
        }


@dataclass
class ConversationSummary:
    """Summary of conversation segment."""
    summary_text: str
    original_message_count: int
    token_count: int
    created_at: datetime
    start_message_index: int
    end_message_index: int  # Exclusive
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary_text": self.summary_text,
            "original_message_count": self.original_message_count,
            "token_count": self.token_count,
            "created_at": self.created_at.isoformat(),
            "start_message_index": self.start_message_index,
            "end_message_index": self.end_message_index,
        }


@dataclass
class ConversationMemoryConfig:
    """Configuration for memory management."""
    
    # Token budgeting (everything is configurable)
    max_context_tokens: int = 6000  # Leave room for response
    summary_trigger_ratio: float = 0.8  # Summarize at 80% of budget
    token_counter_fn: Optional[Any] = None  # fn(text) → token_count
    
    # Rolling window
    keep_recent_messages: int = 5  # Always keep last N messages
    keep_messages_hours: int = 24  # Keep messages from last 24h
    
    # Summary strategy
    importance_threshold: float = 0.5  # Importance score 0-1
    
    # Summarization
    summarizer_fn: Optional[Any] = None  # fn(messages) → summary
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_context_tokens": self.max_context_tokens,
            "summary_trigger_ratio": self.summary_trigger_ratio,
            "keep_recent_messages": self.keep_recent_messages,
            "keep_messages_hours": self.keep_messages_hours,
            "importance_threshold": self.importance_threshold,
        }


@dataclass
class ConversationMemoryState:
    """Complete memory state for a session."""
    messages: List[ConversationMessage] = field(default_factory=list)
    summaries: List[ConversationSummary] = field(default_factory=list)
    total_tokens_used: int = 0
    last_summarization_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_count": len(self.messages),
            "summary_count": len(self.summaries),
            "total_tokens_used": self.total_tokens_used,
            "last_summarization_at": self.last_summarization_at.isoformat() if self.last_summarization_at else None,
            "messages": [m.to_dict() for m in self.messages],
            "summaries": [s.to_dict() for s in self.summaries],
        }


class ConversationMemoryManager:
    """
    Manage conversation memory with rolling summarization.
    
    KEY PRINCIPLE: All limits are configurable, no hardcoding.
    """
    
    def __init__(self, config: Optional[ConversationMemoryConfig] = None):
        self.config = config or ConversationMemoryConfig()
        
        # Ensure token counter is provided
        if not self.config.token_counter_fn:
            # Default: approximate 1 token per 4 characters
            self.config.token_counter_fn = lambda text: max(1, len(text) // 4)
        
        logger.info(f"Memory Manager initialized: max_tokens={self.config.max_context_tokens}, "
                   f"keep_recent={self.config.keep_recent_messages}")
    
    async def add_message(
        self,
        role: MessageRole,
        content: str,
        memory_state: ConversationMemoryState,
    ) -> ConversationMemoryState:
        """
        Add message and manage memory.
        
        Returns:
            Updated memory state (may trigger summarization)
        """
        # Count tokens
        token_count = self.config.token_counter_fn(content)
        
        # Create message
        message = ConversationMessage(
            role=role,
            content=content,
            created_at=datetime.utcnow(),
            token_count=token_count,
        )
        
        memory_state.messages.append(message)
        memory_state.total_tokens_used += token_count
        
        logger.debug(f"Message added: role={role.value}, tokens={token_count}, "
                    f"total={memory_state.total_tokens_used}")
        
        # Check if we need to summarize
        if self._should_summarize(memory_state):
            memory_state = await self._apply_rolling_summarization(memory_state)
        
        return memory_state
    
    def get_context_window(self, memory_state: ConversationMemoryState) -> str:
        """
        Get formatted context for LLM.
        
        Respects token budget and recent message requirement.
        """
        context_messages = self._select_context_messages(memory_state)
        
        # Format as conversation
        lines = []
        for msg in context_messages:
            role_label = msg.role.value.upper()
            lines.append(f"{role_label}: {msg.content}")
        
        context = "\n".join(lines)
        token_count = self.config.token_counter_fn(context)
        
        logger.debug(f"Context window: {len(context_messages)} messages, "
                    f"{token_count} tokens (budget: {self.config.max_context_tokens})")
        
        return context
    
    async def _apply_rolling_summarization(
        self,
        memory_state: ConversationMemoryState,
    ) -> ConversationMemoryState:
        """
        Summarize older messages and remove them.
        
        Keep:
        - Recent N messages (config.keep_recent_messages)
        - Messages from >config.keep_messages_hours ago are candidates
        """
        if not self.config.summarizer_fn:
            logger.warning("No summarizer function provided, skipping summarization")
            return memory_state
        
        now = datetime.utcnow()
        cutoff_time = now - timedelta(hours=self.config.keep_messages_hours)
        
        # Identify old messages
        old_messages = []
        recent_messages = []
        
        for i, msg in enumerate(memory_state.messages):
            # Keep recent N messages
            if i >= len(memory_state.messages) - self.config.keep_recent_messages:
                recent_messages.append(msg)
            # Keep messages from last 24h
            elif msg.created_at >= cutoff_time:
                recent_messages.append(msg)
            else:
                old_messages.append((i, msg))
        
        if not old_messages:
            return memory_state
        
        try:
            # Summarize old messages
            old_text = "\n".join([f"{m.role.value}: {m.content}" for _, m in old_messages])
            summary_text = await self.config.summarizer_fn(old_text)
            
            # Create summary record
            summary = ConversationSummary(
                summary_text=summary_text,
                original_message_count=len(old_messages),
                token_count=self.config.token_counter_fn(summary_text),
                created_at=datetime.utcnow(),
                start_message_index=old_messages[0][0],
                end_message_index=old_messages[-1][0] + 1,
            )
            
            memory_state.summaries.append(summary)
            
            # Replace old messages with recent ones
            memory_state.messages = recent_messages
            memory_state.last_summarization_at = datetime.utcnow()
            
            # Recalculate total tokens
            memory_state.total_tokens_used = (
                sum(m.token_count or 0 for m in memory_state.messages) +
                sum(s.token_count for s in memory_state.summaries)
            )
            
            logger.info(f"Summarization: {len(old_messages)} messages → 1 summary "
                       f"({summary.token_count} tokens)")
            
        except Exception as e:
            logger.error(f"Summarization error: {e}")
        
        return memory_state
    
    def _should_summarize(self, memory_state: ConversationMemoryState) -> bool:
        """Check if we've exceeded token budget."""
        threshold = self.config.max_context_tokens * self.config.summary_trigger_ratio
        return memory_state.total_tokens_used > threshold
    
    def _select_context_messages(self, memory_state: ConversationMemoryState) -> List[ConversationMessage]:
        """
        Select messages for context window.
        
        Strategy:
        1. Include all summaries
        2. Include recent messages
        3. Stop when hitting token budget
        """
        context_messages = []
        tokens_used = 0
        
        # Add summaries first
        for summary in memory_state.summaries:
            summary_msg = ConversationMessage(
                role=MessageRole.SYSTEM,
                content=f"[SUMMARY OF PREVIOUS MESSAGES]\n{summary.summary_text}",
                created_at=summary.created_at,
                token_count=summary.token_count,
            )
            context_messages.append(summary_msg)
            tokens_used += summary.token_count
        
        # Add recent messages in reverse order (newest first, we'll reverse)
        for msg in reversed(memory_state.messages):
            msg_tokens = msg.token_count or 0
            
            # Check if we'd exceed budget
            if tokens_used + msg_tokens > self.config.max_context_tokens:
                # But always include last N messages
                if len(memory_state.messages) - len(context_messages) < self.config.keep_recent_messages:
                    context_messages.insert(0, msg)
                    tokens_used += msg_tokens
                # Stop when budget exceeded and we have recent messages
                break
            
            context_messages.insert(0, msg)
            tokens_used += msg_tokens
        
        return context_messages
    
    def memory_stats(self, memory_state: ConversationMemoryState) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "message_count": len(memory_state.messages),
            "summary_count": len(memory_state.summaries),
            "total_tokens": memory_state.total_tokens_used,
            "token_budget": self.config.max_context_tokens,
            "token_usage_pct": 100 * memory_state.total_tokens_used / self.config.max_context_tokens,
            "last_summarization": memory_state.last_summarization_at.isoformat() if memory_state.last_summarization_at else None,
        }
