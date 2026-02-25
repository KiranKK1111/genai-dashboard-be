"""
Token management for conversation context.

This module handles:
1. Token counting using tiktoken
2. Context window management with truncation
3. Conversation formatting to OpenAI chat message format
4. Token budget calculation for LLM calls

Similar to ChatGPT's approach, it manages conversation history
within token limits and optimizes context for efficiency.
"""

from __future__ import annotations

import tiktoken
from typing import List, Dict, Any, Tuple, Optional
from . import models


# Token limits for different models (conservative estimates)
TOKEN_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 4096,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
}

# Reserve tokens for response generation and prompts
RESPONSE_RESERVE = 2000  # Tokens reserved for assistant response
SYSTEM_PROMPT_RESERVE = 500  # Tokens for system prompt


class TokenManager:
    """Manages token counting and context window for LLM interactions."""
    
    def __init__(self, model: str = "gpt-4o"):
        """Initialize token manager with a specific model.
        
        Args:
            model: Model name (e.g., 'gpt-4o', 'gpt-4-turbo')
        """
        self.model = model
        # Use cl100k_base encoding for most models (OpenAI standard)
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback if tiktoken fails
            self.encoding = None
        
        self.max_tokens = TOKEN_LIMITS.get(model, 4096)
        self.available_tokens = self.max_tokens - RESPONSE_RESERVE - SYSTEM_PROMPT_RESERVE
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not self.encoding:
            # Rough estimate: ~4 characters per token
            return max(1, len(text) // 4)
        
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback to character-based estimation
            return max(1, len(text) // 4)
    
    def count_message_tokens(self, message: Dict[str, str]) -> int:
        """Count tokens for a single message in OpenAI format.
        
        Args:
            message: Message dict with 'role' and 'content'
            
        Returns:
            Number of tokens (includes overhead for message structure)
        """
        # Account for message structure overhead (~4 tokens per message)
        structure_overhead = 4
        content_tokens = self.count_tokens(message.get("content", ""))
        return structure_overhead + content_tokens
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count total tokens for a list of messages.
        
        Args:
            messages: List of message dicts
            
        Returns:
            Total tokens
        """
        return sum(self.count_message_tokens(msg) for msg in messages)
    
    def build_conversation_messages(
        self, 
        conversation_history: List[models.Message],
        system_prompt: str,
        current_query: str
    ) -> Tuple[List[Dict[str, str]], int]:
        """Build OpenAI-format messages with token management.
        
        Similar to ChatGPT's approach:
        1. Start with system prompt
        2. Add as much conversation history as fits
        3. Prioritize recent messages
        4. Respect token window limits
        
        Args:
            conversation_history: List of Message database objects
            system_prompt: System instructions for the LLM
            current_query: Current user query
            
        Returns:
            Tuple of (messages list, total tokens used)
        """
        messages = []
        total_tokens = 0
        
        # Step 1: Add system prompt
        system_msg = {"role": "system", "content": system_prompt}
        system_tokens = self.count_message_tokens(system_msg)
        messages.append(system_msg)
        total_tokens += system_tokens
        
        # Step 2: Add current query
        user_msg = {"role": "user", "content": current_query}
        query_tokens = self.count_message_tokens(user_msg)
        
        # Check if we have room for the query
        if total_tokens + query_tokens > self.available_tokens:
            # Not enough room - return just system + current query
            return messages + [user_msg], total_tokens + query_tokens
        
        messages.append(user_msg)
        total_tokens += query_tokens
        
        # Step 3: Add conversation history (oldest to newest)
        # Process in reverse to prioritize recent messages
        remaining_tokens = self.available_tokens - total_tokens
        history_messages = []
        
        for msg in reversed(conversation_history):
            # Convert database Message object to LLM message format
            # User messages have query field, assistant messages have response field
            if msg.query and msg.responded_at is None:
                # User message
                msg_dict = {
                    "role": "user",
                    "content": msg.query
                }
            elif msg.response and msg.responded_at is not None:
                # Assistant response
                content = ""
                if isinstance(msg.response, dict):
                    content = msg.response.get("message", "")
                msg_dict = {
                    "role": "assistant",
                    "content": content
                }
            else:
                # Skip ambiguous messages
                continue
                
            msg_tokens = self.count_message_tokens(msg_dict)
            
            if msg_tokens > remaining_tokens:
                # Can't fit this message - skip rest of older history
                break
            
            history_messages.insert(0, msg_dict)
            remaining_tokens -= msg_tokens
            total_tokens += msg_tokens
        
        # Insert history between system and current query
        if history_messages:
            messages = messages[:1] + history_messages + messages[1:]
        
        return messages, total_tokens
    
    def truncate_conversation(
        self,
        messages: List[models.Message],
        max_messages: int = 10,
        keep_first: bool = True
    ) -> List[models.Message]:
        """Truncate conversation history to fit token limits.
        
        Strategy:
        - Always keep most recent message
        - Keep oldest message for context (if keep_first=True)
        - Summarize old messages if needed (extensible)
        
        Args:
            messages: List of Message objects
            max_messages: Maximum messages to keep
            keep_first: Whether to keep the first message
            
        Returns:
            Truncated list of messages
        """
        if len(messages) <= max_messages:
            return messages
        
        if not messages:
            return messages
        
        # Always keep the most recent message
        recent = messages[-1:]
        
        if keep_first:
            # Keep first message for context, then recent ones
            return messages[:1] + messages[-(max_messages-1):]
        else:
            # Keep only recent messages
            return messages[-max_messages:]
    
    def get_token_usage_summary(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Get summary of token usage.
        
        Args:
            messages: List of OpenAI-format messages
            
        Returns:
            Dictionary with token usage stats
        """
        total = self.count_messages_tokens(messages)
        percent = (total / self.available_tokens) * 100
        
        return {
            "total_tokens": total,
            "available_tokens": self.available_tokens,
            "percent_used": round(percent, 2),
            "tokens_remaining": self.available_tokens - total,
            "model": self.model,
            "message_count": len(messages)
        }


def get_token_manager(model: str = "gpt-4o") -> TokenManager:
    """Factory function to get a token manager instance.
    
    Args:
        model: Model name
        
    Returns:
        Initialized TokenManager
    """
    return TokenManager(model)
