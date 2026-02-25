"""Data formatting and serialization utilities."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

from .. import models
from ..token_manager import TokenManager


def make_json_serializable(obj: Any) -> Any:
    """Convert non-JSON-serializable objects to JSON-safe format.
    
    Handles common non-serializable types like objects with __dict__,
    numpy types, datetime, etc.
    
    Args:
        obj: Any Python object
        
    Returns:
        JSON-serializable version of the object
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    
    # Handle numpy types
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle datetime objects
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # For custom objects, try to convert to dict
    if hasattr(obj, '__dict__'):
        try:
            return make_json_serializable(obj.__dict__)
        except Exception:
            pass
    
    # If we can't serialize it, convert to string representation
    return str(obj)


def format_conversation_context(messages: List[models.Message]) -> str:
    """Format conversation history for LLM context.
    
    Converts database message records into a formatted string that can be
    included in the LLM prompt so it understands the conversation context.
    
    Args:
        messages: List of Message database objects from the session
        
    Returns:
        Formatted conversation string for LLM context
    """
    if not messages:
        return ""
    
    formatted = "Previous conversation:\n"
    for msg in messages:
        # Determine if this is a user query or assistant response
        if msg.query and msg.responded_at is None:
            # User message
            formatted += f"USER: {msg.query}\n"
        elif msg.response and msg.responded_at is not None:
            # Assistant response
            response_type = msg.response_type if msg.response_type else "response"
            
            # Extract message content from response dict
            if isinstance(msg.response, dict):
                message_content = msg.response.get("message", "")
                if message_content:
                    formatted += f"ASSISTANT: {message_content}\n"
                    
                # Include metadata about data queries
                metadata = msg.response.get("metadata", {})
                if isinstance(metadata, dict) and metadata.get("type") == "data_query":
                    # Could include row counts or other metadata if needed
                    pass
        else:
            # Fallback for ambiguous cases
            content = msg.query or str(msg.response.get("message", "")) if msg.response else ""
            if content:
                role = "USER" if msg.query else "ASSISTANT"
                formatted += f"{role}: {content}\n"
    
    return formatted


def current_timestamp() -> int:
    """Return a millisecond timestamp for API responses.
    
    Returns:
        Current time as milliseconds since epoch
    """
    return int(datetime.utcnow().timestamp() * 1000)


def build_capabilities() -> Dict[str, Any]:
    """Return a dictionary describing the backend's capabilities.

    This mirrors the example shown in the design document. The values
    here are purely illustrative; update them to reflect your actual
    implementation.
    
    Returns:
        Dictionary of supported capabilities and features
    """
    return {
        "supported_visualizations": ["table", "pie", "bar", "line", "heatmap"],
        "response_types": ["data_query", "file_query", "file_lookup", "config_update", "standard"],
        "advantages": [
            "Real-time SQL generation for natural language questions",
            "Upload documents and query them using retrieval augmented generation",
            "Dynamic visualisation configuration via conversational commands",
        ],
    }


def build_messages_with_token_management(
    messages: List[models.Message],
    current_query: str,
    system_prompt: str,
    model: str = "gpt-4o",
    include_metadata: bool = True
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Build OpenAI-format messages with token-aware context window management.
    
    This implements ChatGPT-like context management:
    - Counts tokens in conversation history
    - Respects model's token limits
    - Prioritizes recent messages
    - Truncates old messages if necessary
    - Provides token usage statistics
    
    Args:
        messages: History of Message database objects from the session
        current_query: The current user's question
        system_prompt: System instructions for the LLM
        model: Model name (e.g., 'gpt-4o', 'gpt-4-turbo')
        include_metadata: Whether to include message metadata hints
        
    Returns:
        Tuple of:
        - List of OpenAI-format messages with role and content
        - Dictionary with token usage statistics
    """
    token_mgr = TokenManager(model)
    
    # Build the messages using token manager
    openai_messages, total_tokens = token_mgr.build_conversation_messages(
        messages, system_prompt, current_query
    )
    
    # Get token usage summary
    usage = token_mgr.get_token_usage_summary(openai_messages)
    
    # Add metadata hints if requested (helps understand what data was used)
    if include_metadata and messages:
        # Check if there are data query results
        data_query_count = sum(
            1 for m in messages 
            if m.response_metadata and m.response_metadata.get("query_type") == "data_query"
        )
        if data_query_count > 0:
            usage["has_data_context"] = True
            usage["data_query_results"] = data_query_count
    
    return openai_messages, usage

