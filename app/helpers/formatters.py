"""Data formatting and serialization utilities."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

from .. import models
from ..token_manager import TokenManager


def extract_assistant_message_text(response: Any) -> str:
    """Extract assistant-readable text from a stored response payload.

    The DB stores different response shapes depending on the route/mode.
    This helper normalizes them to a plain-text assistant message for
    conversation context.
    """
    if response is None:
        return ""

    # Pydantic models (best-effort)
    if hasattr(response, "model_dump"):
        try:
            response = response.model_dump(mode="json")
        except Exception:
            pass

    if isinstance(response, str):
        return response.strip()

    if not isinstance(response, dict):
        return str(response).strip()

    # Common legacy/standard shape
    message = response.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()

    # LamaResponse shape: { assistant: { content: [ {type,text/items}, ... ] } }
    assistant = response.get("assistant")
    if isinstance(assistant, dict):
        content = assistant.get("content")
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    if block.strip():
                        parts.append(block.strip())
                    continue
                if not isinstance(block, dict):
                    block_str = str(block).strip()
                    if block_str:
                        parts.append(block_str)
                    continue

                block_text = block.get("text")
                if isinstance(block_text, str) and block_text.strip():
                    parts.append(block_text.strip())

                items = block.get("items")
                if isinstance(items, list) and items:
                    bullet_lines: List[str] = []
                    for item in items:
                        item_str = "" if item is None else str(item).strip()
                        if item_str:
                            bullet_lines.append(f"- {item_str}")
                    if bullet_lines:
                        parts.append("\n".join(bullet_lines))

            return "\n\n".join([p for p in parts if p]).strip()

        # Fallbacks within assistant payload
        assistant_message = assistant.get("message") or assistant.get("text")
        if isinstance(assistant_message, str) and assistant_message.strip():
            return assistant_message.strip()

    # Sometimes nested under a top-level 'response'
    nested = response.get("response")
    if isinstance(nested, dict) and nested is not response:
        nested_text = extract_assistant_message_text(nested)
        if nested_text:
            return nested_text

    return ""


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
    
    CRITICAL: Includes SQL from debug metadata so follow-up queries can find it.
    
    Args:
        messages: List of Message database objects from the session
        
    Returns:
        Formatted conversation string for LLM context with SQL for follow-ups
    """
    if not messages:
        return ""
    
    formatted = "Previous conversation:\n"

    for msg in messages:
        # Skip placeholders/pending rows; they will be added as the current query separately.
        if msg.responded_at is None:
            continue

        if msg.query:
            formatted += f"USER: {msg.query}\n"

        assistant_text = extract_assistant_message_text(msg.response)
        if assistant_text:
            formatted += f"ASSISTANT: {assistant_text}\n"

        # CRITICAL FOR FOLLOW-UPS: Extract SQL from debug metadata
        if isinstance(msg.response, dict):
            debug_info = msg.response.get("debug", {})
            if isinstance(debug_info, dict):
                sql_executed = debug_info.get("sql_executed", "")
                if sql_executed:
                    formatted += f"[SQL] {sql_executed}\n"

                row_count = debug_info.get("row_count")
                if row_count is not None:
                    formatted += f"[Results] {row_count} rows returned\n"

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
            if m.response_type == "data_query"
        )
        if data_query_count > 0:
            usage["has_data_context"] = True
            usage["data_query_results"] = data_query_count
    
    return openai_messages, usage

