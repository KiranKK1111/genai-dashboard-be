"""Intelligent response formatting for follow-up questions and data queries.

This module provides smart formatting that returns simple messages for single values
(like "what is its status?") instead of full tables, while still using tables
for complex multi-row results.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .. import llm


async def should_return_as_message(
    query: str, rows: List[Dict[str, Any]], conversation_history: str = ""
) -> Tuple[bool, Optional[str]]:
    """Determine if results should be returned as a simple message instead of a table.

    This handles follow-up questions like:
    - "what is the status?" → Extract a status-like field from result
    - "what type is it?" → Extract a type/category-like field from result
    - "how much?" → Extract an amount/value-like field from result

    Args:
        query: The user's query
        rows: SQL result rows
        conversation_history: Previous conversation context

    Returns:
        Tuple of (should_use_message, formatted_message)
    """
    # Only return as message for specific conditions
    if not query or not rows:
        return False, None

    # Check if this looks like a follow-up question asking for a property
    followup_patterns = {
        r"\b(is it|what(?:\s+kind)?|what\s+type|which|how much|how many|when|where|who|can you tell|what about|is that)\b": "property_question",
        r"\b(direction|inbound|outbound|incoming|outgoing)\b": "direction_question",
        r"\b(amount|total|balance|price|cost|value)\b": "amount_question",
        r"\b(status|state|category|type|kind)\b": "attribute_question",
        r"\b(time|date|when|created|updated|timestamp)\b": "time_question",
    }

    # Check if query matches any follow-up patterns
    is_followup = False
    question_type = None
    for pattern, qtype in followup_patterns.items():
        if re.search(pattern, query, re.IGNORECASE):
            is_followup = True
            question_type = qtype
            break

    # Only format as message if:
    # 1. This appears to be a follow-up question asking for a specific property
    # 2. We have exactly 1-3 rows (single record or a couple of results)
    if not is_followup or len(rows) > 3:
        return False, None

    # Use LLM to extract key information and format as natural message
    try:
        sample_row = rows[0] if rows else {}
        
        extraction_prompt = f"""Extract a single value or concise answer from this result based on the user's query.

User Query: {query}
Result Data: {json.dumps(sample_row, default=str)}

IMPORTANT: 
- Return ONLY the extracted value or a 1-2 sentence natural response
- NO explanation, NO table format, NO JSON
- Be conversational like ChatGPT
- Examples:
    * Query "what is the status?" → "The status is ACTIVE."
    * Query "how much?" → "The amount is 90016.32."
    * Query "when was it created?" → "It was created on 2024-01-15."

Return only the concise answer:"""

        message_text = await llm.call_llm(
            [
                {
                    "role": "system",
                    "content": "You extract key information from data and provide concise, conversational answers."
                },
                {"role": "user", "content": extraction_prompt},
            ],
            stream=False,
            max_tokens=100
        )
        
        message_text = message_text.strip()
        
        # Verify we got a reasonable response (not a table or JSON)
        if message_text and not message_text.startswith('{') and '|' not in message_text:
            return True, message_text
            
    except Exception as e:
        print(f"⚠️  Failed to format message: {e}")

    return False, None


async def determine_visualization_type(
    query: str, rows: List[Dict[str, Any]], should_use_message: bool = False
) -> Tuple[str, str, str]:
    """Determine the best visualization type and generate a message.

    Args:
        query: The user's query
        rows: SQL result rows
        should_use_message: If True, prioritize message over visualization

    Returns:
        Tuple of (chart_type, message, title)
    """
    if should_use_message or len(rows) <= 2:
        # For single/couple records, use table or message
        return "table", f"Found {len(rows)} record(s) matching your query.", "Results"

    # Check if user explicitly asked for a chart type
    chart_keywords = {
        'pie': 'pie_chart',
        'bar': 'bar',
        'line': 'line',
        'heatmap': 'heatmap',
        'histogram': 'bar',
        'graph': 'line',
        'chart': None,  # Ambiguous - will use default
    }
    
    query_lower = query.lower()
    explicit_chart_type = None
    
    for keyword, chart_type in chart_keywords.items():
        if keyword in query_lower:
            explicit_chart_type = chart_type
            break
    
    # If user explicitly asked for a specific chart type, use visualization logic
    if explicit_chart_type:
        viz_system_prompt = (
            "You are a data visualization expert. Analyze the data and return JSON.\n\n"
            f"User requested: {explicit_chart_type}\n"
            "Return only JSON: {\"visualization_type\": \"...\", \"message\": \"...\"}\n"
        )
        
        try:
            result_summary = f"Query: {query}\nRows: {len(rows)}\nSample: {json.dumps(rows[:2], default=str)}"
            
            viz_messages = [
                {"role": "system", "content": viz_system_prompt},
                {"role": "user", "content": result_summary},
            ]

            viz_response_text = await llm.call_llm(viz_messages, stream=False, max_tokens=200)
            viz_response = json.loads(viz_response_text)
            
            chart_type = viz_response.get("visualization_type", "table").lower()
            message = viz_response.get("message", f"Found {len(rows)} records.")
            
            # Validate
            valid_types = ["table", "pie_chart", "bar", "line", "heatmap"]
            if chart_type not in valid_types:
                chart_type = "table"
                
            title = chart_type.replace("_", " ").title()
            
            return chart_type, message, title
        except Exception as e:
            print(f"⚠️  Visualization determination failed: {e}")
            return "table", f"Found {len(rows)} records matching your query.", "Results"
    else:
        # DEFAULT: No explicit chart type requested - return table with data summary
        title = "Results"
        message = f"Found {len(rows)} record(s) matching your query."
        return "table", message, title


def extract_key_value(row: Dict[str, Any], query: str) -> Optional[str]:
    """Extract a single key value from a row based on the query's intent.

    Args:
        row: Single database row
        query: User's query to determine what to extract

    Returns:
        Formatted key value or None
    """
    if not row:
        return None

    # Dynamic column detection using available data
    # ZERO HARDCODING: Use actual column names from result
    available_columns = list(row.keys())
    
    # Simple heuristic matching - can be enhanced with LLM if needed
    query_lower = query.lower()
    
    # Priority-based matching
    # 1. Amount/Balance queries
    if any(word in query_lower for word in ['amount', 'total', 'balance', 'how much', 'cost', 'price']):
        for col in available_columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['amount', 'balance', 'total', 'price', 'cost', 'value']):
                value = row[col]
                if value is not None:
                    return str(value)
    
    # 2. Type/Category queries
    if any(word in query_lower for word in ['type', 'kind', 'what', 'category']):
        for col in available_columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['type', 'category', 'kind', 'direction', 'status']):
                value = row[col]
                if value is not None and isinstance(value, str):
                    return value.replace('_', ' ').title()
    
    # 3. Time/Date queries
    if any(word in query_lower for word in ['time', 'date', 'when']):
        for col in available_columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['time', 'date', 'created', 'updated', 'timestamp']):
                value = row[col]
                if value is not None:
                    return str(value)
    
    # 4. ID queries
    if any(word in query_lower for word in ['id', 'code', 'number', 'identifier']):
        for col in available_columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['id', 'code', 'number', 'key']):
                value = row[col]
                if value is not None:
                    return str(value)
    
    # 5. Fallback: Return first non-null, non-id column
    for col in available_columns:
        col_lower = col.lower()
        # Skip likely ID columns in fallback
        if not any(skip in col_lower for skip in ['_id', 'id_', 'uuid', 'guid']):
            value = row[col]
            if value is not None:
                if isinstance(value, str):
                    return value.replace('_', ' ').title()
                return str(value)
