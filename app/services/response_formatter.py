"""Intelligent response formatting for follow-up questions and data queries.

This module provides smart formatting that returns simple messages for single values
(like "is it credit or debit?") instead of full tables, while still using tables
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
    - "is it credit or debit?" → Extract direction from result
    - "what type of transaction?" → Extract txn_type from result
    - "how much?" → Extract amount from result

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
        r'\b(is it|what(?:\s+kind)?|what\s+type|which|how much|how many|can you tell|what about|is that)\b': 'property_question',
        r'\b(credit|debit|direction)\b': 'direction_question',
        r'\b(amount|total|balance)\b': 'amount_question',
        r'\b(merchant|customer|account|person)\b': 'entity_question',
        r'\b(time|date|when)\b': 'time_question',
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
    # 1. This appears to be a follow-up question
    # 2. We have exactly 1 or 2 rows (single transaction or couple of results)
    # 3. Conversation history mentions previous data query
    if not is_followup or len(rows) > 3:
        return False, None

    has_context = 'transaction' in conversation_history.lower() or 'customer' in conversation_history.lower()
    if not has_context:
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
  * Query "is it credit or debit?" → "This transaction is a credit (incoming payment)."
  * Query "how much?" → "This transaction is for $90,016.32"
  * Query "what type?" → "This is an IMPS_IN transaction (incoming IMPS transfer)"

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

    # Map of keywords to database column names
    keyword_column_map = {
        r'\b(credit|debit|direction)\b': ['direction', 'txn_type'],
        r'\b(amount|total|balance|how\s+much)\b': ['amount', 'post_balance', 'balance'],
        r'\b(merchant|company)\b': ['merchant_id', 'merchant_code'],
        r'\b(customer|account)\b': ['customer_id', 'customer_code', 'account_id'],
        r'\b(type|kind|what)\b': ['txn_type', 'type', 'direction'],
        r'\b(time|date|when)\b': ['txn_time', 'created_at', 'date'],
        r'\b(note|description|remark)\b': ['note', 'description', 'remarks'],
    }

    # Find matching keywords in query
    for keyword_pattern, column_names in keyword_column_map.items():
        if re.search(keyword_pattern, query, re.IGNORECASE):
            # Try to extract from first matching column
            for col in column_names:
                if col in row and row[col] is not None:
                    value = row[col]
                    # Format the value nicely
                    if isinstance(value, str):
                        return value.replace('_', ' ').title()
                    return str(value)

    return None
