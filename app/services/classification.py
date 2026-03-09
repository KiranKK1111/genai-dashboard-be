"""Query classification service - determines user intent from queries."""

from __future__ import annotations

import json
import re
from typing import Optional, List

from .. import llm, models
from ..helpers import build_messages_with_token_management
import logging

logger = logging.getLogger(__name__)

def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON object from text, handling various formats.
    
    Tries multiple strategies:
    1. Extract from markdown code blocks (```json ... ```)
    2. Find first { and match braces correctly
    3. Extract all text between first { and last }
    4. Return None if no valid JSON found
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        JSON string if found, None otherwise
    """
    text = text.strip()
    
    # Strategy 1: Extract from markdown code blocks
    json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_block:
        return json_block.group(1)
    
    # Strategy 2: Find first opening brace
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    # Try to find matching closing brace by counting
    brace_count = 0
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                # Found matching closing brace
                return text[start_idx:i+1]
    
    # Strategy 3: If no matching brace found, extract from first { to last }
    last_brace = text.rfind('}')
    if last_brace > start_idx:
        return text[start_idx:last_brace+1]
    
    return None


async def classify_query(
    query: str, 
    has_files: bool = False, 
    conversation_history: str = "",
    message_history: Optional[List[models.Message]] = None,
) -> tuple[str, str, float]:
    """Classify the user's intent using LLM.

    Pure LLM-based classification with no hardcoded keyword checks.
    The LLM analyzes the conversation context to determine intent, 
    similar to how ChatGPT understands user queries naturally.
    
    Token-Aware Features:
    - If message_history is provided, uses token management for conversation context
    - Respects model's token limits
    - Prioritizes recent messages for efficiency
    - Similar to ChatGPT's context window management

    Args:
        query: The natural language prompt from the user.
        has_files: Whether one or more files were uploaded with the query.
        conversation_history: Optional formatted conversation history for context (legacy).
        message_history: Optional list of Message objects for token-aware context.

    Returns:
        A tuple of (query_type, explanation, confidence) where:
        - query_type: One of 'data_query', 'file_query', 'file_lookup', 'config_update', 'standard'
        - explanation: LLM's reasoning for the classification
        - confidence: Confidence score (0.0-1.0)
    """
    # Pure LLM-based classification - no hardcoded keyword checks
    # Let the LLM understand context from the conversation naturally
    
    # Build system prompt based on whether files are uploaded
    if has_files:
        system_prompt = (
            "Your job: Classify the user intent into exactly ONE category.\n"
            "You MUST respond with ONLY valid JSON. Nothing else. No explanations. No extra text.\n"
            "If you respond with anything other than JSON, it is a CRITICAL FAILURE.\n\n"
            "USER HAS UPLOADED FILES. Available intents:\n"
            "1. 'file_query': First-time analysis of uploaded files (summarize, extract, analyze content)\n"
            "2. 'file_lookup': Follow-up questions about files already analyzed in this conversation\n"
            "3. 'data_query': Questions about database data (even if files are uploaded)\n"
            "4. 'config_update': Change visualization, settings, or display preferences\n"
            "5. 'standard': General conversation or greetings\n\n"
            "RESPOND WITH THIS EXACT JSON STRUCTURE (single line, nothing else):\n"
            '{"type":"file_query","reasoning":"brief reason","confidence":0.0-1.0}\n'
            "Do not add quotes, don't add comments, don't add markdown. Just JSON.\n\n"
            "Examples:\n"
            '{"type":"file_query","reasoning":"User wants to analyze uploaded CSV file content","confidence":0.92}\n'
            '{"type":"data_query","reasoning":"User asking about database records despite file upload","confidence":0.87}\n'
        )
    else:
        # No files uploaded - let LLM understand intent from conversation
        system_prompt = (
            "Your job: Classify the user intent into exactly ONE category.\n"
            "You MUST respond with ONLY valid JSON. Nothing else. No explanations. No extra text.\n"
            "If you respond with anything other than JSON, it is a CRITICAL FAILURE.\n\n"
            "Available intents:\n"
            "1. 'data_query': User wants database data (show me, get me, retrieve, list records, find by id, etc.)\n"
            "2. 'file_query': User wants to upload and analyze files\n"
            "3. 'file_lookup': Follow-up about previously analyzed files\n"
            "4. 'config_update': Change visualization, layout, or display preferences\n"
            "5. 'standard': General conversation, greetings, or unclear intent\n\n"
            "RESPOND WITH THIS EXACT JSON STRUCTURE (single line, nothing else):\n"
            '{"type":"data_query","reasoning":"brief reason","confidence":0.0-1.0}\n'
            "Do not add quotes, don't add comments, don't add markdown. Just JSON.\n\n"
            "Examples:\n"
            '{"type":"data_query","reasoning":"User asking to look up a specific record by id","confidence":0.95}\n'
            '{"type":"standard","reasoning":"User greeting","confidence":0.88}\n'
        )
    
    # Use token-aware message building if message history is provided
    if message_history:
        messages, token_usage = build_messages_with_token_management(
            message_history,
            query,
            system_prompt,
            model="gpt-4o"
        )
    else:
        # Legacy string-based conversation history
        user_message = f"Classify this query: {query}"
        if conversation_history:
            user_message = f"{conversation_history}\n\nNew follow-up query to classify: {query}\n\nNote: Check if this new query is asking about the data shown in previous messages."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    
    try:
        response = await llm.call_llm(messages, stream=False, max_tokens=300, track_tokens=True)
        # Handle both string and LLMResponse objects
        response_text = str(response) if response else ""
        
        # Clean response before JSON extraction
        # Remove instruction markers [INST0], [INST1], etc.
        response_text = re.sub(r'\[INST\d+\]', '', response_text, flags=re.IGNORECASE)
        
        # Remove common conversational openings (non-greedy, stop at first . or !)
        conversational_patterns = [
            r'^Ah,?\s+I\s+see[!.]\s*',
            r'^Thank\s+you[^.!]*[.!]\s*',
            r'^I\s+appreciate[^.!]*[.!]\s*',
            r'^I\s+understand[^.!]*[.!]\s*',
            r'^Unfortunately[^.!]*[.!]\s*',
            r'^I\s+apologize[^.!]*[.!]\s*',
            r'^Great[^.!]*[.!]\s*',
        ]
        
        for pattern in conversational_patterns:
            response_text = re.sub(pattern, '', response_text, flags=re.IGNORECASE | re.MULTILINE)
        
        response_text = response_text.strip()
        
        # Extract JSON from response (handle cases where LLM includes extra text)
        json_str = extract_json_from_text(response_text)
        if not json_str:
            # If no JSON found, try to extract from LLM response text
            raise ValueError(f"No JSON found in response: {response_text[:100]}")
        
        # Parse JSON response
        try:
            response_json = json.loads(json_str)
        except json.JSONDecodeError as je:
            # If JSON parsing fails, fall through to except block
            raise ValueError(f"Invalid JSON response: {json_str[:100]} (Error: {str(je)})")
        
        query_type = response_json.get("type", "standard").lower()
        reasoning = response_json.get("reasoning", "")
        confidence = float(response_json.get("confidence", 0.5))
        
        # Validate query type
        valid_types = ["data_query", "file_query", "file_lookup", "config_update", "standard"]
        if query_type not in valid_types:
            query_type = "standard"
        
        # POST-PROCESSING: If files are uploaded, adjust classification accordingly
        if has_files:
            # If no prior context (no message history), treat as file_query not file_lookup
            # (file_lookup is for follow-up questions about previously analyzed files)
            if query_type == "file_lookup" and not message_history and not conversation_history:
                query_type = "file_query"
                reasoning = f"First interaction with files; treating as file_query: {reasoning}"
                confidence = min(1.0, confidence + 0.15)
            # If query type is "standard" with files, upgrade to file_query
            elif query_type == "standard":
                query_type = "file_query"
                reasoning = f"File detected; upgrading to file_query: {reasoning}"
                confidence = min(1.0, confidence + 0.20)
        
        return query_type, reasoning, confidence
    except Exception as e:
        # Fallback if LLM response parsing fails
        # Try pattern-based classification as last resort
        
        # First check if this is a follow-up question in conversation context
        if conversation_history or message_history:
            # Use semantic follow-up detection instead of hardcoded patterns
            from .semantic_followup_detector import SemanticFollowUpDetector
            
            detector = SemanticFollowUpDetector()
            
            history_context = conversation_history
            if message_history:
                # Extract text content from Message objects
                msg_texts = []
                for m in message_history[-5:]:  # Last 5 messages for context
                    if m.query:
                        msg_texts.append(m.query)
                    elif m.response and isinstance(m.response, dict):
                        msg_text = m.response.get("message", "")
                        if msg_text:
                            msg_texts.append(msg_text)
                history_context = " ".join(msg_texts)
            
            # Semantic follow-up detection with confidence scoring
            try:
                is_followup, confidence, analysis = detector.detect_followup(query, history_context)
                if is_followup and confidence > 0.6:
                    return "data_query", f"Semantic follow-up detection (LLM failed): {analysis}", confidence
            except Exception as semantic_e:
                logger.warning(f"Semantic follow-up detection failed: {semantic_e}")
                # Minimal fallback - simple linguistic indicators
                simple_followup = any(word in query.lower() for word in ["those", "them", "these", "that", "previous"])
                if simple_followup and len(history_context) > 10:
                    return "data_query", f"Simple follow-up detection (all failed): {str(e)}", 0.5
        
        # Check for file_query patterns when files are uploaded
        if has_files:
            file_query_patterns = r'\b(upload|file|document|csv|excel|what is|summarize|summary|analyze|extract|read|content|parse|information)\b'
            if re.search(file_query_patterns, query, re.IGNORECASE):
                return "file_query", f"Detected file analysis request. Processing file content for insights.", 0.75
        
        # Standard pattern-based classification if no follow-up context
        keywords_data = r'\b(show|get|display|retrieve|select|from|where|group|order|table|column|record|data)\b'
        if re.search(keywords_data, query, re.IGNORECASE):
            return "data_query", f"Pattern fallback (LLM failed): {str(e)}", 0.6
        return "standard", f"Classification failed: {str(e)}", 0.0
