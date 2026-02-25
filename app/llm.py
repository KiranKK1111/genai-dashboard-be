"""
LLM integration utilities.

This module defines helper functions to interact with an OpenAI‑compatible
completion API. The configuration is read from ``app.config.settings``.

Key features:
- Token-aware context management (similar to ChatGPT)
- Support for conversation history with token budgets
- Proper OpenAI chat message format
- Token usage tracking in responses

You can replace these functions with calls to your preferred model or
library as needed. If no AI service is configured, stub responses are
returned to avoid runtime errors.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from .config import settings
from .token_manager import TokenManager

# Configure logging to print to stderr
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(name)s: %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class LLMResponse:
    """Wrapper for LLM responses with token usage tracking."""
    
    def __init__(self, content: str, tokens_used: Optional[int] = None, tokens_remaining: Optional[int] = None):
        self.content = content
        self.tokens_used = tokens_used
        self.tokens_remaining = tokens_remaining
    
    def __str__(self):
        return self.content


async def call_llm(
    messages: List[Dict[str, str]],
    stream: bool = False,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    model: Optional[str] = None,
    track_tokens: bool = False,
) -> Union[str, LLMResponse]:
    """Send a chat completion request to the configured AI service.
    
    Implements ChatGPT-like token awareness:
    - Counts tokens in request (if track_tokens=True)
    - Tracks token usage in response (if track_tokens=True)
    - Supports token budgeting for multi-turn conversations
    - Returns token usage metadata (if track_tokens=True)

    Args:
        messages: List of role/content dictionaries in OpenAI chat format.
        stream: Whether to enable streaming responses.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature for the model.
        model: Override model name (uses settings.ai_factory_model if None)
        track_tokens: Whether to return LLMResponse with token tracking (default: False for backward compatibility)

    Returns:
        If track_tokens=True: LLMResponse object with content and token stats
        Otherwise: Just the assistant's response string (default behavior)
        
        Returns empty response if the call fails or no service is configured.
    """
    logger.debug(f"[LLM] call_llm invoked with {len(messages)} messages to model {model or settings.ai_factory_model}")
    print(f"[LLM] DEBUG: call_llm invoked with {len(messages)} messages", file=sys.stderr, flush=True)
    
    api = settings.ai_factory_api
    if not api:
        logger.error("[LLM] AI Factory API endpoint not configured. Set AI_FACTORY_API environment variable.")
        return LLMResponse("", 0, 0) if track_tokens else ""
    
    headers = {"Content-Type": "application/json"}
    if settings.ai_factory_token:
        headers["Authorization"] = f"Bearer {settings.ai_factory_token}"
    
    model_name = model or settings.ai_factory_model
    
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    
    # Token counting (if enabled)
    input_tokens = 0
    if track_tokens:
        token_mgr = TokenManager(model_name)
        input_tokens = token_mgr.count_messages_tokens(messages)
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(api, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # OpenAI compatible responses return a list of choices
            if not data.get("choices"):
                logger.error(f"[LLM] API response has no 'choices' field. Response: {data}")
                return LLMResponse("", input_tokens, None) if track_tokens else ""
            
            content = data["choices"][0]["message"]["content"]
            
            # Extract token usage from response if available
            output_tokens = 0
            total_tokens = 0
            if "usage" in data:
                output_tokens = data["usage"].get("completion_tokens", 0)
                total_tokens = data["usage"].get("total_tokens", input_tokens + output_tokens)
            
            if track_tokens:
                token_mgr = TokenManager(model_name)
                tokens_remaining = token_mgr.available_tokens - total_tokens
                return LLMResponse(content, total_tokens, tokens_remaining)
            else:
                return content
                
    except httpx.TimeoutException as e:
        logger.error(f"[LLM] Request timeout after 60s to {api}", exc_info=True)
        return LLMResponse("", input_tokens, None) if track_tokens else ""
    except httpx.HTTPStatusError as e:
        logger.error(f"[LLM] HTTP {e.response.status_code} from {api}. Response: {e.response.text[:500]}", exc_info=True)
        return LLMResponse("", input_tokens, None) if track_tokens else ""
    except json.JSONDecodeError as e:
        logger.error(f"[LLM] Failed to parse JSON response from {api}: {str(e)[:200]}", exc_info=True)
        return LLMResponse("", input_tokens, None) if track_tokens else ""
    except Exception as e:
        logger.error(f"[LLM] Unexpected error calling {api}: {type(e).__name__}: {str(e)[:200]}", exc_info=True)
        return LLMResponse("", input_tokens, None) if track_tokens else ""


async def embed_text(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of strings.

    If an embeddings API is configured this function makes a POST
    request to obtain vector representations. Otherwise it returns
    simple bag‑of‑words like vectors by encoding character counts.
    The fallback is deterministic but not useful for semantic search.

    Args:
        texts: A list of input strings.

    Returns:
        A list of embedding vectors (lists of floats).
    """
    if settings.embeddings_api:
        headers = {"Content-Type": "application/json"}
        if settings.embeddings_token:
            headers["Authorization"] = f"Bearer {settings.embeddings_token}"
        payload = {
            "model": settings.embeddings_model,
            "input": texts,
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(settings.embeddings_api, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return [item["embedding"] for item in data.get("data", [])]
        except Exception:
            pass
    # Fallback: very simple vector based on character frequencies
    vectors: List[List[float]] = []
    for t in texts:
        freq: Dict[str, float] = {}
        for ch in t.lower():
            if ch.isalpha():
                freq[ch] = freq.get(ch, 0.0) + 1.0
        # normalise vector by length
        length = sum(freq.values()) or 1.0
        vector = [freq.get(chr(i), 0.0) / length for i in range(ord('a'), ord('z') + 1)]
        vectors.append(vector)
    return vectors