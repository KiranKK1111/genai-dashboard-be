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
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import httpx

from .config import settings
from .token_manager import TokenManager

# Use the global logger configured in main.py (writes to app.log)
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
    messages: Union[str, List[Dict[str, str]]],
    stream: bool = False,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    model: Optional[str] = None,
    track_tokens: bool = False,
    json_mode: bool = False,
    response_format: Optional[Dict[str, Any]] = None,
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
    def _normalize_messages(raw: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        if isinstance(raw, str):
            # Backward compatibility: many services pass a single prompt string.
            # Wrap it into OpenAI-style messages.
            return [
                {"role": "system", "content": settings.llm_system_prompt},
                {"role": "user", "content": raw},
            ]
        return raw

    def _extract_json_fragment(text: str) -> str:
        """Best-effort extraction of the first JSON object/array from text."""
        candidate = (text or "").strip()
        if not candidate:
            return ""

        # Fast-path: already valid JSON
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

        # Find first JSON object
        start_candidates = [candidate.find("{"), candidate.find("[")]
        starts = [i for i in start_candidates if i != -1]
        if not starts:
            return candidate

        start = min(starts)
        opener = candidate[start]
        closer = "}" if opener == "{" else "]"

        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(candidate)):
            ch = candidate[idx]
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == opener:
                depth += 1
                continue
            if ch == closer:
                depth -= 1
                if depth == 0:
                    fragment = candidate[start : idx + 1]
                    return fragment.strip()

        return candidate[start:].strip()

    normalized_messages = _normalize_messages(messages)
    logger.debug(
        f"[LLM] call_llm invoked with {len(normalized_messages)} messages to model {model or settings.ai_factory_model}"
    )
    
    api = settings.ai_factory_api
    if not api:
        logger.error("[LLM] AI Factory API endpoint not configured. Set AI_FACTORY_API environment variable.")
        return LLMResponse("", 0, 0) if track_tokens else ""
    
    headers = {"Content-Type": "application/json"}
    if settings.ai_factory_token:
        headers["Authorization"] = f"Bearer {settings.ai_factory_token}"
    
    model_name = model or settings.ai_factory_model
    
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": normalized_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    # Optional structured output hint (OpenAI-compatible servers may support this).
    if response_format is not None:
        payload["response_format"] = response_format
    
    # Token counting (if enabled)
    input_tokens = 0
    if track_tokens:
        token_mgr = TokenManager(model_name)
        input_tokens = token_mgr.count_messages_tokens(normalized_messages)
    
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
            
            # Clean common prefixes that LLMs sometimes add
            if content:
                content = content.strip()
                for prefix in ["ASSISTANT:", "Assistant:", "assistant:", "AI:", "Bot:"]:
                    if content.startswith(prefix):
                        content = content[len(prefix):].strip()
                        break

            if json_mode:
                content = _extract_json_fragment(content)
            
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


async def call_llm_json(
    *,
    system: str,
    user: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    """Convenience helper for LLM calls that must return JSON.

    This is used by routing components that expect a JSON string.
    It attempts a best-effort fallback when a provider doesn't support
    the OpenAI `response_format` field.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    # Attempt with response_format first (if provided)
    if response_format is not None:
        result = await call_llm(
            messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            json_mode=True,
            response_format=response_format,
        )
        result_text = str(result) if isinstance(result, LLMResponse) else cast(str, result)
        if result_text.strip():
            return result_text

    # Fallback: without response_format (for providers that reject the field)
    result = await call_llm(
        messages,
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
        model=model,
        json_mode=True,
        response_format=None,
    )
    return str(result) if isinstance(result, LLMResponse) else cast(str, result)


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
        except Exception as e:
            logger.warning(f"[EMBEDDINGS] API call failed: {e}. Falling back to simple character-frequency embeddings.")
            pass
    
    # Fallback: very simple vector based on character frequencies
    # Pad to match configured embedding dimensions
    target_dims = settings.embedding_dimensions
    logger.warning(f"[EMBEDDINGS] Using fallback character-frequency embeddings (not suitable for production). Padding to {target_dims} dimensions.")
    vectors: List[List[float]] = []
    for t in texts:
        freq: Dict[str, float] = {}
        for ch in t.lower():
            if ch.isalpha():
                freq[ch] = freq.get(ch, 0.0) + 1.0
        # normalise vector by length
        length = sum(freq.values()) or 1.0
        vector = [freq.get(chr(i), 0.0) / length for i in range(ord('a'), ord('z') + 1)]
        # Pad with zeros to match configured embedding dimensions
        vector.extend([0.0] * (target_dims - len(vector)))
        vectors.append(vector)
    return vectors