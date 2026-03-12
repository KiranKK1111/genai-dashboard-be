"""
Rate Limiting Middleware.

Two-layer protection:
  1. Per-user token-bucket (30 req/min default) via slowapi.
  2. Per-session concurrency limit (1 concurrent query) via asyncio.Semaphore.

Integration in main.py:
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from app.middleware.rate_limiter import limiter, rate_limit_exceeded_handler

    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

Integration in /query route:
    @router.post("/query")
    @limiter.limit(f"{settings.rate_limit_per_minute}/minute")
    async def query(request: Request, ...):
        async with session_semaphore(session_id):
            ...
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# ── Per-session concurrency registry ─────────────────────────────────────────
# Maps session_id → asyncio.Semaphore(1).  Per-process only; sufficient for
# single-worker deployments.  Upgrade to Redis-backed lock for multi-worker.
_session_semaphores: dict[str, asyncio.Semaphore] = {}


@asynccontextmanager
async def session_semaphore(session_id: str | None) -> AsyncIterator[None]:
    """
    Async context manager that enforces a maximum of 1 concurrent query
    per session.  Yields immediately if no session_id (new session flow).
    Raises HTTP 429 if the session already has an in-flight query.
    """
    if not session_id:
        yield
        return

    sem = _session_semaphores.setdefault(session_id, asyncio.Semaphore(1))

    acquired = sem._value > 0  # True = free slot available
    if not acquired:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=429,
            detail={
                "error": "concurrent_query_limit",
                "message": (
                    "A query is already in progress for this session. "
                    "Please wait for it to complete before sending another."
                ),
            },
        )

    async with sem:
        yield


# ── slowapi limiter ───────────────────────────────────────────────────────────
def _get_user_key(request: Request) -> str:
    """
    Extract a unique key for rate limiting.
    Prefers JWT sub claim; falls back to client IP.
    """
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.removeprefix("Bearer ").strip()
    if token:
        try:
            # Decode without verifying signature (already validated upstream)
            import base64, json as _json
            parts = token.split(".")
            if len(parts) == 3:
                padding = "=" * (-len(parts[1]) % 4)
                payload = _json.loads(base64.urlsafe_b64decode(parts[1] + padding))
                sub = payload.get("sub") or payload.get("user_id")
                if sub:
                    return f"user:{sub}"
        except Exception:
            pass
    # Fall back to IP
    forwarded = request.headers.get("X-Forwarded-For")
    return f"ip:{forwarded.split(',')[0].strip() if forwarded else request.client.host}"


try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address  # noqa: F401 (kept for re-export)

    limiter = Limiter(key_func=_get_user_key, default_limits=[])
    _slowapi_available = True
except ImportError:
    # slowapi not installed — create a no-op stub so imports don't break
    logger.warning(
        "[RATE LIMITER] slowapi not installed. "
        "Run `pip install slowapi` to enable per-user rate limiting."
    )

    class _NoOpLimiter:  # type: ignore[no-redef]
        """Stub that makes @limiter.limit(...) a no-op decorator."""
        def limit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

    limiter = _NoOpLimiter()  # type: ignore[assignment]
    _slowapi_available = False


# ── Custom rate-limit exceeded handler ────────────────────────────────────────
async def rate_limit_exceeded_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a ResponseWrapper-compatible 429 instead of slowapi's plain text."""
    body = {
        "success": False,
        "response": {
            "type": "error",
            "intent": "",
            "confidence": 0.0,
            "message": (
                "You are sending requests too quickly. "
                "Please wait a moment before trying again."
            ),
            "metadata": {
                "error": True,
                "error_type": "rate_limit_exceeded",
                "status_code": 429,
                "retry_after_seconds": 60,
            },
        },
        "timestamp": int(time.time() * 1000),
        "original_query": None,
        "session_id": None,
        "message_id": None,
    }
    return JSONResponse(status_code=429, content=body, headers={"Retry-After": "60"})
