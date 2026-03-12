"""
Global Exception Handler Middleware.

Catches ALL unhandled exceptions and returns them as structured
ResponseWrapper-format JSON instead of FastAPI's default 500 HTML/text.

Registered in main.py via:
    app.add_exception_handler(Exception, global_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
"""

from __future__ import annotations

import logging
import traceback
import time
from typing import Any, Dict

from fastapi import Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse

from ..config import settings


def _cors_headers(request: Request) -> dict:
    """
    Build CORS headers for error responses.

    FastAPI's add_exception_handler can short-circuit the CORSMiddleware so
    that 500/422 responses are returned without CORS headers.  Adding them
    here ensures the browser never sees a CORS error hiding the real problem.
    """
    origin = request.headers.get("origin", "")
    allowed = getattr(settings, "cors_origins", [])
    if origin and (origin in allowed or "*" in allowed):
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Vary": "Origin",
        }
    # Fallback: allow all in development so CORS never masks real errors
    if not allowed:
        return {"Access-Control-Allow-Origin": "*"}
    return {}

logger = logging.getLogger(__name__)


def _error_response(
    status_code: int,
    message: str,
    error_type: str = "server_error",
    detail: Any = None,
    session_id: str | None = None,
) -> Dict[str, Any]:
    """Build a ResponseWrapper-compatible error payload."""
    return {
        "success": False,
        "response": {
            "type": "error",
            "intent": "",
            "confidence": 0.0,
            "message": message,
            "metadata": {
                "error": True,
                "error_type": error_type,
                "status_code": status_code,
                # Only expose internal detail in debug mode
                "detail": str(detail)[:500] if (detail and settings.debug) else None,
            },
        },
        "timestamp": int(time.time() * 1000),
        "original_query": None,
        "session_id": session_id,
        "message_id": None,
    }


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all for unhandled exceptions.

    Logs the full traceback and returns a clean JSON error response so
    clients never receive a raw 500 HTML page.
    """
    logger.error(
        "[UNHANDLED_EXCEPTION] %s %s\n%s",
        request.method,
        request.url.path,
        traceback.format_exc(limit=15),
    )
    body = _error_response(
        status_code=500,
        message="An unexpected error occurred. Please try again.",
        error_type=type(exc).__name__,
        detail=exc,
    )
    return JSONResponse(status_code=500, content=body, headers=_cors_headers(request))


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Override FastAPI's default HTTPException handler to return
    ResponseWrapper-format JSON instead of {"detail": "..."}.
    """
    if isinstance(exc.detail, str):
        message = exc.detail
        detail = None
    else:
        message = str(exc.detail) if exc.detail else f"HTTP {exc.status_code}"
        detail = exc.detail

    body = _error_response(
        status_code=exc.status_code,
        message=message,
        error_type="http_error",
        detail=detail,
    )
    extra_headers = getattr(exc, "headers", None) or {}
    return JSONResponse(
        status_code=exc.status_code,
        content=body,
        headers={**extra_headers, **_cors_headers(request)},
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic request validation errors (422 Unprocessable Entity).
    Returns a structured list of field errors.
    """
    errors = exc.errors()
    # Build human-readable summary
    summaries = []
    for err in errors[:5]:  # cap at 5 to avoid huge payloads
        loc = " → ".join(str(l) for l in err.get("loc", []))
        summaries.append(f"{loc}: {err.get('msg', 'invalid')}")
    message = "Request validation failed: " + "; ".join(summaries)

    body = _error_response(
        status_code=422,
        message=message,
        error_type="validation_error",
        detail=errors if settings.debug else None,
    )
    return JSONResponse(status_code=422, content=body, headers=_cors_headers(request))
