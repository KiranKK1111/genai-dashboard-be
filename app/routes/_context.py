"""
QueryContext — shared state passed to every mode handler.

Holds everything the dispatcher has already resolved so that mode-specific
handlers don't need to re-query the DB or re-parse common inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from fastapi import Request
    from ..models import User, ChatSession, Message


@dataclass
class QueryContext:
    # ── Core request fields ────────────────────────────────────────────────────
    query: str
    mode: str
    session_id: str
    message_id: str
    num_variations: int = 5
    safe_files: List[Any] = field(default_factory=list)

    # ── Resolved DB objects ────────────────────────────────────────────────────
    db: Any = None              # AsyncSession
    current_user: Any = None    # models.User
    session: Any = None         # models.ChatSession
    message: Any = None         # models.Message (placeholder, already committed)

    # ── Progress / cancellation helpers ───────────────────────────────────────
    tracker: Any = None
    cancellation_manager: Any = None
