"""
Session State Archival Service.

When a session accumulates more than SESSION_MAX_LIVE_TURNS turns
in chat_sessions.session_state, this service archives the oldest
(total - SESSION_ARCHIVE_KEEP_TURNS) turns to the session_history table
and trims the live JSON so queries stay fast.

Call archive_if_needed() after every message response is persisted.

Design:
  - Non-blocking: runs in the same async context, but failures are silently
    logged so they never break the main query flow.
  - Idempotent: safe to call multiple times for the same session.
  - Threshold-driven: controlled by settings.session_max_live_turns and
    settings.session_archive_keep_turns (defaults 50 / 20).

Usage:
    from app.services.session_archival import archive_if_needed
    await archive_if_needed(db, session)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


async def archive_if_needed(db, session) -> None:
    """
    Check the session's turn count and archive excess turns if needed.

    Args:
        db:      AsyncSession — the open database session (write capable).
        session: ChatSession ORM instance (already loaded, not expired).
    """
    from ..config import settings
    from ..models import SessionHistory

    try:
        state: Dict[str, Any] = session.session_state or {}
        turns: List[Any] = state.get("turns", [])

        if len(turns) <= settings.session_max_live_turns:
            return  # Nothing to archive

        keep = settings.session_archive_keep_turns
        to_archive = turns[:-keep] if keep > 0 else turns
        to_keep = turns[-keep:] if keep > 0 else []

        if not to_archive:
            return

        # Determine starting turn_index for archived rows
        # (re-use already archived count as offset so indices are stable)
        existing_count: int = await _count_archived(db, session.id)

        history_rows = [
            SessionHistory(
                session_id=session.id,
                turn_index=existing_count + i,
                turn_data=turn,
            )
            for i, turn in enumerate(to_archive)
        ]
        db.add_all(history_rows)

        # Trim live state — keep only recent turns
        state["turns"] = to_keep
        session.session_state = state

        # Mark JSON column as modified so SQLAlchemy flushes it
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(session, "session_state")

        await db.flush()

        logger.info(
            "[SESSION ARCHIVAL] session=%s archived=%d kept=%d",
            session.id,
            len(to_archive),
            len(to_keep),
        )

    except Exception as exc:
        # Non-fatal — archival failure must never break a query response
        logger.warning("[SESSION ARCHIVAL] Failed (non-critical): %s", exc)


async def _count_archived(db, session_id) -> int:
    """Return how many turns are already in session_history for this session."""
    from sqlalchemy import select, func
    from ..models import SessionHistory

    result = await db.execute(
        select(func.count()).where(SessionHistory.session_id == session_id)
    )
    return result.scalar() or 0


async def get_full_history(db, session_id) -> List[Dict[str, Any]]:
    """
    Retrieve complete turn history: archived turns + live session_state turns,
    merged in chronological order.  Useful for building full conversation context.
    """
    from sqlalchemy import select
    from ..models import SessionHistory, ChatSession

    # Archived turns
    archived_result = await db.execute(
        select(SessionHistory)
        .where(SessionHistory.session_id == session_id)
        .order_by(SessionHistory.turn_index)
    )
    archived = [row.turn_data for row in archived_result.scalars()]

    # Live turns
    session_result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )
    session = session_result.scalar_one_or_none()
    live: List[Any] = []
    if session and session.session_state:
        live = session.session_state.get("turns", [])

    return archived + live
