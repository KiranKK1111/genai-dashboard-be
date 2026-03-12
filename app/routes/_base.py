"""
Abstract base class for all query-mode handlers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .. import schemas
from ._context import QueryContext


class QueryModeHandler(ABC):
    """
    Each concrete subclass handles exactly one query mode (standard, agentic,
    stream, viz_update, variations).

    The dispatcher in routes.py calls ``await handler.execute(ctx)`` after
    all shared setup (session resolution, message placeholder, security
    checks, progress tracking) is complete.

    For streaming mode the return type is a FastAPI ``StreamingResponse``;
    for all other modes it is a ``schemas.ResponseWrapper``.
    """

    @abstractmethod
    async def execute(self, ctx: QueryContext):
        """
        Run the mode-specific query logic.

        Args:
            ctx: Fully-populated QueryContext (db session, user, session,
                 message placeholder, tracker, etc. are all pre-resolved).

        Returns:
            ``schemas.ResponseWrapper`` for non-streaming modes.
            ``fastapi.responses.StreamingResponse`` for stream mode.
        """
