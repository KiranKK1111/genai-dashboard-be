"""
Stream mode handler.

Returns a FastAPI StreamingResponse using the SSE protocol.
The handler itself does NOT persist a message (streaming persistence
is a future enhancement).
"""

from __future__ import annotations

from fastapi.responses import StreamingResponse
from sqlalchemy import select

from .. import models
from ..services.orchestrator import Orchestrator
from ._base import QueryModeHandler
from ._context import QueryContext


class StreamHandler(QueryModeHandler):
    async def execute(self, ctx: QueryContext) -> StreamingResponse:
        from .main_routes import stream_events  # avoid circular at module level

        previous_messages = (
            await ctx.db.execute(
                select(models.Message)
                .where(models.Message.session_id == ctx.session_id)
                .order_by(models.Message.updated_at.desc())
                .limit(5)
            )
        ).scalars().all()

        conversation_history = "\n".join(
            [
                f"User: {m.query}\nAssistant: {m.response.get('message', '')[:100]}"
                for m in reversed(previous_messages)
            ]
        )

        orchestrator = Orchestrator(db=ctx.db)
        return StreamingResponse(
            stream_events(
                orchestrator=orchestrator,
                user_message=ctx.query,
                user_id=str(ctx.current_user.id),
                session_id=ctx.session_id,
                conversation_history=conversation_history,
                db=ctx.db,
                files=ctx.safe_files,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
