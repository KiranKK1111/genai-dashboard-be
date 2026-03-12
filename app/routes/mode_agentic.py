"""
Agentic mode handler.

Delegates to the multi-tool agentic orchestrator that supports autonomous
reasoning, multi-source queries, and self-correction loops.
"""

from __future__ import annotations

from sqlalchemy import select

from .. import models, schemas
from ._base import QueryModeHandler
from ._context import QueryContext


class AgenticHandler(QueryModeHandler):
    async def execute(self, ctx: QueryContext) -> schemas.ResponseWrapper:
        from ..services import create_agentic_handler
        from ..helpers import extract_assistant_message_text

        previous_messages = (
            await ctx.db.execute(
                select(models.Message)
                .where(models.Message.session_id == ctx.session_id)
                .where(models.Message.responded_at.is_not(None))
                .order_by(models.Message.updated_at.desc())
                .limit(5)
            )
        ).scalars().all()

        conversation_lines: list[str] = []
        for m in reversed(previous_messages):
            if m.query:
                conversation_lines.append(f"User: {m.query}")
            assistant_text = extract_assistant_message_text(m.response)
            if assistant_text:
                conversation_lines.append(f"Assistant: {assistant_text[:500]}")

        conversation_history = "\n".join(conversation_lines)

        handler = await create_agentic_handler(ctx.db)
        wrapper = await handler.handle_query(
            user_query=ctx.query,
            user_id=str(ctx.current_user.id),
            session_id=ctx.session_id,
            conversation_history=conversation_history,
            uploaded_files=ctx.safe_files,
        )
        return wrapper
