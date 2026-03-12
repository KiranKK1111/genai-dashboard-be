"""
Standard mode handler.

Runs the pre-query clarification check, then delegates to the unified
semantic routing pipeline (execute_with_session_state).
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy.orm.attributes import flag_modified

from .. import schemas
from ..helpers import current_timestamp
from ..services.query_handler import build_data_query_response
from ..services.session_query_handler import execute_with_session_state
from ._base import QueryModeHandler
from ._context import QueryContext

logger = logging.getLogger(__name__)


class StandardHandler(QueryModeHandler):
    async def execute(self, ctx: QueryContext) -> schemas.ResponseWrapper:
        # ── Pre-query clarification check ──────────────────────────────────
        try:
            from ..services.clarification_engine import get_clarification_engine
            from ..services.response_composer import build_clarification_lama_response

            clarification_engine = await get_clarification_engine(ctx.db)
            ambiguity = await clarification_engine.analyze_query(
                user_query=ctx.query,
                conversation_history="",
            )

            if ambiguity.has_ambiguity and not ambiguity.can_proceed:
                clarification_dict = build_clarification_lama_response(
                    session_id=ctx.session_id,
                    user_query=ctx.query,
                    ambiguity_analysis=ambiguity,
                    message_id=ctx.message_id,
                )
                ctx.message.response_type = "clarification"
                ctx.message.response = clarification_dict
                ctx.message.responded_at = datetime.utcnow()
                flag_modified(ctx.message, "response")
                await ctx.db.commit()
                await ctx.db.refresh(ctx.message)

                if ctx.session.session_state is None:
                    ctx.session.session_state = {}
                ctx.session.session_state["last_response_type"] = "clarification"
                ctx.session.session_state["pending_clarification_query"] = ctx.query
                flag_modified(ctx.session, "session_state")
                await ctx.db.commit()

                ctx.tracker.complete("Clarification returned")
                return schemas.ResponseWrapper(
                    success=True,
                    response=schemas.StandardResponse(
                        type="clarification",
                        intent=ctx.query,
                        confidence=float(ambiguity.confidence),
                        message=(
                            clarification_dict.get("assistant", {})
                            .get("content", [{}])[0]
                            .get("text", "Please clarify your question.")
                        ),
                        metadata=clarification_dict,
                    ),
                    timestamp=current_timestamp(),
                    original_query=ctx.query,
                    session_id=ctx.session_id,
                    message_id=ctx.message_id,
                )
        except Exception as exc:
            logger.warning(
                "[CLARIFICATION] Pre-query check failed (non-critical): %s", exc
            )

        # ── Unified semantic routing ────────────────────────────────────────
        wrapper = await execute_with_session_state(
            session_id=ctx.session_id,
            user_id=str(ctx.current_user.id),
            user_query=ctx.query,
            db=ctx.db,
            current_user=ctx.current_user,
            handler_func=build_data_query_response,
            files=ctx.safe_files,
            message_id=ctx.message_id,
        )
        return wrapper
