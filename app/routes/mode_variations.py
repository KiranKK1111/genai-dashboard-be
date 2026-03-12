"""
Variations mode handler.

Generates multiple response style variations (friendly, professional,
enthusiastic, etc.) for conversational queries using DynamicResponseGenerator.
"""

from __future__ import annotations

from .. import schemas
from ..helpers import current_timestamp
from ..services.response_generator import DynamicResponseGenerator
from ._base import QueryModeHandler
from ._context import QueryContext


class VariationsHandler(QueryModeHandler):
    async def execute(self, ctx: QueryContext) -> schemas.ResponseWrapper:
        from ..services.response_generator import create_conversation_state

        state = await create_conversation_state(
            session_id=ctx.session_id,
            user_id=str(ctx.current_user.id),
            db=ctx.db,
            exclude_message_id=ctx.message_id,
        )

        response_generator = DynamicResponseGenerator()
        variations = await response_generator.generate_multiple_responses(
            query=ctx.query,
            query_type="chat",
            db=ctx.db,
            session_id=ctx.session_id,
            user_id=str(ctx.current_user.id),
            conversation_state=state,
            num_responses=ctx.num_variations,
        )

        response_obj = schemas.StandardResponse(
            type="standard",
            intent=ctx.query,
            confidence=0.95,
            message=variations[0] if variations else "I'm here to help!",
            variations=variations if len(variations) > 1 else None,
            related_queries=["Tell me more", "Give me examples", "How does this compare?"],
            metadata={
                "type": "chat_variations",
                "mode": "variations",
                "variations_count": len(variations),
                "num_requested": ctx.num_variations,
            },
        )

        return schemas.ResponseWrapper(
            success=True,
            response=response_obj,
            timestamp=current_timestamp(),
            original_query=ctx.query,
            session_id=ctx.session_id,
            message_id=ctx.message_id,
        )
