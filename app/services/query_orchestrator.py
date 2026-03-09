"""Main query orchestrator - unified entry point for all query types.

This module provides handle_dynamic_query() which:
1. Uses SemanticIntentRouter to route (RUN_SQL / ANALYZE_FILE / CHAT / MIXED)
2. Executes the appropriate handler
3. Returns a consistent ResponseWrapper
"""

from __future__ import annotations

from typing import List, Optional
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from .. import schemas
from ..helpers import current_timestamp
from .query_handler import (
    build_data_query_response,
    build_file_query_response,
    build_file_lookup_response,
    build_standard_response,
)
from .semantic_routing_integration import SemanticRoutingIntegration


async def handle_dynamic_query(
    user_message: str,
    db: AsyncSession,
    user_id: str,
    session_id: str,
    uploaded_files: Optional[List[UploadFile]] = None,
    conversation_history: str = "",
) -> schemas.ResponseWrapper:
    """
    Unified query handler - intelligently routes to appropriate service.
    
    This is the main entry point for all user queries. It:
    1. Decides what to do (SQL / FILES / CHAT) using DecisionEngine
    2. Routes to appropriate handler
    3. Returns consistent ResponseWrapper
    
    Args:
        user_message: User's natural language query
        db: Database session
        user_id: Current user ID
        session_id: Chat session ID
        uploaded_files: Optional files to analyze
        conversation_history: Previous conversation context
    
    Returns:
        ResponseWrapper with appropriate response type
    """
    
    files_uploaded = bool(uploaded_files)

    # Step 1: Semantic routing decision (single source of truth)
    routing_integration = SemanticRoutingIntegration(db)
    router_decision = await routing_integration.make_routing_decision(
        user_query=user_message,
        session_id=session_id,
        user_id=user_id,
        current_request_has_files=files_uploaded,
    )

    print(
        f"[ROUTER] Tool: {router_decision.tool.value}, Confidence: {router_decision.confidence:.0%}"
    )
    print(f"         Reasoning: {router_decision.reasoning}")

    # Clarification gating
    if router_decision.needs_clarification:
        first_q = router_decision.clarification_questions[0] if router_decision.clarification_questions else None
        clarification_text = getattr(first_q, "question", None) if first_q else None
        clarification_text = clarification_text or "I need more information to proceed."

        clarification_response = schemas.StandardResponse(
            type="clarification",
            intent=user_message,
            confidence=router_decision.confidence,
            message=clarification_text,
            needs_clarification=True,
            clarification_options=None,
            metadata={
                "type": "clarification_request",
                "router": router_decision.model_dump(mode="json"),
            },
        )

        wrapper = schemas.ResponseWrapper(
            success=True,
            response=clarification_response,
            timestamp=current_timestamp(),
            original_query=user_message,
        )
        wrapper.intent = {
            "domain": router_decision.tool.value,
            "confidence": router_decision.confidence,
            "reasoning": router_decision.reasoning,
            "action": router_decision.tool.value,
        }
        return wrapper

    # Step 2: Execute based on routing decision
    try:
        if router_decision.tool == schemas.Tool.MIXED:
            from .agentic_query_handler import create_agentic_handler

            handler = await create_agentic_handler(db)
            wrapper = await handler.handle_query(
                user_query=user_message,
                user_id=user_id,
                session_id=session_id,
                conversation_history=conversation_history,
                uploaded_files=uploaded_files or [],
            )

        elif router_decision.tool == schemas.Tool.RUN_SQL:
            # Route to data query handler
            wrapper = await build_data_query_response(
                db=db,
                user_id=user_id,
                session_id=session_id,
                query=user_message,
                conversation_history=conversation_history,
            )

        elif router_decision.tool == schemas.Tool.ANALYZE_FILE:
            # Route to file query handler
            if uploaded_files:
                wrapper = await build_file_query_response(
                    db=db,
                    user_id=user_id,
                    session_id=session_id,
                    query=user_message,
                    files=uploaded_files,
                    conversation_history=conversation_history,
                )
            else:
                # No files, fall back to file lookup
                wrapper = await build_file_lookup_response(
                    db=db,
                    user_id=user_id,
                    session_id=session_id,
                    query=user_message,
                    conversation_history=conversation_history,
                )

        else:  # CHAT
            # Route to standard/chat response handler or clarification flow
            wrapper = await build_standard_response(
                db=db,
                user_id=user_id,
                session_id=session_id,
                query=user_message,
            )

        wrapper.intent = {
            "domain": router_decision.tool.value,
            "confidence": router_decision.confidence,
            "reasoning": router_decision.reasoning,
            "action": router_decision.tool.value,
        }
        return wrapper
    
    except Exception as e:
        # Handle errors gracefully
        print(f"[ERROR] Query handling failed: {str(e)}")
        
        error_response = schemas.StandardResponse(
            intent=user_message,
            confidence=0.0,
            message=f"Error processing your query: {str(e)[:100]}",
            related_queries=[],
            metadata={"error": True, "type": "error"},
        )
        return schemas.ResponseWrapper(
            success=False,
            response=error_response,
            timestamp=current_timestamp(),
            original_query=user_message,
        )
