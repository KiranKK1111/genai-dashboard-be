"""Main query orchestrator - unified entry point for all query types.

This module provides handle_dynamic_query() which:
1. Uses DecisionEngine to route (SQL / FILES / CHAT)
2. Processes the appropriate service
3. Returns consistent ResponseWrapper

Works with the new modular architecture.
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
from .decision_engine import create_decision_engine


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
    
    # Step 1: Decide what to do
    decision_engine = await create_decision_engine()
    
    files_uploaded = uploaded_files is not None and len(uploaded_files) > 0
    database_available = db is not None
    
    decision = await decision_engine.decide(
        user_message=user_message,
        files_uploaded=files_uploaded,
        database_available=database_available,
    )
    
    print(f"[DECISION] Action: {decision.action.value}, Confidence: {decision.confidence:.0%}")
    print(f"           Reasoning: {decision.reasoning}")
    
    # Step 2: Route to appropriate handler
    try:
        if decision.action.value == "RUN_SQL":
            # Route to data query handler
            response = await build_data_query_response(
                db=db,
                user_id=user_id,
                session_id=session_id,
                query=user_message,
                conversation_history=conversation_history,
            )
            return response
        
        elif decision.action.value == "ANALYZE_FILES":
            # Route to file query handler
            if uploaded_files:
                response = await build_file_query_response(
                    db=db,
                    user_id=user_id,
                    session_id=session_id,
                    query=user_message,
                    files=uploaded_files,
                )
                return response
            else:
                # No files, fall back to file lookup
                response = await build_file_lookup_response(
                    db=db,
                    user_id=user_id,
                    session_id=session_id,
                    query=user_message,
                )
                return response
        
        else:  # CHAT or uncertainty-driven clarification (needs_clarification)
            # Route to standard/chat response handler or clarification flow
            response = await build_standard_response(
                db=db,
                user_id=user_id,
                session_id=session_id,
                query=user_message,
            )
            return response
    
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
