"""
Integrated Session Query Handler - Bridges routes with SessionStateManager.

This module ensures:
1. Dynamically determines query domain (GENERAL, DATABASE, FILES)
2. Routes to appropriate handler based on domain
3. Maintains session state for database queries
4. Classifies follow-ups BEFORE SQL generation
5. Merges state based on follow-up type
6. Persists updated state back to DB

This is the glue that makes ChatGPT-style follow-ups and dynamic routing work.
"""

from __future__ import annotations

import inspect
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models, schemas
from .session_state_manager import (
    SessionStateManager,
    QueryState,
    FollowUpType,
    SelectiveRetriever,
)
from .decision_engine import DecisionEngine, Action


async def execute_with_session_state(
    session_id: str,
    user_id: str,
    user_query: str,
    db: AsyncSession,
    current_user: models.User,
    handler_func=None,  # Optional: used if provided, but may be overridden by intent
    **handler_kwargs
) -> schemas.ResponseWrapper:
    """
    Execute a query while maintaining session state (ChatGPT-style).

    DYNAMIC ROUTING: This function now:
    1. Determines query domain (GENERAL, DATABASE, FILES) dynamically
    2. Routes to the appropriate handler based on domain
    3. For DATABASE queries: Maintains session state, classifies follow-ups
    4. For GENERAL queries: Returns conversational response
    5. For FILE queries: Routes to file analysis
    
    This is completely database-agnostic and makes NO hardcoded assumptions.

    Args:
        session_id: Chat session ID
        user_id: User ID
        user_query: The current user query
        db: Database session
        current_user: Current user object
        handler_func: Optional default handler (may be overridden by intent)
        **handler_kwargs: Additional arguments to pass to handler

    Returns:
        ResponseWrapper with the query response
    """
    
    # ======================================================================
    # STEP 0: Load session and get conversation context
    # ======================================================================
    session = (
        await db.execute(
            select(models.ChatSession)
            .where(models.ChatSession.id == session_id)
            .where(models.ChatSession.user_id == current_user.id)
        )
    ).scalars().first()
    
    if not session:
        raise Exception(f"Session {session_id} not found")
    
    # Get previous messages for context
    previous_messages = (
        await db.execute(
            select(models.Message)
            .where(models.Message.session_id == session_id)
            .order_by(models.Message.created_at.desc())
            .limit(20)
        )
    ).scalars().all()
    
    previous_messages = list(reversed(previous_messages))
    
    from ..helpers import format_conversation_context
    conversation_history = format_conversation_context(previous_messages)
    
    # Get previous state
    previous_state_dict = session.session_state or {}
    previous_domain = previous_state_dict.get("domain") if previous_state_dict else None
    
    # ======================================================================
    # STEP 1: USE DECISION ENGINE TO DETERMINE QUERY TYPE
    # ======================================================================
    decision_engine = DecisionEngine()
    decision_result = await decision_engine.decide(
        user_message=user_query,
        files_uploaded=bool(handler_kwargs.get('files')),
        conversation_history=conversation_history,
        database_available=True,
    )
    
    print(f"\n[DECISION] Action: {decision_result.action.value} (confidence: {decision_result.confidence:.0%})")
    print(f"[DECISION] Reasoning: {decision_result.reasoning}")
    
    # ======================================================================
    # STEP 2: ROUTE BASED ON DECISIONENGINE ACTION
    # ======================================================================
    
    # Route CHAT queries and uncertainty-driven clarification (needs_clarification flag)
    if decision_result.action == Action.CHAT or decision_result.needs_clarification:
        print(f"[ROUTE] Conversational query detected → Using CHAT handler")
        
        # Import here to avoid circular imports
        from .query_handler import build_standard_response
        
        try:
            response = await build_standard_response(
                db=db,
                user_id=user_id,
                session_id=session_id,
                query=user_query,
            )
            # Add decision details to response
            response.intent = {
                "domain": decision_result.action.value,
                "confidence": decision_result.confidence,
                "reasoning": decision_result.reasoning,
            }
            return response
        except Exception as e:
            print(f"[ERROR] Conversational handler failed: {e}")
            error_response = schemas.StandardResponse(
                intent=user_query,
                confidence=0.0,
                message=f"I encountered an error processing your request: {str(e)[:100]}",
                related_queries=[],
                metadata={"error": True},
            )
            return schemas.ResponseWrapper(
                success=False,
                response=error_response,
                timestamp=datetime.utcnow().isoformat(),
                original_query=user_query,
            )
    
    elif decision_result.action == Action.ANALYZE_FILES:
        print(f"[ROUTE] File analysis detected → Using ANALYZE_FILES handler")
        
        from .query_handler import build_file_query_response, build_file_lookup_response
        
        try:
            files = handler_kwargs.get('files')
            if files:
                response = await build_file_query_response(
                    db=db,
                    user_id=user_id,
                    session_id=session_id,
                    query=user_query,
                    files=files,
                )
            else:
                response = await build_file_lookup_response(
                    db=db,
                    user_id=user_id,
                    session_id=session_id,
                    query=user_query,
                )
            # Add decision details to response
            response.intent = {
                "domain": decision_result.action.value,
                "confidence": decision_result.confidence,
                "reasoning": decision_result.reasoning,
            }
            return response
        except Exception as e:
            print(f"[ERROR] File handler failed: {e}")
            error_response = schemas.StandardResponse(
                intent=user_query,
                confidence=0.0,
                message=f"I encountered an error processing files: {str(e)[:100]}",
                related_queries=[],
                metadata={"error": True},
            )
            return schemas.ResponseWrapper(
                success=False,
                response=error_response,
                timestamp=datetime.utcnow().isoformat(),
                original_query=user_query,
            )
    
    # ======================================================================
    # STEP 3: DATABASE DOMAIN - Full session state management
    # ======================================================================
    # Only reach here for DATABASE queries
    
    session_manager = SessionStateManager.from_session_dict(
        previous_state_dict,
        session_id=session_id,
        user_id=user_id,
    )
    
    print(f"[SESSION] Loading state: {len(previous_state_dict)} bytes")
    if session_manager.last_query_state:
        print(f"[SESSION] Previous tables: {session_manager.last_query_state.selected_entities}")
        print(f"[SESSION] Previous filters: {len(session_manager.last_query_state.filters)} conditions")
    else:
        print(f"[SESSION] No previous state (new session)")
    
    # ======================================================================
    # STEP 4: Detect the follow-up type using DecisionEngine results
    # ======================================================================
    # DecisionEngine.decide() already detected follow-ups from conversation_history
    # Extract follow-up info from decision_result
    followup_type = decision_result.followup_type
    is_followup = decision_result.is_followup

    print(f"\n[FOLLOWUP] Is Follow-up: {is_followup}")
    if is_followup and followup_type:
        print(f"[FOLLOWUP] Type: {followup_type.name}")
        print(f"[FOLLOWUP] Reasoning: {decision_result.followup_reasoning}")
    else:
        print(f"[FOLLOWUP] New conversation - no follow-up detected")

    # ======================================================================
    # STEP 5: Merge state based on follow-up type
    # ======================================================================
    if not is_followup or followup_type is None:
        print(f"[MERGE] NEW: Starting fresh, clearing previous state")
        session_manager.reset_state()
    
    elif followup_type == FollowUpType.NEW:
        print(f"[MERGE] NEW: Clearing previous state")
        session_manager.reset_state()
    
    elif followup_type == FollowUpType.REFINEMENT:
        print(f"[MERGE] REFINEMENT: Keeping tables/joins, adding filters")
        pass
    
    elif followup_type == FollowUpType.EXPANSION:
        print(f"[MERGE] EXPANSION: Keeping filters, broadening scope")
        pass
    
    elif followup_type == FollowUpType.CLARIFICATION:
        print(f"[MERGE] CLARIFICATION: Drilling into detail from previous results")
        if session_manager.last_query_state:
            session_manager.last_query_state.aggregation = None
    
    elif followup_type == FollowUpType.PIVOT:
        print(f"[MERGE] PIVOT: Related table, building on previous context")
        pass
    
    elif followup_type == FollowUpType.CONTINUATION:
        print(f"[MERGE] CONTINUATION: Same table, different analytical angle")
        pass
    
    # ======================================================================
    # STEP 7: Build enriched context for handler
    # ======================================================================
    retriever = SelectiveRetriever()
    
    selective_history = await retriever.retrieve(
        session_id=session_id,
        messages=session_manager.messages,
        tool_calls=session_manager.tool_calls,
        last_query_state=session_manager.last_query_state,
    )
    
    print(f"\n[RETRIEVAL] Tier 1 context: {len(selective_history.recent_messages)} recent messages")
    print(f"[RETRIEVAL] Tier 2 context: {selective_history.last_query_state is not None}")
    
    # ======================================================================
    # STEP 7b: DYNAMIC FUNCTION PARAMETER INJECTION (ChatGPT-style)
    # ======================================================================
    # Dynamically inspect what parameters build_data_query_response actually accepts
    # This is 100% dynamic - no hardcoding of parameter names
    from .query_handler import build_data_query_response
    
    sig = inspect.signature(build_data_query_response)
    accepted_params = set(sig.parameters.keys())
    
    # Build handler kwargs with ONLY the parameters the function accepts
    final_handler_kwargs = {
        'db': db,
        'user_id': user_id,
        'session_id': session_id,
        'query': user_query,
        'conversation_history': selective_history.to_prompt_context(),
    }
    
    # Dynamically add any extra kwargs that the function accepts (never hardcode param names)
    for key, value in handler_kwargs.items():
        if key in accepted_params:
            final_handler_kwargs[key] = value
        else:
            print(f"[DYNAMIC FILTERING] Parameter '{key}' not accepted by handler - automatically filtered out")
    
    # ======================================================================
    # STEP 8: Call the data query handler (only for DATABASE domain)
    # ======================================================================
    print(f"\n[HANDLER] Calling database handler with enriched context")
    print(f"[HANDLER] Parameters being passed: {', '.join(final_handler_kwargs.keys())}")
    
    response_wrapper = await build_data_query_response(**final_handler_kwargs)
    
    # ======================================================================
    # STEP 9: Record the tool call
    # ======================================================================
    tool_input = {
        "query": user_query,
        "followup_type": followup_type.name if followup_type else "NEW",
        "intent_domain": decision_result.action.value,
        "decision_action": decision_result.action.value,
        "state_before": {
            "tables": session_manager.last_query_state.selected_entities if session_manager.last_query_state else [],
            "filters": len(session_manager.last_query_state.filters) if session_manager.last_query_state else 0,
        }
    }
    
    tool_output = {
        "response_type": response_wrapper.response.type if hasattr(response_wrapper.response, 'type') else "standard",
        "success": response_wrapper.success,
    }
    
    session_manager.record_tool_call(
        tool_type="sql_generation",
        input_json=tool_input,
        output_json=tool_output,
        success=response_wrapper.success,
    )
    
    # ======================================================================
    # STEP 10: Persist updated QueryState to database
    # ======================================================================
    if session_manager.last_query_state:
        print(f"\n[PERSIST] Saving QueryState to database")
        state_dict = session_manager.to_session_dict()
        state_dict["domain"] = decision_result.action.value  # Track the domain determined by DecisionEngine
        session.session_state = state_dict
        session.state_updated_at = datetime.utcnow()
        session.tool_calls_log = state_dict.get("tool_calls", [])
        
        await db.commit()
        print(f"[PERSIST] QueryState saved: {len(str(session.session_state))} bytes")
    
    # Add decision engine results to response
    response_wrapper.intent = {
        "domain": decision_result.action.value,
        "confidence": decision_result.confidence,
        "reasoning": decision_result.reasoning,
        "action": decision_result.action.value,
    }
    
    return response_wrapper
