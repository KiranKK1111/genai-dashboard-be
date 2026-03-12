"""
Integrated Session Query Handler - Bridges routes with SessionStateManager.

This module ensures:
1. Dynamically determines query domain (GENERAL, DATABASE, FILES)
2. Routes to appropriate handler based on domain
3. Maintains session state for database queries
4. Classifies follow-ups BEFORE SQL generation
5. Merges state based on follow-up type
6. Persists updated state back to DB
7. Uses RAG (vector embeddings) for semantic context retrieval

This is the glue that makes ChatGPT-style follow-ups and dynamic routing work.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import uuid
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models, schemas
from .session_state_manager import (
    SessionStateManager,
    QueryState,
    SelectiveRetriever,
    QueryDomain,
)
from .followup_manager import FollowUpType, get_followup_analyzer
from .rag_context_retriever import get_rag_retriever
from .query_embeddings import get_embedding_store, get_embedding_generator
from .intelligent_followup_value_mapper import (
    get_followup_value_mapper,
    create_context_for_followup,
    create_context_for_initial_query,
)
from .intelligent_query_orchestrator import get_query_orchestrator
from .semantic_value_grounding_enhanced import (
    get_semantic_value_grounder_enhanced,
    GroundedValueWithRelationship,
)
from .semantic_routing_integration import SemanticRoutingIntegration
from ..helpers import current_timestamp

logger = logging.getLogger(__name__)

# ============================================================================
# VERSION MARKER FOR DEBUGGING VERSION MISMATCH ISSUES
# ============================================================================
SESSION_QUERY_HANDLER_VERSION = "projection_bypass_v3_FINAL"
logger.debug(f"[INIT] =========================================================")
logger.debug(f"[INIT] session_query_handler.py VERSION: {SESSION_QUERY_HANDLER_VERSION}")
logger.debug(f"[INIT] =========================================================")


def _extract_table_from_sql(sql: str) -> Optional[str]:
    """Extract the main table name from SQL query."""
    if not sql:
        return None
    
    # Look for FROM clause or first table mention
    from_match = re.search(r"\bFROM\s+(?:\w+\.)?(\w+)", sql, re.IGNORECASE)
    if from_match:
        return from_match.group(1).lower()
    
    # Fallback: look for any table name pattern
    return None


def _extract_filters_from_sql(sql: str) -> List[Dict[str, str]]:
    """Extract WHERE clause filters from SQL query."""
    if not sql or 'WHERE' not in sql.upper():
        return []
    
    filters = []
    # Simple extraction of WHERE conditions
    where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|;|$)', sql, re.IGNORECASE)
    if where_match:
        where_clause = where_match.group(1)
        # Split by AND/OR (simple approach)
        conditions = re.split(r'\s+AND\s+|\s+OR\s+', where_clause, flags=re.IGNORECASE)
        for condition in conditions:
            condition = condition.strip()
            if not condition:
                continue
            # Try to parse "column operator value"
            match = re.match(r'(\w+)\s*(=|!=|<>|<|>|<=|>=|LIKE|IN)\s*(.+?)$', condition, re.IGNORECASE)
            if match:
                filters.append({
                    "column": match.group(1),
                    "operator": match.group(2),
                    "value": match.group(3).strip(),
                })
    
    return filters


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
            .order_by(models.Message.updated_at.desc())
            .limit(20)
        )
    ).scalars().all()
    
    previous_messages = list(reversed(previous_messages))
    
    from ..helpers import format_conversation_context
    conversation_history = format_conversation_context(previous_messages)
    
    # Get previous state
    previous_state_dict = session.session_state or {}
    previous_domain = previous_state_dict.get("domain") if previous_state_dict else None
    
    # DEBUG: Log session state loading details
    logger.debug(f"[DEBUG] Session ID being loaded: {session_id}")
    logger.debug(f"[DEBUG] Session.session_state raw size: {len(str(session.session_state)) if session.session_state else 0} bytes")
    logger.debug(f"[DEBUG] Session.session_state content type: {type(session.session_state)}")
    if session.session_state:
        logger.debug(f"[DEBUG] Session.session_state keys: {list(session.session_state.keys())}")
        if 'messages' in session.session_state:
            logger.debug(f"[DEBUG] Session has {len(session.session_state['messages'])} messages")
        else:
            logger.debug(f"[DEBUG] Session has NO messages in session_state")
    logger.debug(f"[DEBUG] previous_state_dict size: {len(str(previous_state_dict))} bytes")
    
    # ======================================================================
    # STEP 1: SEMANTIC ROUTING - SINGLE SOURCE OF TRUTH
    # ======================================================================
    current_request_has_files = bool(handler_kwargs.get("files"))
    routing_integration = SemanticRoutingIntegration(db)

    try:
        router_decision = await routing_integration.make_routing_decision(
            user_query=user_query,
            session_id=session_id,
            user_id=user_id,
            current_request_has_files=current_request_has_files,
        )

        logger.info(
            f"[SESSION_HANDLER] RouterDecision tool={router_decision.tool.value} "
            f"followup_type={router_decision.followup_type.value} "
            f"confidence={router_decision.confidence:.2f} "
            f"needs_clarification={router_decision.needs_clarification}"
        )

        if router_decision.needs_clarification:
            logger.info("[SESSION_HANDLER] Router requested clarification → returning clarification response")
            return routing_integration.build_clarification_response(
                decision=router_decision,
                session_id=session_id,
            )

        # MIXED queries are executed by the agentic orchestrator.
        if router_decision.tool == schemas.Tool.MIXED:
            from .agentic_query_handler import create_agentic_handler

            handler = await create_agentic_handler(db)
            wrapper = await handler.handle_query(
                user_query=user_query,
                user_id=user_id,
                session_id=session_id,
                conversation_history=conversation_history,
                uploaded_files=handler_kwargs.get("files") or [],
            )
            wrapper.intent = {
                "domain": router_decision.tool.value,
                "confidence": router_decision.confidence,
                "reasoning": router_decision.reasoning,
                "action": router_decision.tool.value,
            }
            return wrapper

        # CHAT routing
        if router_decision.tool == schemas.Tool.CHAT:
            from .query_handler import build_standard_response

            message_id = handler_kwargs.get("message_id")
            response = await build_standard_response(
                db=db,
                user_id=user_id,
                session_id=session_id,
                query=user_query,
                message_id=message_id,
            )

            response.intent = {
                "domain": router_decision.tool.value,
                "confidence": router_decision.confidence,
                "reasoning": router_decision.reasoning,
                "action": router_decision.tool.value,
            }

            # Persist semantic turn-state (best-effort) without overwriting SQL QueryState.
            try:
                assistant_text = getattr(response.response, "message", "") if getattr(response, "response", None) else ""
                assistant_text = assistant_text.strip() if isinstance(assistant_text, str) else ""
                assistant_summary = assistant_text[:500] if assistant_text else "Chat response"

                await routing_integration.save_turn_state_after_tool(
                    session_id=session_id,
                    user_query=user_query,
                    tool_used=schemas.Tool.CHAT,
                    assistant_summary=assistant_summary,
                    artifacts={
                        "tool_used": schemas.Tool.CHAT.value,
                        "chat_summary": assistant_text[:2000] if assistant_text else None,
                    },
                    confidence=router_decision.confidence,
                )
            except Exception as e:
                logger.warning(f"[SESSION_HANDLER] Failed to persist turn state (CHAT): {e}")

            db_tool_call = models.ToolCall(
                session_id=uuid.UUID(session_id) if isinstance(session_id, str) else session_id,
                tool_type="chat_response",
                input_json={"query": user_query, "type": "conversational"},
                output_json={
                    "success": response.success,
                    "response_type": response.response.type if hasattr(response.response, "type") else "standard",
                },
                success=response.success,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
            )
            db.add(db_tool_call)
            await db.commit()
            logger.debug("[PERSIST] Chat response logged to tool_calls table")

            return response

        # FILE routing
        if router_decision.tool == schemas.Tool.ANALYZE_FILE:
            from .query_handler import build_file_query_response, build_file_lookup_response

            message_id = handler_kwargs.get("message_id")
            files = handler_kwargs.get("files")

            if files:
                response = await build_file_query_response(
                    db=db,
                    user_id=user_id,
                    session_id=session_id,
                    query=user_query,
                    files=files,
                    conversation_history=conversation_history,
                    message_id=message_id,
                )
                tool_type = "file_upload"
                output_metadata = {"files_processed": len(files)}
            else:
                response = await build_file_lookup_response(
                    db=db,
                    user_id=user_id,
                    session_id=session_id,
                    query=user_query,
                    conversation_history=conversation_history,
                    message_id=message_id,
                )
                tool_type = "file_lookup"
                output_metadata = {
                    "chunks_found": response.response.metadata.get("chunks_found", 0)
                    if hasattr(response.response, "metadata")
                    else 0
                }

            response.intent = {
                "domain": router_decision.tool.value,
                "confidence": router_decision.confidence,
                "reasoning": router_decision.reasoning,
                "action": router_decision.tool.value,
            }

            # Persist semantic turn-state (best-effort). Include file_ids if possible.
            try:
                file_ids: List[str] = []
                try:
                    file_rows = await db.execute(
                        select(models.UploadedFile.id)
                        .where(models.UploadedFile.session_id == session.id)
                        .order_by(models.UploadedFile.upload_time.desc())
                        .limit(20)
                    )
                    file_ids = [str(fid) for fid in file_rows.scalars().all()]
                except Exception:
                    file_ids = []

                assistant_summary = f"File operation: {tool_type}"[:500]

                await routing_integration.save_turn_state_after_tool(
                    session_id=session_id,
                    user_query=user_query,
                    tool_used=schemas.Tool.ANALYZE_FILE,
                    assistant_summary=assistant_summary,
                    artifacts={
                        "tool_used": schemas.Tool.ANALYZE_FILE.value,
                        "file_ids": file_ids or None,
                        **output_metadata,
                    },
                    confidence=router_decision.confidence,
                )
            except Exception as e:
                logger.warning(f"[SESSION_HANDLER] Failed to persist turn state (ANALYZE_FILE): {e}")

            db_tool_call = models.ToolCall(
                session_id=uuid.UUID(session_id) if isinstance(session_id, str) else session_id,
                tool_type=tool_type,
                input_json={"query": user_query, "type": "file_analysis"},
                output_json={"success": response.success, **output_metadata},
                success=response.success,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
            )
            db.add(db_tool_call)
            await db.commit()
            logger.debug(f"[PERSIST] File operation ({tool_type}) logged to tool_calls table")

            # ===== FALLBACK LOGIC: If file analysis produces weak results, try database =====
            chunks_found = output_metadata.get("chunks_found", 0)
            is_weak_result = (
                chunks_found == 0 or  # No relevant chunks found
                (chunks_found <= 2 and "database" in user_query.lower()) or  # Few chunks + database keywords
                (chunks_found <= 2 and any(kw in user_query.lower() for kw in ["clients", "sales", "records", "all", "get me", "show me", "list"]))
            )
            
            if is_weak_result and response.success:
                logger.debug(f"[FALLBACK] File analysis found {chunks_found} chunks for database-like query. Attempting SQL fallback...")
                try:
                    from .enhanced_analysis_planner import EnhancedAnalysisPlanner
                    
                    # Create fresh session manager for SQL analysis (use already imported SessionStateManager)
                    session_manager = SessionStateManager.from_session_dict(
                        session.session_state or {}, session_id, user_id
                    )
                    
                    planner = EnhancedAnalysisPlanner(db)
                    sql_result = await planner.execute_full_pipeline(
                        user_query=user_query,
                        user_id=user_id,
                        session_manager=session_manager,
                        conversation_history=conversation_history,
                        **handler_kwargs
                    )
                    
                    # If SQL analysis succeeds and finds data, use it instead
                    if (sql_result and sql_result.success and 
                        hasattr(sql_result.response, 'artifacts') and
                        sql_result.response.artifacts.get('sql')):
                        
                        logger.debug("[FALLBACK] SQL analysis successful. Using database results instead of weak file results.")
                        
                        # Update routing info to indicate fallback
                        sql_result.intent = {
                            "domain": "RUN_SQL",
                            "confidence": 0.8,
                            "reasoning": "Fallback from weak file analysis to database query",
                            "action": "RUN_SQL",
                        }
                        
                        return sql_result
                    else:
                        logger.debug("[FALLBACK] SQL analysis did not improve results. Keeping file analysis.")
                        
                except Exception as e:
                    logger.debug(f"[FALLBACK] SQL fallback failed: {e}. Keeping file analysis results.")

            return response

        # Otherwise: RUN_SQL → fall through to database domain handler below.

    except Exception as e:
        logger.error(f"[SESSION_HANDLER] Semantic routing/execution failed: {e}")
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
            timestamp=current_timestamp(),
            original_query=user_query,
        )
    
    # ======================================================================
    # STEP 3: DATABASE DOMAIN - Full session state management
    # ======================================================================
    # Only reach here for DATABASE queries
    
    # FIX: Load TurnState FIRST as the CANONICAL source of last turn context
    # This consolidates state management around TurnState instead of multiple sources
    from .turn_state_manager import TurnStateManager as TurnStateMgr
    turn_state_mgr = TurnStateMgr(db)
    last_turn_state = await turn_state_mgr.get_last_turn_state(session_id)
    
    if last_turn_state:
        logger.debug(f"[SESSION] ✅ Loaded TurnState #{last_turn_state.turn_id} as canonical context")
        logger.debug(f"[SESSION]   Tool used: {last_turn_state.tool_used}")
        logger.debug(f"[SESSION]   Has SQL: {bool(last_turn_state.artifacts.get('sql') if last_turn_state.artifacts else False)}")
    else:
        logger.debug(f"[SESSION] No previous TurnState found (new session)")
    
    session_manager = SessionStateManager.from_session_dict(
        previous_state_dict,
        session_id=session_id,
        user_id=user_id,
    )
    
    # FIX: Hydrate SessionStateManager from TurnState if available
    # This ensures TurnState is the primary source, with session_state as backup
    if last_turn_state and last_turn_state.artifacts:
        artifacts = last_turn_state.artifacts
        if artifacts.get('sql') and session_manager.last_query_state:
            # Ensure generated_sql is populated from TurnState (canonical source)
            if not session_manager.last_query_state.generated_sql:
                session_manager.last_query_state.generated_sql = artifacts.get('sql')
                logger.debug(f"[SESSION] Hydrated generated_sql from TurnState")
            
            # Ensure tables are populated from TurnState
            if not session_manager.last_query_state.selected_entities and artifacts.get('tables'):
                session_manager.last_query_state.selected_entities = artifacts.get('tables')
                logger.debug(f"[SESSION] Hydrated selected_entities from TurnState")
    
    # ✅ FIX: Show actual loaded session state, not raw dict size
    logger.debug(f"[SESSION] Loading state: {len(session_manager.messages)} messages loaded")
    if session_manager.last_query_state:
        logger.debug(f"[SESSION] Previous tables: {session_manager.last_query_state.selected_entities}")
        logger.debug(f"[SESSION] Previous filters: {len(session_manager.last_query_state.filters)} conditions")
    else:
        logger.debug(f"[SESSION] No previous state (new session)")
    
    # ======================================================================
    # STEP 4: PROVISIONAL follow-up detection from router (NOT FINAL)
    # ======================================================================
    # CRITICAL: This decision is PROVISIONAL. The final decision will be made
    # by the arbiter AFTER RAG + LLM analysis. DO NOT use this to gate behavior.
    from .followup_manager import FollowUpType as FollowUpTypeAnalyzer

    preliminary_router_followup = bool(
        router_decision
        and router_decision.followup_type == schemas.FollowupType.RUN_SQL_FOLLOW_UP
    )
    preliminary_followup_type = FollowUpTypeAnalyzer.REFINEMENT if preliminary_router_followup else None
    followup_reasoning = router_decision.reasoning if router_decision else ""
    
    # Detect if this is truly a new session (no previous context at all)
    is_truly_new_session = len(previous_messages) == 0 and len(session_manager.messages) == 0
    has_previous_context = not is_truly_new_session and (
        len(previous_messages) > 0 or 
        len(session_manager.messages) > 0 or
        (session_manager.last_query_state and session_manager.last_query_state.generated_sql)
    )
    
    logger.debug(f"[FOLLOWUP] preliminary_router_followup={preliminary_router_followup}, is_truly_new_session={is_truly_new_session}")
    logger.debug(f"[FOLLOWUP] has_previous_context={has_previous_context}")

    logger.debug(f"\n[FOLLOWUP] Preliminary Router Signal: {preliminary_router_followup}")
    logger.debug(f"[FOLLOWUP] Has Previous Context: {has_previous_context}")
    logger.debug(f"[FOLLOWUP] Is Truly New Session: {is_truly_new_session}")

    # ======================================================================
    # STEP 5: DO NOT RESET STATE HERE - Deferred until after arbiter decision
    # ======================================================================
    # CRITICAL FIX: State reset was happening here based on preliminary detection.
    # This was the core architectural bug. State management now happens ONLY
    # after the arbiter makes the final decision at STEP 6.7.
    logger.debug(f"[STATE] ⏸️ State management DEFERRED until arbiter decision (STEP 6.7)")
    
    # ======================================================================
    # STEP 6: Build enriched context for handler
    # ======================================================================
    retriever = SelectiveRetriever()
    
    selective_history = await retriever.retrieve(
        session_id=session_id,
        messages=session_manager.messages,
        tool_calls=session_manager.tool_calls,
        last_query_state=session_manager.last_query_state,
    )
    
    logger.debug(f"\n[RETRIEVAL] Tier 1 context: {len(selective_history.recent_messages)} recent messages")
    logger.debug(f"[RETRIEVAL] Tier 2 context: {selective_history.last_query_state is not None}")
    
    # ======================================================================
    # STEP 6.5: MULTI-STAGE Follow-Up Detection (RAG + LLM)
    # ======================================================================
    # CRITICAL FIX: This analysis runs UNCONDITIONALLY for sessions with previous context.
    # The preliminary router decision does NOT gate this. We need the deeper analysis
    # to inform the arbiter, which makes the FINAL decision.
    #
    # STAGE 1: Try RAG-based semantic search for similar previous queries
    # STAGE 2: Run LLM-based analysis (ALWAYS for non-new sessions)
    # STAGE 3: Fall back to preliminary detection only if both fail
    
    from .followup_manager import FollowUpContext, FollowUpType as FollowUpTypeAnalyzer, PreviousQueryContext
    from .session_state_manager import FollowUpType as FollowUpTypeState
    
    # Initialize followup_context to None - will be populated by analysis
    followup_context = None
    rag_context = None
    
    # STAGE 1: Try RAG retrieval (UNCONDITIONALLY for sessions with previous context)
    # FIX: Removed dependency on preliminary `is_followup` - RAG runs for ANY session with context
    if has_previous_context:
        logger.debug(f"\n[FOLLOWUP-STAGE1] Starting RAG retrieval for: {user_query[:50]}")
        logger.debug(f"[FOLLOWUP-STAGE1] ✅ Running unconditionally (has_previous_context=True)")
        try:
            rag_retriever = await get_rag_retriever(similarity_threshold=0.15, top_k=3, db_session=db)
            rag_context = await rag_retriever.retrieve_context_for_followup(
                session_id=session_id,
                current_query=user_query,
            )
        except Exception as e:
            logger.debug(f"[FOLLOWUP-STAGE1] ⚠️ RAG retrieval failed: {e}")
            rag_context = None
    else:
        logger.debug(f"\n[FOLLOWUP-STAGE1] Skipping RAG (new session - no previous context)")
        rag_context = None
    
    if rag_context and rag_context.is_relevant:
        logger.debug(f"[FOLLOWUP-RAG] ✅ Found semantically similar previous query!")
        logger.debug(f"[FOLLOWUP-RAG] Similarity: {rag_context.similarity_score:.0%}")
        logger.debug(f"[FOLLOWUP-RAG] Previous: {rag_context.previous_query}")
        logger.debug(f"[FOLLOWUP-RAG] SQL: {rag_context.previous_sql}")
        
        # Extract table name and filters from SQL
        table_name = _extract_table_from_sql(rag_context.previous_sql)
        filters = _extract_filters_from_sql(rag_context.previous_sql)
        
        # Construct PreviousQueryContext from RAG results
        previous_context = PreviousQueryContext(
            query_text=rag_context.previous_query,
            generated_sql=rag_context.previous_sql,
            table_name=table_name,
            filters=filters,
            columns_selected=rag_context.column_names or [],
            result_count=rag_context.result_count,
        )
        
        # Construct followup_context from RAG results WITH previous_context
        followup_context = FollowUpContext(
            is_followup=True,
            followup_type=FollowUpTypeAnalyzer.REFINEMENT,  # Assume refinement for follow-ups
            confidence=rag_context.similarity_score,
            reasoning=f"RAG-detected follow-up (similarity: {rag_context.similarity_score:.0%})",
            previous_context=previous_context,  # NOW POPULATED FROM RAG
        )
        
        logger.debug(f"[FOLLOWUP-RAG] ✅ Follow-up context created with table: {table_name}, filters: {len(filters)}")
        
        # Inject RAG context into selective_history for downstream processing
        rag_context_str = f"\n\n[RAG CONTEXT]\n{rag_context.context_text}"
        selective_history.recent_messages.append({
            "role": "system",
            "content": rag_context_str,
        })
        
        # Store the previous SQL from RAG for use in the query handler
        if session_manager.last_query_state:
            session_manager.last_query_state.generated_sql = rag_context.previous_sql
    
    # STAGE 2: CONTEXT EXTRACTION (not LLM analysis - that's handled by router + arbiter)
    # REFACTORED: The followup_manager no longer does LLM analysis.
    # The router already provides LLM-based follow-up signals in router_decision.followup_type.
    # Here we only extract context for the arbiter to use.
    if has_previous_context and followup_context is None:
        logger.debug(f"\n[FOLLOWUP-STAGE2] Extracting context (router provides LLM signal)")
        try:
            followup_analyzer = await get_followup_analyzer()
            
            # Extract previous SQL from conversation history for context (table-aware)
            from .query_handler import _extract_table_mentions_from_query, _extract_previous_sql_by_table
            candidate_tables: Optional[List[str]] = None
            try:
                from .database_adapter import get_global_adapter

                adapter = get_global_adapter()
                schema_name = adapter.get_default_schema() if hasattr(adapter, "get_default_schema") else None
                candidate_tables = await adapter.get_available_tables(db, schema_name=schema_name)
            except Exception:
                candidate_tables = None

            target_tables = _extract_table_mentions_from_query(user_query, candidate_tables=candidate_tables)
            previous_sql_tuple = _extract_previous_sql_by_table(selective_history.to_prompt_context(), target_tables=target_tables if target_tables else None)
            previous_sql = previous_sql_tuple[0] if previous_sql_tuple else None
            previous_result_count = previous_sql_tuple[1] if previous_sql_tuple else 0
            
            # Build simple conversation history for context extraction
            simple_conversation_history = ""
            if session_manager and session_manager.tool_calls:
                query_parts = []
                for tool_call in session_manager.tool_calls[-3:]:  # Last 3 interactions
                    historical_query = None
                    if hasattr(tool_call, 'input_json') and tool_call.input_json:
                        historical_query = (tool_call.input_json.get('query') or 
                                           tool_call.input_json.get('user_query') or 
                                           tool_call.input_json.get('user_input'))
                    
                    if historical_query:
                        query_parts.append(f"USER: {historical_query}")
                
                if query_parts:
                    simple_conversation_history = "\n".join(query_parts)
                    logger.debug(f"[DEBUG] Session simple conversation history: {simple_conversation_history}")
            
            # Extract previous query from history
            previous_query = followup_analyzer.extract_previous_query(
                simple_conversation_history if simple_conversation_history else selective_history.to_prompt_context()
            )
            
            # Extract context using the streamlined method (no LLM)
            previous_context = None
            if previous_sql and previous_query:
                previous_context = followup_analyzer.extract_context_only(
                    previous_query=previous_query,
                    previous_sql=previous_sql,
                    previous_result_count=previous_result_count,
                )
                logger.debug(f"[FOLLOWUP-STAGE2] Extracted context: table={previous_context.table_name if previous_context else 'N/A'}")
            
            # Build followup_context using ROUTER'S signal (not redundant LLM call)
            # The router already did LLM-based follow-up classification in stage 3
            router_is_followup = (router_decision.followup_type == schemas.FollowupType.RUN_SQL_FOLLOW_UP) if router_decision else False
            
            followup_context = FollowUpContext(
                is_followup=router_is_followup or preliminary_router_followup,
                followup_type=FollowUpTypeAnalyzer.REFINEMENT if (router_is_followup or preliminary_router_followup) else FollowUpTypeAnalyzer.NEW,
                confidence=router_decision.confidence if router_decision else 0.5,
                previous_context=previous_context,
                reasoning=f"Context extracted, router signal: {router_decision.followup_type.value if router_decision else 'N/A'}"
            )
            
            logger.debug(f"[FOLLOWUP-STAGE2] Router follow-up signal: {router_is_followup}")
            logger.debug(f"[FOLLOWUP-STAGE2] Confidence: {followup_context.confidence:.0%}")
                
        except Exception as e:
            logger.debug(f"\n[FOLLOWUP-STAGE2] ⚠️ Context extraction failed: {e}")
            # If extraction fails and we don't have RAG context, create a default
            if followup_context is None:
                logger.debug(f"[FOLLOWUP-STAGE2] Creating default context (extraction failed)")
                followup_context = FollowUpContext(
                    is_followup=preliminary_router_followup,  # Use router signal as fallback
                    followup_type=preliminary_followup_type if preliminary_followup_type else FollowUpTypeAnalyzer.NEW,
                    confidence=0.3,  # Low confidence for fallback
                    reasoning="Fallback to router signal (context extraction failed)",
                    previous_context=None
                )
    
    # For truly new sessions, create a NEW context
    if followup_context is None:
        logger.debug(f"[FOLLOWUP] Creating NEW context (first query in session)")
        followup_context = FollowUpContext(
            is_followup=False,
            followup_type=FollowUpTypeAnalyzer.NEW,
            confidence=1.0,
            reasoning="New session - first query",
            previous_context=None
        )
    
    # Map follow-up type for potential future refinements
    def map_followup_type(analyzer_type: FollowUpTypeAnalyzer) -> FollowUpTypeState:
        """Map analyzer's follow-up type to state manager's follow-up type."""
        mapping = {
            FollowUpTypeAnalyzer.NEW: FollowUpTypeState.NEW_REQUEST,
            FollowUpTypeAnalyzer.REFINEMENT: FollowUpTypeState.REFINE,
            FollowUpTypeAnalyzer.EXPANSION: FollowUpTypeState.EXPAND,
            FollowUpTypeAnalyzer.CLARIFICATION: FollowUpTypeState.DRILL_DOWN,
            FollowUpTypeAnalyzer.PIVOT: FollowUpTypeState.TRANSFORM,
            FollowUpTypeAnalyzer.CONTINUATION: FollowUpTypeState.TRANSFORM,
        }
        return mapping.get(analyzer_type, FollowUpTypeState.NEW_REQUEST)
    
    state_followup_type = map_followup_type(followup_context.followup_type)
    state_followup_type_val = state_followup_type.value if hasattr(state_followup_type, 'value') else str(state_followup_type)
    logger.debug(f"[FOLLOWUP-LLM] Mapped to state type: {state_followup_type_val}")
    
    # ======================================================================
    # STEP 6.7: FINAL ARBITER DECISION (Single Authority)
    # ======================================================================
    # FIX: Use the DecisionArbiter as the SINGLE SOURCE OF TRUTH for routing decisions.
    # This resolves the "split-brain" issue where multiple components make conflicting decisions.
    
    from .decision_arbiter import get_decision_arbiter, ArbiterDecision, TurnClass
    
    arbiter = await get_decision_arbiter(db)
    
    # Collect last turn state for arbiter
    last_turn_state_dict = None
    if session_manager.last_query_state:
        last_turn_state_dict = {
            "artifacts": {
                "sql": session_manager.last_query_state.generated_sql,
                "tables": session_manager.last_query_state.selected_entities,
                "filters": [f.to_dict() for f in session_manager.last_query_state.filters] if session_manager.last_query_state.filters else [],
            }
        }

    # Fallback: if last_turn_state has no SQL (e.g. after logout/re-login),
    # use RAG-recovered previous context so the arbiter can detect follow-ups.
    if (not last_turn_state_dict or not last_turn_state_dict.get("artifacts", {}).get("sql")):
        if followup_context and followup_context.previous_context:
            pc = followup_context.previous_context
            last_turn_state_dict = {
                "artifacts": {
                    "sql": pc.generated_sql,
                    "tables": [pc.table_name] if pc.table_name else [],
                    "filters": [],
                }
            }
    
    # Prepare followup_context dict for arbiter
    followup_context_for_arbiter = {
        "is_followup": followup_context.is_followup,
        "followup_type": followup_context.followup_type.value if hasattr(followup_context.followup_type, 'value') else str(followup_context.followup_type),
        "confidence": followup_context.confidence,
        "reasoning": followup_context.reasoning,
    } if followup_context else None
    
    # Get hard signals from router
    hard_signals = None
    if router_decision and hasattr(router_decision, 'signals_used'):
        # Reconstruct hard signals if available
        try:
            hard_signals = schemas.RouterSignals(
                has_uploaded_files=bool(handler_kwargs.get("files")),
                db_connected=True,
                last_tool=None,
                last_sql_exists=bool(session_manager.last_query_state and session_manager.last_query_state.generated_sql),
                last_file_context_exists=False,
                time_since_last_turn=None,
            )
        except Exception:
            hard_signals = None
    
    # Call the arbiter for the FINAL decision
    arbiter_decision = await arbiter.arbitrate(
        user_query=user_query,
        router_decision=router_decision,
        followup_context=followup_context_for_arbiter,
        last_turn_state=last_turn_state_dict,
        hard_signals=hard_signals,
        session_id=session_id,
        user_id=user_id,
        is_new_session=is_truly_new_session,
        previous_messages_count=len(previous_messages),
    )
    
    logger.debug(f"\n[ARBITER] ✅ Final Decision Made:")
    logger.debug(f"[ARBITER]   Turn Class: {arbiter_decision.final_turn_class.value}")
    logger.debug(f"[ARBITER]   Tool: {arbiter_decision.final_tool.value}")
    logger.debug(f"[ARBITER]   Subtype: {arbiter_decision.final_followup_subtype.value}")
    logger.debug(f"[ARBITER]   Confidence: {arbiter_decision.confidence:.0%}")
    logger.debug(f"[ARBITER]   Merge State: {arbiter_decision.should_merge_state}")
    logger.debug(f"[ARBITER]   Reset State: {arbiter_decision.should_reset_state}")
    logger.debug(f"[ARBITER]   Reasoning: {arbiter_decision.reasoning}")
    
    # FIX: NOW apply state management based on arbiter's decision (not early router decision)
    if arbiter_decision.should_reset_state:
        logger.debug(f"[STATE] Arbiter says RESET → clearing previous state")
        session_manager.reset_state()
    elif arbiter_decision.should_merge_state:
        logger.debug(f"[STATE] Arbiter says MERGE → preserving context for follow-up")
        # State is preserved, will be merged during SQL generation
    
    # Update is_followup based on arbiter's final decision
    is_followup = arbiter_decision.final_turn_class == TurnClass.FOLLOW_UP

    # ======================================================================
    # ARBITER TOOL OVERRIDE — honour arbiter's final_tool
    # ======================================================================
    # The arbiter may have determined a different tool than the initial router
    # decision (e.g. the query looks like chat to the router but the arbiter
    # knows it's a SQL follow-up, or vice-versa).  Redirect NOW so downstream
    # code always uses the authoritative decision.
    if arbiter_decision.final_tool == schemas.Tool.CHAT:
        logger.info(
            "[ARBITER] Overriding router decision to CHAT "
            f"(router said {router_decision.tool.value}, arbiter confidence {arbiter_decision.confidence:.0%})"
        )
        from .query_handler import build_standard_response
        chat_response = await build_standard_response(
            db=db,
            user_id=user_id,
            session_id=session_id,
            query=user_query,
            message_id=handler_kwargs.get("message_id"),
        )
        chat_response.intent = {
            "domain": schemas.Tool.CHAT.value,
            "confidence": arbiter_decision.confidence,
            "reasoning": arbiter_decision.reasoning,
            "action": schemas.Tool.CHAT.value,
            "arbiter_override": True,
        }
        return chat_response

    elif arbiter_decision.final_tool == schemas.Tool.ANALYZE_FILE:
        logger.info(
            "[ARBITER] Overriding router decision to ANALYZE_FILE "
            f"(router said {router_decision.tool.value}, arbiter confidence {arbiter_decision.confidence:.0%})"
        )
        from .query_handler import build_file_query_response, build_file_lookup_response
        files = handler_kwargs.get("files")
        if files:
            file_response = await build_file_query_response(
                db=db,
                user_id=user_id,
                session_id=session_id,
                query=user_query,
                files=files,
                message_id=handler_kwargs.get("message_id"),
            )
        else:
            file_response = await build_file_lookup_response(
                db=db,
                user_id=user_id,
                session_id=session_id,
                query=user_query,
                message_id=handler_kwargs.get("message_id"),
            )
        file_response.intent = {
            "domain": schemas.Tool.ANALYZE_FILE.value,
            "confidence": arbiter_decision.confidence,
            "reasoning": arbiter_decision.reasoning,
            "action": schemas.Tool.ANALYZE_FILE.value,
            "arbiter_override": True,
        }
        return file_response

    # ======================================================================
    # STEP 7.7a: PROJECTION CHANGE DETECTION (NEW)
    # ======================================================================
    # For follow-ups like "list all those clients" after a COUNT, transform COUNT→SELECT *
    # This is critical for drill-down from aggregate to detail views
    
    projection_change_sql = None
    
    if is_followup and session_manager.last_query_state and session_manager.last_query_state.generated_sql:
        previous_sql = session_manager.last_query_state.generated_sql
        query_lower = user_query.lower()
        
        # Detect drill-down intent (user wants rows, not aggregate)
        drill_down_keywords = {'list', 'show', 'display', 'give', 'get', 'detail', 'details', 'all'}
        reference_keywords = {'those', 'them', 'these', 'the', 'client', 'customer', 'record', 'row', 'result'}
        
        has_drill_down_intent = any(kw in query_lower for kw in drill_down_keywords)
        has_reference = any(kw in query_lower for kw in reference_keywords)
        
        # Check if previous SQL was an aggregate (COUNT, SUM, etc.)
        previous_sql_upper = previous_sql.upper()
        is_previous_aggregate = any(agg in previous_sql_upper for agg in ['COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX('])
        
        if has_drill_down_intent and has_reference and is_previous_aggregate:
            logger.debug(f"\n[STEP 7.7a] 🔄 PROJECTION CHANGE DETECTED")
            logger.debug(f"[STEP 7.7a]   Intent: drill-down (aggregate → rows)")
            logger.debug(f"[STEP 7.7a]   Previous SQL type: aggregate")
            
            # Transform: COUNT(*) FROM ... WHERE ... → SELECT * FROM ... WHERE ...
            import re
            
            # Extract FROM and WHERE clauses from previous SQL
            # Pattern: SELECT ... FROM table WHERE conditions [LIMIT ...]
            from_match = re.search(r'\bFROM\s+(.+?)(?:\s+WHERE|\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|\s*;|\s*$)', previous_sql, re.I | re.S)
            where_match = re.search(r'\bWHERE\s+(.+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|\s*;|\s*$)', previous_sql, re.I | re.S)
            
            if from_match:
                table_clause = from_match.group(1).strip()
                where_clause = where_match.group(1).strip() if where_match else None
                
                # Build transformed SQL
                if where_clause:
                    projection_change_sql = f"SELECT * FROM {table_clause} WHERE {where_clause} LIMIT 1000"
                else:
                    projection_change_sql = f"SELECT * FROM {table_clause} LIMIT 1000"
                
                logger.debug(f"[STEP 7.7a] ✅ Transformed SQL: {projection_change_sql}")
                logger.debug(f"[STEP 7.7a]   This will be used instead of re-generating from scratch")
                
                # Store the projection-changed SQL for later use
                session_manager._projection_change_sql = projection_change_sql
            else:
                logger.debug(f"[STEP 7.7a] ⚠️ Could not extract FROM clause, falling back to normal generation")
        else:
            if is_followup:
                logger.debug(f"[STEP 7.7a] No projection change needed (drill_down={has_drill_down_intent}, ref={has_reference}, aggregate={is_previous_aggregate})")
    
    # ======================================================================
    # STEP 7.7: ENHANCED SEMANTIC VALUE GROUNDING (FK-Aware)
    # ======================================================================
    # NEW: Apply intelligent value grounding with FK relationship awareness
    # This handles multi-table scenarios and auto-generates JOINs
    
    # FIX: Use the current query's followup_context (just computed in STAGES 1-2 + arbiter)
    # NOT session_manager.followup_context_from_rag which is stale from previous query
    enhanced_followup_context = followup_context  # Use the freshly computed context
    grounded_filter_values = []
    
    try:
        logger.debug(f"\n[STEP 7.7] 🧠 Applying enhanced semantic value grounding...")
        
        # Get enhanced grounder instance
        grounder = get_semantic_value_grounder_enhanced()
        
        # Ensure grounder is initialized
        if not grounder.initialized:
            logger.debug(f"[STEP 7.7] Initializing grounder (first use)...")
            await grounder.initialize_for_tables(db, sample_size=100)
        
        # Determine context for grounding
        # FIX: Use arbiter's decision (is_followup), NOT RAG-based followup_context.is_followup
        if is_followup:
            # Follow-up query: Use the table from previous context
            logger.debug(f"[STEP 7.7] Query Type: FOLLOW-UP (using previous context)")
            # FIX: Use followup_context.previous_context (current), not session_manager (stale)
            rag_context_for_grounding = followup_context.previous_context if followup_context else None
            target_tables = [rag_context_for_grounding.table_name] if rag_context_for_grounding and rag_context_for_grounding.table_name else []
            logger.debug(f"[STEP 7.7] Target table from current context: {target_tables}")
        else:
            # Initial query: Will be handled during SQL generation
            logger.debug(f"[STEP 7.7] Query Type: NEW_QUERY (arbiter said reset, no previous context)")
            target_tables = []
        
        # For follow-up queries with table context, attempt value grounding
        if target_tables and grounder.initialized:
            try:
                # Try to extract filter values from query
                # This is a simplified approach - the main grounding happens in query_handler
                logger.debug(f"[STEP 7.7] Hold value grounding for query_handler (context ready)")
                
                # Store grounder info in context for query_handler to use
                # FIX: Store on followup_context (which will be stored later), not stale session value
                if followup_context and not hasattr(followup_context, 'value_grounding_ready'):
                    followup_context.value_grounding_ready = True
                    followup_context.grounder_initialized = True
                    
            except Exception as e:
                logger.debug(f"[STEP 7.7] ⚠️ Value grounding prep failed: {e}")
        
        logger.debug(f"[STEP 7.7] ✅ Value grounding context prepared")
        
    except Exception as e:
        logger.debug(f"[STEP 7.7] ⚠️ Value grounding phase failed (non-critical): {e}")
        # Continue without grounding - query_handler will handle fallback
    
    # ======================================================================
    # STEP 7.8: INTELLIGENT QUERY ORCHESTRATION (NEW) - WITH PERFORMANCE OPTIMIZATION
    # ======================================================================
    # PERFORMANCE OPTIMIZATION: Skip heavy orchestration for high-confidence simple queries
    # The efficient router already identified the query with high confidence, so we can
    # bypass the expensive semantic analysis for simple patterns
    
    orchestrator_context = None
    
    # ======================================================================
    # NEW: EXECUTION POLICY ENGINE - ChatGPT-Grade Performance Optimization
    # ======================================================================
    # Determine optimal execution path (FAST_SQL, FULL_SQL, FOLLOWUP_SQL, etc.)
    # This intelligently routes queries to avoid expensive processing for simple cases
    logger.debug(f"\n[POLICY ENGINE] 🎯 Determining execution policy...")
    
    from .execution_policy_engine import get_execution_policy_engine, ExecutionPath
    policy_engine = get_execution_policy_engine()
    
    # Get schema info for context
    from ..helpers import get_database_schema
    try:
        schema_info = await get_database_schema(db)
    except Exception as e:
        logger.debug(f"[POLICY ENGINE] ⚠️ Could not retrieve schema: {e}")
        schema_info = ""
    
    # Prepare context for policy decision
    arbiter_context = {
        "turn_class": arbiter_decision.final_turn_class.value if arbiter_decision else "new_query",
        "confidence": arbiter_decision.confidence if arbiter_decision else 0.0,
        "intent": arbiter_decision.final_tool.value if arbiter_decision else "RUN_SQL",
        "can_skip_orchestration": arbiter_decision.can_skip_orchestration if arbiter_decision else False,
    }
    
    conversation_context_dict = {
        "turn_count": len(previous_messages),
        "has_history": bool(conversation_history),
        "is_followup": is_followup,
    }
    
    schema_context_dict = {
        "table_count": schema_info.count("CREATE TABLE") if schema_info else 0,
    }
    
    try:
        execution_policy = await policy_engine.determine_execution_path(
            user_query=user_query,
            arbiter_decision=arbiter_context,
            conversation_context=conversation_context_dict,
            schema_context=schema_context_dict,
        )
        
        logger.debug(f"[POLICY ENGINE] ✅ Policy determined: {execution_policy.path.value}")
        logger.debug(f"[POLICY ENGINE]   Estimated duration: ~{execution_policy.estimated_duration_seconds}s")
        logger.debug(f"[POLICY ENGINE]   Skip orchestration: {execution_policy.skip_orchestration}")
        logger.debug(f"[POLICY ENGINE]   Skip value grounding: {execution_policy.skip_grounding}")
        logger.debug(f"[POLICY ENGINE]   Rationale: {execution_policy.rationale}")
        
        # Override orchestration skip based on policy
        if execution_policy.skip_orchestration:
            skip_orchestration = True
            orchestrator_context = {
                'skipped_for_performance': True,
                'reason': 'execution_policy',
                'policy_path': execution_policy.path.value,
                'reasoning': execution_policy.rationale
            }
            logger.debug(f"[POLICY ENGINE] ⚡ Overriding skip_orchestration = True (policy decision)")
        
    except Exception as e:
        logger.debug(f"[POLICY ENGINE] ⚠️ Policy engine failed (non-critical): {e}")
        # Fallback to arbiter decision
        execution_policy = None
        skip_orchestration = arbiter_decision.can_skip_orchestration
    
    # FIX: Use arbiter's decision for orchestration skip, NOT early router decision
    # The arbiter has already combined all signals (router, followup classifier, hard signals)
    # to make the authoritative decision about whether orchestration can be skipped.
    if not execution_policy:
        skip_orchestration = arbiter_decision.can_skip_orchestration
    
    # CRITICAL FIX: Also skip orchestration when projection-change SQL is already computed
    # This prevents unnecessary DB work and latency
    has_projection_change_sql = hasattr(session_manager, '_projection_change_sql') and session_manager._projection_change_sql
    
    if has_projection_change_sql:
        logger.debug(f"\n[STEP 7.8] ⚡⚡⚡ PROJECTION_BYPASS_V3 - SKIPPING ORCHESTRATION ⚡⚡⚡")
        logger.debug(f"[STEP 7.8] ⚡ SKIPPING ORCHESTRATION (projection-change SQL already computed)")
        logger.debug(f"[STEP 7.8] ✅ Fast path - will use pre-computed drill-down SQL")
        skip_orchestration = True
        orchestrator_context = {
            'skipped_for_performance': True,
            'reason': 'projection_change_bypass',
            'reasoning': 'Projection-change SQL already computed in STEP 7.7a'
        }
    
    if has_projection_change_sql:
        # Already handled above, just ensuring we don't fall into the else block
        pass
    elif skip_orchestration:
        logger.debug(f"\n[STEP 7.8] ⚡ SKIPPING ORCHESTRATION (arbiter decision)")
        logger.debug(f"[STEP 7.8] ✅ Fast path - arbiter determined orchestration unnecessary")
        logger.debug(f"[STEP 7.8]   Turn class: {arbiter_decision.final_turn_class.value}")
        logger.debug(f"[STEP 7.8]   Confidence: {arbiter_decision.confidence:.0%}")
        orchestrator_context = {
            'skipped_for_performance': True,
            'arbiter_confidence': arbiter_decision.confidence,
            'arbiter_turn_class': arbiter_decision.final_turn_class.value,
            'reasoning': f'Arbiter determined orchestration unnecessary ({arbiter_decision.reasoning})'
        }
    else:
        logger.debug(f"\n[STEP 7.8] 🧠 ORCHESTRATING INTELLIGENT QUERY ANALYSIS (confidence: {getattr(router_decision, 'confidence', 'unknown')})")
        try:
            orchestrator = get_query_orchestrator()
            
            # Prepare orchestrator inputs
            # CRITICAL: Honor arbiter's decision - only pass previous context if arbiter says follow-up
            previous_query_context = None
            if is_followup and enhanced_followup_context and enhanced_followup_context.previous_context:
                previous_query_context = {
                    'table': enhanced_followup_context.previous_context.table_name,
                    'columns_used': enhanced_followup_context.previous_context.columns_selected,
                    'filters': enhanced_followup_context.previous_context.filters,
                    'query': enhanced_followup_context.previous_context.generated_sql,
                }
                logger.debug(f"[STEP 7.8] 📋 Passing previous query context to orchestrator (arbiter approved)")
                logger.debug(f"[STEP 7.8]   Table: {previous_query_context['table']}")
            else:
                logger.debug(f"[STEP 7.8] 🆕 No previous context for orchestrator (arbiter decision: is_followup={is_followup})")
            
            # Prepare conversation history for orchestrator
            # SelectiveHistory has 'recent_messages' (list of dicts), not 'messages'
            orchestrator_conversation_history = []
            if selective_history:
                # For new sessions, don't include conversation history
                if not is_truly_new_session and selective_history.recent_messages:
                    for msg in selective_history.recent_messages:
                        if isinstance(msg, dict) and 'content' in msg:
                            orchestrator_conversation_history.append(msg['content'])
                        elif isinstance(msg, dict) and 'message' in msg:
                            orchestrator_conversation_history.append(msg['message'])
                        elif hasattr(msg, 'content'):
                            orchestrator_conversation_history.append(msg.content)
            
            # Run orchestrator pipeline
            orch_result = await orchestrator.orchestrate_query(
                user_prompt=user_query,
                db=db,
                previous_query_context=previous_query_context,
                conversation_history=orchestrator_conversation_history
            )
            
            # Store orchestrator result for downstream use
            orchestrator_context = {
                'semantic_analysis': orch_result.semantic_analysis,
                'value_mappings': orch_result.value_mappings,
                'generated_sql': orch_result.generated_sql,
                'sql_patterns': orch_result.sql_patterns,
                'sql_complexity': orch_result.sql_complexity,
                'reasoning': orch_result.reasoning,
            }
            
            logger.debug(f"[STEP 7.8] ✅ Orchestration complete")
            logger.debug(f"[STEP 7.8]   Query Type: {orch_result.semantic_analysis.query_type}")
            logger.debug(f"[STEP 7.8]   Tables: {[t.table_name for t in orch_result.semantic_analysis.relevant_tables]}")
            logger.debug(f"[STEP 7.8]   Patterns: {', '.join(orch_result.sql_patterns)}")
            logger.debug(f"[STEP 7.8]   SQL: {orch_result.generated_sql}")
            
        except Exception as e:
            logger.warning(f"[STEP 7.8] ⚠️ Orchestrator phase failed (non-critical): {e}")
            logger.debug(f"[STEP 7.8] ⚠️ Orchestrator analysis skipped, continuing with existing flow: {e}")
            # Continue without orchestrator - not critical
    
    # ======================================================================
    # STEP 7.6: Store RAG-based followup context for query_handler to use
    # ======================================================================
    # CRITICAL: Honor arbiter's decision - only store if arbiter says it's a follow-up
    # The arbiter has already considered RAG context + entity changes + standalone patterns
    # IMPORTANT: Do this BEFORE building handler_kwargs so the context is available

    # FALLBACK: If arbiter says follow-up but followup_context has no previous_context
    # (e.g. RAG similarity too low after server restart), build it from last_query_state.
    # last_query_state is always persisted to the DB so it survives restarts.
    if (
        is_followup
        and session_manager.last_query_state
        and session_manager.last_query_state.generated_sql
        and not (followup_context and followup_context.previous_context)
    ):
        _prev_sql = session_manager.last_query_state.generated_sql
        _table_name = _extract_table_from_sql(_prev_sql)
        _filters = _extract_filters_from_sql(_prev_sql)
        _prev_ctx = PreviousQueryContext(
            query_text=getattr(session_manager.last_query_state, 'query', '') or '',
            generated_sql=_prev_sql,
            table_name=_table_name,
            filters=_filters,
            columns_selected=[],
            result_count=getattr(session_manager.last_query_state, 'result_count', 0) or 0,
        )
        followup_context = FollowUpContext(
            is_followup=True,
            followup_type=FollowUpTypeAnalyzer.REFINEMENT,
            confidence=0.80,
            reasoning="Follow-up context rebuilt from last_query_state (RAG similarity below threshold)",
            previous_context=_prev_ctx,
        )
        logger.info(
            "[STEP 7.6] Built previous_context from last_query_state: table=%s, sql=%s",
            _table_name, _prev_sql[:80],
        )

    if is_followup and followup_context and followup_context.previous_context:
        logger.debug(f"[STEP 7.6] ✅ Storing followup context for query handler (arbiter approved)")
        logger.debug(f"[STEP 7.6]   Table: {followup_context.previous_context.table_name}")
        logger.debug(f"[STEP 7.6]   Filters: {len(followup_context.previous_context.filters)}")
        session_manager.followup_context_from_rag = followup_context
    else:
        # Clear it if arbiter says reset or no follow-up
        logger.debug(f"[STEP 7.6] 🔄 Clearing followup context (arbiter decision: is_followup={is_followup})")
        session_manager.followup_context_from_rag = None
    
    # ======================================================================
    # STEP 7b: DYNAMIC FUNCTION PARAMETER INJECTION (ChatGPT-style)
    # ======================================================================
    # Dynamically inspect what parameters build_data_query_response actually accepts
    # This is 100% dynamic - no hardcoding of parameter names
    from .query_handler import build_data_query_response
    
    sig = inspect.signature(build_data_query_response)
    accepted_params = set(sig.parameters.keys())
    
    # Build handler kwargs with ONLY the parameters the function accepts
    # FIX: Use session_manager.followup_context_from_rag DIRECTLY (not the early snapshot)
    # STEP 7.6 just updated this, so we get the final arbitrated followup context
    final_handler_kwargs = {
        'db': db,
        'user_id': user_id,
        'session_id': session_id,
        'query': user_query,
        'conversation_history': selective_history.to_prompt_context(),
        'followup_context_from_rag': session_manager.followup_context_from_rag,  # FIX: Use latest, not stale snapshot
        'orchestrator_context': orchestrator_context,  # NEW: Pass orchestrator analysis results
        'session_manager': session_manager,  # NEW: Pass session manager for follow-up rewriting
    }
    
    # Dynamically add any extra kwargs that the function accepts (never hardcode param names)
    for key, value in handler_kwargs.items():
        if key in accepted_params:
            final_handler_kwargs[key] = value
        else:
            logger.debug(f"[DYNAMIC FILTERING] Parameter '{key}' not accepted by handler - automatically filtered out")
    
    # ======================================================================
    # STEP 7.5: Initialize QueryState if not already set
    # ======================================================================
    # CRITICAL: Create a new QueryState for this database query if one doesn't exist
    # This is needed for STEP 9 to store the generated SQL and result metadata
    if session_manager.last_query_state is None:
        logger.debug(f"[STEP 7.5] Creating new QueryState for database query")
        session_manager.last_query_state = QueryState(
            user_query=user_query,
            domain=QueryDomain.DATABASE,
        )
        logger.debug(f"[STEP 7.5] ✅ QueryState created and ready for STEP 9")
    else:
        logger.debug(f"[STEP 7.5] Reusing existing QueryState for follow-up")
    
    # ======================================================================
    # STEP 8: Call the data query handler (only for DATABASE domain)
    # ======================================================================
    # CRITICAL FIX: If we have a projection-change SQL from STEP 7.7a, 
    # BYPASS the query_handler entirely and execute that SQL directly.
    # This prevents the semantic pipeline from regenerating a fresh query without filters.
    
    response_wrapper = None
    projection_change_bypass = False
    
    if hasattr(session_manager, '_projection_change_sql') and session_manager._projection_change_sql:
        projection_change_sql = session_manager._projection_change_sql
        logger.debug(f"\n[STEP 8] 🚀🚀🚀 PROJECTION_BYPASS_V3 - DIRECT EXECUTION 🚀🚀🚀")
        logger.debug(f"[STEP 8] 🚀 PROJECTION CHANGE BYPASS - Executing transformed SQL directly")
        logger.debug(f"[STEP 8] SQL: {projection_change_sql}")
        
        # Clear the projection change SQL so it doesn't get reused incorrectly
        session_manager._projection_change_sql = None
        projection_change_bypass = True
        
        try:
            from sqlalchemy import text
            
            # Execute the pre-computed projection-changed SQL
            result = await db.execute(text(projection_change_sql))
            rows = result.fetchall()
            row_count = len(rows)
            
            logger.debug(f"[STEP 8] ✅ Executed projection-changed SQL, got {row_count} rows")
            
            # Convert to dict format
            result_data = [dict(row._mapping) for row in rows] if rows else []
            column_names = list(result_data[0].keys()) if result_data else []
            
            # Build a minimal response wrapper for the rest of the flow
            response_wrapper = schemas.ResponseWrapper(
                success=True,
                response=schemas.DataQueryResponse(
                    intent=user_query,
                    confidence=0.95,
                    message=f"Found {row_count} matching records.",
                    datasets=[schemas.DataQueryDataset(
                        id="projection_change_results",
                        description="Query results from projection change (detail view of previous aggregate)",
                        data=result_data,
                        count=row_count,
                    )] if result_data else [],
                    visualizations=[],
                    layout=schemas.Layout(type="table", arrangement="single"),
                    related_queries=[],
                    debug={
                        "normalized_user_request": user_query,
                        "sql_executed": projection_change_sql,
                        "row_count": row_count,
                        "complexity": "projection_change_followup",
                        "response_type": "data",
                        "projection_change_bypass": True,
                    },
                    metadata={
                        "projection_change": True,
                        "original_aggregate_sql": session_manager.last_query_state.generated_sql if session_manager.last_query_state else None,
                    },
                ),
                timestamp=current_timestamp(),
                original_query=user_query,
            )
            
            # Update session state with the new SQL
            session_manager.last_query_state.generated_sql = projection_change_sql
            session_manager.last_query_state.result_count = row_count
            session_manager.last_query_state.user_query = user_query
            
            logger.debug(f"[STEP 8] ✅ Projection change bypass complete - returning {row_count} rows")
            
        except Exception as e:
            logger.debug(f"[STEP 8] ⚠️ Projection change SQL execution failed: {e}")
            logger.debug(f"[STEP 8] Falling back to normal query_handler flow")
            projection_change_bypass = False
            # Clear any partial state
            session_manager._projection_change_sql = None
    
    if not projection_change_bypass:
        logger.debug(f"\n[HANDLER] Calling database handler with enriched context")
        logger.debug(f"[HANDLER] Parameters being passed: {', '.join(final_handler_kwargs.keys())}")
        
        response_wrapper = await build_data_query_response(**final_handler_kwargs)
    
    # ======================================================================
    # STEP 9: Extract generated SQL from response and update QueryState
    # ======================================================================
    # CRITICAL: Store the generated SQL in QueryState so it's available for follow-up queries
    generated_sql = None
    result_count = 0
    column_names = []
    sample_row = None
    all_result_rows = []  # Store ALL result rows for pgvector embeddings
    
    logger.debug(f"\n[STEP 9] EXTRACTION STARTING")
    logger.debug(f"[STEP 9] response_wrapper type: {type(response_wrapper)}")
    logger.debug(f"[STEP 9] response_wrapper.response type: {type(response_wrapper.response)}")
    logger.debug(f"[STEP 9] session_manager: {session_manager}")
    logger.debug(f"[STEP 9] last_query_state: {session_manager.last_query_state if session_manager else 'N/A'}")
    
    if session_manager.last_query_state and response_wrapper.response:
        logger.debug(f"[STEP 9] ✅ Both conditions met - extracting SQL")
        
        # Check what attributes the response has
        logger.debug(f"[STEP 9] response attributes: {dir(response_wrapper.response)}")
        
        # Extract SQL from debug metadata (now stored as dict due to extra="allow")
        if hasattr(response_wrapper.response, 'debug'):
            debug_info = response_wrapper.response.debug
            logger.debug(f"[STEP 9] Found debug attribute: {debug_info is not None}")
            logger.debug(f"[STEP 9] debug_info type: {type(debug_info)}, value: {debug_info}")
            
            # Handle both dict and object cases (Pydantic extra fields are stored as dicts)
            try:
                if isinstance(debug_info, dict):
                    # Extract from dict
                    generated_sql = debug_info.get('sql_executed', None)
                    result_count = debug_info.get('row_count', 0)
                    column_names = debug_info.get('columns', [])
                    logger.debug(f"[STEP 9] Extracted from dict - sql: {generated_sql is not None}, rows: {result_count}, cols: {len(column_names)}")
                else:
                    # Extract from object attributes
                    generated_sql = getattr(debug_info, 'sql_executed', None)
                    result_count = getattr(debug_info, 'row_count', 0)
                    column_names = getattr(debug_info, 'columns', []) or []
                    logger.debug(f"[STEP 9] Extracted from object - sql: {generated_sql is not None}, rows: {result_count}, cols: {len(column_names)}")
                
                if generated_sql:
                    session_manager.last_query_state.generated_sql = generated_sql
                    session_manager.last_query_state.result_count = result_count
                    logger.debug(f"[STEP 9] ✅ Extracted SQL: {generated_sql} ({result_count} rows)")
                else:
                    logger.debug(f"[STEP 9] ⚠️ No sql_executed in debug_info. Keys: {list(debug_info.keys()) if isinstance(debug_info, dict) else 'N/A'}")
            except Exception as e:
                import traceback
                logger.debug(f"[STEP 9] ⚠️ Error extracting from debug: {e}")
                logger.debug(f"[STEP 9] Traceback: {traceback.format_exc()}")
        else:
            logger.debug(f"[STEP 9] ❌ No debug attribute found on response")
            logger.debug(f"[STEP 9] Available attributes: {[attr for attr in dir(response_wrapper.response) if not attr.startswith('_')]}")
            
            # FALLBACK: For follow-ups, use the previously generated SQL
            if session_manager.last_query_state and session_manager.last_query_state.generated_sql:
                logger.debug(f"[STEP 9] 📋 Fallback: Using previous SQL from session for follow-up")
                generated_sql = session_manager.last_query_state.generated_sql
                result_count = session_manager.last_query_state.result_count
                column_names = session_manager.last_query_state.selected_columns or []
                logger.debug(f"[STEP 9] ✅ Using previous SQL: {generated_sql} ({result_count} rows)")
    
    # IMPORTANT: Extract actual result rows from response
    # Check multiple locations: datasets (old format), or debug.result_rows (new format)
    
    # Add comprehensive debugging
    logger.debug(f"[STEP 9] Response type: {type(response_wrapper.response)}")
    logger.debug(f"[STEP 9] Has debug attr: {hasattr(response_wrapper.response, 'debug')}")
    if hasattr(response_wrapper.response, 'debug'):
        debug_obj = response_wrapper.response.debug
        logger.debug(f"[STEP 9] Debug type: {type(debug_obj)}")
        logger.debug(f"[STEP 9] Debug is None: {debug_obj is None}")
        if debug_obj is not None:
            # Handle both dict and Pydantic model
            if hasattr(debug_obj, 'result_rows'):
                logger.debug(f"[STEP 9] Debug has result_rows attribute")
                result_rows_value = debug_obj.result_rows if hasattr(debug_obj, 'result_rows') else None
                logger.debug(f"[STEP 9] result_rows value: {type(result_rows_value)}, length: {len(result_rows_value) if result_rows_value else 0}")
            elif isinstance(debug_obj, dict) and 'result_rows' in debug_obj:
                logger.debug(f"[STEP 9] Debug is dict with result_rows key")
    
    # Try debug.result_rows first (LamaResponse format - most common now)
    # Handle both Pydantic model and dict
    if hasattr(response_wrapper.response, 'debug') and response_wrapper.response.debug:
        debug_obj = response_wrapper.response.debug
        # Try Pydantic model attribute access first
        if hasattr(debug_obj, 'result_rows') and debug_obj.result_rows:
            all_result_rows = debug_obj.result_rows
            logger.debug(f"[STEP 9] ✅ Extracted {len(all_result_rows)} result rows from debug.result_rows (Pydantic)")
            if all_result_rows and len(all_result_rows) > 0 and isinstance(all_result_rows[0], dict):
                sample_row = all_result_rows[0]
                logger.debug(f"[STEP 9] First row keys: {list(sample_row.keys())}")
        # Fallback to dict access
        elif isinstance(debug_obj, dict) and 'result_rows' in debug_obj:
            all_result_rows = debug_obj.get('result_rows', [])
            if all_result_rows and len(all_result_rows) > 0:
                logger.debug(f"[STEP 9] ✅ Extracted {len(all_result_rows)} result rows from debug.result_rows (dict)")
                if isinstance(all_result_rows[0], dict):
                    sample_row = all_result_rows[0]
                    logger.debug(f"[STEP 9] First row keys: {list(sample_row.keys())}")
    # Fallback to datasets (DataQueryResponse format - legacy)
    if not all_result_rows and hasattr(response_wrapper.response, 'datasets') and response_wrapper.response.datasets:
        datasets = response_wrapper.response.datasets
        if len(datasets) > 0:
            first_dataset = datasets[0]
            logger.debug(f"[STEP 9] Found datasets: {len(datasets)} dataset(s)")
            if hasattr(first_dataset, 'data') and first_dataset.data:
                all_result_rows = first_dataset.data
                logger.debug(f"[STEP 9] ✅ Extracted {len(all_result_rows)} result rows from datasets")
                if len(all_result_rows) > 0 and isinstance(first_dataset.data[0], dict):
                    sample_row = all_result_rows[0]
                    logger.debug(f"[STEP 9] First row keys: {list(sample_row.keys())}")
    
    # Log if no rows found anywhere
    if not all_result_rows:
        logger.debug(f"[STEP 9] ⚠️ No result rows extracted from response (will try re-execution if SQL available)")
    
    # FALLBACK: For follow-ups, use the previously generated SQL if needed
    # PRIORITY: projection_change_sql > previous SQL unchanged
    if not generated_sql:
        # First priority: Use projection-changed SQL if we detected drill-down intent
        if hasattr(session_manager, '_projection_change_sql') and session_manager._projection_change_sql:
            logger.debug(f"[STEP 9] 🔄 Using PROJECTION-CHANGED SQL (drill-down from aggregate to rows)")
            generated_sql = session_manager._projection_change_sql
            # Clear it after use
            session_manager._projection_change_sql = None
            logger.debug(f"[STEP 9] ✅ Transformed SQL: {generated_sql}")
        # Second priority: Fall back to previous SQL unchanged
        elif session_manager.last_query_state and session_manager.last_query_state.generated_sql:
            logger.debug(f"[STEP 9] 📋 Fallback: Using previous SQL from session for follow-up")
            generated_sql = session_manager.last_query_state.generated_sql
            result_count = session_manager.last_query_state.result_count
            column_names = session_manager.last_query_state.selected_columns or []
            logger.debug(f"[STEP 9] ✅ Using previous SQL: {generated_sql} ({result_count} rows)")
    
    # FINAL FALLBACK: If we have SQL but NO result rows, execute the SQL to get the actual data
    # PERFORMANCE OPTIMIZATION: Skip re-execution for fast-path handler calls (no orchestration)
    # The handler already executed the SQL, so rows should be in the response already
    skip_reexecution_for_fastpath = (orchestrator_context is None and not projection_change_bypass)
    
    if generated_sql and not all_result_rows:
        if skip_reexecution_for_fastpath:
            logger.debug(f"\n[STEP 9.1] ⚡ SKIPPING SQL re-execution for fast-path query (handler already executed)")
            logger.debug(f"[STEP 9.1]    orchestrator_context=None, projection_bypass={projection_change_bypass}")
            logger.debug(f"[STEP 9.1]    Rows should be in response.datasets already (if empty, handler returned 0 rows)")
        else:
            logger.debug(f"\n[STEP 9.1] EXECUTING SQL TO FETCH ACTUAL RESULT ROWS (LLM response didn't include them)")
            logger.debug(f"[STEP 9.1] SQL: {generated_sql}")
            try:
                from sqlalchemy import text
                result = await db.execute(text(generated_sql))
                rows = result.fetchall()
                logger.debug(f"[STEP 9.1] ✅ Executed SQL, got {len(rows)} rows from database")
                
                if len(rows) > 0:
                    # Convert SQLAlchemy Row objects to dicts
                    all_result_rows = [dict(row._mapping) for row in rows]
                    result_count = len(all_result_rows)
                    if len(all_result_rows) > 0:
                        sample_row = all_result_rows[0]
                        if isinstance(sample_row, dict):
                            column_names = list(sample_row.keys())
                            logger.debug(f"[STEP 9.1] ✅ Converted rows to dicts with {len(column_names)} columns: {list(sample_row.keys())}")
            except Exception as e:
                logger.debug(f"[STEP 9.1] ⚠️ Failed to execute SQL for row extraction: {e}")
                logger.debug(f"[STEP 9.1]   Continuing with available metadata (embeddings won't include actual data)")
    
    # ======================================================================
    # STEP 9.5: Store query embedding — BACKGROUND (off critical path)
    # ======================================================================
    # Fired as a non-blocking task so embedding generation (~2 s) does not
    # delay the response.  Uses its own DB session (see _bg_store_embedding).
    if generated_sql:
        logger.debug("[RAG-STORE] Scheduling background embedding storage")
        asyncio.create_task(_bg_store_embedding(
            session_id=session_id,
            user_query=user_query,
            generated_sql=generated_sql,
            result_count=result_count,
            column_names=column_names,
            all_result_rows=all_result_rows,
            sample_row=sample_row,
        ))
    else:
        logger.debug("[RAG-STORE] No SQL — skipping embedding")
    
    # ======================================================================
    # STEP 10: Record the tool call
    # ======================================================================
    tool_input = {
        "query": user_query,
        "followup_type": (followup_context.followup_type.value if hasattr(followup_context.followup_type, 'value') else str(followup_context.followup_type)) if is_followup else "NEW",
        "intent_domain": router_decision.tool.value if router_decision else "RUN_SQL",
        "decision_action": router_decision.tool.value if router_decision else "RUN_SQL",
        "state_before": {
            "tables": session_manager.last_query_state.selected_entities if session_manager.last_query_state else [],
            "filters": len(session_manager.last_query_state.filters) if session_manager.last_query_state else 0,
        }
    }
    
    tool_output = {
        "response_type": response_wrapper.response.type if hasattr(response_wrapper.response, 'type') else "standard",
        "success": response_wrapper.success,
        "generated_sql": generated_sql,  # Include SQL for followup detection
        "result_count": result_count,    # Include result count for context
    }
    
    # Record tool call in memory with proper SQL-focused data
    from .session_state_manager import ToolType
    tool_call_record = session_manager.record_tool_call(
        tool_type=ToolType.SQL_QUERY,  # Use proper enum value
        input_json=tool_input,
        output_json=tool_output,
        success=response_wrapper.success,
    )
    
    # ChatGPT-Level: Add SQL conversation entry for follow-up detection
    session_manager.add_sql_conversation_entry(
        user_query=user_query,
        generated_sql=generated_sql,
        result_count=result_count,
        success=response_wrapper.success
    )
    
    # Persist tool call — BACKGROUND (off critical path)
    # Uses its own DB session (see _bg_log_tool_call).
    asyncio.create_task(_bg_log_tool_call(
        session_id=session_id,
        tool_type=tool_call_record.tool_type.value,
        input_json=tool_call_record.input_json,
        output_json=tool_call_record.output_json,
        success=tool_call_record.success,
        error_message=tool_call_record.error_message,
        start_time=tool_call_record.start_time,
        end_time=tool_call_record.end_time,
    ))
    logger.debug("[PERSIST] Tool call scheduled for background logging")
    
    # ======================================================================
    # STEP 10.5: Store structured query plan for follow-up rewriting
    # ======================================================================
    if session_manager.last_query_state and orchestrator_context:
        logger.debug(f"\n[PERSIST] Adding orchestrator context to QueryState for follow-ups")
        
        # Store semantic analysis as structured query plan if available
        if orchestrator_context.get('semantic_analysis'):
            semantic_analysis = orchestrator_context['semantic_analysis']
            
            # Create a structured plan for follow-up rewriting
            structured_plan = {
                "primary_table": getattr(semantic_analysis, 'relevant_tables', [None])[0].table_name if hasattr(semantic_analysis, 'relevant_tables') and semantic_analysis.relevant_tables else None,
                "where_conditions": [],
                "select_clauses": [{"type": "aggregate", "function": "COUNT", "column": "*"}] if "count" in user_query.lower() else [{"type": "wildcard", "column": "*"}],
                "source": "orchestrator_semantic_analysis"
            }
            
            # Extract filter information generically from the generated SQL
            if generated_sql:
                try:
                    import re as _re
                    # Isolate the WHERE clause
                    where_match = _re.search(
                        r'\bWHERE\b(.*?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|$)',
                        generated_sql, _re.IGNORECASE | _re.DOTALL
                    )
                    if where_match:
                        where_clause = where_match.group(1).strip()
                        # Simple equality: col = 'val' or col = val (no schema prefix)
                        for m in _re.finditer(
                            r'\b(?:\w+\.)?(\w+)\s*=\s*(?:\'([^\']*?)\'|"([^"]*?)"|(\d+(?:\.\d+)?))',
                            where_clause
                        ):
                            col = m.group(1).lower()
                            val = m.group(2) if m.group(2) is not None else (m.group(3) if m.group(3) is not None else m.group(4))
                            if col not in ('and', 'or', 'not', 'null', 'true', 'false'):
                                structured_plan["where_conditions"].append({
                                    "column": col,
                                    "operator": "=",
                                    "value": val,
                                })
                        # IN clause: col IN ('v1', 'v2')
                        for m in _re.finditer(
                            r'\b(?:\w+\.)?(\w+)\s+IN\s*\(([^)]+)\)',
                            where_clause, _re.IGNORECASE
                        ):
                            col = m.group(1).lower()
                            vals = [v.strip().strip("'\"") for v in m.group(2).split(",")]
                            structured_plan["where_conditions"].append({
                                "column": col,
                                "operator": "IN",
                                "values": vals,
                            })
                except Exception as e:
                    logger.debug(f"[PERSIST] Error extracting filters from SQL: {e}")
            
            session_manager.last_query_state.query_plan_json = structured_plan
            logger.debug(f"[PERSIST] ✅ Stored structured plan with {len(structured_plan['where_conditions'])} filters")
    
    # ======================================================================
    # STEP 11 + QUESTION-BACK: Concurrent execution
    # ======================================================================
    # session persist (DB write) and question-back (LLM call) are independent —
    # run them together so the slower one doesn't block the faster one.
    # Intent metadata is stamped first (pure dict assignment, no I/O).
    response_wrapper.intent = {
        "domain": router_decision.tool.value if router_decision else "RUN_SQL",
        "confidence": (
            arbiter_decision.confidence if arbiter_decision
            else (router_decision.confidence if router_decision else 0.0)
        ),
        "reasoning": (
            arbiter_decision.reasoning if arbiter_decision
            else (router_decision.reasoning if router_decision else "")
        ),
        "action": (
            arbiter_decision.final_tool.value if arbiter_decision
            else (router_decision.tool.value if router_decision else "RUN_SQL")
        ),
        "arbiter_turn_class": arbiter_decision.final_turn_class.value if arbiter_decision else "unknown",
        "arbiter_followup_subtype": arbiter_decision.final_followup_subtype.value if arbiter_decision else "none",
        "arbiter_merged_state": arbiter_decision.should_merge_state if arbiter_decision else False,
        "execution_policy": (
            execution_policy.path.value
            if "execution_policy" in locals() and execution_policy
            else "unknown"
        ),
        "policy_estimated_duration": (
            execution_policy.estimated_duration_seconds
            if "execution_policy" in locals() and execution_policy
            else None
        ),
    }

    await asyncio.gather(
        _persist_session_and_turn(
            session=session,
            session_manager=session_manager,
            router_decision=router_decision,
            routing_integration=routing_integration,
            db=db,
            response_wrapper=response_wrapper,
            user_query=user_query,
        ),
        _generate_questions(
            user_query=user_query,
            session_manager=session_manager,
            arbiter_decision=arbiter_decision,
            response_wrapper=response_wrapper,
        ),
        return_exceptions=True,
    )

    return response_wrapper


# ============================================================================
# BACKGROUND TASK HELPERS
# These run fire-and-forget after the response is already returned, keeping
# slow I/O (embedding generation, tool call DB writes) off the critical path.
# Each helper creates its own DB session — never share the request session.
# ============================================================================

async def _bg_store_embedding(
    session_id: str,
    user_query: str,
    generated_sql: str,
    result_count: int,
    column_names: List[str],
    all_result_rows: List[Dict[str, Any]],
    sample_row: Optional[Dict[str, Any]],
) -> None:
    """Fire-and-forget: generate and persist a query embedding for RAG retrieval."""
    from ..database import async_session_factory
    try:
        embedding_generator = await get_embedding_generator()
        embedding = await embedding_generator.generate_embedding(
            query=user_query,
            sql=generated_sql,
            result_data=sample_row,
            column_names=column_names,
            result_count=result_count,
        )
        async with async_session_factory() as bg_db:
            embedding_store = await get_embedding_store()
            await embedding_store.store_embedding(
                query_id=str(uuid.uuid4()),
                session_id=session_id,
                user_query=user_query,
                generated_sql=generated_sql,
                result_count=result_count,
                column_names=column_names,
                all_result_rows=all_result_rows,
                embedding=embedding,
                db_session=bg_db,  # Pass explicitly — avoids using stale self.db_session
            )
            await bg_db.commit()
        logger.debug("[BG-EMBED] ✅ Embedding stored successfully")
    except Exception as e:
        logger.debug(f"[BG-EMBED] Error (non-critical): {e}")


async def _bg_log_tool_call(
    session_id: str,
    tool_type: str,
    input_json: Dict[str, Any],
    output_json: Dict[str, Any],
    success: bool,
    error_message: Optional[str],
    start_time: datetime,
    end_time: datetime,
) -> None:
    """Fire-and-forget: write a ToolCall record to the database."""
    from ..database import async_session_factory
    try:
        async with async_session_factory() as bg_db:
            db_tool_call = models.ToolCall(
                session_id=uuid.UUID(session_id) if isinstance(session_id, str) else session_id,
                tool_type=tool_type,
                input_json=input_json,
                output_json=output_json,
                success=success,
                error_message=error_message,
                start_time=start_time,
                end_time=end_time,
            )
            bg_db.add(db_tool_call)
            await bg_db.commit()
        logger.debug("[BG-TOOL-CALL] ✅ Tool call logged")
    except Exception as e:
        logger.debug(f"[BG-TOOL-CALL] Error (non-critical): {e}")


async def _persist_session_and_turn(
    session: models.ChatSession,
    session_manager,
    router_decision,
    routing_integration,
    db: AsyncSession,
    response_wrapper,
    user_query: str,
) -> None:
    """Persist QueryState + semantic turn-state to DB (STEP 11)."""
    if not session_manager.last_query_state:
        await db.commit()
        return

    state_dict = session_manager.to_session_dict()
    state_dict["domain"] = router_decision.tool.value if router_decision else "RUN_SQL"

    session.session_state = state_dict
    session.state_updated_at = datetime.now(timezone.utc)
    session.tool_calls_log = state_dict.get("tool_calls", [])

    try:
        last_state = session_manager.last_query_state

        result_schema = None
        if isinstance(getattr(last_state, "last_result_schema", None), list):
            cols = [c for c in (last_state.last_result_schema or []) if isinstance(c, str) and c]
            if cols:
                result_schema = [{"name": c, "type": ""} for c in cols]

        assistant_text = (
            getattr(response_wrapper.response, "message", "")
            if getattr(response_wrapper, "response", None)
            else ""
        )
        assistant_text = assistant_text.strip() if isinstance(assistant_text, str) else ""
        assistant_summary = assistant_text[:500] if assistant_text else "Executed SQL query"

        await routing_integration.save_turn_state_after_tool(
            session_id=session_manager.session_id,
            user_query=user_query,
            tool_used=schemas.Tool.RUN_SQL,
            assistant_summary=assistant_summary,
            artifacts={
                "tool_used": schemas.Tool.RUN_SQL.value,
                "sql": getattr(last_state, "generated_sql", None),
                "tables": getattr(last_state, "selected_entities", None),
                "filters": [f.to_dict() for f in (getattr(last_state, "filters", None) or [])],
                "row_count": getattr(last_state, "result_count", None),
                "result_schema": result_schema,
            },
            confidence=router_decision.confidence if router_decision else 0.8,
        )
    except Exception as e:
        logger.warning(f"[PERSIST] Failed to save turn state: {e}")

    try:
        from .session_archival import archive_if_needed
        await archive_if_needed(db, session)
    except Exception as _arc_err:
        logger.debug("[PERSIST] Archival skipped: %s", _arc_err)

    await db.commit()
    logger.debug("[PERSIST] ✅ Session state committed")


async def _generate_questions(
    user_query: str,
    session_manager,
    arbiter_decision,
    response_wrapper,
) -> None:
    """Generate contextual follow-up suggestions (question-back engine)."""
    try:
        from .question_back_engine import get_question_back_engine
        question_engine = get_question_back_engine()

        query_context_dict = {
            "user_query": user_query,
            "intent": arbiter_decision.final_tool.value if arbiter_decision else "list",
            "entity": (
                session_manager.last_query_state.selected_entities[0]
                if session_manager.last_query_state
                and session_manager.last_query_state.selected_entities
                else "data"
            ),
            "has_filters": bool(
                session_manager.last_query_state and session_manager.last_query_state.filters
            ),
            "sql": (
                session_manager.last_query_state.generated_sql
                if session_manager.last_query_state
                else ""
            ),
        }

        result_context_dict = {
            "row_count_returned": (
                session_manager.last_query_state.result_count
                if session_manager.last_query_state
                else 0
            ),
            "total_count_estimate": (
                session_manager.last_query_state.result_count
                if session_manager.last_query_state
                else 0
            ),
            "limit_applied": (
                "LIMIT"
                in (session_manager.last_query_state.generated_sql or "").upper()
                if session_manager.last_query_state
                else False
            ),
            "columns": (
                session_manager.last_query_state.selected_columns
                if session_manager.last_query_state
                else []
            ),
            "has_numeric_columns": True,
            "visualization_generated": bool(
                response_wrapper.response
                and hasattr(response_wrapper.response, "visualizations")
                and response_wrapper.response.visualizations
            ),
        }

        question_result = await question_engine.generate_questions(
            query_context=query_context_dict,
            result_context=result_context_dict,
            schema_context={"filterable_columns": []},
        )

        logger.debug(
            f"[QUESTION-BACK] ✅ Generated {len(question_result.questions)} questions, "
            f"{len(question_result.actions)} actions"
        )

        if response_wrapper.response and hasattr(response_wrapper.response, "suggested_questions"):
            response_wrapper.response.suggested_questions = question_result.questions
            response_wrapper.response.suggested_actions = [
                {
                    "type": action.action_type,
                    "label": action.label,
                    "description": action.description,
                    "parameters": action.parameters,
                }
                for action in question_result.actions
            ]
    except Exception as e:
        logger.debug(f"[QUESTION-BACK] Error (non-critical): {e}")


async def _check_files_in_session(db: AsyncSession, session_id: str) -> bool:
    """
    Check if any files were uploaded in this session (ChatGPT-like follow-up detection).
    
    Enables routing of follow-up questions about previously analyzed files
    to the file lookup handler instead of pure chat.
    
    Args:
        db: Database session
        session_id: Chat session ID
        
    Returns:
        bool: True if files exist in session, False otherwise
    """
    if not db or not session_id:
        return False
    
    try:
        result = await db.execute(
            select(models.UploadedFile).where(
                models.UploadedFile.session_id == session_id
            ).limit(1)
        )
        has_files = result.scalars().first() is not None
        return has_files
    except Exception as e:
        logger.debug(f"[SESSION] Error checking files in session: {e}")
        return False
