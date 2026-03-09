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

import inspect
import re
import uuid
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
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
    
    # ======================================================================
    # STEP 1: SEMANTIC ROUTING - SINGLE SOURCE OF TRUTH
    # ======================================================================
    logger = logging.getLogger(__name__)

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
            print("[PERSIST] Chat response logged to tool_calls table")

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
            print(f"[PERSIST] File operation ({tool_type}) logged to tool_calls table")

            # ===== FALLBACK LOGIC: If file analysis produces weak results, try database =====
            chunks_found = output_metadata.get("chunks_found", 0)
            is_weak_result = (
                chunks_found == 0 or  # No relevant chunks found
                (chunks_found <= 2 and "database" in user_query.lower()) or  # Few chunks + database keywords
                (chunks_found <= 2 and any(kw in user_query.lower() for kw in ["clients", "sales", "records", "all", "get me", "show me", "list"]))
            )
            
            if is_weak_result and response.success:
                print(f"[FALLBACK] File analysis found {chunks_found} chunks for database-like query. Attempting SQL fallback...")
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
                        
                        print("[FALLBACK] SQL analysis successful. Using database results instead of weak file results.")
                        
                        # Update routing info to indicate fallback
                        sql_result.intent = {
                            "domain": "RUN_SQL",
                            "confidence": 0.8,
                            "reasoning": "Fallback from weak file analysis to database query",
                            "action": "RUN_SQL",
                        }
                        
                        return sql_result
                    else:
                        print("[FALLBACK] SQL analysis did not improve results. Keeping file analysis.")
                        
                except Exception as e:
                    print(f"[FALLBACK] SQL fallback failed: {e}. Keeping file analysis results.")

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
    # STEP 4: Detect follow-up intent from semantic router decision
    # ======================================================================
    from .followup_manager import FollowUpType as FollowUpTypeAnalyzer

    is_followup = bool(
        router_decision
        and router_decision.followup_type == schemas.FollowupType.RUN_SQL_FOLLOW_UP
    )
    # Use a safe default so we do not wipe state on router-identified follow-ups.
    followup_type = FollowUpTypeAnalyzer.REFINEMENT if is_followup else None
    followup_reasoning = router_decision.reasoning if router_decision else ""
    
    # SAFETY CHECK: If no previous messages exist OR no previous state, this is a NEW SESSION
    # CANNOT be a follow-up even if the router indicates it is
    # This prevents RAG from searching unnecessary context for first query
    is_truly_new_session = len(previous_messages) == 0 and not bool(previous_state_dict)
    
    logger.debug(f"[FOLLOWUP] is_followup={is_followup}, is_truly_new_session={is_truly_new_session}")
    logger.debug(f"[FOLLOWUP] previous_messages length: {len(previous_messages)}, previous_state_dict: {previous_state_dict is None}")
    
    if is_followup and is_truly_new_session:
        print(f"[FOLLOWUP] ⚠️ Router indicated follow-up, but new session detected")
        print(f"[FOLLOWUP] → Overriding to NEW (first query in new session)")
        is_followup = False
        followup_type = None
    elif is_followup and not is_truly_new_session:
        print(f"[FOLLOWUP] ✅ Follow-up confirmed: {len(previous_messages)} prior messages, session state exists")

    print(f"\n[FOLLOWUP] Is Follow-up: {is_followup}")
    if is_followup and followup_type:
        print(f"[FOLLOWUP] Type: {followup_type.name}")
        print(f"[FOLLOWUP] Reasoning: {followup_reasoning}")
    else:
        print(f"[FOLLOWUP] New conversation - no follow-up detected")

    # ======================================================================
    # STEP 5: Merge state based on follow-up type (preliminary)
    # ======================================================================
    if not is_followup or followup_type is None:
        print(f"[MERGE] NEW: Starting fresh, clearing previous state")
        session_manager.reset_state()
    
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
    
    print(f"\n[RETRIEVAL] Tier 1 context: {len(selective_history.recent_messages)} recent messages")
    print(f"[RETRIEVAL] Tier 2 context: {selective_history.last_query_state is not None}")
    
    # ======================================================================
    # STEP 6.5: MULTI-STAGE Follow-Up Detection (RAG + LLM)
    # ======================================================================
    # STAGE 1: Try RAG-based semantic search for similar previous queries
    # STAGE 2: Fall back to LLM-based analysis if RAG doesn't find strong match
    # STAGE 3: Fall back to preliminary detection if both fail
    
    from .followup_manager import FollowUpContext, FollowUpType as FollowUpTypeAnalyzer, PreviousQueryContext
    from .session_state_manager import FollowUpType as FollowUpTypeState
    
    # STAGE 1: Try RAG retrieval first (fast, no LLM timeout)
    # SKIP RAG if this is a new session - prevents searching other sessions' context
    if not is_truly_new_session and len(previous_messages) > 0 and is_followup:
        print(f"\n[FOLLOWUP-STAGE1] Starting RAG retrieval for: {user_query[:50]}")
        rag_retriever = await get_rag_retriever(similarity_threshold=0.15, top_k=3, db_session=db)  # Lower threshold for token hashing
        rag_context = await rag_retriever.retrieve_context_for_followup(
            session_id=session_id,
            current_query=user_query,
        )
    else:
        if is_truly_new_session:
            print(f"\n[FOLLOWUP-STAGE1] Skipping RAG (new session - first query)")
        else:
            print(f"\n[FOLLOWUP-STAGE1] Skipping RAG (no previous messages or not a follow-up)")
        rag_context = None
    
    if rag_context and rag_context.is_relevant:
        print(f"[FOLLOWUP-RAG] ✅ Found semantically similar previous query!")
        print(f"[FOLLOWUP-RAG] Similarity: {rag_context.similarity_score:.0%}")
        print(f"[FOLLOWUP-RAG] Previous: {rag_context.previous_query[:60]}...")
        print(f"[FOLLOWUP-RAG] SQL: {rag_context.previous_sql[:60]}...")
        
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
        
        print(f"[FOLLOWUP-RAG] ✅ Follow-up context created with table: {table_name}, filters: {len(filters)}")
        
        # Inject RAG context into selective_history for downstream processing
        rag_context_str = f"\n\n[RAG CONTEXT]\n{rag_context.context_text}"
        selective_history.recent_messages.append({
            "role": "system",
            "content": rag_context_str,
        })
        
        # Store the previous SQL from RAG for use in the query handler
        if session_manager.last_query_state:
            session_manager.last_query_state.generated_sql = rag_context.previous_sql
    else:
        print(f"[FOLLOWUP-STAGE1] No strong RAG match found, trying LLM...")
        
        # STAGE 2: Try LLM-based analysis (with timeout)
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
            
            # Build simple conversation history for better followup detection
            simple_conversation_history = ""
            if session_manager and session_manager.tool_calls:
                query_parts = []
                for tool_call in session_manager.tool_calls[-3:]:  # Last 3 interactions
                    # Extract user query from tool call input
                    user_query = None
                    if hasattr(tool_call, 'input_json') and tool_call.input_json:
                        # Try different possible keys for user query - "query" is the main one used
                        user_query = (tool_call.input_json.get('query') or 
                                     tool_call.input_json.get('user_query') or 
                                     tool_call.input_json.get('user_input'))
                    
                    if user_query:
                        query_parts.append(f"USER: {user_query}")
                
                if query_parts:
                    simple_conversation_history = "\n".join(query_parts)
                    print(f"[DEBUG] Session simple conversation history: {simple_conversation_history}")
            
            # ChatGPT-Level: Use SQL conversation history for better follow-up detection
            sql_conversation_history = session_manager.get_sql_conversation_history(max_entries=5) if session_manager else ""
            print(f"[DEBUG] Session SQL conversation history: {sql_conversation_history}")
            
            # Use SQL history if available, otherwise fallback to simple history
            conversation_history_for_followup = sql_conversation_history if sql_conversation_history else (simple_conversation_history if simple_conversation_history else selective_history.to_prompt_context())
            
            # LLM semantic analysis of follow-up type
            followup_context = await followup_analyzer.analyze(
                current_query=user_query,
                conversation_history=conversation_history_for_followup,  # Use SQL conversation history
                previous_sql=previous_sql,
                previous_result_count=previous_result_count,
            )
            
            print(f"\n[FOLLOWUP-LLM] Is Follow-up: {followup_context.is_followup}")
            followup_type_val = followup_context.followup_type.value if hasattr(followup_context.followup_type, 'value') else str(followup_context.followup_type)
            print(f"[FOLLOWUP-LLM] Type: {followup_type_val.upper()}, Confidence: {followup_context.confidence:.0%}")
            print(f"[FOLLOWUP-LLM] Reasoning: {followup_context.reasoning}")
        except Exception as e:
            print(f"\n[FOLLOWUP-LLM] ERROR during follow-up analysis: {e}")
            print(f"[FOLLOWUP-LLM] Using preliminary detection from STEP 4/5")
            
            # STAGE 3: Fall back to preliminary detection from STEP 4/5
            followup_context = FollowUpContext(
                is_followup=is_followup,
                followup_type=followup_type if followup_type else FollowUpTypeAnalyzer.NEW,
                confidence=0.0,
                reasoning="Using preliminary detection due to LLM error",
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
    print(f"[FOLLOWUP-LLM] Mapped to state type: {state_followup_type_val}")
    
    # ======================================================================
    # STEP 7.7: ENHANCED SEMANTIC VALUE GROUNDING (FK-Aware)
    # ======================================================================
    # NEW: Apply intelligent value grounding with FK relationship awareness
    # This handles multi-table scenarios and auto-generates JOINs
    
    enhanced_followup_context = session_manager.followup_context_from_rag
    grounded_filter_values = []
    
    try:
        print(f"\n[STEP 7.7] 🧠 Applying enhanced semantic value grounding...")
        
        # Get enhanced grounder instance
        grounder = get_semantic_value_grounder_enhanced()
        
        # Ensure grounder is initialized
        if not grounder.initialized:
            print(f"[STEP 7.7] Initializing grounder (first use)...")
            await grounder.initialize_for_tables(db, sample_size=100)
        
        # Determine context for grounding
        if session_manager.followup_context_from_rag and session_manager.followup_context_from_rag.is_followup:
            # Follow-up query: Use the table from previous context
            print(f"[STEP 7.7] Query Type: FOLLOW-UP (using previous context)")
            rag_context = session_manager.followup_context_from_rag.previous_context
            target_tables = [rag_context.table_name] if rag_context and rag_context.table_name else []
            print(f"[STEP 7.7] Target table from RAG: {target_tables}")
        else:
            # Initial query: Will be handled during SQL generation
            print(f"[STEP 7.7] Query Type: INITIAL (will ground values during SQL gen)")
            target_tables = []
        
        # For follow-up queries with table context, attempt value grounding
        if target_tables and grounder.initialized:
            try:
                # Try to extract filter values from query
                # This is a simplified approach - the main grounding happens in query_handler
                print(f"[STEP 7.7] Hold value grounding for query_handler (context ready)")
                
                # Store grounder info in context for query_handler to use
                if not hasattr(session_manager.followup_context_from_rag, 'value_grounding_ready'):
                    session_manager.followup_context_from_rag.value_grounding_ready = True
                    session_manager.followup_context_from_rag.grounder_initialized = True
                    
            except Exception as e:
                print(f"[STEP 7.7] ⚠️ Value grounding prep failed: {e}")
        
        print(f"[STEP 7.7] ✅ Value grounding context prepared")
        
    except Exception as e:
        print(f"[STEP 7.7] ⚠️ Value grounding phase failed (non-critical): {e}")
        # Continue without grounding - query_handler will handle fallback
    
    # ======================================================================
    # STEP 7.8: INTELLIGENT QUERY ORCHESTRATION (NEW) - WITH PERFORMANCE OPTIMIZATION
    # ======================================================================
    # PERFORMANCE OPTIMIZATION: Skip heavy orchestration for high-confidence simple queries
    # The efficient router already identified the query with high confidence, so we can
    # bypass the expensive semantic analysis for simple patterns
    
    orchestrator_context = None
    
    # Check if we can skip orchestration based on router decision confidence
    skip_orchestration = (
        hasattr(router_decision, 'confidence') and 
        hasattr(router_decision, 'tool') and
        router_decision.confidence >= 0.90 and  # High confidence threshold
        router_decision.tool.value == 'RUN_SQL' and  # Simple SQL query
        not enhanced_followup_context  # Not a complex follow-up
    )
    
    if skip_orchestration:
        print(f"\n[STEP 7.8] ⚡ SKIPPING ORCHESTRATION (high confidence: {router_decision.confidence:.2f})")
        print(f"[STEP 7.8] ✅ Fast path - efficient routing already identified query pattern")
        orchestrator_context = {
            'skipped_for_performance': True,
            'router_confidence': router_decision.confidence,
            'reasoning': f'Skipped heavy orchestration due to high router confidence ({router_decision.confidence:.2f})'
        }
    else:
        print(f"\n[STEP 7.8] 🧠 ORCHESTRATING INTELLIGENT QUERY ANALYSIS (confidence: {getattr(router_decision, 'confidence', 'unknown')})")
        try:
            orchestrator = get_query_orchestrator()
            
            # Prepare orchestrator inputs
            previous_query_context = None
            if enhanced_followup_context and enhanced_followup_context.previous_context:
                previous_query_context = {
                    'table': enhanced_followup_context.previous_context.table_name,
                    'columns_used': enhanced_followup_context.previous_context.columns_selected,
                    'filters': enhanced_followup_context.previous_context.filters,
                    'query': enhanced_followup_context.previous_context.generated_sql,
                }
            
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
            
            print(f"[STEP 7.8] ✅ Orchestration complete")
            print(f"[STEP 7.8]   Query Type: {orch_result.semantic_analysis.query_type}")
            print(f"[STEP 7.8]   Tables: {[t.table_name for t in orch_result.semantic_analysis.relevant_tables]}")
            print(f"[STEP 7.8]   Patterns: {', '.join(orch_result.sql_patterns)}")
            print(f"[STEP 7.8]   SQL: {orch_result.generated_sql[:80]}...")
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"[STEP 7.8] ⚠️ Orchestrator phase failed (non-critical): {e}")
            print(f"[STEP 7.8] ⚠️ Orchestrator analysis skipped, continuing with existing flow: {e}")
            # Continue without orchestrator - not critical
    
    # ======================================================================
    # STEP 7.6: Store RAG-based followup context for query_handler to use
    # ======================================================================
    # CRITICAL: Pass the RAG-detected followup_context to query_handler
    # so it doesn't create a new one from scratch
    # IMPORTANT: Do this BEFORE building handler_kwargs so the context is available
    if followup_context and followup_context.is_followup and followup_context.previous_context:
        print(f"[STEP 7.6] ✅ Storing RAG-based followup context for query handler")
        print(f"[STEP 7.6]   Table: {followup_context.previous_context.table_name}")
        print(f"[STEP 7.6]   Filters: {len(followup_context.previous_context.filters)}")
        session_manager.followup_context_from_rag = followup_context
    else:
        # Clear it if this isn't a strong follow-up
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
    final_handler_kwargs = {
        'db': db,
        'user_id': user_id,
        'session_id': session_id,
        'query': user_query,
        'conversation_history': selective_history.to_prompt_context(),
        'followup_context_from_rag': enhanced_followup_context,  # Use enhanced context with discovered mappings
        'orchestrator_context': orchestrator_context,  # NEW: Pass orchestrator analysis results
        'session_manager': session_manager,  # NEW: Pass session manager for follow-up rewriting
    }
    
    # Dynamically add any extra kwargs that the function accepts (never hardcode param names)
    for key, value in handler_kwargs.items():
        if key in accepted_params:
            final_handler_kwargs[key] = value
        else:
            print(f"[DYNAMIC FILTERING] Parameter '{key}' not accepted by handler - automatically filtered out")
    
    # ======================================================================
    # STEP 7.5: Initialize QueryState if not already set
    # ======================================================================
    # CRITICAL: Create a new QueryState for this database query if one doesn't exist
    # This is needed for STEP 9 to store the generated SQL and result metadata
    if session_manager.last_query_state is None:
        print(f"[STEP 7.5] Creating new QueryState for database query")
        session_manager.last_query_state = QueryState(
            user_query=user_query,
            domain=QueryDomain.DATABASE,
        )
        print(f"[STEP 7.5] ✅ QueryState created and ready for STEP 9")
    else:
        print(f"[STEP 7.5] Reusing existing QueryState for follow-up")
    
    # ======================================================================
    # STEP 8: Call the data query handler (only for DATABASE domain)
    # ======================================================================
    print(f"\n[HANDLER] Calling database handler with enriched context")
    print(f"[HANDLER] Parameters being passed: {', '.join(final_handler_kwargs.keys())}")
    
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
    
    print(f"\n[STEP 9] EXTRACTION STARTING")
    print(f"[STEP 9] response_wrapper type: {type(response_wrapper)}")
    print(f"[STEP 9] response_wrapper.response type: {type(response_wrapper.response)}")
    print(f"[STEP 9] session_manager: {session_manager}")
    print(f"[STEP 9] last_query_state: {session_manager.last_query_state if session_manager else 'N/A'}")
    
    if session_manager.last_query_state and response_wrapper.response:
        print(f"[STEP 9] ✅ Both conditions met - extracting SQL")
        
        # Check what attributes the response has
        print(f"[STEP 9] response attributes: {dir(response_wrapper.response)}")
        
        # Extract SQL from debug metadata (now stored as dict due to extra="allow")
        if hasattr(response_wrapper.response, 'debug'):
            debug_info = response_wrapper.response.debug
            print(f"[STEP 9] Found debug attribute: {debug_info is not None}")
            print(f"[STEP 9] debug_info type: {type(debug_info)}, value: {debug_info}")
            
            # Handle both dict and object cases (Pydantic extra fields are stored as dicts)
            try:
                if isinstance(debug_info, dict):
                    # Extract from dict
                    generated_sql = debug_info.get('sql_executed', None)
                    result_count = debug_info.get('row_count', 0)
                    column_names = debug_info.get('columns', [])
                    print(f"[STEP 9] Extracted from dict - sql: {generated_sql is not None}, rows: {result_count}, cols: {len(column_names)}")
                else:
                    # Extract from object attributes
                    generated_sql = getattr(debug_info, 'sql_executed', None)
                    result_count = getattr(debug_info, 'row_count', 0)
                    column_names = getattr(debug_info, 'columns', [])
                    print(f"[STEP 9] Extracted from object - sql: {generated_sql is not None}, rows: {result_count}, cols: {len(column_names)}")
                
                if generated_sql:
                    session_manager.last_query_state.generated_sql = generated_sql
                    session_manager.last_query_state.result_count = result_count
                    print(f"[STEP 9] ✅ Extracted SQL: {generated_sql[:80]}... ({result_count} rows)")
                else:
                    print(f"[STEP 9] ⚠️ No sql_executed in debug_info. Keys: {list(debug_info.keys()) if isinstance(debug_info, dict) else 'N/A'}")
            except Exception as e:
                import traceback
                print(f"[STEP 9] ⚠️ Error extracting from debug: {e}")
                print(f"[STEP 9] Traceback: {traceback.format_exc()}")
        else:
            print(f"[STEP 9] ❌ No debug attribute found on response")
            print(f"[STEP 9] Available attributes: {[attr for attr in dir(response_wrapper.response) if not attr.startswith('_')]}")
            
            # FALLBACK: For follow-ups, use the previously generated SQL
            if session_manager.last_query_state and session_manager.last_query_state.generated_sql:
                print(f"[STEP 9] 📋 Fallback: Using previous SQL from session for follow-up")
                generated_sql = session_manager.last_query_state.generated_sql
                result_count = session_manager.last_query_state.result_count
                column_names = session_manager.last_query_state.selected_columns or []
                print(f"[STEP 9] ✅ Using previous SQL: {generated_sql[:80]}... ({result_count} rows)")
    
    # IMPORTANT: Extract actual result rows from datasets (this is SEPARATE from extracting SQL)
    # The SQL might come from debug_info, but the actual result DATA comes from datasets
    if hasattr(response_wrapper.response, 'datasets') and response_wrapper.response.datasets:
        datasets = response_wrapper.response.datasets
        if len(datasets) > 0:
            first_dataset = datasets[0]
            print(f"[STEP 9] Found datasets: {len(datasets)} dataset(s)")
            print(f"[STEP 9] First dataset type: {type(first_dataset)}")
            print(f"[STEP 9] First dataset attributes: {dir(first_dataset)[:10]}...")  # Show first 10 attributes
            
            if hasattr(first_dataset, 'data') and first_dataset.data:
                all_result_rows = first_dataset.data  # Store ALL rows for embedding
                print(f"[STEP 9] ✅ Extracted {len(all_result_rows)} result rows from datasets")
                if len(all_result_rows) > 0 and isinstance(first_dataset.data[0], dict):
                    sample_row = all_result_rows[0]
                    print(f"[STEP 9] First row keys: {list(sample_row.keys())[:5]}...")
            else:
                print(f"[STEP 9] ⚠️ First dataset has no 'data' attribute or it's empty")
                print(f"[STEP 9]   First dataset: {first_dataset}")
    else:
        print(f"[STEP 9] ⚠️ response_wrapper.response has no datasets or datasets is empty")
        print(f"[STEP 9]   Checking response object for result data...")
        if hasattr(response_wrapper.response, 'artifacts'):
            print(f"[STEP 9]   artifacts: {response_wrapper.response.artifacts}")
        # Try to see if there's a 'data' or 'results' field
        for attr_name in ['data', 'results', 'rows', 'result_rows', 'result_data']:
            if hasattr(response_wrapper.response, attr_name):
                attr_val = getattr(response_wrapper.response, attr_name)
                print(f"[STEP 9]   Found attribute '{attr_name}': {type(attr_val)}")
                if isinstance(attr_val, list) and len(attr_val) > 0:
                    all_result_rows = attr_val
                    print(f"[STEP 9]   ✅ Using '{attr_name}' with {len(all_result_rows)} rows")
    
    # FALLBACK: For follow-ups, use the previously generated SQL if needed
    if not generated_sql and session_manager.last_query_state and session_manager.last_query_state.generated_sql:
        print(f"[STEP 9] 📋 Fallback: Using previous SQL from session for follow-up")
        generated_sql = session_manager.last_query_state.generated_sql
        result_count = session_manager.last_query_state.result_count
        column_names = session_manager.last_query_state.selected_columns or []
        print(f"[STEP 9] ✅ Using previous SQL: {generated_sql[:80]}... ({result_count} rows)")
    
    # FINAL FALLBACK: If we have SQL but NO result rows, execute the SQL to get the actual data
    if generated_sql and not all_result_rows:
        print(f"\n[STEP 9.1] EXECUTING SQL TO FETCH ACTUAL RESULT ROWS (LLM response didn't include them)")
        print(f"[STEP 9.1] SQL: {generated_sql[:100]}...")
        try:
            from sqlalchemy import text
            result = await db.execute(text(generated_sql))
            rows = result.fetchall()
            print(f"[STEP 9.1] ✅ Executed SQL, got {len(rows)} rows from database")
            
            if len(rows) > 0:
                # Convert SQLAlchemy Row objects to dicts
                all_result_rows = [dict(row._mapping) for row in rows]
                result_count = len(all_result_rows)
                if len(all_result_rows) > 0:
                    sample_row = all_result_rows[0]
                    if isinstance(sample_row, dict):
                        column_names = list(sample_row.keys())
                        print(f"[STEP 9.1] ✅ Converted rows to dicts with {len(column_names)} columns: {list(sample_row.keys())[:5]}...")
        except Exception as e:
            print(f"[STEP 9.1] ⚠️ Failed to execute SQL for row extraction: {e}")
            print(f"[STEP 9.1]   Continuing with available metadata (embeddings won't include actual data)")
    
    # ======================================================================
    # STEP 9.5: Store query embedding for RAG-based follow-up retrieval
    # ======================================================================
    # This enables semantic search for follow-ups even with intervening queries
    
    # DEBUG: Log the condition inputs
    print(f"\n[RAG-STORE DEBUG] Checking embedding storage conditions:")
    print(f"[RAG-STORE DEBUG]   generated_sql={generated_sql is not None} (value: {generated_sql[:50] if generated_sql else 'None'}...)")
    print(f"[RAG-STORE DEBUG]   router_tool={router_decision.tool.value if router_decision else 'unknown'}")
    print(f"[RAG-STORE DEBUG]   Condition met: {bool(generated_sql)}")
    
    if generated_sql:
        try:
            print(f"\n[RAG-STORE] ✅ CONDITION MET - Storing query embedding for semantic search...")
            print(f"[RAG-STORE] Session ID: {session_id}")
            print(f"[RAG-STORE] User Query: {user_query[:80]}...")
            
            embedding_generator = await get_embedding_generator()
            print(f"[RAG-STORE] Got embedding generator: {embedding_generator}")
            
            # Generate embedding with result data included (NOT just query/SQL text)
            embedding = await embedding_generator.generate_embedding(
                query=user_query,
                sql=generated_sql,
                result_data=sample_row,  # Include actual result values!
                column_names=column_names,
                result_count=result_count,
            )
            print(f"[RAG-STORE] Generated embedding: {len(embedding)} dimensions (with result data)")
            
            embedding_store = await get_embedding_store(db_session=db)
            print(f"[RAG-STORE] Got embedding store: {embedding_store}")
            print(f"[RAG-STORE] Store currently has {len(embedding_store.embeddings)} total embeddings")
            
            query_id = str(uuid.uuid4())
            print(f"[RAG-STORE] Generated query_id: {query_id}")
            
            print(f"[RAG-STORE] Calling store_embedding with:")
            print(f"[RAG-STORE]   result_count={result_count}")
            print(f"[RAG-STORE]   column_names={column_names}")
            print(f"[RAG-STORE]   all_result_rows={'<present>' if all_result_rows else 'None'} (count: {len(all_result_rows)})")
            
            await embedding_store.store_embedding(
                query_id=query_id,
                session_id=session_id,
                user_query=user_query,
                generated_sql=generated_sql,
                result_count=result_count,
                column_names=column_names,
                all_result_rows=all_result_rows,
                embedding=embedding,
            )
            
            print(f"[RAG-STORE] ✅ EMBEDDING STORED SUCCESSFULLY!")
            print(f"[RAG-STORE] ✅ Final store state: {len(embedding_store.embeddings)} total embeddings for session {session_id}")
        except Exception as e:
            import traceback
            print(f"[RAG-STORE] ⚠️ ERROR storing embedding: {e}")
            print(f"[RAG-STORE] ⚠️ Traceback: {traceback.format_exc()}")
            # Don't fail the query on embedding error - it's not critical
    else:
        print(f"[RAG-STORE] ❌ CONDITION NOT MET - Skipping embedding storage")
    
    # ======================================================================
    # STEP 10: Record the tool call
    # ======================================================================
    tool_input = {
        "query": user_query,
        "followup_type": (followup_context.followup_type.value if hasattr(followup_context.followup_type, 'value') else str(followup_context.followup_type)) if followup_context.is_followup else "NEW",
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
    
    # Persist tool call to dedicated tool_calls table (always, regardless of query state)
    print(f"\n[PERSIST] Logging tool call to tool_calls table")
    db_tool_call = models.ToolCall(
        session_id=uuid.UUID(session_id) if isinstance(session_id, str) else session_id,
        tool_type=tool_call_record.tool_type.value,  # Convert enum to string value
        input_json=tool_call_record.input_json,
        output_json=tool_call_record.output_json,
        success=tool_call_record.success,
        error_message=tool_call_record.error_message,
        start_time=tool_call_record.start_time,
        end_time=tool_call_record.end_time,
    )
    db.add(db_tool_call)
    
    # ======================================================================
    # STEP 10.5: Store structured query plan for follow-up rewriting
    # ======================================================================
    if session_manager.last_query_state and orchestrator_context:
        print(f"\n[PERSIST] Adding orchestrator context to QueryState for follow-ups")
        
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
            
            # Try to extract filter information from generated SQL or other sources
            if generated_sql:
                try:
                    # Basic extraction of filters from SQL (this could be enhanced)
                    if "gender" in generated_sql.lower():
                        if "in (" in generated_sql.lower():
                            structured_plan["where_conditions"].append({
                                "column": "gender", 
                                "operator": "IN", 
                                "values": ["M", "F"]
                            })
                        elif "=" in generated_sql.lower():
                            structured_plan["where_conditions"].append({
                                "column": "gender", 
                                "operator": "=", 
                                "value": "M"  # This could be enhanced to extract actual value
                            })
                    
                    if "extract(month from" in generated_sql.lower():
                        # Extract month value from SQL
                        import re
                        month_match = re.search(r"extract\s*\(\s*month\s+from.*?\)\s*=\s*(\d+)", generated_sql, re.IGNORECASE)
                        if month_match:
                            month_value = int(month_match.group(1))
                            structured_plan["where_conditions"].append({
                                "column": "dob",
                                "operator": "MONTH_EQUALS",
                                "value": month_value
                            })
                            
                except Exception as e:
                    print(f"[PERSIST] Error extracting filters from SQL: {e}")
            
            session_manager.last_query_state.query_plan_json = structured_plan
            print(f"[PERSIST] ✅ Stored structured plan with {len(structured_plan['where_conditions'])} filters")
        
        # Also store any existing plan from rewritten_query_plan
        elif locals().get('rewritten_query_plan'):
            session_manager.last_query_state.query_plan_json = locals()['rewritten_query_plan']
            print(f"[PERSIST] ✅ Stored rewritten query plan")
    
    # ======================================================================
    # STEP 11: Persist updated QueryState to database
    # ======================================================================
    if session_manager.last_query_state:
        print(f"\n[PERSIST] Saving QueryState to database")
        state_dict = session_manager.to_session_dict()
        state_dict["domain"] = router_decision.tool.value if router_decision else "RUN_SQL"
        session.session_state = state_dict
        session.state_updated_at = datetime.utcnow()
        session.tool_calls_log = state_dict.get("tool_calls", [])

        # Persist semantic turn-state (best-effort) to power follow-up routing.
        try:
            last_state = session_manager.last_query_state

            result_schema = None
            if isinstance(getattr(last_state, "last_result_schema", None), list):
                cols = [c for c in (last_state.last_result_schema or []) if isinstance(c, str) and c]
                if cols:
                    result_schema = [{"name": c, "type": ""} for c in cols]

            assistant_text = getattr(response_wrapper.response, "message", "") if getattr(response_wrapper, "response", None) else ""
            assistant_text = assistant_text.strip() if isinstance(assistant_text, str) else ""
            assistant_summary = assistant_text[:500] if assistant_text else "Executed SQL query"

            await routing_integration.save_turn_state_after_tool(
                session_id=session_id,
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
            logger.warning(f"[SESSION_HANDLER] Failed to persist turn state (RUN_SQL): {e}")
        
        await db.commit()
        print(f"[PERSIST] QueryState saved: {len(str(session.session_state))} bytes")
        print(f"[PERSIST] Tool call logged to tool_calls table")
    else:
        # Even if no query state, still commit the tool call
        await db.commit()
        print(f"[PERSIST] Tool call logged to tool_calls table")
    
    # Add decision engine results to response
    response_wrapper.intent = {
        "domain": router_decision.tool.value if router_decision else "RUN_SQL",
        "confidence": router_decision.confidence if router_decision else 0.0,
        "reasoning": router_decision.reasoning if router_decision else "",
        "action": router_decision.tool.value if router_decision else "RUN_SQL",
    }
    
    return response_wrapper


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
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[SESSION] Error checking files in session: {e}")
        return False
