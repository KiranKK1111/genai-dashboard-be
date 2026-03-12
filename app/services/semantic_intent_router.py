"""
Semantic Intent Router - Dynamic tool routing without keyword hacks.

This service implements a fully semantic routing system:

1. Hard Signals: Deterministic checks (fast, safe)
   - has_uploaded_files
   - db_connected
   - last_tool_used
   - time_since_last_turn

2. LLM Router: Structured JSON decision
   - Classification: CHAT | ANALYZE_FILE | RUN_SQL | MIXED
   - Follow-up: NEW_QUERY | {TOOL}_FOLLOW_UP
   - Follow-up subtypes (e.g., ADD_FILTER, CHANGE_GROUPING)
   - Confidence + Clarification gating

3. Deterministic Corrections: Safety checks
   - If ANALYZE_FILE but no files → clarification needed
   - If RUN_SQL_FOLLOW_UP but no last SQL → treat as NEW_QUERY
   - If confidence < 0.60 → ask clarifying question

NO keyword matching. Pure learned routing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from .. import llm, models, schemas
from ..config import settings

logger = logging.getLogger(__name__)


class SemanticIntentRouter:
    """
    Semantic routing signal generator.
    
    NOTE: This router produces PROVISIONAL SIGNALS for the DecisionArbiter.
    It does NOT make final routing decisions - that authority belongs to the arbiter.
    
    Takes: user_query + session_state
    Returns: RouterDecision (provisional tool, followup_type, confidence, clarification_questions)
    
    The returned decision should be treated as a SIGNAL, not a final decision.
    DecisionArbiter combines this signal with other signals to make the final call.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        # Use the configured model by default to avoid hardcoding model names.
        self.router_model = settings.ai_factory_model
        self.temperature = 0.1  # Low temperature for deterministic decisions
        
    async def route_turn(
        self,
        user_query: str,
        session_id: str,
        user_id: str,
        current_request_has_files: bool = False,
    ) -> schemas.RouterDecision:
        """
        Main routing function. Route a single turn to the appropriate tool.
        
        Args:
            user_query: The user's question/request
            session_id: Session ID
            user_id: User ID
            
        Returns:
            RouterDecision with tool, followup_type, confidence, clarification_questions
        """
        # Step 1: Load session context
        session = await self._load_session(session_id, user_id)
        if not session:
            logger.error(f"Session {session_id} not found for user {user_id}")
            return self._default_decision_error()
        
        # Step 2: Get hard signals
        hard_signals = await self._compute_hard_signals(
            session,
            current_request_has_files=current_request_has_files,
        )
        
        # Step 3: Get last turn state
        last_turn_summary = None
        last_turn_artifacts = None
        if session.session_state:
            last_turn_summary = session.session_state.get("assistant_summary")
            if session.tool_calls_log and len(session.tool_calls_log) > 0:
                last_tool_call = session.tool_calls_log[-1]
                last_turn_artifacts = self._extract_artifacts_from_tool_call(last_tool_call)

            # Fallback: prefer richer persisted artifacts from session_state
            # (TurnState persistence) when tool_calls_log is missing/sparse.
            if not last_turn_artifacts:
                try:
                    artifacts_dict = session.session_state.get("artifacts")
                    if isinstance(artifacts_dict, dict) and artifacts_dict:
                        fake_tool_call = {
                            "tool_type": session.session_state.get("tool_used")
                            or artifacts_dict.get("tool_used")
                            or "UNKNOWN",
                            "output": artifacts_dict,
                        }
                        last_turn_artifacts = self._extract_artifacts_from_tool_call(fake_tool_call)
                except Exception:
                    last_turn_artifacts = None
        
        # Step 4: Build router input
        router_input = schemas.RouterInput(
            user_query=user_query,
            hard_signals=hard_signals,
            last_turn_summary=last_turn_summary,
            last_turn_artifacts=last_turn_artifacts,
            session_state=session.session_state,
        )
        
        # Step 5: Call LLM router
        decision = await self._call_router_llm(router_input)
        
        # Step 6: Apply deterministic corrections
        decision = await self._apply_safety_corrections(decision, hard_signals, session, user_query)
        
        return decision
    
    async def _load_session(
        self,
        session_id: str,
        user_id: str,
    ) -> Optional[models.ChatSession]:
        """Load session from database."""
        stmt = select(models.ChatSession).where(
            models.ChatSession.id == session_id,
            models.ChatSession.user_id == user_id,
        )
        result = await self.db.execute(stmt)
        return result.scalars().first()
    
    async def _compute_hard_signals(
        self,
        session: models.ChatSession,
        current_request_has_files: bool = False,
    ) -> schemas.RouterSignals:
        """
        Compute hard (deterministic) signals for routing.
        These are NOT ML-based, just objective facts.
        """
        # Signal 1: Has uploaded files?
        # Treat files uploaded on the current request as available for routing,
        # even if they haven't been persisted to the session yet.
        has_files_in_session = False
        try:
            stmt = select(models.UploadedFile.id).where(
                models.UploadedFile.session_id == session.id
            ).limit(1)
            result = await self.db.execute(stmt)
            has_files_in_session = result.scalar_one_or_none() is not None
        except Exception:
            # Fallback: relationship-based check (may be unloaded in async contexts)
            has_files_in_session = len(session.files) > 0 if session.files else False

        has_files = bool(current_request_has_files) or bool(has_files_in_session)
        
        # Signal 2: DB connected? (always for now, could check connectivity)
        db_connected = True
        
        # Signal 3: Last tool used?
        last_tool = None
        last_sql_exists = False
        last_file_context_exists = False
        time_since_last_turn = None
        
        if session.session_state:
            last_tool_str = session.session_state.get("tool_used")
            if last_tool_str:
                try:
                    last_tool = schemas.Tool(last_tool_str)
                except ValueError:
                    pass
            
            # Signal 4: Last SQL query exists?
            artifacts = session.session_state.get("artifacts", {})
            last_sql_exists = bool(artifacts.get("sql"))
            
            # Signal 5: Last file context exists?
            last_file_context_exists = bool(artifacts.get("file_ids"))
            
            # Signal 6: Time since last turn
            if session.state_updated_at:
                # Handle timezone-aware vs naive datetime comparison
                try:
                    if session.state_updated_at.tzinfo is not None:
                        # state_updated_at is timezone-aware, make utcnow() aware too
                        from datetime import timezone
                        current_time = datetime.now(timezone.utc)
                    else:
                        # state_updated_at is naive, use naive utcnow()
                        current_time = datetime.utcnow()
                    delta = current_time - session.state_updated_at
                    time_since_last_turn = int(delta.total_seconds())
                except Exception as e:
                    logger.warning(f"Error computing time since last turn: {e}")
                    time_since_last_turn = None
        
        return schemas.RouterSignals(
            has_uploaded_files=has_files,
            db_connected=db_connected,
            last_tool_used=last_tool,
            last_sql_exists=last_sql_exists,
            last_file_context_exists=last_file_context_exists,
            time_since_last_turn_seconds=time_since_last_turn,
        )
    
    def _extract_artifacts_from_tool_call(
        self,
        tool_call: Dict[str, Any]
    ) -> Optional[schemas.TurnStateArtifacts]:
        """Extract artifacts from stored tool call."""
        try:
            output = tool_call.get("output", {})
            if not output:
                return None
            
            # Build artifacts from output
            artifacts_dict = {
                "tool_used": tool_call.get("tool_type", "UNKNOWN"),
                "sql": output.get("sql"),
                "sql_plan_json": output.get("sql_plan_json"),
                "tables": output.get("tables"),
                "filters": output.get("filters"),
                "result_schema": output.get("result_schema"),
                "row_count": output.get("row_count"),
                "result_sample": output.get("result_sample"),
                "file_ids": output.get("file_ids"),
                "extracted_chunks": output.get("extracted_chunks"),
                "extracted_summary": output.get("extracted_summary"),
                "chat_summary": output.get("chat_summary"),
            }
            
            return schemas.TurnStateArtifacts(**artifacts_dict)
        except Exception as e:
            logger.error(f"Error extracting artifacts: {e}")
            return None
    
    async def _call_router_llm(
        self,
        router_input: schemas.RouterInput,
    ) -> schemas.RouterDecision:
        """Call LLM to make routing decision."""
        
        system_prompt = """You are a semantic intent router for a data query system.

Your job: Classify each user query into:
1. Tool: CHAT | ANALYZE_FILE | RUN_SQL | MIXED
2. Follow-up: NEW_QUERY | CHAT_FOLLOW_UP | ANALYZE_FILE_FOLLOW_UP | RUN_SQL_FOLLOW_UP
3. Follow-up subtype (if follow-up):
   - RUN_SQL: ADD_FILTER, REMOVE_FILTER, CHANGE_GROUPING, CHANGE_METRIC, SORT_OR_TOPK, EXPAND_COLUMNS, DRILLDOWN, PAGINATION, SWITCH_ENTITY, FIX_ERROR
   - ANALYZE_FILE: ASK_MORE_DETAIL, ASK_SUMMARY_DIFFERENT_STYLE, ASK_SOURCE_CITATION, COMPARE_SECTIONS, EXTRACT_TABLE_ENTITIES
   - CHAT: CLARIFY, CONTINUE, APPLY_PREVIOUS_ADVICE, REPHRASE, NEW_TOPIC_SAME_SESSION

KEY RULES:
- Use hard_signals to avoid mistakes (if DB not connected, no RUN_SQL)
- **FILES**: If query asks about "this file", "document content", "upload" → ANALYZE_FILE
- **DATABASE**: If query asks for "all clients", "sales data", "show me records" → RUN_SQL  
- **BOTH**: If query could use files AND database → MIXED
- If user says "those", "same", "previous", "as above" → likely FOLLOW_UP
- Implicit reference to last result → FOLLOW_UP
- Explicit new question → NEW_QUERY
- If confidence < 0.60 → needs_clarification = true
- If ANALYZE_FILE but no files in session → needs_clarification = true
- If RUN_SQL_FOLLOW_UP but no last_sql_exists → treat as NEW_QUERY

Output MUST be valid JSON matching the schema:
{
  "tool": "CHAT|ANALYZE_FILE|RUN_SQL|MIXED",
  "followup_type": "NEW_QUERY|CHAT_FOLLOW_UP|ANALYZE_FILE_FOLLOW_UP|RUN_SQL_FOLLOW_UP",
  "followup_subtype": "subtype_or_null",
  "needs_clarification": bool,
  "clarification_questions": [],
  "confidence": 0.0-1.0,
  "reasoning": "explanation",
  "signals_used": {}
}
"""
        
        user_prompt = self._build_router_prompt(router_input)
        
        try:
            response = await llm.call_llm_json(
                system=system_prompt,
                user=user_prompt,
                model=self.router_model,
                temperature=self.temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "RouterDecision",
                        "strict": True,
                        "schema": self._json_schema_for_router_decision(),
                    }
                }
            )
            
            decision_json = json.loads(response)
            # Mark this as a provisional signal (not final decision)
            decision_json["signals_used"] = decision_json.get("signals_used", {})
            decision_json["signals_used"]["is_provisional"] = True
            decision_json["signals_used"]["source"] = "semantic_intent_router"
            decision = schemas.RouterDecision(**decision_json)
            logger.info(f"Router signal: {decision.tool.value} ({decision.confidence:.2f}) [PROVISIONAL]")
            return decision
            
        except Exception as e:
            logger.error(f"Router LLM error: {e}")
            return self._default_decision_error()
    
    def _build_router_prompt(self, router_input: schemas.RouterInput) -> str:
        """Build the user prompt for the router LLM."""
        
        signals_summary = f"""
Hard Signals:
- Files in session: {router_input.hard_signals.has_uploaded_files}
- DB connected: {router_input.hard_signals.db_connected}
- Last tool used: {router_input.hard_signals.last_tool_used}
- Last SQL exists: {router_input.hard_signals.last_sql_exists}
- Last file context exists: {router_input.hard_signals.last_file_context_exists}
- Time since last turn: {router_input.hard_signals.time_since_last_turn_seconds}s
"""
        
        last_turn_summary = ""
        if router_input.last_turn_summary:
            last_turn_summary = f"""
Last Turn Summary:
{router_input.last_turn_summary}
"""
        
        last_artifacts = ""
        if router_input.last_turn_artifacts:
            artifacts = router_input.last_turn_artifacts
            last_artifacts = f"""
Last Turn Artifacts:
- Tool: {artifacts.tool_used}
- [SQL] Rows: {artifacts.row_count}, Tables: {artifacts.tables}
- [Files] Chunks: {len(artifacts.extracted_chunks) if artifacts.extracted_chunks else 0}
- Result Schema: {artifacts.result_schema}
"""
        
        return f"""Classify this user query based on hard signals and session context.

Current User Query:
"{router_input.user_query}"

{signals_summary}{last_turn_summary}{last_artifacts}

Now decide: What tool should handle this? Is it a follow-up to last turn or a new query?
Return valid JSON."""
    
    def _json_schema_for_router_decision(self) -> Dict[str, Any]:
        """Return JSON schema for RouterDecision for structured output."""
        return {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "enum": ["CHAT", "ANALYZE_FILE", "RUN_SQL", "MIXED"],
                },
                "followup_type": {
                    "type": "string",
                    "enum": [
                        "NEW_QUERY",
                        "CHAT_FOLLOW_UP",
                        "ANALYZE_FILE_FOLLOW_UP",
                        "RUN_SQL_FOLLOW_UP"
                    ],
                },
                "followup_subtype": {
                    "type": ["string", "null"],
                },
                "needs_clarification": {
                    "type": "boolean",
                },
                "clarification_questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "question": {"type": "string"},
                        },
                    },
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "reasoning": {
                    "type": "string",
                },
                "signals_used": {
                    "type": "object",
                },
            },
            "required": [
                "tool",
                "followup_type",
                "confidence",
                "reasoning",
                "needs_clarification",
                "clarification_questions",
                "signals_used",
            ],
            "additionalProperties": False,
        }
    
    async def _apply_safety_corrections(
        self,
        decision: schemas.RouterDecision,
        hard_signals: schemas.RouterSignals,
        session: models.ChatSession,
        user_query: str = "",
    ) -> schemas.RouterDecision:
        """
        Deterministic safety corrections.
        
        Fixes:
        1. If ANALYZE_FILE but no files → needs clarification
        2. If RUN_SQL_FOLLOW_UP but no last SQL → NEW_QUERY
        3. If confidence < 0.60 → needs clarification
        4. If tool needs resource not available → fallback
        """
        
        # Correction 1: ANALYZE_FILE without files
        if decision.tool == schemas.Tool.ANALYZE_FILE and not hard_signals.has_uploaded_files:
            decision.needs_clarification = True
            decision.clarification_questions = [
                schemas.ClarificationQuestionMissingParameter(
                    question="Please upload a file to analyze.",
                    required_field="file",
                    field_type="file",
                )
            ]
            decision.confidence = min(decision.confidence, 0.55)
            logger.warning("Corrected ANALYZE_FILE decision: no files in session")
        
        # Correction 1.5: Database queries should not be forced to ANALYZE_FILE
        # Only correct CHAT→ANALYZE_FILE for file-specific questions
        if (decision.tool == schemas.Tool.CHAT 
            and hard_signals.has_uploaded_files):
            # Check if query is actually about files (not database)
            file_keywords = ["file", "document", "upload", "this", "content", "analyze"]
            try:
                from app.services.schema_intelligence_service import get_schema_intelligence as _get_schema_intel
                _svc = _get_schema_intel()
                db_keywords = list(_svc.table_profiles.keys()) if _svc.table_profiles else []
                db_keywords += ["data", "records", "get me", "show me", "list", "all"]
            except Exception:
                db_keywords = ["data", "records", "get me", "show me", "list", "all"]
            
            query_lower = user_query.lower()
            has_file_intent = any(kw in query_lower for kw in file_keywords)
            has_db_intent = any(kw in query_lower for kw in db_keywords)
            
            if has_file_intent and not has_db_intent:
                decision.tool = schemas.Tool.ANALYZE_FILE
                decision.confidence = min(decision.confidence + 0.2, 1.0)
                logger.info("Corrected CHAT to ANALYZE_FILE: file-specific query detected")
        
        # Correction 2: RUN_SQL_FOLLOW_UP without last SQL
        if (decision.followup_type == schemas.FollowupType.RUN_SQL_FOLLOW_UP
                and not hard_signals.last_sql_exists):
            decision.followup_type = schemas.FollowupType.NEW_QUERY
            decision.followup_subtype = None
            # Cap confidence when downgrading: the LLM thought this was a follow-up
            # but there is no prior SQL to anchor to, so we are less certain.
            decision.confidence = min(decision.confidence, 0.6)
            logger.warning(
                "Corrected RUN_SQL_FOLLOW_UP to NEW_QUERY: no last SQL exists "
                f"(confidence capped at {decision.confidence:.2f})"
            )
        
        # Correction 3: Low confidence gating
        if decision.confidence < 0.60:
            decision.needs_clarification = True
            if not decision.clarification_questions:
                # Add a binary choice
                decision.clarification_questions = [
                    schemas.ClarificationQuestionMultipleChoice(
                        question="What would you like to do?",
                        options=[
                            "Run a SQL query",
                            "Analyze uploaded files",
                            "Chat with me about your data",
                        ],
                    )
                ]
            logger.warning(f"Low confidence ({decision.confidence:.2f}): triggering clarification")
        
        # Correction 4: DB not connected
        if not hard_signals.db_connected and decision.tool == schemas.Tool.RUN_SQL:
            decision.needs_clarification = True
            decision.clarification_questions = [
                schemas.ClarificationQuestionBinary(
                    question="Database is not available. Would you like to chat instead?",
                )
            ]
            decision.tool = schemas.Tool.CHAT  # Fallback
            decision.confidence = min(decision.confidence, 0.50)
            logger.error("DB not connected: falling back from RUN_SQL to CHAT")
        
        return decision
    
    def _default_decision_error(self) -> schemas.RouterDecision:
        """Return a safe default decision on error."""
        return schemas.RouterDecision(
            tool=schemas.Tool.CHAT,
            followup_type=schemas.FollowupType.NEW_QUERY,
            confidence=0.3,
            reasoning="Router failed - defaulting to CHAT",
            needs_clarification=True,
            clarification_questions=[
                schemas.ClarificationQuestionBinary(
                    question="I had trouble understanding your query. Could you rephrase it?",
                )
            ],
            signals_used={},
        )


async def create_router(db: AsyncSession) -> SemanticIntentRouter:
    """Factory function to create a router instance."""
    return SemanticIntentRouter(db)
