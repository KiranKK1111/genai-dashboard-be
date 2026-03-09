"""
UNIFIED SEMANTIC ROUTER (P0)

Single authoritative router implementing clean hierarchy:
1. Hard signals (deterministic facts)
2. Schema-derived signals (database capabilities)
3. LLM decision (learned routing)
4. Safety gating (confidence + missing context)

Zero hardcoding. Everything schema-driven or LLM-learned.
"""

from __future__ import annotations

import logging
import json
from typing import Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession

from .. import llm, models
from ..config import settings
from .router_decision import (
    RouterDecision, Tool, RequestType, FollowupTool,
    RunSQLFollowupSubtype, AnalyzeFileFollowupSubtype, ChatFollowupSubtype,
    ROUTER_DECISION_JSON_SCHEMA, ClarificationOption
)
from .hard_signals import HardSignalsExtractor, HardSignals
from .schema_derived_signals import SchemaDerivedSignalExtractor, SchemaDerivedSignals

logger = logging.getLogger(__name__)


class UnifiedSemanticRouter:
    """
    Single router with clear separation:
    - Hard signals: deterministic session facts
    - Schema signals: database capabilities
    - LLM: learns routing decisions
    - Safety gates: catches missing context before execution
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.hard_signals_extractor = HardSignalsExtractor(db)
        self.schema_signals_extractor = SchemaDerivedSignalExtractor(db)
        # Use the configured model by default to avoid hardcoding model names.
        self.router_model = settings.ai_factory_model
        self.temperature = 0.1
        self.confidence_threshold = 0.70  # Configurable
    
    async def route(
        self,
        user_query: str,
        session: models.ChatSession,
        schema_discovery_service,  # Schema service for table/column metadata
        current_request_has_files: bool = False,
    ) -> RouterDecision:
        """
        Route a user query to the appropriate tool.
        
        Args:
            user_query: The user's question
            session: Current ChatSession
            schema_discovery_service: Service providing DB schema metadata
            current_request_has_files: Whether files were uploaded this turn
            
        Returns:
            RouterDecision with tool, request_type, confidence, clarification needs
        """
        
        # Step 1: Extract hard signals (deterministic facts)
        hard_signals = await self.hard_signals_extractor.extract(
            session,
            current_request_has_files=current_request_has_files,
        )
        
        # Step 2: Extract schema signals (database capabilities)
        schema_signals = await self.schema_signals_extractor.extract_signals(
            user_query=user_query,
            schema_discovery_service=schema_discovery_service,
        )
        
        # Step 3: Get LLM to make decision (learned routing)
        decision = await self._call_router_llm(
            user_query=user_query,
            hard_signals=hard_signals,
            schema_signals=schema_signals,
            session_state=session.session_state,
        )
        
        # Step 4: Apply safety gates (clarification if needed)
        decision = self._apply_safety_gates(decision, hard_signals, schema_signals)
        
        # Log for debugging
        logger.info(f"Routing decision: {decision.tool.value} "
                   f"(request_type={decision.request_type.value}, "
                   f"confidence={decision.confidence:.2f})")
        
        return decision
    
    async def _call_router_llm(
        self,
        user_query: str,
        hard_signals: HardSignals,
        schema_signals: SchemaDerivedSignals,
        session_state: Optional[Dict[str, Any]],
    ) -> RouterDecision:
        """Call LLM to make routing decision."""
        
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            user_query=user_query,
            hard_signals=hard_signals,
            schema_signals=schema_signals,
            session_state=session_state,
        )
        
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
                        "schema": ROUTER_DECISION_JSON_SCHEMA,
                    }
                }
            )
            
            decision_dict = json.loads(response)
            decision = RouterDecision.from_dict(decision_dict)
            return decision
            
        except Exception as e:
            logger.error(f"Router LLM error: {e}")
            return self._default_decision_chat_error()
    
    def _build_system_prompt(self) -> str:
        """System prompt for router (instruction to LLM)."""
        return """You are a semantic router for a GenAI backend with three tools:
1. CHAT: conversation and knowledge questions
2. ANALYZE_FILE: analyze uploaded documents (PDF, DOCX, XLSX, CSV, text)
3. RUN_SQL: query a database

Your job is to DECIDE (not execute).

You must output ONLY valid JSON matching the RouterDecision schema.

Rules for decision:
1. Tool selection:
   - If files uploaded this turn → ANALYZE_FILE
   - If clear data/SQL intent → RUN_SQL
   - Otherwise (or if ambiguous) → CHAT

2. Request type (NEW_QUERY vs FOLLOW_UP):
   - NEW_QUERY: completely new question, or prior context unavailable
   - FOLLOW_UP: user explicitly references prior result, tables, or files

3. If FOLLOW_UP: determine followup_tool domain:
   - CHAT_FOLLOW_UP: continuing conversation
   - ANALYZE_FILE_FOLLOW_UP: referencing file context
   - RUN_SQL_FOLLOW_UP: modifying prior SQL query

4. If RUN_SQL_FOLLOW_UP: determine subtype:
   - ADD_FILTER, REMOVE_FILTER: WHERE clause changes
   - CHANGE_GROUPING, CHANGE_METRIC: aggregation changes
   - SORT_OR_TOPK, PAGINATION: result ordering/limiting
   - DRILLDOWN: zoom into subset
   - SWITCH_ENTITY: change which table(s)
   - EXPAND_COLUMNS: add/remove SELECT columns
   - FIX_ERROR: prior query had error

5. Confidence (0-1):
   - 0.9+ : very clear decision
   - 0.7-0.9: reasonable confidence
   - <0.7: ambiguous, needs clarification

6. Safety gates (set needs_clarification=true if):
   - RUN_SQL requested but db_available=false
   - ANALYZE_FILE requested but has_files_in_session=false
   - RUN_SQL_FOLLOW_UP requested but has_last_sql=false
   - confidence < 0.70
   - followup_tool requested but missing required context

OUTPUT MUST BE VALID JSON.
"""
    
    def _build_user_prompt(
        self,
        user_query: str,
        hard_signals: HardSignals,
        schema_signals: SchemaDerivedSignals,
        session_state: Optional[Dict[str, Any]],
    ) -> str:
        """Build user prompt with signals and context."""
        
        hard_signals_str = json.dumps(hard_signals.to_dict(), indent=2)
        schema_signals_str = json.dumps(schema_signals.to_dict(), indent=2)
        
        last_turn_summary = ""
        if session_state and "assistant_summary" in session_state:
            last_turn_summary = f"""
Last turn summary:
{session_state.get('assistant_summary', '')}
"""
        
        return f"""CURRENT QUERY:
"{user_query}"

HARD SIGNALS (deterministic facts):
{hard_signals_str}

SCHEMA-DERIVED SIGNALS (database capabilities):
{schema_signals_str}

{last_turn_summary}

TASK: Decide tool, request_type, followup info, confidence, and clarification needs.

Return JSON matching RouterDecision schema.
"""
    
    def _apply_safety_gates(
        self,
        decision: RouterDecision,
        hard_signals: HardSignals,
        schema_signals: SchemaDerivedSignals,
    ) -> RouterDecision:
        """
        Apply safety gates to ensure decision is executable.
        If something is missing, set needs_clarification=true.
        """
        
        # Gate 1: ANALYZE_FILE requires files in session
        if decision.tool == Tool.ANALYZE_FILE:
            if not hard_signals.has_files_in_session:
                decision.needs_clarification = True
                decision.clarification_question = "I don't see any uploaded files. Could you upload a file first?"
                return decision
        
        # Gate 2: RUN_SQL requires database availability
        if decision.tool == Tool.RUN_SQL:
            if not hard_signals.db_available:
                decision.needs_clarification = True
                decision.clarification_question = "The database is currently unavailable. I can only chat for now."
                decision.tool = Tool.CHAT
                return decision
            
            # Also check if schema has tables
            if not schema_signals.explicit_tables and decision.request_type == RequestType.NEW_QUERY:
                if not schema_signals.aggregatable_columns:
                    # No clear SQL intent from schema
                    decision.tool = Tool.CHAT
                    decision.reasoning += " [No SQL tables/columns detected; falling back to CHAT]"
        
        # Gate 3: RUN_SQL_FOLLOW_UP requires last SQL
        if (decision.request_type == RequestType.FOLLOW_UP and 
            decision.followup_tool == FollowupTool.RUN_SQL_FOLLOW_UP):
            if not hard_signals.has_last_sql:
                decision.needs_clarification = True
                decision.clarification_question = "I don't have a prior SQL query to modify. Could you run a SQL query first?"
                decision.request_type = RequestType.NEW_QUERY
                decision.followup_tool = None
                return decision
        
        # Gate 4: ANALYZE_FILE_FOLLOW_UP requires file context
        if (decision.request_type == RequestType.FOLLOW_UP and
            decision.followup_tool == FollowupTool.ANALYZE_FILE_FOLLOW_UP):
            if not hard_signals.has_last_file_context:
                decision.needs_clarification = True
                decision.clarification_question = "I don't have prior file context to follow up on. Please upload or reference a file."
                decision.request_type = RequestType.NEW_QUERY
                decision.followup_tool = None
                return decision
        
        # Gate 5: Low confidence always triggers clarification
        if decision.confidence < self.confidence_threshold:
            if not decision.needs_clarification:
                decision.needs_clarification = True
                decision.clarification_question = "I'm not entirely sure what you're asking. Could you clarify?"
        
        return decision
    
    def _default_decision_chat_error(self) -> RouterDecision:
        """Fallback decision when router fails."""
        return RouterDecision(
            tool=Tool.CHAT,
            request_type=RequestType.NEW_QUERY,
            confidence=0.1,
            needs_clarification=True,
            clarification_question="I encountered an error processing your request. Could you rephrase?",
            reasoning="Router LLM call failed; defaulting to CHAT",
        )
