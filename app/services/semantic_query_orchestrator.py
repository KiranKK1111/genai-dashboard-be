"""
Semantic Query Orchestrator - Coordinates the 5-component semantic query pipeline.

Pipeline Flow:
1. Load/refresh semantic catalog (schema metadata + profiles)
2. Embed user query â†’ retrieve Top-K tables/columns as context
3. LLM generates canonical QueryPlan (JSON, not raw SQL)
4. Confidence gate: If confidence < 60%, ask clarification
5. Plan â†’ SQL rendering (dialect-aware per database)
6. Safety validation + execution

This orchestrator bridges all semantic components into the active query flow.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from .semantic_schema_catalog import SemanticSchemaCatalog, TableMetadata, ColumnMetadata
from .embedding_retriever import EmbeddingBasedRetriever
from .query_plan_generator import QueryPlanGenerator  # Generator only - no internal types
from .query_plan import QueryPlan, ColumnSelectionAnalysis, ColumnSelectionIntent  # Canonical plan types
from .confidence_gate import ConfidenceGate
from .dialect_adapter import get_adapter

# NEW PHASE 1-5: Universal schema agnostic services
from .schema_discovery_engine import SchemaDiscoveryEngine
from .intent_classifier import IntentClassifier
from .universal_value_grounder import UniversalValueGrounder
from .semantic_concept_mapper import SemanticConceptMapper
from ..config import settings

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """Pipeline execution stages."""
    CATALOG_LOAD = "catalog_load"
    RETRIEVAL = "retrieval"
    PLAN_GENERATION = "plan_generation"
    CONFIDENCE_CHECK = "confidence_check"
    RENDERING = "rendering"
    SAFETY_VALIDATION = "safety_validation"
    EXECUTION = "execution"


@dataclass
class RetrievalContext:
    """Context from semantic retrieval phase."""
    top_tables: List[str]  # Top-N table names
    top_columns_per_table: Dict[str, List[str]]  # {table: [cols]}
    join_candidates: List[Tuple[str, str]]  # [(table1, table2), ...]
    confidence_score: float  # 0.0-1.0
    reasoning: str  # Why these candidates were selected


@dataclass
class SemanticQueryResult:
    """Result from semantic orchestrator."""
    success: bool
    sql: str  # Rendered SQL ready for execution
    plan: Optional[QueryPlan]  # The generated canonical plan (if successful)
    retrieval_context: Optional[RetrievalContext]  # What was retrieved
    confidence_score: float  # Final confidence (0.0-1.0)
    clarification_needed: bool  # If true, user asked for clarification
    clarification_question: Optional[str]  # The question, if needed
    error: Optional[str]  # Error message if failed
    pipeline_trace: Dict[str, Any]  # Debug trace of each stage


class SemanticQueryOrchestrator:
    """Orchestrates the 5-component semantic query pipeline."""
    
    def __init__(
        self,
        db_session: Optional[AsyncSession] = None,
        catalog_ttl_minutes: int = 60,
        confidence_threshold: float = 0.60,
        enable_confidence_gate: bool = True,
    ):
        """Initialize orchestrator.
        
        Args:
            db_session: Database session for schema discovery (optional for testing)
            catalog_ttl_minutes: How long to cache catalog before refresh
            confidence_threshold: Threshold for asking clarification
            enable_confidence_gate: Whether to use confidence gate
        """
        self.db_session = db_session
        self.catalog_ttl = timedelta(minutes=catalog_ttl_minutes)
        self.confidence_threshold = confidence_threshold
        self.enable_confidence_gate = enable_confidence_gate
        
        # Components (lazy-initialized)
        self._catalog: Optional[SemanticSchemaCatalog] = None
        self._retriever: Optional[EmbeddingBasedRetriever] = None
        self._plan_generator = QueryPlanGenerator()
        self._confidence_gate = ConfidenceGate()
        self._catalog_loaded_at: Optional[datetime] = None
        
        # NEW PHASE 1-5: Universal schema agnostic services (feature-flagged)
        self.schema_discovery_engine: Optional[SchemaDiscoveryEngine] = None
        self.intent_classifier: Optional[IntentClassifier] = None
        self.value_grounder: Optional[UniversalValueGrounder] = None
        self.concept_mapper: Optional[SemanticConceptMapper] = None
        
        # Initialize universal services if enabled and db_session available
        if db_session:
            if settings.enable_schema_discovery_engine:
                self.schema_discovery_engine = SchemaDiscoveryEngine(db_session)
            if settings.enable_semantic_intent_classifier:
                self.intent_classifier = IntentClassifier()
            if settings.enable_universal_value_grounder:
                self.value_grounder = UniversalValueGrounder(db_session)
            if settings.enable_semantic_concept_mapper:
                self.concept_mapper = SemanticConceptMapper()
        
        logger.info(
            f"[SEMANTIC ORCHESTRATOR] Initialized "
            f"(TTL={catalog_ttl_minutes}m, threshold={confidence_threshold}, gate={enable_confidence_gate})"
        )
    
    async def initialize_catalog(self) -> None:
        """Initialize/refresh semantic catalog from database."""
        if self._catalog is None:
            logger.info("[SEMANTIC] Initializing semantic catalog...")
            self._catalog = SemanticSchemaCatalog()  # âœ… Fixed API: no args to __init__
            
            # âœ… Fixed API: pass adapter and session to populate_from_database
            from .database_adapter import get_global_adapter
            adapter = get_global_adapter()
            await self._catalog.populate_from_database(adapter, self.db_session)
            
            self._catalog_loaded_at = datetime.now()
            logger.info(f"[SEMANTIC] Catalog loaded with {len(self._catalog.tables)} tables")
        else:
            # Check if TTL expired
            if self._catalog_loaded_at and datetime.now() - self._catalog_loaded_at > self.catalog_ttl:
                logger.info("[SEMANTIC] Catalog TTL expired, refreshing...")
                # âœ… Fixed API: pass adapter and session to populate_from_database
                from .database_adapter import get_global_adapter
                adapter = get_global_adapter()
                await self._catalog.populate_from_database(adapter, self.db_session)
                self._catalog_loaded_at = datetime.now()
    
    async def retrieve_schema_context(
        self,
        user_query: str,
    ) -> RetrievalContext:
        """Retrieve Top-K tables and columns relevant to user query.
        
        Args:
            user_query: Natural language user query
            
        Returns:
            RetrievalContext with candidate tables and columns
        """
        if self._retriever is None:
            self._retriever = EmbeddingBasedRetriever(self._catalog)
        
        # Retrieve candidates
        top_tables = await self._retriever.retrieve_table_candidates(
            user_query, k=3, threshold=0.4
        )
        
        # For each table, retrieve relevant columns
        top_columns_per_table = {}
        for table_result in top_tables:
            table_name = table_result.name
            cols = await self._retriever.retrieve_column_candidates(
                user_query, table_name=table_name, k=3, threshold=0.3
            )
            top_columns_per_table[table_name] = [c.name for c in cols]
        
        # Retrieve join candidates
        joins = await self._retriever.retrieve_join_candidates(
            table_names=[t.name for t in top_tables], k=2
        )
        
        # Compute average confidence
        avg_confidence = sum(t.confidence for t in top_tables) / len(top_tables) if top_tables else 0.0
        
        context = RetrievalContext(
            top_tables=[t.name for t in top_tables],
            top_columns_per_table=top_columns_per_table,
            join_candidates=[(j.left_table, j.right_table) for j in joins],
            confidence_score=avg_confidence,
            reasoning=f"Retrieved {len(top_tables)} tables, {sum(len(c) for c in top_columns_per_table.values())} columns"
        )
        
        logger.info(f"[SEMANTIC RETRIEVAL] {context.reasoning} (conf={avg_confidence:.2f})")
        return context
    
    async def analyze_column_selection_intent(
        self,
        user_query: str,
        available_tables: List[str],
        available_columns: Dict[str, List[str]],
    ) -> ColumnSelectionAnalysis:
        """
        Use LLM to semantically analyze what columns user wants in result.
        
        This is 100% LLM-driven, not hardcoded keyword matching.
        
        Args:
            user_query: User's natural language query
            available_tables: Tables available in database
            available_columns: Dict of {table: [columns]}
            
        Returns:
            ColumnSelectionAnalysis with LLM-determined intent
        """
        from .. import llm
        
        # Prepare column list for LLM
        column_descriptions = []
        for table, cols in list(available_columns.items())[:3]:  # Limit to top 3 tables
            column_descriptions.append(f"  {table}: {', '.join(cols[:10])}")  # Top 10 cols per table
        columns_str = "\n".join(column_descriptions)
        
        analysis_prompt = f"""Analyze the user's query to determine what columns they should see in the database result.

Available tables and their columns:
{columns_str}

User query: "{user_query}"

CLASSIFY the user's intent into ONE category:

1. ALL_COLUMNS - User wants to see all available columns
   Indicators: "all", "everything", "complete", "full", show/get/retrieve", "details"
    KEY: "get all <table>" = wants ALL columns for that entity, not just a few

2. SPECIFIC_COLUMNS - User explicitly names specific columns they want
   Indicators: Exact column names mentioned like "name and email", "ID and status"
   NOTE: Only use if user lists actual column names!

3. COUNT_ONLY - User wants aggregate count values
   Indicators: "count", "how many", "total", "number of"

4. DISTINCT_VALUES - User wants unique/different values from a column
   Indicators: "distinct", "unique", "what are all", "list all", "different"

5. FIRST_N_COLUMNS - User wants top N records (uses LIMIT)
   Indicators: "top N", "first N", "show N records"

RULE: When in doubt, choose ALL_COLUMNS (shows more, better UX)

Respond ONLY with valid JSON:
{{
  "intent": "all_columns" | "specific_columns" | "count_only" | "distinct_values" | "first_n_columns",
  "requested_columns": ["col1", "col2"],  # ONLY for specific_columns intent
  "reasoning": "Based on user's query, why this intent",
  "user_mentions_columns": true/false,
  "user_mentions_all": true/false,
  "confidence": 0.5-1.0
}}"""
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert semantic SQL analyst. Your sole job is to understand what columns the user wants.

100% LLM-DRIVEN: All decisions come from your semantic understanding. No hardcoded rules, no keyword matching.

Classify column selection intent based on deep understanding of the user's query, not surface pattern matching.
Always return valid JSON with your best semantic judgment."""
            },
            {
                "role": "user",
                "content": analysis_prompt
            }
        ]
        
        # Call LLM for semantic analysis
        response = await llm.call_llm(messages, stream=False, max_tokens=256, temperature=0.2)
        
        try:
            import json
            import re
            # Robustly extract the first top-level JSON object from the LLM response.
            # find('{') / rfind('}') fails when the response contains multiple objects or
            # surrounding text with braces.  Using a regex with DOTALL is more reliable.
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group(0))

                intent_str = analysis_data.get("intent", "all_columns")
                try:
                    intent = ColumnSelectionIntent(intent_str)
                except ValueError:
                    logger.warning("[SEMANTIC] Unknown intent '%s' from LLM, defaulting to ALL_COLUMNS", intent_str)
                    intent = ColumnSelectionIntent.ALL_COLUMNS

                result = ColumnSelectionAnalysis(
                    intent=intent,
                    requested_columns=analysis_data.get("requested_columns", []),
                    reasoning=analysis_data.get("reasoning", ""),
                    user_mentions_columns=analysis_data.get("user_mentions_columns", False),
                    user_mentions_all=analysis_data.get("user_mentions_all", False),
                    confidence=float(analysis_data.get("confidence", 0.8)),
                )

                logger.info(
                    "[SEMANTIC] Column Selection Analysis: intent=%s, columns=%d, confidence=%.2f",
                    result.intent, len(result.requested_columns), result.confidence,
                )
                return result
        except json.JSONDecodeError as e:
            logger.warning("[SEMANTIC] JSON parse error in column analysis: %s", e)
        except Exception as e:
            logger.warning("[SEMANTIC] Failed to parse column analysis: %s", e)
        
        # Fallback: If LLM parsing completely fails, default to ALL_COLUMNS for better UX
        # No hardcoded keyword matching - all semantic intelligence from LLM
        return ColumnSelectionAnalysis(
            intent=ColumnSelectionIntent.ALL_COLUMNS,
            reasoning="LLM column analysis encountered error, defaulting to all columns for optimal user experience",
            confidence=0.5
        )
    
    async def execute_semantic_query(
        self,
        user_query: str,
        clarification_response: Optional[str] = None,
        followup_anchor: Optional[Dict[str, Any]] = None,
    ) -> SemanticQueryResult:
        """Execute full semantic query pipeline.
        
        Pipeline:
        1. Load/refresh catalog
        2. Retrieve schema context (optionally biased by followup_anchor)
        3. Generate query plan (LLM)
        4. Check confidence (gate)
        5. Render to SQL
        6. Validate safety
        
        Args:
            user_query: Natural language query from user
            clarification_response: User's response to clarification question (if any)
            followup_anchor: Optional follow-up context {previous_table, previous_filters, previous_sql}
                            Used to bias plan generation toward previous query context
            
        Returns:
            SemanticQueryResult with SQL, plan, confidence, and trace
        """
        trace: Dict[str, Any] = {}
        
        try:
            # Determine if this is a follow-up with context
            # Used throughout pipeline to adjust behavior (skip clarifications, force execution)
            is_followup_with_context = followup_anchor and (
                followup_anchor.get("previous_table") or 
                followup_anchor.get("inferred_previous_table") or
                followup_anchor.get("previous_sql")
            )
            
            # Stage 1: Load catalog
            trace[PipelineStage.CATALOG_LOAD] = "starting"
            await self.initialize_catalog()
            trace[PipelineStage.CATALOG_LOAD] = "complete"
            
            # Stage 2: Retrieve schema context
            trace[PipelineStage.RETRIEVAL] = "starting"
            
            # NEW: If this is a follow-up query, bias retrieval toward previous table
            if followup_anchor:
                logger.info(f"[SEMANTIC RETRIEVAL] Using follow-up anchor context")
                prev_table = followup_anchor.get("previous_table") or followup_anchor.get("inferred_previous_table")
                # Include previous table in retrieval to anchor follow-up focus
                retrieval_context = await self.retrieve_schema_context(user_query)
                # Boost previous table to top if it exists in catalog
                if prev_table and prev_table in retrieval_context.top_tables:
                    retrieval_context.top_tables.remove(prev_table)
                    retrieval_context.top_tables.insert(0, prev_table)
                    logger.info(f"[SEMANTIC RETRIEVAL] Boosted previous table '{prev_table}' to top")
                
                # NEW: Fallback for follow-ups with 0 retrieval results
                # If retrieval got nothing, bias toward previous table as last resort
                if not retrieval_context.top_tables and prev_table:
                    logger.info(f"[SEMANTIC RETRIEVAL] FALLBACK: Using previous table '{prev_table}' (retrieval got 0 results)")
                    retrieval_context.top_tables = [prev_table]
                    if prev_table in self._catalog.tables:
                        table_meta = self._catalog.tables[prev_table]
                        # table_meta.columns is a dict, so get values and slice
                        all_columns = list(table_meta.columns.values()) if hasattr(table_meta.columns, 'values') else []
                        retrieval_context.top_columns_per_table[prev_table] = [
                            col.name for col in all_columns[:10]
                        ]
                    retrieval_context.confidence_score = 0.5  # Medium confidence (assisted by anchor)
            else:
                retrieval_context = await self.retrieve_schema_context(user_query)
            
            trace[PipelineStage.RETRIEVAL] = {
                "tables": retrieval_context.top_tables,
                "confidence": retrieval_context.confidence_score
            }
            
            # Stage 3: Generate plan (LLM) - REQUIREMENT C: Plan-first generation
            # âœ… LLM now emits canonical QueryPlan JSON (not raw SQL), enabling deterministic rendering
            trace[PipelineStage.PLAN_GENERATION] = "starting"
            
            # âœ… NEW: Pass table metadata with actual sample values to plan generator
            # This enables dynamic filter value extraction instead of hardcoding
            table_metadata = {
                table_name: {
                    "columns": table_meta.columns
                }
                for table_name, table_meta in self._catalog.tables.items()
            }
            
            plan = await self._plan_generator.create_query_plan(
                user_query,
                available_tables=retrieval_context.top_tables,
                available_columns_per_table=retrieval_context.top_columns_per_table,
                possible_joins=retrieval_context.join_candidates,
                table_metadata=table_metadata,
                catalog=self._catalog,  # Pass catalog for sample data grounding
            )
            trace[PipelineStage.PLAN_GENERATION] = {"plan_type": "query_plan", "table": plan.from_table}
            
            # âœ… NEW: Check if LLM indicates clarification is needed (ChatGPT-style)
            # BUT: For follow-ups with context, SKIP this and force execution with previous table
            # This allows refinements like "what about those in delhi" to work immediately
            skip_clarification_for_followup = is_followup_with_context
            
            if plan.clarification_needed and not skip_clarification_for_followup:
                logger.info(
                    f"[SEMANTIC PIPELINE] LLM requested clarification: {plan.clarification_question}"
                )
                trace[PipelineStage.PLAN_GENERATION] = {
                    "plan_type": "clarification_request",
                    "question": plan.clarification_question,
                    "options": plan.clarification_options
                }
                
                # Return immediately with clarification question
                return SemanticQueryResult(
                    success=True,  # Not a failure - just asking for clarification
                    sql="",
                    plan=plan,  # Include the plan so UI can use options if provided
                    retrieval_context=retrieval_context,
                    confidence_score=0.0,  # Low confidence until clarified
                    clarification_needed=True,
                    clarification_question=plan.clarification_question,
                    error=None,
                    pipeline_trace=trace,
                )
            
            # NEW: Analyze column selection intent semantically using LLM
            logger.info("[SEMANTIC] Analyzing column selection intent...")
            column_analysis = await self.analyze_column_selection_intent(
                user_query,
                retrieval_context.top_tables,
                retrieval_context.top_columns_per_table,
            )
            plan.column_selection = column_analysis
            intent_val = column_analysis.intent.value if hasattr(column_analysis.intent, 'value') else str(column_analysis.intent)
            logger.info(
                f"[SEMANTIC] Column selection recorded in plan: "
                f"intent={intent_val}, confidence={column_analysis.confidence:.2f}"
            )
            
            # Stage 4: Confidence gate
            trace[PipelineStage.CONFIDENCE_CHECK] = "starting"
            plan_confidence = retrieval_context.confidence_score  # Could be plan-specific
            
            needs_clarification = (
                self.enable_confidence_gate and 
                plan_confidence < self.confidence_threshold and
                not is_followup_with_context  # SKIP gate for follow-ups with context
            )
            
            if needs_clarification:
                clarification_q = f"Did you mean to query {', '.join(retrieval_context.top_tables[:2])}?"
                logger.info(f"[CONFIDENCE GATE] Confidence {plan_confidence:.2f} < {self.confidence_threshold}, asking: {clarification_q}")
                trace[PipelineStage.CONFIDENCE_CHECK] = {"needs_clarification": True, "question": clarification_q}
                
                # Return partial result with question
                return SemanticQueryResult(
                    success=False,
                    sql="",
                    plan=None,
                    retrieval_context=retrieval_context,
                    confidence_score=plan_confidence,
                    clarification_needed=True,
                    clarification_question=clarification_q,
                    error=None,
                    pipeline_trace=trace,
                )
            
            trace[PipelineStage.CONFIDENCE_CHECK] = {"needs_clarification": False}
            
            # Stage 5: Render to SQL via CANONICAL pipeline (dialect-aware)
            trace[PipelineStage.RENDERING] = "starting"
            # Detect database dialect from session
            dialect_name = "postgresql"  # default
            if self.db_session:
                try:
                    bind = self.db_session.get_bind()
                    if hasattr(bind, 'dialect'):
                        dialect_name = bind.dialect.name
                except Exception:
                    pass  # fallback to postgresql default
            adapter = get_adapter(dialect_name)
            dialect_val = adapter.dialect.value if hasattr(adapter.dialect, 'value') else str(adapter.dialect)
            dialect = dialect_val
            
            # CANONICAL PIPELINE: plan is already canonical QueryPlan - compile directly
            from .query_plan_compiler import compile_query_plan
            sql = compile_query_plan(plan, dialect=dialect)
            trace[PipelineStage.RENDERING] = {"dialect": dialect, "sql_length": len(sql)}
            
            # Stage 6: Safety validation
            trace[PipelineStage.SAFETY_VALIDATION] = "starting"
            from .sql_safety_validator import SQLSafetyValidator
            validator = SQLSafetyValidator(allowed_schemas=None, max_rows=500)
            is_safe, error, rewritten_sql = validator.validate_and_rewrite(sql)
            
            if not is_safe:
                trace[PipelineStage.SAFETY_VALIDATION] = {"safe": False, "error": error}
                return SemanticQueryResult(
                    success=False,
                    sql="",
                    plan=plan,
                    retrieval_context=retrieval_context,
                    confidence_score=plan_confidence,
                    clarification_needed=False,
                    clarification_question=None,
                    error=f"Safety validation failed: {error}",
                    pipeline_trace=trace,
                )
            
            trace[PipelineStage.SAFETY_VALIDATION] = {"safe": True}
            
            logger.info(f"[SEMANTIC PIPELINE] Success: {len(rewritten_sql)} char SQL")
            
            return SemanticQueryResult(
                success=True,
                sql=rewritten_sql,
                plan=plan,
                retrieval_context=retrieval_context,
                confidence_score=plan_confidence,
                clarification_needed=False,
                clarification_question=None,
                error=None,
                pipeline_trace=trace,
            )
            
        except Exception as e:
            logger.error(f"[SEMANTIC PIPELINE] Error: {str(e)}", exc_info=True)
            
            # CRITICAL: Rollback failed transaction to prevent InFailedSQLTransactionError
            # If any DB operation failed (like MissingGreenlet), the transaction is marked as failed
            # until we explicitly rollback. This prevents subsequent queries from failing.
            if self.db_session:
                try:
                    await self.db_session.rollback()
                    logger.debug("[SEMANTIC PIPELINE] Rolled back failed transaction")
                except Exception as rollback_err:
                    logger.warning(f"[SEMANTIC PIPELINE] Failed to rollback: {rollback_err}")
            
            return SemanticQueryResult(
                success=False,
                sql="",
                plan=None,
                retrieval_context=None,
                confidence_score=0.0,
                clarification_needed=False,
                clarification_question=None,
                error=str(e),
                pipeline_trace=trace,
            )
    
    # ===== NEW: Follow-up Anchor Selection (Plan-Diff System) =====
    
    async def select_followup_anchor(
        self,
        followup_query: str,
        memory_store,  # QueryMemoryStore from SessionStateManager
        top_k: int = 3,
    ) -> Optional[Tuple[Any, float, List]]:
        """
        Select which past SQL query a follow-up should anchor to.
        
        Uses semantic similarity + recency + entity overlap.
        If ambiguous (top candidates close), returns list for clarification.
        
        Args:
            followup_query: The follow-up user query
            memory_store: QueryMemoryStore with past executions
            top_k: How many candidates to retrieve
            
        Returns:
            Tuple of (anchor_execution, confidence, alternatives) or None if no history
        """
        if not memory_store or not memory_store.queries:
            logger.debug("[FOLLOW-UP] No past queries in memory, treating as new request")
            return None
        
        # Get top-K similar queries
        candidates = memory_store.get_top_k_similar(followup_query, k=top_k)
        
        if not candidates:
            logger.info("[FOLLOW-UP] No similar past queries found")
            return None
        
        anchor, top_score = candidates[0]
        
        # Check if ambiguous (top-1 and top-2 are close)
        if len(candidates) > 1:
            second_score = candidates[1][1]
            if abs(top_score - second_score) < 0.08:
                logger.info(
                    f"[FOLLOW-UP] Ambiguous anchor selection (diff={abs(top_score - second_score):.3f}), "
                    f"returning top candidates for clarification"
                )
                alternatives = [(c[0].query_id, c[0].user_query, c[1]) for c in candidates[:3]]
                return (anchor, top_score, alternatives)
        
        logger.info(f"[FOLLOW-UP] Selected anchor query (confidence={top_score:.2f})")
        return (anchor, top_score, None)
    
    async def generate_plan_edits(
        self,
        anchor_plan: QueryPlan,
        followup_query: str,
        retrieval_context: RetrievalContext,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate plan edits for a follow-up instead of full regeneration.
        
        Calls LLM to output structured edits:
        - add_filter, remove_filter
        - set_limit, set_offset
        - add_order_by, change_order_by
        - add_group_by, modify_aggregates
        
        This is more stable than full SQL regeneration.
        
        Args:
            anchor_plan: The canonical plan from previous query
            followup_query: The follow-up user request
            retrieval_context: Retrieved schema context
            
        Returns:
            Dict with "apply_to", "edits", "needs_clarification" fields, or None if can't edit
        """
        from .. import llm
        
        # Build anchor plan description for LLM
        anchor_desc = f"""
ANCHOR QUERY: {anchor_plan.user_query if hasattr(anchor_plan, 'user_query') else 'Previous query'}
ANCHOR PLAN:
- Tables: {', '.join([anchor_plan.from_table] + [j.right_table for j in anchor_plan.joins])}
- Filters: {len(anchor_plan.where_conditions)} conditions
- Group By: {anchor_plan.group_by if anchor_plan.group_by else 'none'}
- Aggregates: {len(anchor_plan.select_aggregates)} aggregates
- Order By: {len(anchor_plan.order_by)} fields
- Limit/Offset: {anchor_plan.limit}/{anchor_plan.offset}
"""
        
        llm_prompt = f"""You are helping a user modify a previous database query.

{anchor_desc}

FOLLOW-UP REQUEST: "{followup_query}"

Your job: Generate structured EDITS to the anchor plan, not a new query.

COMMON EDITS:
- Add filter: {{"op":"add_filter","left":"column","operator":"=","right":"value","right_kind":"literal"}}
- Remove filter: {{"op":"remove_filter","filter_index":0}}
- Set limit: {{"op":"set_limit","value":100}}
- Add order by: {{"op":"add_order_by","column":"name","direction":"DESC"}}
- Set group by: {{"op":"set_group_by","columns":["status","type"]}}

RESPOND WITH ONLY THIS JSON (no markdown):
{{
  "apply_to": "anchor_query_id",
  "edits": [
    {{"op":"..."," ...}}
  ],
  "needs_clarification": false,
  "clarification_question": null
}}"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a plan editor. Output ONLY valid JSON, no other text."
            },
            {
                "role": "user",
                "content": llm_prompt
            }
        ]
        
        try:
            response = await llm.call_llm(messages, stream=False, max_tokens=500)
            
            # Parse JSON
            import json
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            if response.endswith("```"):
                response = response[:-3].strip()
            
            result = json.loads(response)
            logger.info(f"[PLAN EDITS] Generated {len(result.get('edits', []))} edits")
            
            return result
            
        except Exception as e:
            logger.warning(f"[PLAN EDITS] Generation failed: {e}")
            return None
    
    # NOTE: apply_plan_edits has been REMOVED (was dead code using internal types).
    # Plan modifications should be done on canonical QueryPlan directly.
    # See query_plan.py for the canonical QueryPlan type.


async def create_semantic_orchestrator(
    db_session: AsyncSession,
    catalog_ttl_minutes: int = 60,
    confidence_threshold: float = 0.60,
    enable_confidence_gate: bool = True,
) -> SemanticQueryOrchestrator:
    """Factory function to create and initialize orchestrator."""
    orchestrator = SemanticQueryOrchestrator(
        db_session=db_session,
        catalog_ttl_minutes=catalog_ttl_minutes,
        confidence_threshold=confidence_threshold,
        enable_confidence_gate=enable_confidence_gate,
    )
    await orchestrator.initialize_catalog()
    return orchestrator

