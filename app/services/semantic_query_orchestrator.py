"""
Semantic Query Orchestrator - Coordinates the 5-component semantic query pipeline.

Pipeline Flow:
1. Load/refresh semantic catalog (schema metadata + profiles)
2. Embed user query → retrieve Top-K tables/columns as context
3. LLM generates QueryPlan (JSON, not raw SQL)
4. Confidence gate: If confidence < 60%, ask clarification
5. Plan → SQL rendering (dialect-aware per database)
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
from .query_plan_generator import QueryPlanGenerator, QueryPlan, QueryPlanRenderer, ColumnSelectionAnalysis, ColumnSelectionIntent
from .confidence_gate import ConfidenceGate
from .dialect_adapter import get_adapter

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
    plan: Optional[QueryPlan]  # The generated plan (if successful)
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
        
        logger.info(
            f"[SEMANTIC ORCHESTRATOR] Initialized "
            f"(TTL={catalog_ttl_minutes}m, threshold={confidence_threshold}, gate={enable_confidence_gate})"
        )
    
    async def initialize_catalog(self) -> None:
        """Initialize/refresh semantic catalog from database."""
        if self._catalog is None:
            logger.info("[SEMANTIC] Initializing semantic catalog...")
            self._catalog = SemanticSchemaCatalog()  # ✅ Fixed API: no args to __init__
            
            # ✅ Fixed API: pass adapter and session to populate_from_database
            from .database_adapter import get_global_adapter
            adapter = get_global_adapter()
            await self._catalog.populate_from_database(adapter, self.db_session)
            
            self._catalog_loaded_at = datetime.now()
            logger.info(f"[SEMANTIC] Catalog loaded with {len(self._catalog.tables)} tables")
        else:
            # Check if TTL expired
            if self._catalog_loaded_at and datetime.now() - self._catalog_loaded_at > self.catalog_ttl:
                logger.info("[SEMANTIC] Catalog TTL expired, refreshing...")
                # ✅ Fixed API: pass adapter and session to populate_from_database
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
   KEY: "get all customers" = wants ALL customer columns, not just a few!

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
            # Extract JSON from response (LLM might add text around it)
            json_match = response.find('{')
            json_end = response.rfind('}') + 1
            if json_match >= 0 and json_end > json_match:
                json_str = response[json_match:json_end]
                analysis_data = json.loads(json_str)
                
                intent_str = analysis_data.get("intent", "all_columns")
                intent = ColumnSelectionIntent(intent_str)
                
                result = ColumnSelectionAnalysis(
                    intent=intent,
                    requested_columns=analysis_data.get("requested_columns", []),
                    reasoning=analysis_data.get("reasoning", ""),
                    user_mentions_columns=analysis_data.get("user_mentions_columns", False),
                    user_mentions_all=analysis_data.get("user_mentions_all", False),
                    confidence=analysis_data.get("confidence", 0.8),
                )
                
                logger.info(
                    f"[SEMANTIC] Column Selection Analysis: intent={result.intent}, "
                    f"columns={len(result.requested_columns)}, confidence={result.confidence:.2f}"
                )
                return result
        except Exception as e:
            logger.warning(f"[SEMANTIC] Failed to parse column analysis: {e}")
        
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
    ) -> SemanticQueryResult:
        """Execute full semantic query pipeline.
        
        Pipeline:
        1. Load/refresh catalog
        2. Retrieve schema context
        3. Generate query plan (LLM)
        4. Check confidence (gate)
        5. Render to SQL
        6. Validate safety
        
        Args:
            user_query: Natural language query from user
            clarification_response: User's response to clarification question (if any)
            
        Returns:
            SemanticQueryResult with SQL, plan, confidence, and trace
        """
        trace: Dict[str, Any] = {}
        
        try:
            # Stage 1: Load catalog
            trace[PipelineStage.CATALOG_LOAD] = "starting"
            await self.initialize_catalog()
            trace[PipelineStage.CATALOG_LOAD] = "complete"
            
            # Stage 2: Retrieve schema context
            trace[PipelineStage.RETRIEVAL] = "starting"
            retrieval_context = await self.retrieve_schema_context(user_query)
            trace[PipelineStage.RETRIEVAL] = {
                "tables": retrieval_context.top_tables,
                "confidence": retrieval_context.confidence_score
            }
            
            # Stage 3: Generate plan (LLM) - REQUIREMENT C: Plan-first generation
            # ✅ LLM now emits QueryPlan JSON (not raw SQL), enabling deterministic rendering
            trace[PipelineStage.PLAN_GENERATION] = "starting"
            
            # ✅ NEW: Pass table metadata with actual sample values to plan generator
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
            
            # NEW: Analyze column selection intent semantically using LLM
            logger.info("[SEMANTIC] Analyzing column selection intent...")
            column_analysis = await self.analyze_column_selection_intent(
                user_query,
                retrieval_context.top_tables,
                retrieval_context.top_columns_per_table,
            )
            plan.column_selection = column_analysis
            logger.info(
                f"[SEMANTIC] Column selection recorded in plan: "
                f"intent={column_analysis.intent.value}, confidence={column_analysis.confidence:.2f}"
            )
            
            # Stage 4: Confidence gate
            trace[PipelineStage.CONFIDENCE_CHECK] = "starting"
            plan_confidence = retrieval_context.confidence_score  # Could be plan-specific
            needs_clarification = self.enable_confidence_gate and plan_confidence < self.confidence_threshold
            
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
            
            # Stage 5: Render to SQL (dialect-aware)
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
            dialect = adapter.dialect.value
            renderer = QueryPlanRenderer(dialect=dialect)
            sql = renderer.render(plan)
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
