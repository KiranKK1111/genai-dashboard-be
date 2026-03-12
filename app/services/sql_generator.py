"""
DEPRECATED: SQL generation service - legacy direct LLM-to-SQL bypass path.

================================================================================
⚠️  WARNING: THIS MODULE IS DEPRECATED AND SHOULD NOT BE USED
================================================================================

This module is kept only for backward compatibility during migration.
All code has been removed - functions raise NotImplementedError.

CANONICAL ARCHITECTURE (use this instead):
    NL Query → SemanticQueryOrchestrator → QueryPlan → query_plan_compiler → SQL

OR for manual plan construction:
    QueryPlanGenerator → canonical QueryPlan → compile_query_plan() → SQL

This file will be fully deleted in the next major version.
================================================================================
"""

from __future__ import annotations

import warnings
from typing import Tuple, Optional

from sqlalchemy.ext.asyncio import AsyncSession

# Emit deprecation warning when module is imported
warnings.warn(
    "sql_generator module is deprecated. Use SemanticQueryOrchestrator or "
    "QueryPlanGenerator → query_plan_compiler for SQL generation.",
    DeprecationWarning,
    stacklevel=2
)


async def generate_sql_with_analysis(
    query: str, 
    session: AsyncSession, 
    conversation_history: str = "",
    force_join: bool = False,
    followup_context = None,
    semantic_context = None,
) -> Tuple[str, Optional[str]]:
    """
    DEPRECATED AND REMOVED: This function is no longer available.
    
    This function previously generated SQL directly from LLM without using 
    the canonical QueryPlan pipeline, which bypassed validation and safety checks.
    
    MIGRATION GUIDE:
    ================
    Use the canonical plan-first pipeline instead:
    
    1. SemanticQueryOrchestrator (recommended):
       ```python
       from app.services.semantic_query_orchestrator import create_semantic_orchestrator
       orchestrator = await create_semantic_orchestrator(db_session)
       result = await orchestrator.process_query(query)
       sql = result.sql
       ```
    
    2. QueryPlan → compile_query_plan (for manual plan construction):
       ```python
       from app.services.query_plan import QueryPlan, SelectClause, FromClause
       from app.services.query_plan_compiler import compile_query_plan
       
       plan = QueryPlan(
           intent="data_query",
           select=SelectClause(fields=["*"]),
           from_=FromClause(table="my_table")
       )
       sql = compile_query_plan(plan, dialect="postgresql")
       ```
    
    See: session_query_handler.py → build_data_query_response for the active implementation.
    """
    raise NotImplementedError(
        "generate_sql_with_analysis has been REMOVED. "
        "Use SemanticQueryOrchestrator → QueryPlan → query_plan_compiler pipeline. "
        "See session_query_handler.py → build_data_query_response for the canonical implementation."
    )


async def generate_sql(query: str, session: AsyncSession, conversation_history: str = "", followup_context = None, semantic_context = None) -> str:
    """
    DEPRECATED AND REMOVED: This function is no longer available.
    
    This function previously generated SQL directly from LLM without using 
    the canonical QueryPlan pipeline, which bypassed validation and safety checks.
    
    MIGRATION GUIDE:
    ================
    Use the canonical plan-first pipeline instead:
    
    1. SemanticQueryOrchestrator (recommended):
       ```python
       from app.services.semantic_query_orchestrator import create_semantic_orchestrator
       orchestrator = await create_semantic_orchestrator(db_session)
       result = await orchestrator.process_query(query)
       sql = result.sql
       ```
    
    2. QueryPlan → compile_query_plan (for manual plan construction):
       ```python
       from app.services.query_plan import QueryPlan, SelectClause, FromClause
       from app.services.query_plan_compiler import compile_query_plan
       
       plan = QueryPlan(
           intent="data_query",
           select=SelectClause(fields=["*"]),
           from_=FromClause(table="my_table")
       )
       sql = compile_query_plan(plan, dialect="postgresql")
       ```
    
    See: session_query_handler.py → build_data_query_response for the active implementation.
    """
    raise NotImplementedError(
        "generate_sql has been REMOVED. "
        "Use SemanticQueryOrchestrator → QueryPlan → query_plan_compiler pipeline. "
        "See session_query_handler.py → build_data_query_response for the canonical implementation."
    )


# =============================================================================
# LEGACY CODE REMOVED
# =============================================================================
# The original implementation of generate_sql() has been removed.
# The canonical pipeline is now:
#   NL Query → SemanticQueryOrchestrator → QueryPlan → query_plan_compiler → SQL
#
# For reference on the old implementation, see git history.
# This file will be fully deleted in the next major version.
# =============================================================================
