"""
FastAPI dependency providers.

Wraps the global singletons into typed FastAPI Depends() providers so that:
  - Route handlers declare exact dependencies (self-documenting)
  - Tests can override any dependency via app.dependency_overrides
  - Services can be swapped without touching route code

Usage in a route:
    from app.dependencies import get_schema_intel, get_qb_engine

    @router.get("/foo")
    async def foo(
        schema: SchemaIntelligenceService = Depends(get_schema_intel),
        qb: QuestionBackEngine         = Depends(get_qb_engine),
    ):
        ...

Usage in tests:
    def override_schema():
        return MockSchemaIntelligenceService()

    app.dependency_overrides[get_schema_intel] = override_schema
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_session


# ── Schema Intelligence ────────────────────────────────────────────────────────

def get_schema_intel():
    """
    Return the SchemaIntelligenceService singleton.
    Initialized at startup; read-only at request time.
    """
    from .services.schema_intelligence_service import get_schema_intelligence
    return get_schema_intelligence()


# ── Question-Back Engine ───────────────────────────────────────────────────────

def get_qb_engine():
    """Return the QuestionBackEngine singleton."""
    from .services.question_back_engine import get_question_back_engine
    return get_question_back_engine()


# ── Execution Policy Engine ────────────────────────────────────────────────────

def get_policy_engine():
    """Return the ExecutionPolicyEngine singleton."""
    from .services.execution_policy_engine import get_execution_policy_engine
    return get_execution_policy_engine()


# ── Clarification Engine ───────────────────────────────────────────────────────

async def get_clarification_engine_dep(db: AsyncSession = Depends(get_session)):
    """
    Return a ClarificationEngine scoped to the current DB session.
    This is async because the engine may need a DB reference.
    """
    from .services.clarification_engine import get_clarification_engine
    return await get_clarification_engine(db)


# ── Decision Arbiter ───────────────────────────────────────────────────────────

def get_decision_arbiter():
    """Return the DecisionArbiter singleton."""
    from .services.decision_arbiter import DecisionArbiter
    _key = "_arbiter_instance"
    import app.services.decision_arbiter as _mod
    if getattr(_mod, _key, None) is None:
        setattr(_mod, _key, DecisionArbiter())
    return getattr(_mod, _key)


# ── Progress Tracker Manager ───────────────────────────────────────────────────

def get_progress_tracker_manager():
    """Return the global ProgressTrackerManager."""
    from .services import progress_tracker_manager
    return progress_tracker_manager


# ── Cancellation Manager ───────────────────────────────────────────────────────

def get_cancellation_manager():
    """Return the global CancellationManager."""
    from .services import cancellation_manager
    return cancellation_manager


# ── Prompt Injection Guardian (per-request, stateless) ────────────────────────

def get_injection_guardian():
    """Return a configured PromptInjectionGuardian (cheap to construct)."""
    from .services.prompt_injection_guardian import (
        GuardianConfig,
        InjectionRiskLevel,
        PromptInjectionGuardian,
    )
    return PromptInjectionGuardian(
        config=GuardianConfig(
            block_threshold=InjectionRiskLevel.HIGH,
            sanitize_threshold=InjectionRiskLevel.MEDIUM,
            enable_pattern_detection=True,
            enable_encoding_detection=True,
            enable_semantic_analysis=True,
        )
    )


# ── Advanced SQL Generator ─────────────────────────────────────────────────────

def get_sql_generator():
    """Return the AdvancedSQLGenerator singleton."""
    from .services.advanced_sql_generator import get_advanced_sql_generator
    return get_advanced_sql_generator()


# ── Auto Retry Executor ────────────────────────────────────────────────────────

def get_retry_executor():
    """Return the AutoRetryExecutor singleton."""
    from .services.auto_retry_logic import get_auto_retry_executor
    return get_auto_retry_executor()


# ── Semantic Concept Extractor ─────────────────────────────────────────────────

def get_concept_extractor():
    """Return the SemanticConceptExtractor singleton."""
    from .services.semantic_concept_extractor import get_concept_extractor as _get
    return _get()


# ── Query Context Extractor ────────────────────────────────────────────────────

def get_query_ctx_extractor():
    """Return the QueryContextExtractor singleton."""
    from .services.query_context_extractor import get_query_context_extractor
    return get_query_context_extractor()


# ── Result Interpreter ─────────────────────────────────────────────────────────

def get_result_interp():
    """Return the ResultInterpreter singleton."""
    from .services.result_interpreter import get_result_interpreter
    return get_result_interpreter()


# ── Database Adapter ───────────────────────────────────────────────────────────

def get_db_adapter():
    """Return the global DatabaseAdapter singleton."""
    from .services.database_adapter import get_global_adapter
    return get_global_adapter()


# ── Task Queue helpers ─────────────────────────────────────────────────────────

def get_task_dispatcher():
    """Return the dispatch_query_task callable (for injection / mocking)."""
    from .services.task_queue import dispatch_query_task
    return dispatch_query_task


# ── Convenience bundle ─────────────────────────────────────────────────────────
# Named tuple of all core infrastructure providers, useful for service-layer
# classes that need multiple dependencies injected at once.

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class CoreDependencies:
    """
    Bundle of the most commonly needed infrastructure singletons.

    Construct via the FastAPI dependency:
        core: CoreDependencies = Depends(get_core_deps)
    """
    schema_intel: Any
    qb_engine: Any
    policy_engine: Any
    progress_manager: Any
    cancellation_manager: Any


def get_core_deps(
    schema_intel=Depends(get_schema_intel),
    qb_engine=Depends(get_qb_engine),
    policy_engine=Depends(get_policy_engine),
    progress_manager=Depends(get_progress_tracker_manager),
    cancel_manager=Depends(get_cancellation_manager),
) -> CoreDependencies:
    """
    Inject the CoreDependencies bundle.

    Usage:
        @router.post("/query")
        async def query(core: CoreDependencies = Depends(get_core_deps)):
            core.schema_intel.resolve(...)
    """
    return CoreDependencies(
        schema_intel=schema_intel,
        qb_engine=qb_engine,
        policy_engine=policy_engine,
        progress_manager=progress_manager,
        cancellation_manager=cancel_manager,
    )
