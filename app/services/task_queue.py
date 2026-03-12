"""
Background Task Queue — arq integration.

Design
------
When settings.arq_enabled=False (default):
  - dispatch_query_task() runs the query INLINE and returns the result normally.
  - Zero runtime dependency on arq or Redis.

When settings.arq_enabled=True:
  - Queries classified as "heavy" are offloaded to an arq worker process.
  - dispatch_query_task() enqueues the job and returns immediately with
    job_id.  The frontend polls /messages/{id}/progress while the worker
    runs and writes the result back to the DB.
  - "Light" queries still run inline (fast path).

Heavy query classification
--------------------------
A query is considered heavy when either:
  (a) One or more files are attached  OR
  (b) The pre-generated SQL string contains ≥ settings.heavy_query_join_threshold
      JOIN keywords.

arq Worker
----------
Start the worker as a separate process:
    arq app.services.task_queue.WorkerSettings

Or via Makefile / docker-compose (recommended).

Redis URL is read from settings.redis_url (default "redis://localhost:6379").

Usage (in mode handlers)
------------------------
    from .task_queue import dispatch_query_task, is_heavy_query

    if is_heavy_query(files=ctx.safe_files, sql_hint=None):
        # returns immediately; frontend polls progress endpoint
        wrapper = await dispatch_query_task(ctx, execute_fn)
    else:
        wrapper = await execute_fn(ctx)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Optional

from ..config import settings

logger = logging.getLogger(__name__)


# ── Heaviness classifier ───────────────────────────────────────────────────────

def is_heavy_query(
    *,
    files: list | None = None,
    sql_hint: str | None = None,
) -> bool:
    """
    Return True when the query warrants background processing.

    Args:
        files:     Uploaded files attached to this request.
        sql_hint:  Optional pre-generated SQL to count JOINs in.
    """
    if not settings.arq_enabled:
        return False  # arq disabled — always inline

    if files:  # any file upload is heavy
        return True

    if sql_hint:
        join_count = sql_hint.upper().count(" JOIN ")
        if join_count >= settings.heavy_query_join_threshold:
            return True

    return False


# ── Arq pool (lazily created) ──────────────────────────────────────────────────

_arq_pool = None


async def _get_arq_pool():
    """Return the shared arq Redis pool, creating it on first call."""
    global _arq_pool
    if _arq_pool is None:
        try:
            from arq import create_pool
            from arq.connections import RedisSettings

            _arq_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
            logger.info("[TASK_QUEUE] arq pool created (redis=%s)", settings.redis_url)
        except ImportError:
            logger.error("[TASK_QUEUE] arq not installed. Run: pip install arq")
            raise
    return _arq_pool


# ── Dispatcher ─────────────────────────────────────────────────────────────────

async def dispatch_query_task(
    ctx,                          # QueryContext
    execute_fn: Callable,         # async callable (ctx) → ResponseWrapper
    *,
    force_background: bool = False,
) -> Any:
    """
    Dispatch a query task.

    If arq is enabled and the query is heavy (or force_background=True),
    enqueues the job and returns a "processing" ResponseWrapper immediately.
    Otherwise runs execute_fn(ctx) inline and returns its result.

    Args:
        ctx:              QueryContext populated by the /query dispatcher.
        execute_fn:       The mode handler's core execution coroutine.
        force_background: Always background even for light queries.

    Returns:
        schemas.ResponseWrapper
    """
    heavy = force_background or is_heavy_query(
        files=getattr(ctx, "safe_files", None),
    )

    if not heavy or not settings.arq_enabled:
        # ── Inline (fast path) ────────────────────────────────────────────
        return await execute_fn(ctx)

    # ── Background (arq) path ─────────────────────────────────────────────
    from .. import schemas
    from ..helpers import current_timestamp

    try:
        pool = await _get_arq_pool()
        job = await pool.enqueue_job(
            "run_query_task",
            ctx.session_id,
            ctx.user_id if hasattr(ctx, "user_id") else str(ctx.current_user.id),
            ctx.query,
            ctx.mode,
            ctx.message_id,
            _job_id=ctx.message_id,  # use message_id as arq job ID for easy lookup
        )
        logger.info(
            "[TASK_QUEUE] Enqueued job %s for message %s", job.job_id, ctx.message_id
        )
    except Exception as exc:
        logger.warning(
            "[TASK_QUEUE] Enqueue failed (%s) — falling back to inline", exc
        )
        return await execute_fn(ctx)

    # Return an immediate "accepted" wrapper; frontend polls /progress
    return schemas.ResponseWrapper(
        success=True,
        response=schemas.StandardResponse(
            type="processing",
            intent=ctx.query,
            confidence=1.0,
            message="Your query is being processed in the background. "
                    "Track progress via the /messages/{id}/progress endpoint.",
            metadata={
                "background": True,
                "job_id": job.job_id,
                "message_id": ctx.message_id,
            },
        ),
        timestamp=current_timestamp(),
        original_query=ctx.query,
        session_id=ctx.session_id,
        message_id=ctx.message_id,
    )


# ── arq task functions ─────────────────────────────────────────────────────────

async def run_query_task(
    arq_ctx: dict,
    session_id: str,
    user_id: str,
    query: str,
    mode: str,
    message_id: str,
) -> dict:
    """
    arq worker entry-point.

    Re-creates the DB session, re-fetches the user/session, builds a
    QueryContext, then delegates to the correct mode handler exactly as the
    /query endpoint would.

    The result is written back to the Message row in the database so that
    /history and /messages/{id}/progress reflect the final state.
    """
    import uuid
    from datetime import datetime

    from sqlalchemy import select
    from sqlalchemy.orm.attributes import flag_modified

    logger.info(
        "[TASK_QUEUE] run_query_task starting: message=%s mode=%s", message_id, mode
    )

    try:
        from ..database import SessionLocal
        from .. import models
        from ..routes import DISPATCH_TABLE
        from ..routes._context import QueryContext

        async with SessionLocal() as db:
            # Re-fetch user and session
            user = (
                await db.execute(
                    select(models.User).where(models.User.id == uuid.UUID(user_id))
                )
            ).scalars().first()

            sess_obj = (
                await db.execute(
                    select(models.ChatSession).where(
                        models.ChatSession.id == uuid.UUID(session_id)
                    )
                )
            ).scalars().first()

            msg_obj = (
                await db.execute(
                    select(models.Message).where(
                        models.Message.id == uuid.UUID(message_id)
                    )
                )
            ).scalars().first()

            if not (user and sess_obj and msg_obj):
                logger.error(
                    "[TASK_QUEUE] Missing DB objects for message %s", message_id
                )
                return {"status": "error", "message_id": message_id}

            # Build a lightweight tracker stub (progress tracker may not exist in worker)
            class _TrackerStub:
                def complete(self, _): pass
                def error(self, _): pass
                def cancelled(self, _): pass

            ctx = QueryContext(
                query=query,
                mode=mode,
                session_id=session_id,
                message_id=message_id,
                safe_files=[],
                db=db,
                current_user=user,
                session=sess_obj,
                message=msg_obj,
                tracker=_TrackerStub(),
            )

            handler_cls = DISPATCH_TABLE.get(mode)
            if not handler_cls:
                raise ValueError(f"Unknown mode '{mode}'")

            wrapper = await handler_cls().execute(ctx)

            # Persist response
            response_dict = (
                wrapper.response.model_dump(mode="json")
                if hasattr(wrapper.response, "model_dump")
                else {}
            )
            response_dict["id"] = message_id
            msg_obj.response_type = getattr(wrapper.response, "type", "standard")
            msg_obj.response = response_dict
            msg_obj.responded_at = datetime.utcnow()
            flag_modified(msg_obj, "response")
            await db.commit()

            logger.info("[TASK_QUEUE] run_query_task complete: message=%s", message_id)
            return {"status": "complete", "message_id": message_id}

    except Exception as exc:
        logger.exception("[TASK_QUEUE] run_query_task failed: %s", exc)
        return {"status": "error", "message_id": message_id, "error": str(exc)}


# ── Worker settings ────────────────────────────────────────────────────────────

class WorkerSettings:
    """
    arq WorkerSettings.

    Start the worker:
        arq app.services.task_queue.WorkerSettings

    Environment variables consumed:
        REDIS_URL            (default redis://localhost:6379)
        ARQ_MAX_JOBS         (default 10)
    """
    functions = [run_query_task]
    redis_settings = None  # set dynamically below
    max_jobs: int = 10
    job_timeout: int = 300  # 5 minutes max per job

    def __init_subclass__(cls, **kwargs):  # pragma: no cover
        super().__init_subclass__(**kwargs)

    @classmethod
    def _configure(cls) -> None:
        """Apply settings.* values to this class (called at import time)."""
        try:
            from arq.connections import RedisSettings as _RS
            cls.redis_settings = _RS.from_dsn(settings.redis_url)
            cls.max_jobs = settings.arq_max_jobs
        except ImportError:
            pass


# Apply settings when arq is available
try:
    WorkerSettings._configure()
except Exception:
    pass


# ── Startup / shutdown helpers (called from main.py lifespan) ─────────────────

async def start_task_queue() -> None:
    """
    Initialise the arq Redis pool during application startup.
    No-op if arq is disabled or Redis is unavailable.
    """
    if not settings.arq_enabled:
        logger.debug("[TASK_QUEUE] arq disabled (ARQ_ENABLED=false)")
        return
    try:
        await _get_arq_pool()
        logger.info("[TASK_QUEUE] Background task queue ready")
    except Exception as exc:
        logger.warning("[TASK_QUEUE] Could not connect to Redis: %s", exc)


async def stop_task_queue() -> None:
    """Close the arq pool on shutdown."""
    global _arq_pool
    if _arq_pool is not None:
        try:
            await _arq_pool.close()
            _arq_pool = None
            logger.info("[TASK_QUEUE] arq pool closed")
        except Exception as exc:
            logger.debug("[TASK_QUEUE] Pool close error: %s", exc)
