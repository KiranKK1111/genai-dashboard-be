"""
Schema Change Detector.

Runs a lightweight background poll every N seconds (default 60) that
compares information_schema table/column counts against the last snapshot.
When a change is detected it invalidates:
  - SchemaIntelligenceService (forces re-introspection on next use)
  - SemanticSchemaCatalog (forces catalog reload on next query)

Design:
  - No heavy queries — only COUNT(*) on information_schema.tables /
    information_schema.columns (indexed, sub-millisecond).
  - Integrated into the FastAPI lifespan via asyncio.create_task().
  - Graceful shutdown on cancellation.

Usage (main.py startup):
    from app.services.schema_change_detector import start_schema_change_detector
    asyncio.create_task(start_schema_change_detector(SessionLocal))
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Snapshot ──────────────────────────────────────────────────────────────────

class _SchemaSnapshot:
    """Lightweight fingerprint of current schema state."""
    __slots__ = ("table_count", "column_count", "checksum")

    def __init__(self, table_count: int, column_count: int) -> None:
        self.table_count = table_count
        self.column_count = column_count
        self.checksum = (table_count, column_count)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _SchemaSnapshot):
            return False
        return self.checksum == other.checksum

    def __repr__(self) -> str:
        return f"<SchemaSnapshot tables={self.table_count} cols={self.column_count}>"


async def _take_snapshot(db_session, schema_name: str) -> Optional[_SchemaSnapshot]:
    """Query information_schema for current table/column counts."""
    from sqlalchemy import text
    try:
        t_result = await db_session.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = :schema AND table_type = 'BASE TABLE'"
            ),
            {"schema": schema_name},
        )
        table_count: int = t_result.scalar() or 0

        c_result = await db_session.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_schema = :schema"
            ),
            {"schema": schema_name},
        )
        column_count: int = c_result.scalar() or 0

        return _SchemaSnapshot(table_count, column_count)
    except Exception as exc:
        logger.debug("[SCHEMA DETECTOR] Snapshot failed: %s", exc)
        return None


def _invalidate_caches() -> None:
    """Reset all schema-derived caches so they rebuild on next access."""
    invalidated = []

    # 1. SchemaIntelligenceService singleton
    try:
        from .schema_intelligence_service import get_schema_intelligence
        svc = get_schema_intelligence()
        svc.initialized = False
        svc.table_profiles.clear()
        svc.column_profiles.clear()
        svc.join_graph.clear()
        invalidated.append("SchemaIntelligenceService")
    except Exception as exc:
        logger.debug("[SCHEMA DETECTOR] Could not reset SchemaIntelligenceService: %s", exc)

    # 2. SemanticSchemaCatalog (global catalog instance)
    try:
        from .semantic_schema_catalog import get_catalog
        catalog = get_catalog()
        # Force reload by clearing tables dict — catalog.initialize() will
        # repopulate on the next query.
        catalog.tables.clear()
        invalidated.append("SemanticSchemaCatalog")
    except Exception as exc:
        logger.debug("[SCHEMA DETECTOR] Could not reset SemanticSchemaCatalog: %s", exc)

    # 3. ValueBasedColumnGrounder (if initialized)
    try:
        from .value_based_column_grounding import ValueBasedColumnGrounder
        # The grounder caches per-table profiles; clearing forces re-profile.
        if hasattr(ValueBasedColumnGrounder, "_instance") and ValueBasedColumnGrounder._instance:
            ValueBasedColumnGrounder._instance.column_profiles.clear()
            invalidated.append("ValueBasedColumnGrounder")
    except Exception as exc:
        logger.debug("[SCHEMA DETECTOR] Could not reset ValueBasedColumnGrounder: %s", exc)

    # 4. Value-scan cache in plan_first_sql_generator
    try:
        from .plan_first_sql_generator import clear_value_scan_cache
        clear_value_scan_cache()
        invalidated.append("ValueScanCache")
    except Exception as exc:
        logger.debug("[SCHEMA DETECTOR] Could not clear ValueScanCache: %s", exc)

    if invalidated:
        logger.info(
            "[SCHEMA DETECTOR] Schema change detected — invalidated: %s",
            ", ".join(invalidated),
        )


# ── Main loop ─────────────────────────────────────────────────────────────────

async def start_schema_change_detector(
    session_factory,
    schema_name: Optional[str] = None,
    poll_interval_sec: Optional[int] = None,
) -> None:
    """
    Long-running async task. Poll information_schema and invalidate caches
    when table/column counts change.

    Args:
        session_factory: SQLAlchemy async session factory (e.g. SessionLocal).
        schema_name:     Database schema to watch (defaults to settings.postgres_schema).
        poll_interval_sec: Override for settings.schema_change_poll_interval_sec.
    """
    from ..config import settings
    if schema_name is None:
        schema_name = settings.postgres_schema

    interval = poll_interval_sec or settings.schema_change_poll_interval_sec
    logger.info(
        "[SCHEMA DETECTOR] Started (schema=%s, interval=%ds)", schema_name, interval
    )

    last_snapshot: Optional[_SchemaSnapshot] = None

    while True:
        try:
            async with session_factory() as db:
                snapshot = await _take_snapshot(db, schema_name)

            if snapshot is None:
                # DB unavailable — skip this cycle silently
                pass
            elif last_snapshot is None:
                # First run — baseline, no invalidation needed
                last_snapshot = snapshot
                logger.debug("[SCHEMA DETECTOR] Baseline: %s", snapshot)
            elif snapshot != last_snapshot:
                logger.info(
                    "[SCHEMA DETECTOR] Change: %s -> %s",
                    last_snapshot, snapshot,
                )
                _invalidate_caches()
                last_snapshot = snapshot
            # else: no change — nothing to do

        except asyncio.CancelledError:
            logger.info("[SCHEMA DETECTOR] Stopped (cancelled)")
            return
        except Exception as exc:
            # Non-fatal — log and keep polling
            logger.warning("[SCHEMA DETECTOR] Poll error: %s", exc)

        await asyncio.sleep(interval)
