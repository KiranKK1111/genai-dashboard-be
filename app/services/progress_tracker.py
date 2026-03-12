"""
Progress Tracker - Real-time query progress tracking.

Dynamic progress system: each query path declares its own step plan at
runtime. Percentages are auto-computed from position in the plan.
Each step carries: step_id, label, status, percentage, duration_ms, metadata.

Step lifecycle:
  tracker.set_plan(["routing", "sql_generation", ...])  # declare plan upfront
  tracker.update("routing", "Routing: SQL query detected")             # status=running
  tracker.done("routing", "Routed to SQL engine")                      # status=done
  tracker.skip("schema_discovery", "Schema cached (12 tables)")        # status=skipped
  tracker.complete("Returned 42 rows in 0.3s")                         # step=complete

SSE payload per step:
  { step, label, status, percentage, duration_ms, timestamp, metadata }
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backward-compatible enum (kept so existing imports don't break)
# ---------------------------------------------------------------------------

class ProgressStep(str, Enum):
    """Legacy step identifiers — kept for backward compatibility only.
    New code should pass free-form strings to tracker.update()."""
    STARTING = "starting"
    VALIDATING = "validating"
    ROUTING = "routing"
    ANALYZING_INTENT = "analyzing_intent"
    DISCOVERING_SCHEMA = "discovering_schema"
    MATCHING_TABLES = "matching_tables"
    GENERATING_SQL = "generating_sql"
    VALIDATING_SQL = "validating_sql"
    EXECUTING_QUERY = "executing_query"
    PROCESSING_RESULTS = "processing_results"
    GENERATING_VISUALIZATION = "generating_visualization"
    FORMATTING_RESPONSE = "formatting_response"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Pre-defined step plans for each query path
# ---------------------------------------------------------------------------

# Each plan is an ordered list of (step_id, display_label) tuples.
# Percentages are evenly distributed across the plan (0 → 95); complete = 100.

PLAN_SQL = [
    ("routing",            "Routing query"),
    ("schema_discovery",   "Loading schema"),
    ("table_matching",     "Matching tables"),
    ("sql_generation",     "Generating SQL"),
    ("sql_validation",     "Validating SQL"),
    ("query_execution",    "Executing query"),
    ("result_processing",  "Processing results"),
    ("visualization",      "Building visualizations"),
    ("formatting",         "Formatting response"),
]

PLAN_FILE_STRUCTURED = [
    ("file_upload",      "Uploading file"),
    ("file_parsing",     "Parsing file"),
    ("pandas_query",     "Querying data"),
    ("insight_analysis", "Analysing insights"),
    ("visualization",    "Building visualizations"),
    ("formatting",       "Formatting response"),
]

PLAN_FILE_UNSTRUCTURED = [
    ("file_upload",         "Uploading file"),
    ("embedding_extract",   "Extracting text chunks"),
    ("intent_analysis",     "Analysing intent"),
    ("summary_generation",  "Generating summary"),
    ("insight_analysis",    "Analysing insights"),
    ("formatting",          "Formatting response"),
]

PLAN_FILE_LOOKUP_STRUCTURED = [
    ("routing",        "Routing to file"),
    ("file_lookup",    "Loading file"),
    ("pandas_query",   "Querying data"),
    ("visualization",  "Building visualizations"),
    ("formatting",     "Formatting response"),
]

PLAN_FILE_LOOKUP_UNSTRUCTURED = [
    ("routing",           "Routing to file"),
    ("chunk_retrieval",   "Retrieving relevant chunks"),
    ("context_analysis",  "Analysing context"),
    ("answer_generation", "Generating answer"),
    ("formatting",        "Formatting response"),
]

PLAN_CHAT = [
    ("routing",             "Routing query"),
    ("response_generation", "Generating response"),
]

PLAN_VARIATIONS = [
    ("routing",             "Routing query"),
    ("response_generation", "Generating variations"),
    ("formatting",          "Formatting response"),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProgressUpdate:
    """A single progress update emitted as an SSE event."""
    step: str                          # Free-form step ID (e.g. "sql_generation")
    label: str                         # Human-readable description
    timestamp: datetime
    status: str = "running"            # running | done | skipped | error
    percentage: Optional[int] = None   # 0-100 (auto-computed from plan position)
    duration_ms: Optional[float] = None  # Time since previous step in ms
    metadata: Optional[Dict] = None

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "label": self.label,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "percentage": self.percentage,
            "duration_ms": round(self.duration_ms, 1) if self.duration_ms is not None else None,
            "metadata": self.metadata or {},
        }


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class ProgressTracker:
    """
    Tracks query execution progress for real-time SSE streaming.

    Usage:
        tracker = progress_tracker_manager.start_tracking(message_id)

        # Declare the pipeline steps upfront (auto-computes percentages)
        tracker.set_plan(PLAN_SQL)

        # Emit steps as they execute
        tracker.update("routing", "Detected SQL query")
        tracker.done("routing", "Routed to SQL engine (0.02s)")
        tracker.skip("schema_discovery", "Schema cached — 12 tables")
        tracker.update("sql_generation", "Writing SQL for customers table")
        tracker.done("sql_generation", "SQL ready", metadata={"sql_preview": "SELECT *"})
        tracker.complete("Returned 42 rows in 0.3s")
    """

    def __init__(self, message_id: str):
        self.message_id = message_id
        self.updates: List[ProgressUpdate] = []
        self.current_step: Optional[str] = None
        self.is_complete = False
        self.is_error = False
        self.error_message: Optional[str] = None
        self.started_at = datetime.now(timezone.utc)
        self.completed_at: Optional[datetime] = None

        # Dynamic plan: step_id → percentage
        self._step_pct: Dict[str, int] = {}
        self._last_step_time: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Plan management
    # ------------------------------------------------------------------

    def set_plan(self, plan: List[Tuple[str, str]]):
        """Declare the ordered pipeline steps.

        Args:
            plan: List of (step_id, display_label) tuples. Percentages are
                  evenly spread from 0 to 95; 'complete' is always 100.

        Example:
            tracker.set_plan(PLAN_SQL)
        """
        n = len(plan)
        if n == 0:
            return
        for i, (step_id, _) in enumerate(plan):
            pct = int(i / max(n - 1, 1) * 95) if n > 1 else 0
            self._step_pct[step_id] = pct
        # Terminal steps always 100
        self._step_pct["complete"] = 100
        self._step_pct["error"] = 100
        self._step_pct["cancelled"] = 100
        logger.debug("[PROGRESS] Plan set for %s: %s", self.message_id, list(self._step_pct.keys()))

    # ------------------------------------------------------------------
    # Core update method
    # ------------------------------------------------------------------

    def update(
        self,
        step: Union[str, ProgressStep],
        label: str,
        metadata: Optional[Dict] = None,
        status: str = "running",
    ):
        """Emit a progress update.

        Args:
            step:     Step identifier (free-form string or ProgressStep enum).
            label:    Human-readable description shown to the user.
            metadata: Optional structured data (table names, row counts, etc.)
            status:   "running" | "done" | "skipped" | "error"
        """
        step_id = step.value if isinstance(step, ProgressStep) else step

        now = datetime.now(timezone.utc)
        duration_ms: Optional[float] = None
        if self._last_step_time is not None:
            duration_ms = (now - self._last_step_time).total_seconds() * 1000
        self._last_step_time = now

        pct = self._step_pct.get(step_id)

        update = ProgressUpdate(
            step=step_id,
            label=label,
            timestamp=now,
            status=status,
            percentage=pct,
            duration_ms=duration_ms,
            metadata=metadata,
        )
        self.updates.append(update)
        self.current_step = step_id
        logger.info("[PROGRESS] %s → %s (%s): %s", self.message_id, step_id, status, label)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def done(self, step: Union[str, ProgressStep], label: str, metadata: Optional[Dict] = None):
        """Mark a step as completed successfully."""
        self.update(step, label, metadata=metadata, status="done")

    def skip(self, step: Union[str, ProgressStep], reason: str, metadata: Optional[Dict] = None):
        """Mark a step as skipped (e.g. schema cached, SQL reused)."""
        self.update(step, reason, metadata=metadata, status="skipped")

    def complete(self, message: str = "Query completed successfully", metadata: Optional[Dict] = None):
        """Mark the whole query as complete (percentage = 100)."""
        self.update("complete", message, metadata=metadata, status="done")
        self.is_complete = True
        self.completed_at = datetime.now(timezone.utc)
        duration = (self.completed_at - self.started_at).total_seconds()
        logger.info("[PROGRESS] %s completed in %.2fs", self.message_id, duration)

    def error(self, message: str, metadata: Optional[Dict] = None):
        """Mark as failed."""
        self.update("error", message, metadata=metadata, status="error")
        self.is_error = True
        self.error_message = message
        self.completed_at = datetime.now(timezone.utc)
        logger.error("[PROGRESS] %s error: %s", self.message_id, message)

    def cancelled(self, message: str = "Query cancelled by user"):
        """Mark as cancelled."""
        self.update("cancelled", message, status="done")
        self.is_complete = True
        self.completed_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_latest_update(self) -> Optional[ProgressUpdate]:
        return self.updates[-1] if self.updates else None

    def get_all_updates(self) -> List[ProgressUpdate]:
        return self.updates.copy()

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "current_step": self.current_step,
            "is_complete": self.is_complete,
            "is_error": self.is_error,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "updates": [u.to_dict() for u in self.updates],
        }


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class ProgressTrackerManager:
    """Singleton that manages progress trackers for all ongoing queries."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._trackers = {}   # Dict[str, ProgressTracker]
            cls._instance._subscribers = {}  # Dict[str, List[asyncio.Queue]]
        return cls._instance

    def start_tracking(self, message_id: str, initial_label: str = "Starting") -> ProgressTracker:
        tracker = ProgressTracker(message_id)
        tracker.update("starting", initial_label, status="done")
        self._trackers[message_id] = tracker
        logger.info("[PROGRESS] Started tracking for message %s", message_id)
        return tracker

    def get_tracker(self, message_id: str) -> Optional[ProgressTracker]:
        return self._trackers.get(message_id)

    def remove_tracker(self, message_id: str):
        if message_id in self._trackers:
            del self._trackers[message_id]
            logger.debug("[PROGRESS] Removed tracker for message %s", message_id)

    async def subscribe(self, message_id: str) -> asyncio.Queue:
        if message_id not in self._subscribers:
            self._subscribers[message_id] = []
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[message_id].append(queue)
        tracker = self._trackers.get(message_id)
        if tracker:
            for update in tracker.get_all_updates():
                await queue.put(update)
        logger.debug("[PROGRESS] New subscriber for message %s", message_id)
        return queue

    async def publish_update(self, message_id: str, update: ProgressUpdate):
        if message_id in self._subscribers:
            for queue in self._subscribers[message_id]:
                try:
                    await queue.put(update)
                except Exception as exc:
                    logger.error("[PROGRESS] Error publishing to subscriber: %s", exc)

    def unsubscribe(self, message_id: str, queue: asyncio.Queue):
        if message_id in self._subscribers:
            try:
                self._subscribers[message_id].remove(queue)
                if not self._subscribers[message_id]:
                    del self._subscribers[message_id]
            except ValueError:
                pass

    def cleanup_completed(self, max_age_seconds: int = 300):
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
        to_remove = [
            mid for mid, t in self._trackers.items()
            if t.completed_at and t.completed_at < cutoff
        ]
        for mid in to_remove:
            del self._trackers[mid]
            self._subscribers.pop(mid, None)
        if to_remove:
            logger.info("[PROGRESS] Cleaned up %d completed trackers", len(to_remove))

    def get_active_count(self) -> int:
        return sum(1 for t in self._trackers.values() if not t.is_complete)


# Global singleton
progress_tracker_manager = ProgressTrackerManager()
