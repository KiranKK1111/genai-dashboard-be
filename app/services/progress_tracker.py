"""
Progress Tracker - Real-time query progress tracking.

This service tracks query execution progress and allows streaming updates to clients.
In production, this should use Redis Pub/Sub for real-time updates across workers.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class ProgressStep(str, Enum):
    """Query processing steps."""
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


@dataclass
class ProgressUpdate:
    """A single progress update."""
    step: ProgressStep
    label: str
    timestamp: datetime
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "step": self.step.value,
            "label": self.label,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }


class ProgressTracker:
    """
    Tracks query execution progress for real-time updates.
    
    Usage:
        tracker = progress_tracker.start_tracking(message_id)
        tracker.update(ProgressStep.ROUTING, "Determining query type...")
        tracker.update(ProgressStep.GENERATING_SQL, "Writing SQL query...", {"tables": ["users"]})
        tracker.complete("Query completed successfully")
    """
    
    def __init__(self, message_id: str):
        self.message_id = message_id
        self.updates: List[ProgressUpdate] = []
        self.current_step: Optional[ProgressStep] = None
        self.is_complete = False
        self.is_error = False
        self.error_message: Optional[str] = None
        self.started_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
    
    def update(self, step: ProgressStep, label: str, metadata: Optional[Dict] = None):
        """Add a progress update."""
        update = ProgressUpdate(
            step=step,
            label=label,
            timestamp=datetime.utcnow(),
            metadata=metadata
        )
        self.updates.append(update)
        self.current_step = step
        logger.info(f"[PROGRESS] {self.message_id} -> {step.value}: {label}")
    
    def complete(self, message: str = "Query completed successfully"):
        """Mark as complete."""
        self.update(ProgressStep.COMPLETE, message)
        self.is_complete = True
        self.completed_at = datetime.utcnow()
        duration = (self.completed_at - self.started_at).total_seconds()
        logger.info(f"[PROGRESS] {self.message_id} completed in {duration:.2f}s")
    
    def error(self, message: str, metadata: Optional[Dict] = None):
        """Mark as error."""
        self.update(ProgressStep.ERROR, message, metadata)
        self.is_error = True
        self.error_message = message
        self.completed_at = datetime.utcnow()
        logger.error(f"[PROGRESS] {self.message_id} error: {message}")
    
    def cancelled(self, message: str = "Query cancelled by user"):
        """Mark as cancelled."""
        self.update(ProgressStep.CANCELLED, message)
        self.is_complete = True
        self.completed_at = datetime.utcnow()
        logger.info(f"[PROGRESS] {self.message_id} cancelled")
    
    def get_latest_update(self) -> Optional[ProgressUpdate]:
        """Get the most recent update."""
        return self.updates[-1] if self.updates else None
    
    def get_all_updates(self) -> List[ProgressUpdate]:
        """Get all updates."""
        return self.updates.copy()
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "current_step": self.current_step.value if self.current_step else None,
            "is_complete": self.is_complete,
            "is_error": self.is_error,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "updates": [u.to_dict() for u in self.updates]
        }


class ProgressTrackerManager:
    """
    Manages progress trackers for all ongoing queries.
    
    Singleton pattern to ensure consistent state across the application.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._trackers = {}  # Dict[str, ProgressTracker]
            cls._instance._subscribers = {}  # Dict[str, List[asyncio.Queue]]
        return cls._instance
    
    def start_tracking(self, message_id: str, initial_label: str = "Starting query...") -> ProgressTracker:
        """Start tracking progress for a message."""
        tracker = ProgressTracker(message_id)
        tracker.update(ProgressStep.STARTING, initial_label)
        self._trackers[message_id] = tracker
        logger.info(f"[PROGRESS] Started tracking for message {message_id}")
        return tracker
    
    def get_tracker(self, message_id: str) -> Optional[ProgressTracker]:
        """Get tracker for a message."""
        return self._trackers.get(message_id)
    
    def remove_tracker(self, message_id: str):
        """Remove tracker after query completion (cleanup)."""
        if message_id in self._trackers:
            del self._trackers[message_id]
            logger.debug(f"[PROGRESS] Removed tracker for message {message_id}")
    
    async def subscribe(self, message_id: str) -> asyncio.Queue:
        """
        Subscribe to progress updates for a message.
        Returns a queue that will receive updates.
        """
        if message_id not in self._subscribers:
            self._subscribers[message_id] = []
        
        queue = asyncio.Queue()
        self._subscribers[message_id].append(queue)
        
        # Send existing updates to new subscriber
        tracker = self._trackers.get(message_id)
        if tracker:
            for update in tracker.get_all_updates():
                await queue.put(update)
        
        logger.debug(f"[PROGRESS] New subscriber for message {message_id}")
        return queue
    
    async def publish_update(self, message_id: str, update: ProgressUpdate):
        """Publish an update to all subscribers."""
        if message_id in self._subscribers:
            for queue in self._subscribers[message_id]:
                try:
                    await queue.put(update)
                except Exception as e:
                    logger.error(f"[PROGRESS] Error publishing to subscriber: {e}")
    
    def unsubscribe(self, message_id: str, queue: asyncio.Queue):
        """Remove a subscriber."""
        if message_id in self._subscribers:
            try:
                self._subscribers[message_id].remove(queue)
                if not self._subscribers[message_id]:
                    del self._subscribers[message_id]
            except ValueError:
                pass
    
    def cleanup_completed(self, max_age_seconds: int = 300):
        """Clean up completed trackers older than max_age_seconds."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(seconds=max_age_seconds)
        to_remove = []
        
        for message_id, tracker in self._trackers.items():
            if tracker.completed_at and tracker.completed_at < cutoff:
                to_remove.append(message_id)
        
        for message_id in to_remove:
            del self._trackers[message_id]
            # Also clean up subscribers
            if message_id in self._subscribers:
                del self._subscribers[message_id]
        
        if to_remove:
            logger.info(f"[PROGRESS] Cleaned up {len(to_remove)} completed trackers")
    
    def get_active_count(self) -> int:
        """Get count of active (incomplete) trackers."""
        return sum(1 for t in self._trackers.values() if not t.is_complete)


# Global singleton instance
progress_tracker_manager = ProgressTrackerManager()
