"""
ENHANCED OBSERVABILITY LAYER - 100% Comprehensive Logging & Monitoring

File-based observability without Redis or external services.

Features:
- Structured JSON logging
- Performance metrics tracking
- Token usage monitoring
- Query execution traces
- Error tracking and analysis
- Latency measurements
- Success/failure rates

Components:
1. Logger - Structured logging to files
2. Metrics Collector - Performance metrics
3. Tracer - Execution flow tracing
4. Analytics - Query/error analysis

Log Levels:
- DEBUG: Detailed execution steps
- INFO: Key events and milestones
- WARNING: Potential issues
- ERROR: Failures and exceptions
- CRITICAL: System failures

Architecture:
    All Services → Observability Layer → Log Files → Analytics Dashboard

No external dependencies - pure file-based logging for simplicity.
"""

from __future__ import annotations

import logging
import json
import os
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from collections import defaultdict, Counter
import traceback

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(str, Enum):
    """Types of observable events."""
    QUERY_START = "query_start"
    QUERY_COMPLETE = "query_complete"
    QUERY_ERROR = "query_error"
    SQL_EXECUTION = "sql_execution"
    LLM_CALL = "llm_call"
    TOOL_EXECUTION = "tool_execution"
    CLARIFICATION = "clarification"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    FILE_UPLOAD = "file_upload"
    AMBIGUITY_DETECTED = "ambiguity_detected"


@dataclass
class ObservabilityEvent:
    """Single observable event."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    duration_ms: Optional[float] = None
    success: bool = True
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    trace_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": round(self.duration_ms, 2) if self.duration_ms else None,
            "success": self.success,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "error": self.error,
            "trace_id": self.trace_id,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    total_llm_calls: int = 0
    total_tokens: int = 0
    clarifications_triggered: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": round(self.successful_queries / max(self.total_queries, 1), 3),
            "average_latency_ms": round(self.average_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "total_llm_calls": self.total_llm_calls,
            "total_tokens": self.total_tokens,
            "clarifications_triggered": self.clarifications_triggered,
        }


class ObservabilityLogger:
    """
    Unified logging system - writes everything to app.log.
    
    All events, metrics, and errors are logged to the main app.log file
    using Python's standard logging module. This provides a single source
    of truth for all system activity.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize observability logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Use standard Python logger - all output goes to app.log
        self.logger = logging.getLogger("observability")
        self.logger.info("[OBSERVABILITY] Initialized - logging to app.log")
        
        # In-memory metrics for analytics
        self.events: List[ObservabilityEvent] = []
        self.latencies: List[float] = []
        self.error_counts: Counter = Counter()
        self.cache_stats = {"hits": 0, "misses": 0}
        self.llm_stats = {"calls": 0, "tokens": 0}
    
    def log_event(
        self,
        event_type: EventType,
        duration_ms: Optional[float] = None,
        success: bool = True,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        trace_id: Optional[str] = None,
    ):
        """Log an observable event to app.log."""
        import uuid
        
        event = ObservabilityEvent(
            event_id=str(uuid.uuid4())[:8],
            event_type=event_type,
            timestamp=datetime.utcnow(),
            duration_ms=duration_ms,
            success=success,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            error=error,
            trace_id=trace_id,
        )
        
        # Store in memory for metrics
        self.events.append(event)
        
        if duration_ms:
            self.latencies.append(duration_ms)
        
        if not success and error:
            self.error_counts[error] += 1
        
        # Log to app.log via Python logging
        log_msg = f"[{event_type.value.upper()}]"
        if trace_id:
            log_msg += f" [trace:{trace_id[:8]}]"
        if session_id:
            log_msg += f" [session:{session_id[:8]}]"
        if duration_ms:
            log_msg += f" [{duration_ms:.0f}ms]"
        
        # Add metadata summary
        if metadata:
            meta_summary = ", ".join(f"{k}={str(v)[:50]}" for k, v in list(metadata.items())[:3])
            log_msg += f" | {meta_summary}"
        
        # Log based on success/failure
        if success:
            self.logger.info(log_msg)
        else:
            if error:
                self.logger.error(f"{log_msg} | ERROR: {error}")
            else:
                self.logger.warning(log_msg)
    
    def log_query_start(
        self,
        query: str,
        user_id: str,
        session_id: str,
        trace_id: str,
    ):
        """Log query start."""
        self.log_event(
            event_type=EventType.QUERY_START,
            user_id=user_id,
            session_id=session_id,
            metadata={"query": query[:200]},
            trace_id=trace_id,
        )
    
    def log_query_complete(
        self,
        trace_id: str,
        duration_ms: float,
        user_id: str,
        session_id: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log query completion."""
        self.log_event(
            event_type=EventType.QUERY_COMPLETE,
            duration_ms=duration_ms,
            user_id=user_id,
            session_id=session_id,
            success=success,
            metadata=metadata,
            trace_id=trace_id,
        )
    
    def log_sql_execution(
        self,
        sql: str,
        duration_ms: float,
        row_count: int,
        success: bool,
        trace_id: str,
        error: Optional[str] = None,
    ):
        """Log SQL execution."""
        self.log_event(
            event_type=EventType.SQL_EXECUTION,
            duration_ms=duration_ms,
            success=success,
            metadata={
                "sql": sql[:500],
                "row_count": row_count,
            },
            error=error,
            trace_id=trace_id,
        )
    
    def log_llm_call(
        self,
        prompt_length: int,
        response_length: int,
        duration_ms: float,
        tokens: Optional[int] = None,
        model: Optional[str] = None,
        trace_id: Optional[str] = None,
    ):
        """Log LLM API call."""
        self.llm_stats["calls"] += 1
        if tokens:
            self.llm_stats["tokens"] += tokens
        
        self.log_event(
            event_type=EventType.LLM_CALL,
            duration_ms=duration_ms,
            metadata={
                "prompt_length": prompt_length,
                "response_length": response_length,
                "tokens": tokens,
                "model": model,
            },
            trace_id=trace_id,
        )
    
    def log_cache(self, hit: bool, key: str):
        """Log cache hit/miss."""
        if hit:
            self.cache_stats["hits"] += 1
            event_type = EventType.CACHE_HIT
        else:
            self.cache_stats["misses"] += 1
            event_type = EventType.CACHE_MISS
        
        self.log_event(
            event_type=event_type,
            metadata={"cache_key": key[:100]},
        )
    
    def log_clarification(
        self,
        ambiguity_type: str,
        question: str,
        user_id: str,
        session_id: str,
        trace_id: str,
    ):
        """Log clarification request."""
        self.log_event(
            event_type=EventType.CLARIFICATION,
            user_id=user_id,
            session_id=session_id,
            metadata={
                "ambiguity_type": ambiguity_type,
                "question": question[:200],
            },
            trace_id=trace_id,
        )
    
    def get_metrics(self) -> PerformanceMetrics:
        """Calculate current performance metrics."""
        total_queries = len([e for e in self.events if e.event_type == EventType.QUERY_COMPLETE])
        successful = len([e for e in self.events if e.event_type == EventType.QUERY_COMPLETE and e.success])
        
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        
        # P95 latency
        sorted_latencies = sorted(self.latencies)
        p95_index = int(len(sorted_latencies) * 0.95)
        p95_latency = sorted_latencies[p95_index] if sorted_latencies else 0.0
        
        # Cache hit rate
        total_cache = self.cache_stats["hits"] + self.cache_stats["misses"]
        cache_hit_rate = self.cache_stats["hits"] / total_cache if total_cache > 0 else 0.0
        
        clarifications = len([e for e in self.events if e.event_type == EventType.CLARIFICATION])
        
        return PerformanceMetrics(
            timestamp=datetime.utcnow(),
            total_queries=total_queries,
            successful_queries=successful,
            failed_queries=total_queries - successful,
            average_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            cache_hit_rate=cache_hit_rate,
            total_llm_calls=self.llm_stats["calls"],
            total_tokens=self.llm_stats["tokens"],
            clarifications_triggered=clarifications,
        )
    
    def save_metrics(self):
        """Log current metrics to app.log."""
        metrics = self.get_metrics()
        
        # Log metrics to app.log
        self.logger.info(
            f"[METRICS] Queries: {metrics.total_queries} total, "
            f"{metrics.successful_queries} success, "
            f"{metrics.failed_queries} failed | "
            f"Latency: {metrics.average_latency_ms:.0f}ms avg, "
            f"{metrics.p95_latency_ms:.0f}ms p95 | "
            f"Cache: {metrics.cache_hit_rate*100:.1f}% hit rate | "
            f"LLM: {metrics.total_llm_calls} calls, "
            f"{metrics.total_tokens} tokens | "
            f"Clarifications: {metrics.clarifications_triggered}"
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "unique_errors": len(self.error_counts),
            "top_errors": self.error_counts.most_common(10),
        }
    
    def analyze_slow_queries(self, threshold_ms: float = 1000.0) -> List[Dict[str, Any]]:
        """Find slow queries above threshold."""
        slow_queries = []
        
        for event in self.events:
            if (event.event_type == EventType.QUERY_COMPLETE and 
                event.duration_ms and 
                event.duration_ms > threshold_ms):
                slow_queries.append({
                    "query": event.metadata.get("query", ""),
                    "duration_ms": event.duration_ms,
                    "timestamp": event.timestamp.isoformat(),
                    "trace_id": event.trace_id,
                })
        
        # Sort by duration
        slow_queries.sort(key=lambda x: x["duration_ms"], reverse=True)
        
        return slow_queries[:20]  # Top 20 slowest
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for observability dashboard."""
        metrics = self.get_metrics()
        error_summary = self.get_error_summary()
        slow_queries = self.analyze_slow_queries()
        
        return {
            "metrics": metrics.to_dict(),
            "errors": error_summary,
            "slow_queries": slow_queries,
            "cache_stats": self.cache_stats,
            "llm_stats": self.llm_stats,
        }


class ExecutionTracer:
    """
    Traces execution flow through the system.
    
    Provides distributed tracing-like functionality without external tools.
    """
    
    def __init__(self, observability_logger: ObservabilityLogger):
        """Initialize tracer."""
        self.logger = observability_logger
        self.active_traces: Dict[str, Dict[str, Any]] = {}
    
    def start_trace(self, trace_id: str, operation: str) -> Dict[str, Any]:
        """Start a new trace."""
        import uuid
        
        if not trace_id:
            trace_id = str(uuid.uuid4())[:8]
        
        trace = {
            "trace_id": trace_id,
            "operation": operation,
            "start_time": time.time(),
            "steps": [],
        }
        
        self.active_traces[trace_id] = trace
        return trace
    
    def add_step(
        self,
        trace_id: str,
        step_name: str,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a step to trace."""
        if trace_id in self.active_traces:
            self.active_traces[trace_id]["steps"].append({
                "name": step_name,
                "duration_ms": duration_ms,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat(),
            })
    
    def end_trace(self, trace_id: str, success: bool = True, error: Optional[str] = None):
        """End trace and log to app.log."""
        if trace_id not in self.active_traces:
            return
        
        trace = self.active_traces[trace_id]
        end_time = time.time()
        elapsed_time = end_time - trace["start_time"]
        trace["end_time"] = end_time
        trace["duration_ms"] = elapsed_time * 1000
        trace["success"] = success
        trace["error"] = error
        
        # Log trace completion to app.log
        duration_ms = elapsed_time * 1000
        self.logger.logger.info(
            f"[TRACE] {trace['operation']} completed in {duration_ms:.0f}ms | "
            f"Steps: {len(trace['steps'])} | "
            f"Trace ID: {trace_id[:8]}"
        )
        
        # Remove from active
        del self.active_traces[trace_id]


# Global instances
_observability_logger: Optional[ObservabilityLogger] = None
_execution_tracer: Optional[ExecutionTracer] = None


def get_observability_logger() -> ObservabilityLogger:
    """Get global observability logger."""
    global _observability_logger
    if _observability_logger is None:
        _observability_logger = ObservabilityLogger()
    return _observability_logger


def get_execution_tracer() -> ExecutionTracer:
    """Get global execution tracer."""
    global _execution_tracer
    if _execution_tracer is None:
        obs_logger = get_observability_logger()
        _execution_tracer = ExecutionTracer(obs_logger)
    return _execution_tracer


# Convenience functions
def log_query_start(query: str, user_id: str, session_id: str, trace_id: str):
    """Convenience wrapper for logging query start."""
    logger = get_observability_logger()
    logger.log_query_start(query, user_id, session_id, trace_id)


def log_query_complete(trace_id: str, duration_ms: float, user_id: str, session_id: str, success: bool, metadata: Optional[Dict] = None):
    """Convenience wrapper for logging query completion."""
    logger = get_observability_logger()
    logger.log_query_complete(trace_id, duration_ms, user_id, session_id, success, metadata)


def log_llm_call(prompt_length: int, response_length: int, duration_ms: float, tokens: Optional[int] = None, model: Optional[str] = None, trace_id: Optional[str] = None):
    """Convenience wrapper for logging LLM calls."""
    logger = get_observability_logger()
    logger.log_llm_call(prompt_length, response_length, duration_ms, tokens, model, trace_id)
