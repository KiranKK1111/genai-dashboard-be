"""
PRIVACY & AUDIT LAYER (P2 - Zero Hardcoding)

PII Detection, redaction, retention policies, audit logging:
- Pattern-based and ML-based PII detection
- Configurable redaction rules
- Retention policy enforcement
- Immutable audit trail

All rules are config-driven, not hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Pattern
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class PiiType(str, Enum):
    """Types of PII we detect."""
    SSN = "ssn"
    PAYMENT_CARD_NUMBER = "payment_card_number"
    EMAIL = "email"
    PHONE = "phone"
    NAME = "name"
    ADDRESS = "address"
    PASSPORT = "passport"
    MEDICAL_RECORD = "medical_record"
    CUSTOM = "custom"


@dataclass
class PiiDetection:
    """Result of PII detection."""
    pii_type: PiiType
    value: str
    start_pos: int
    end_pos: int
    confidence: float  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.pii_type.value,
            "value": self.value,
            "start": self.start_pos,
            "end": self.end_pos,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class PrivacyConfig:
    """Configuration for privacy and PII handling."""
    
    # Redaction
    redact_pii_in_logs: bool = True
    redact_pii_in_responses: bool = True
    redaction_pattern: str = "[REDACTED:{}]"  # {pii_type} placeholder
    
    # Detection thresholds
    pii_confidence_threshold: float = 0.85
    
    # Retention
    retention_days_default: int = 90  # Delete after 90 days
    retention_days_audit_log: int = 365  # Audit logs for 1 year
    
    # PII patterns (regex, customizable)
    pii_patterns: Dict[PiiType, List[str]] = field(default_factory=lambda: {
        PiiType.SSN: [r"\b\d{3}-\d{2}-\d{4}\b"],
        PiiType.PAYMENT_CARD_NUMBER: [r"\b(?:\d[ \-]*?){13,19}\b"],
        PiiType.EMAIL: [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
        PiiType.PHONE: [r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"],
    })
    
    # ML-based PII detector (optional)
    ml_pii_detector: Optional[Callable] = None  # fn(text) → List[PiiDetection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "redact_pii_in_logs": self.redact_pii_in_logs,
            "redact_pii_in_responses": self.redact_pii_in_responses,
            "redaction_pattern": self.redaction_pattern,
            "pii_confidence_threshold": self.pii_confidence_threshold,
            "retention_days_default": self.retention_days_default,
            "retention_days_audit_log": self.retention_days_audit_log,
        }


@dataclass
class AuditLogEntry:
    """Single entry in audit log."""
    timestamp: datetime
    event_type: str  # "QUERY", "FILE_UPLOAD", "TOOL_EXECUTION", "ERROR"
    user_id: str
    session_id: str
    tool_used: Optional[str] = None
    request_text: Optional[str] = None  # Original (unredacted for audit)
    result_summary: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # PII detected in this event
    pii_detected: List[PiiDetection] = field(default_factory=list)
    
    # For audit trail integrity
    is_redacted: bool = False
    redacted_at: Optional[datetime] = None
    
    def to_dict(self, redact: bool = False) -> Dict[str, Any]:
        request = self.request_text
        result = self.result_summary
        
        if redact and self.is_redacted:
            # Replace with redaction note
            request = "[REDACTED: PII REMOVED]" if request else None
            result = "[REDACTED: PII REMOVED]" if result else None
        
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "tool_used": self.tool_used,
            "request": request,
            "result": result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "pii_detected_count": len(self.pii_detected),
            "is_redacted": self.is_redacted,
        }


class PiiDetector:
    """Detect PII in text."""
    
    def __init__(self, config: Optional[PrivacyConfig] = None):
        self.config = config or PrivacyConfig()
        
        # Compile regex patterns
        self.compiled_patterns: Dict[PiiType, List[Pattern]] = {}
        for pii_type, patterns in self.config.pii_patterns.items():
            self.compiled_patterns[pii_type] = [re.compile(p) for p in patterns]
        
        logger.info(f"PII Detector initialized with {len(self.compiled_patterns)} pattern types")
    
    def detect(self, text: str) -> List[PiiDetection]:
        """
        Detect PII in text.
        
        Returns:
            List of PiiDetection sorted by position
        """
        detections = []
        
        # Pattern-based detection
        for pii_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    detection = PiiDetection(
                        pii_type=pii_type,
                        value=match.group(0),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.95,  # Regex matches are high confidence
                    )
                    detections.append(detection)
        
        # ML-based detection (if available)
        if self.config.ml_pii_detector:
            try:
                ml_detections = self.config.ml_pii_detector(text)
                # Filter by confidence threshold
                detections.extend([
                    d for d in ml_detections
                    if d.confidence >= self.config.pii_confidence_threshold
                ])
            except Exception as e:
                logger.warning(f"ML PII detector error: {e}")
        
        # Sort by position
        detections.sort(key=lambda d: d.start_pos)
        
        # Deduplicate overlapping detections
        deduplicated = []
        for d in detections:
            # Check if overlaps with existing
            overlaps = any(
                not (d.end_pos <= existing.start_pos or d.start_pos >= existing.end_pos)
                for existing in deduplicated
            )
            if not overlaps:
                deduplicated.append(d)
        
        return deduplicated
    
    def redact(self, text: str, detections: Optional[List[PiiDetection]] = None) -> str:
        """
        Redact PII from text.
        
        Args:
            text: Original text
            detections: If None, will detect automatically
            
        Returns:
            Redacted text
        """
        if detections is None:
            detections = self.detect(text)
        
        if not detections:
            return text
        
        # Replace from end to start (so positions don't shift)
        redacted = text
        for detection in reversed(detections):
            redaction = self.config.redaction_pattern.format(detection.pii_type.value)
            redacted = (
                redacted[:detection.start_pos] +
                redaction +
                redacted[detection.end_pos:]
            )
        
        return redacted


class AuditLogger:
    """Immutable audit log."""
    
    def __init__(self, db, pii_detector: Optional[PiiDetector] = None, config: Optional[PrivacyConfig] = None):
        self.db = db
        self.config = config or PrivacyConfig()
        self.pii_detector = pii_detector or PiiDetector(self.config)
        logger.info("AuditLogger initialized")
    
    async def log_event(
        self,
        event_type: str,
        user_id: str,
        session_id: str,
        request_text: Optional[str] = None,
        tool_used: Optional[str] = None,
        result_summary: Optional[str] = None,
        error: Optional[str] = None,
        duration_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLogEntry:
        """
        Log an event to the audit trail.
        
        Detects PII and redacts if configured.
        """
        timestamp = datetime.utcnow()
        
        # Detect PII in request and result
        pii_detected = []
        if request_text:
            pii_detected.extend(self.pii_detector.detect(request_text))
        if result_summary:
            pii_detected.extend(self.pii_detector.detect(result_summary))
        
        # Create entry
        entry = AuditLogEntry(
            timestamp=timestamp,
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            tool_used=tool_used,
            request_text=request_text,
            result_summary=result_summary,
            error=error,
            duration_ms=duration_ms,
            metadata=metadata or {},
            pii_detected=pii_detected,
            is_redacted=False,
        )
        
        # Redact PII in logs if configured
        if self.config.redact_pii_in_logs and pii_detected:
            entry.request_text = self.pii_detector.redact(entry.request_text or "", pii_detected)
            entry.result_summary = self.pii_detector.redact(entry.result_summary or "", pii_detected)
            entry.is_redacted = True
            entry.redacted_at = timestamp
        
        # Store to database (parameterized to prevent injection)
        try:
            from sqlalchemy import text
            
            insert_sql = """
            INSERT INTO audit_log 
            (timestamp, event_type, user_id, session_id, tool_used, request_text,
             result_summary, error, duration_ms, metadata, pii_detected_count, is_redacted)
            VALUES 
            (:timestamp, :event_type, :user_id, :session_id, :tool_used, :request_text,
             :result_summary, :error, :duration_ms, :metadata::jsonb, :pii_detected_count, :is_redacted)
            """
            
            async with self.db() as session:
                await session.execute(
                    text(insert_sql),
                    {
                        "timestamp": entry.timestamp,
                        "event_type": entry.event_type,
                        "user_id": entry.user_id,
                        "session_id": entry.session_id,
                        "tool_used": entry.tool_used,
                        "request_text": entry.request_text,
                        "result_summary": entry.result_summary,
                        "error": entry.error,
                        "duration_ms": entry.duration_ms,
                        "metadata": str(entry.metadata),
                        "pii_detected_count": len(pii_detected),
                        "is_redacted": entry.is_redacted,
                    }
                )
                await session.commit()
        
        except Exception as e:
            logger.error(f"Error logging to audit trail: {e}")
        
        return entry
    
    async def enforce_retention_policy(self) -> int:
        """
        Delete audit logs beyond retention period.
        
        Returns:
            Number of records deleted
        """
        from sqlalchemy import text
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_days_audit_log)
        
        try:
            async with self.db() as session:
                result = await session.execute(
                    text("DELETE FROM audit_log WHERE timestamp < :cutoff_date"),
                    {"cutoff_date": cutoff_date}
                )
                await session.commit()
                deleted = result.rowcount
                logger.info(f"Retention policy: deleted {deleted} audit logs older than {self.config.retention_days_audit_log} days")
                return deleted
        
        except Exception as e:
            logger.error(f"Error enforcing retention policy: {e}")
            return 0
