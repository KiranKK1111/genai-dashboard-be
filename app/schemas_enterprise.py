"""
Production-grade ChatGPT-style response schemas.

This module defines the complete response architecture for enterprise-level
query handling with streaming, tool execution tracing, metadata, and observability.

Aligned with ChatGPT-style architecture:
- Intent detection with confidence
- Tool execution with full tracing
- Safety checks and compliance
- Streaming event format
- Clarification gates
- Follow-up awareness
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ==================== INTENT & DECISION MODELS ====================

class IntentType(str, Enum):
    """Core intent types for routing."""
    DATA_RETRIEVAL = "data_retrieval"
    FILE_ANALYSIS = "file_analysis"
    CLARIFICATION = "clarification"
    GENERAL_CHAT = "general_chat"
    TOOL_EXECUTION = "tool_execution"
    UNKNOWN = "unknown"


class IntentInfo(BaseModel):
    """Intent detection with confidence and follow-up awareness."""
    name: IntentType
    confidence: float = Field(..., ge=0.0, le=1.0, description="Intent confidence 0-1")
    reasoning: Optional[str] = Field(None, description="Why this intent was chosen")
    followup_detected: bool = Field(False, description="Is this a follow-up query")
    followup_type: Optional[str] = Field(None, description="Type: refinement, expansion, clarification, pivot")
    previous_query_reference: Optional[str] = Field(None, description="Reference to previous execution_id")


# ==================== TOOL EXECUTION MODELS ====================

class ToolType(str, Enum):
    """Types of tools that can be executed."""
    SQL_EXECUTOR = "sql_executor"
    RAG_RETRIEVAL = "rag_retrieval"
    FILE_ANALYZER = "file_analyzer"
    WEB_SEARCH = "web_search"
    INTERNAL_API = "internal_api"


class ToolExecutionStep(BaseModel):
    """Single step in tool execution."""
    step_number: int
    tool_type: ToolType
    tool_name: str
    input_params: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[Dict[str, Any]] = Field(None)
    error: Optional[str] = Field(None)
    duration_ms: int = Field(0, ge=0)
    success: bool


class ToolExecution(BaseModel):
    """Complete tool execution trace."""
    execution_id: str = Field(..., description="Unique execution identifier")
    tool_used: ToolType
    tool_name: str = Field(..., description="Friendly name of tool")
    
    # For SQL execution
    sql: Optional[str] = Field(None, description="Generated/executed SQL")
    sql_hash: Optional[str] = Field(None, description="Hash for deduplication")
    
    # Execution details
    steps: List[ToolExecutionStep] = Field(default_factory=list)
    row_count: Optional[int] = Field(None, description="Result row count")
    tables_involved: List[str] = Field(default_factory=list)
    columns_involved: Optional[Dict[str, List[str]]] = Field(None, description="Table -> columns mapping")
    
    # Execution safety
    safety_checks_passed: bool
    safety_violations: List[str] = Field(default_factory=list)
    
    # Performance
    execution_time_ms: int = Field(0, ge=0)
    timeout_occurred: bool = Field(False)
    
    # Results cached/retrieved
    cached: bool = Field(False)
    cache_key: Optional[str] = Field(None)


# ==================== DATA & VISUALIZATION MODELS ====================

class Column(BaseModel):
    """Column metadata."""
    name: str
    type: str = Field(..., description="SQL type: integer, text, timestamp, etc.")
    nullable: bool = Field(True)
    size: Optional[int] = Field(None, description="Size in bytes")


class DataSet(BaseModel):
    """Structured data result."""
    dataset_id: str = Field(..., description="Unique dataset identifier")
    name: str
    description: Optional[str] = None
    
    columns: List[Column]
    rows: List[List[Any]]  # List of [col1, col2, ...]
    
    row_count: int
    truncated: bool = Field(False, description="Was result truncated")
    truncation_reason: Optional[str] = Field(None)
    
    # Metadata
    source: str = Field(..., description="sql, rag, file, etc.")
    query_used: Optional[str] = Field(None)


class VisualizationSpec(BaseModel):
    """Visualization specification."""
    viz_id: str = Field(..., description="Unique visualization ID")
    type: str = Field(..., description="bar, line, pie, table, heatmap, etc.")
    title: str
    subtitle: Optional[str] = None
    
    # Data mapping
    x_field: Optional[str] = None
    y_field: Optional[str] = None
    y_fields: Optional[List[str]] = None  # For multi-series
    label_field: Optional[str] = None
    value_field: Optional[str] = None
    
    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict)
    
    # UI hints
    emoji: str = Field("📊")
    show_raw_data: bool = True
    exportable: bool = True


# ==================== CLARIFICATION MODELS ====================

class ClarificationType(str, Enum):
    """Clarification question types."""
    BINARY = "binary"
    MULTIPLE_CHOICE = "multiple_choice"
    VALUE_INPUT = "value_input"
    ENTITY_DISAMBIGUATION = "entity_disambiguation"
    MISSING_PARAMETER = "missing_parameter"


class Clarification(BaseModel):
    """Clarification needed before proceeding."""
    clarification_id: str = Field(..., description="Unique clarification ID")
    type: ClarificationType
    blocking: bool = Field(True, description="Block SQL execution until answered")
    
    question: str
    options: Optional[List[str]] = Field(None, description="For multiple choice")
    
    field_name: Optional[str] = Field(None, description="The field being clarified")
    field_type: Optional[str] = Field(None, description="Type hint: date, number, select, etc.")
    
    placeholder: Optional[str] = Field(None, description="UI placeholder text")
    default_value: Optional[Any] = Field(None, description="Default/suggested value")


# ==================== SAFETY & COMPLIANCE MODELS ====================

class SafetyCheck(BaseModel):
    """Individual safety check."""
    check_type: str = Field(..., description="policy, jailbreak, pii, sql_injection, etc.")
    passed: bool
    severity: Literal["low", "medium", "high", "critical"] = "low"
    message: Optional[str] = None
    remediation: Optional[str] = Field(None, description="How to fix the issue")


class ComplianceInfo(BaseModel):
    """Compliance and safety information."""
    safety_checks: List[SafetyCheck] = Field(default_factory=list)
    all_passed: bool = Field(True)
    pii_detected: bool = Field(False)
    pii_redacted: bool = Field(False)
    access_denied_reason: Optional[str] = Field(None)


# ==================== FOLLOW-UP MODELS ====================

class FollowUpContext(BaseModel):
    """Information about previous queries for follow-up handling."""
    detected: bool
    followup_type: Optional[str] = None  # refinement, expansion, clarification, pivot
    confidence: float = 0.0
    previous_execution_id: Optional[str] = None
    merged_filters: Optional[Dict[str, Any]] = None
    preserved_context: Optional[List[str]] = Field(None, description="What context was preserved")


# ==================== METADATA & OBSERVABILITY ====================

class ModelMetadata(BaseModel):
    """LLM model information."""
    model_used: str = Field(..., description="Model ID or name")
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tokens_used: int = Field(0)
    context_window_utilization: float = Field(0.0, ge=0.0, le=1.0, description="Percent of context used")


class AnalysisMetadata(BaseModel):
    """Analysis and decision information."""
    entities_detected: List[str] = Field(default_factory=list)
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    ambiguities: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)


class DebugInfo(BaseModel):
    """Debug information (development only)."""
    analysis: Optional[AnalysisMetadata] = None
    prompt_hash: Optional[str] = None
    decision_path: List[str] = Field(default_factory=list, description="Tracing the decision making")
    warnings: List[str] = Field(default_factory=list)


class ResponseMetadata(BaseModel):
    """Complete metadata about response."""
    # Scores
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)
    complexity_score: float = Field(0.0, ge=0.0, le=1.0, description="Query complexity")
    
    # Model
    model: ModelMetadata
    
    # Timing
    total_duration_ms: int = Field(0, ge=0)
    lm_duration_ms: int = Field(0, ge=0)
    tool_duration_ms: int = Field(0, ge=0)
    
    # Compliance
    compliance: ComplianceInfo = Field(default_factory=ComplianceInfo)
    
    # Debug
    debug: Optional[DebugInfo] = None
    
    # Follow-up
    followup: FollowUpContext = Field(default_factory=lambda: FollowUpContext(detected=False))


# ==================== RESPONSE MESSAGES ====================

class ChatGPTStyleResponse(BaseModel):
    """Complete ChatGPT-style response (non-streaming)."""
    
    # Basics
    success: bool
    session_id: str
    message_id: str
    conversation_turn: int
    timestamp: datetime
    
    # Intent & Routing
    intent: IntentInfo
    
    # Clarification (if needed)
    clarification: Optional[Clarification] = None
    
    # Tool Execution (if tools were used)
    tool_execution: Optional[ToolExecution] = None
    
    # Results
    datasets: List[DataSet] = Field(default_factory=list)
    visualizations: List[VisualizationSpec] = Field(default_factory=list)
    
    # Response text (LLM generated summary/explanation)
    response_text: str = Field(..., description="Main text response to user")
    
    # Metadata & Observability
    metadata: ResponseMetadata
    
    # Debug info (development only)
    debug: Optional[DebugInfo] = None


# ==================== STREAMING EVENT MODELS ====================

class StreamingEvent(BaseModel):
    """Base class for streaming events."""
    event: str = Field(..., description="Event type")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StreamingEventMessageStart(StreamingEvent):
    """Stream started."""
    event: Literal["message_start"] = "message_start"
    message_id: str
    session_id: str


class StreamingEventIntentDetected(StreamingEvent):
    """Intent detected."""
    event: Literal["intent_detected"] = "intent_detected"
    intent: IntentInfo


class StreamingEventToolCall(StreamingEvent):
    """Tool call initiated."""
    event: Literal["tool_call"] = "tool_call"
    tool_name: str
    tool_type: ToolType
    arguments: Dict[str, Any]


class StreamingEventToolResult(StreamingEvent):
    """Tool execution completed."""
    event: Literal["tool_result"] = "tool_result"
    execution_id: str
    success: bool
    error: Optional[str] = None
    row_count: Optional[int] = None
    duration_ms: int = 0


class StreamingEventTextDelta(StreamingEvent):
    """Text response chunk (streamed)."""
    event: Literal["text_delta"] = "text_delta"
    delta: str
    finish_reason: Optional[str] = None  # "stop", "length", etc.


class StreamingEventMessageEnd(StreamingEvent):
    """Stream completed."""
    event: Literal["message_end"] = "message_end"
    success: bool
    message_id: str


class StreamingEventClarificationNeeded(StreamingEvent):
    """Clarification required."""
    event: Literal["clarification_needed"] = "clarification_needed"
    clarification: Clarification


# Union of all streaming events
StreamingEventPayload = Union[
    StreamingEventMessageStart,
    StreamingEventIntentDetected,
    StreamingEventToolCall,
    StreamingEventToolResult,
    StreamingEventTextDelta,
    StreamingEventClarificationNeeded,
    StreamingEventMessageEnd,
]


# ==================== BACKWARD COMPATIBILITY ====================

# Keep existing schemas for backward compatibility
DataQueryResponse = ChatGPTStyleResponse
FileQueryResponse = ChatGPTStyleResponse
StandardResponse = ChatGPTStyleResponse

ResponseWrapper = ChatGPTStyleResponse
