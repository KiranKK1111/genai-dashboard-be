"""
Pydantic schema definitions for request and response bodies.

These classes are used to validate incoming data and to generate
structured responses that match the expected format used by the
front‑end. Where possible we mirror the sample responses provided in
``frontend/SDM/Sample Responses``. Additional helper classes model
datasets, visualisations and layouts.
"""

from __future__ import annotations

from datetime import datetime
import time
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ----------------------------- Clarification Types & Models ----

class ClarificationType(str, Enum):
    """Enumeration of clarification question types for deterministic handling."""
    BINARY = "binary"
    MULTIPLE_CHOICE = "multiple_choice"
    MISSING_PARAMETER = "missing_parameter"
    VALUE_INPUT = "value_input"
    ENTITY_DISAMBIGUATION = "entity_disambiguation"


class ClarificationQuestionBinary(BaseModel):
    """Binary yes/no clarification question."""
    type: Literal[ClarificationType.BINARY] = ClarificationType.BINARY
    question: str = Field(..., description="The clarification question")
    # Example: "Did you mean 'transactions' instead of 'customer'?"


class ClarificationQuestionMultipleChoice(BaseModel):
    """Multiple choice clarification question."""
    type: Literal[ClarificationType.MULTIPLE_CHOICE] = ClarificationType.MULTIPLE_CHOICE
    question: str = Field(..., description="The clarification question")
    options: List[str] = Field(..., description="Available options to choose from")
    # Example: "Which account type?" with options ["Savings", "Current", "Loan"]


class ClarificationQuestionMissingParameter(BaseModel):
    """Missing required parameter clarification question."""
    type: Literal[ClarificationType.MISSING_PARAMETER] = ClarificationType.MISSING_PARAMETER
    question: str = Field(..., description="The clarification question")
    required_field: str = Field(..., description="The field name that is required")
    field_type: Optional[str] = Field(None, description="Type of the required field (e.g., date_range, string)")
    # Example: "Please provide a date range for transactions." with required_field "date_range"


class ClarificationQuestionValueInput(BaseModel):
    """Value input clarification question."""
    type: Literal[ClarificationType.VALUE_INPUT] = ClarificationType.VALUE_INPUT
    question: str = Field(..., description="The clarification question")
    input_type: str = Field(..., description="Type of input expected (number, string, date, etc.)")
    placeholder: Optional[str] = Field(None, description="Placeholder text for input field")
    # Example: "What minimum amount should be considered high value?" with input_type "number"


class ClarificationQuestionEntityDisambiguation(BaseModel):
    """Entity disambiguation clarification question."""
    type: Literal[ClarificationType.ENTITY_DISAMBIGUATION] = ClarificationType.ENTITY_DISAMBIGUATION
    question: str = Field(..., description="The clarification question")
    options: List[str] = Field(..., description="Entity options to disambiguate")
    entity_field: Optional[str] = Field(None, description="The field being disambiguated")
    # Example: "Do you mean branch city or branch name?" with options ["branch_city", "branch_name"]


# Union type for all clarification questions
ClarificationQuestion = Union[
    ClarificationQuestionBinary,
    ClarificationQuestionMultipleChoice,
    ClarificationQuestionMissingParameter,
    ClarificationQuestionValueInput,
    ClarificationQuestionEntityDisambiguation
]


# ConfirmationRequest for responding to clarification
class ClarificationConfirmationRequest(BaseModel):
    """Request to confirm clarification question response."""
    session_id: str = Field(..., description="Session ID")
    clarification_type: ClarificationType = Field(..., description="Type of clarification being answered")
    response: Any = Field(..., description="User's response (choice, value, true/false, etc.)")
    confirmed: bool = Field(default=True, description="Whether user confirmed the clarification")


# ----------------------------- Request models -----------------------------

class NewSessionRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")


class QueryRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    session_id: str = Field(..., description="Unique identifier for the chat session")
    query: str = Field(..., description="The user's natural language prompt")
    # Files will be handled via FastAPI UploadFile objects; they are not
    # declared here because Pydantic cannot parse UploadFile.


# ----------------------------- Response models ---------------------------

class BaseResponse(BaseModel):
    type: str = Field(..., description="Type of the response e.g. data_query, file_query")
    intent: str = Field(..., description="High level intent extracted from the query")
    confidence: Union[float, str] = Field(..., description="Model confidence score")
    message: str = Field(..., description="User‑visible explanation of the result")
    related_queries: Optional[List[str]] = Field(
        None, description="Suggestions for follow‑up questions"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Arbitrary metadata about the response"
    )
    variations: Optional[List[str]] = Field(
        None, description="Optional list of alternative response variations for conversational responses"
    )


class DataQueryDataset(BaseModel):
    id: str
    data: List[Dict[str, Any]]
    count: int
    description: str


class VisualizationConfig(BaseModel):
    """
    Flexible configuration container for any visualization type.
    Can hold chart-specific or generic configuration options.
    """
    legend: Optional[Dict[str, Any]] = None
    colors: Optional[List[str]] = None
    labels: Optional[Dict[str, Any]] = None
    size: Optional[Dict[str, Any]] = None
    # Table config
    columns: Optional[List[str]] = None
    row_limit: Optional[int] = None
    sortable: Optional[bool] = None
    filterable: Optional[bool] = None
    paginated: Optional[bool] = None
    page_size: Optional[int] = None
    # Chart config
    category_field: Optional[str] = None
    value_field: Optional[str] = None
    x_field: Optional[str] = None
    y_field: Optional[str] = None
    y_fields: Optional[List[str]] = None
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    horizontal: Optional[bool] = None
    stacked: Optional[bool] = None
    smooth: Optional[bool] = None
    show_points: Optional[bool] = None
    show_values: Optional[bool] = None
    show_percentage: Optional[bool] = None
    show_legend: Optional[bool] = None
    donut_hole: Optional[bool] = None
    color_scheme: Optional[str] = None
    color_scale: Optional[str] = None
    # Multi-viz config
    primary_view: Optional[str] = None
    available_views: Optional[List[str]] = None
    
    class Config:
        extra = "allow"  # Allow arbitrary configuration options


class Visualization(BaseModel):
    """
    Dynamic, AI-centric visualization specification.
    Generated by LLM analysis of query results and schema.
    Includes intelligent aggregators, filters, and controls.
    """
    chart_id: str = Field(..., description="Unique identifier for this visualization")
    type: str = Field(..., description="Chart type: table, bar, line, pie, area, scatter, heatmap, multi_viz")
    title: str = Field(..., description="Display title for the visualization")
    subtitle: Optional[str] = Field(None, description="Optional subtitle")
    description: Optional[str] = Field(None, description="Chart description/explanation")
    emoji: Optional[str] = Field("📊", description="Emoji indicator for chart type")
    
    # Dynamic configuration with field_schema, aggregators, and controls
    config: Optional[Dict[str, Any]] = Field(None, description="Chart-specific configuration")
    
    # AI-generated field_schema analysis of result columns and their roles
    field_schema: Optional[Dict[str, Any]] = Field(None, description="Field schema with types and role hints (id, category, measure, time, geo_admin1, etc.)")
    
    # AI-generated aggregation suggestions for each chart type
    aggregators: Optional[Dict[str, Any]] = Field(None, description="Dynamic aggregators for bar/line/pie charts with controls and defaults")
    
    # Presentation options
    show_raw_data: Optional[bool] = Field(True, description="Show raw data table option")
    exportable: Optional[bool] = Field(True, description="Allow data export")
    full_screen_enabled: Optional[bool] = Field(True, description="Allow full-screen view")
    
    class Config:
        extra = "allow"  # Allow additional dynamic fields


class Layout(BaseModel):
    type: str
    arrangement: str


# ============================================================================
# QueryPlan Artifacts (NEW - for composable, multi-dialect queries)
# ============================================================================

class QueryPlanArtifact(BaseModel):
    """
    Persisted QueryPlan AST for deterministic query execution.
    
    Enables:
    - Multi-dialect SQL generation (PostgreSQL, MySQL, SQLite, SQL Server)
    - Intelligent follow-up queries (understand previous query structure)
    - Query repair and validation (check before execution)
    - Predictable performance optimization
    
    Structure is dialect-neutral JSON that can be deserialized and
    recompiled to any SQL dialect.
    """
    plan_json: Dict[str, Any] = Field(
        ..., 
        description="QueryPlan AST as JSON (dialect-neutral intermediate representation)"
    )
    intent: str = Field(..., description="Query intent (data_query, file_query, chat)")
    tables_used: List[str] = Field(default_factory=list, description="Tables referenced in query")
    columns_used: List[str] = Field(default_factory=list, description="Columns referenced")
    joins_used: List[Dict[str, Any]] = Field(default_factory=list, description="JOIN operations used")
    where_conditions: List[str] = Field(default_factory=list, description="WHERE clause predicates")
    group_by_fields: List[str] = Field(default_factory=list, description="GROUP BY fields")
    has_subqueries: bool = Field(False, description="Whether query contains subqueries")
    is_aggregated: bool = Field(False, description="Whether query uses aggregation")


class QueryArtifactsSection(BaseModel):
    """
    Complete artifacts for a data query response, including QueryPlan.
    """
    query_plan: Optional[QueryPlanArtifact] = Field(
        None,
        description="QueryPlan AST (used for follow-ups, dialect translation, debugging)"
    )
    sql_generated: str = Field(..., description="The generated SQL that was executed")
    sql_dialect: str = Field(default="postgresql", description="SQL dialect used")
    execution_time_ms: Optional[float] = Field(None, description="Milliseconds to execute query")
    row_count: int = Field(..., description="Number of rows returned")
    is_truncated: bool = Field(False, description="Whether result was truncated (has more rows)")


class DataQueryResponse(BaseResponse):
    type: Literal["data_query"] = "data_query"
    datasets: List[DataQueryDataset]
    visualizations: List[Visualization]
    layout: Optional[Layout] = None
    artifacts: Optional[QueryArtifactsSection] = Field(
        None,
        description="Query artifacts including QueryPlan AST, SQL, and execution metadata"
    )


class FileQueryResponse(BaseResponse):
    type: Literal["file_query"] = "file_query"
    files: List[Dict[str, Any]]


class FileLookupResponse(BaseResponse):
    type: Literal["file_lookup"] = "file_lookup"
    files: Optional[List[Dict[str, Any]]] = None


class ConfigUpdateResponse(BaseResponse):
    type: Literal["config_update"] = "config_update"
    visualizations: List[Visualization]
    layout: Optional[Layout] = None


class StandardResponse(BaseResponse):
    type: Literal["standard"] = "standard"


ResponsePayload = Union[
    DataQueryResponse,
    FileQueryResponse,
    FileLookupResponse,
    ConfigUpdateResponse,
    StandardResponse,
    "LamaResponse",
]


class ResponseWrapper(BaseModel):
    success: bool
    response: ResponsePayload
    timestamp: Optional[int] = None
    original_query: Optional[str] = None
    intent: Optional[Dict[str, Any]] = None  # Add intent classification result


class SessionSummary(BaseModel):
    session_id: str
    created_at: datetime


class MessageSchema(BaseModel):
    """Message schema for chat history retrieval.
    
    Stores both user queries and assistant responses in a structured format.
    For assistant messages, the response contains a complete LamaResponse with:
    - Assistant message block with content
    - Artifacts (SQL, files, citations)
    - Visualizations with chart configurations
    - Follow-up suggestions
    - Routing and debug information
    """
    response_type: str  # 'user_query' or 'assistant_response'
    query: Optional[str] = None  # The user's query (for user_query messages)
    queried_at: Optional[datetime] = None  # When the query was received
    responded_at: Optional[datetime] = None  # When the response was sent (for assistant messages)
    response: Dict[str, Any]  # Full response object (LamaResponse dict for assistant messages, {} for user messages)
    created_at: datetime


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: List[MessageSchema]


class SessionsResponse(BaseModel):
    user_id: str
    sessions: List[SessionSummary]


class NewSessionResponse(BaseModel):
    success: bool
    session_id: str


class HealthResponse(BaseModel):
    status: str


# ----------------------------- Auth models -----------------------------

class RegisterRequest(BaseModel):
    username: str = Field(..., description="Username for registration")
    password: str = Field(..., description="Password for the user")


class LoginRequest(BaseModel):
    username: str = Field(..., description="Username for login")
    password: str = Field(..., description="Password for the user")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    username: str
    created_at: datetime


# ----------------------------- Intelligent Modal Models ----

class IntelligentModalRequest(BaseModel):
    """Request for intelligent SQL generation, execution, or file analysis."""
    query: str = Field(..., description="Natural language query")
    session_id: Optional[str] = Field(None, description="Optional conversation session ID")
    conversation_history: Optional[str] = Field(None, description="Previous conversation context")
    # Files are handled separately via FastAPI UploadFile


class QueryMetadata(BaseModel):
    """Metadata about executed query."""
    row_count: Optional[int] = None
    execution_time_ms: Optional[float] = None
    column_names: Optional[List[str]] = None
    data_types: Optional[Dict[str, str]] = None
    table_names: Optional[List[str]] = None
    # File-specific metadata
    file_name: Optional[str] = None
    file_size_bytes: Optional[int] = None
    chunk_count: Optional[int] = None
    relevance_score: Optional[float] = None
    matched_chunks: Optional[int] = None
    # 🧠 NLP Analysis Metadata (NEW!)
    confidence_score: Optional[float] = None  # 0-1 confidence in query understanding
    confidence_level: Optional[str] = None    # VERY_HIGH, HIGH, MEDIUM, LOW, VERY_LOW
    complexity_score: Optional[float] = None  # 0-1 query complexity assessment
    analysis: Optional[Dict[str, Any]] = None # Full NLP analysis details


class FileQueryMetadata(BaseModel):
    """Enhanced metadata for file-based queries."""
    file_name: str
    total_size_bytes: int
    total_characters: int
    total_words: int
    total_lines: int
    chunk_count: int
    strategy: str
    avg_chunk_size: int
    relevant_matches: Optional[List[Dict[str, Any]]] = None


class IntelligentModalResponse(BaseModel):
    """Response from intelligent SQL modal (SQL or file-based)."""
    success: bool = Field(..., description="Whether query was successful")
    # SQL-specific
    sql: Optional[str] = Field(None, description="Generated SQL query")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Query results as table rows")
    # Visualizations (NEW!)
    visualizations: Optional[List[Visualization]] = Field(
        None, 
        description="Chart configurations for visualizing the data (table, bar, line, pie, multi_viz, etc.)"
    )
    # File-specific
    relevant_content: Optional[str] = Field(None, description="Relevant content extracted from file")
    matched_chunks: Optional[int] = Field(None, description="Number of chunks matched in file")
    relevance_score: Optional[float] = Field(None, description="Relevance score for file matches")
    # Common
    message: Optional[str] = Field(None, description="Beautiful formatted response with emojis")
    metadata: Optional[QueryMetadata] = Field(None, description="Execution metadata")
    clarifying_question: Optional[ClarificationQuestion] = Field(None, description="Structured clarification question if more context needed")
    status: Optional[str] = Field(None, description="Status: success, clarification_needed, error")
    error: Optional[str] = Field(None, description="Error message if failed")
    recovered: Optional[bool] = Field(None, description="Whether query was recovered from error")


# ----------------------------- ChatGPT-like Response Models ----

class ContentBlock(BaseModel):
    """Single content block in a response."""
    type: str = Field(..., description="Block type: paragraph, heading, bullets, numbered, callout, table, code")
    text: Optional[str] = Field(None, description="Text content")
    items: Optional[List[str]] = Field(None, description="List items (for bullets/numbered)")
    variant: Optional[str] = Field(None, description="Callout variant: info, success, warning, next, error")
    headers: Optional[List[str]] = Field(None, description="Table headers")
    rows: Optional[List[List[str]]] = Field(None, description="Table rows")
    language: Optional[str] = Field(None, description="Code block language (python, sql, bash, etc.)")
    emoji: Optional[str] = Field(None, description="Emoji for visual appeal")


class AssistantMessageBlock(BaseModel):
    """Structured assistant message with content blocks."""
    role: str = Field(default="assistant", description="Always 'assistant'")
    title: Optional[str] = Field(None, description="Message title/headline")
    content: List[ContentBlock] = Field(..., description="List of content blocks")


class FileArtifact(BaseModel):
    """Reference to a file used in the response."""
    file_id: str
    filename: str


class ArtifactsSection(BaseModel):
    """Metadata about artifacts used in the response."""
    files_used: List[FileArtifact] = Field(default_factory=list, description="Files referenced")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Citation references")
    sql: Optional[str] = Field(None, description="SQL query if applicable")


class RoutingInfo(BaseModel):
    """Routing and intent information."""
    type: str = Field(..., description="Response type: file_query, data_query, chat, clarification")
    intent: str = Field(..., description="Normalized intent")
    confidence: float = Field(..., description="Confidence score 0-1")


class FollowUpSuggestion(BaseModel):
    """Follow-up question suggestion."""
    id: str
    text: str


class DebugInfo(BaseModel):
    """Debug information for development."""
    normalized_user_request: Optional[str] = None
    requires_date: Optional[bool] = None
    complexity: Optional[str] = None
    
    class Config:
        extra = "allow"


class LamaResponse(BaseModel):
    """
    Unified ChatGPT-like response format.
    
    Combines:
    - Conversational assistant message with content blocks
    - Structured artifacts and routing
    - Single visualization (multi_viz by default)
    - Follow-up suggestions
    - Debug information
    """
    id: str = Field(..., description="Response ID (e.g., msg_...)")
    object: str = Field(default="chat.response", description="Object type")
    created_at: int = Field(..., description="Timestamp in milliseconds")
    session_id: str = Field(..., description="Session ID")
    
    mode: str = Field(..., description="Response mode: file, sql, chat")
    assistant: AssistantMessageBlock = Field(..., description="Assistant message with content blocks")
    artifacts: ArtifactsSection = Field(..., description="Artifacts used in response")
    visualizations: Optional[Visualization] = Field(None, description="Visualization configuration (single multi_viz by default)")
    routing: RoutingInfo = Field(..., description="Routing and intent info")
    followups: List[FollowUpSuggestion] = Field(default_factory=list, description="Follow-up suggestions")
    debug: Optional[DebugInfo] = Field(None, description="Debug information")

# Rebuild ResponseWrapper model to resolve forward reference to LamaResponse
ResponseWrapper.model_rebuild()