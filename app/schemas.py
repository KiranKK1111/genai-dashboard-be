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
    # Example: "Did you mean '<option_a>' instead of '<option_b>'?"


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
    # Example: "Please provide a date range." with required_field "date_range"


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


# ============================================================================
# Semantic Intent Routing (NEW - fully dynamic classification)
# ============================================================================

class Tool(str, Enum):
    """Available tools/actions."""
    CHAT = "CHAT"
    ANALYZE_FILE = "ANALYZE_FILE"
    RUN_SQL = "RUN_SQL"
    MIXED = "MIXED"


class FollowupType(str, Enum):
    """Follow-up classification types."""
    NEW_QUERY = "NEW_QUERY"
    CHAT_FOLLOW_UP = "CHAT_FOLLOW_UP"
    ANALYZE_FILE_FOLLOW_UP = "ANALYZE_FILE_FOLLOW_UP"
    RUN_SQL_FOLLOW_UP = "RUN_SQL_FOLLOW_UP"


class RunSQLFollowupSubtype(str, Enum):
    """RUN_SQL follow-up subtypes (query modification operations)."""
    ADD_FILTER = "ADD_FILTER"
    REMOVE_FILTER = "REMOVE_FILTER"
    CHANGE_GROUPING = "CHANGE_GROUPING"
    CHANGE_METRIC = "CHANGE_METRIC"
    SORT_OR_TOPK = "SORT_OR_TOPK"
    EXPAND_COLUMNS = "EXPAND_COLUMNS"
    DRILLDOWN = "DRILLDOWN"
    PAGINATION = "PAGINATION"
    SWITCH_ENTITY = "SWITCH_ENTITY"
    FIX_ERROR = "FIX_ERROR"


class AnalyzeFileFollowupSubtype(str, Enum):
    """ANALYZE_FILE follow-up subtypes."""
    ASK_MORE_DETAIL = "ASK_MORE_DETAIL"
    ASK_SUMMARY_DIFFERENT_STYLE = "ASK_SUMMARY_DIFFERENT_STYLE"
    ASK_SOURCE_CITATION = "ASK_SOURCE_CITATION"
    COMPARE_SECTIONS = "COMPARE_SECTIONS"
    EXTRACT_TABLE_ENTITIES = "EXTRACT_TABLE_ENTITIES"


class ChatFollowupSubtype(str, Enum):
    """CHAT follow-up subtypes."""
    CLARIFY = "CLARIFY"
    CONTINUE = "CONTINUE"
    APPLY_PREVIOUS_ADVICE = "APPLY_PREVIOUS_ADVICE"
    REPHRASE = "REPHRASE"
    NEW_TOPIC_SAME_SESSION = "NEW_TOPIC_SAME_SESSION"


class TurnStateArtifacts(BaseModel):
    """
    Artifacts persisted after tool execution.
    Used to detect and classify follow-ups reliably.
    """
    # Common
    tool_used: str = Field(..., description="Tool that was executed (RUN_SQL, ANALYZE_FILE, CHAT)")
    
    # SQL artifacts
    sql: Optional[str] = Field(None, description="Generated SQL query")
    sql_plan_json: Optional[Dict[str, Any]] = Field(None, description="QueryPlan AST (structured)")
    tables: Optional[List[str]] = Field(None, description="Tables referenced in query")
    filters: Optional[List[Dict[str, Any]]] = Field(None, description="WHERE clause filters")
    grouping: Optional[List[str]] = Field(None, description="GROUP BY fields")
    having: Optional[List[Dict[str, Any]]] = Field(None, description="HAVING clause conditions")
    order_by: Optional[List[Dict[str, str]]] = Field(None, description="ORDER BY specification")
    limit: Optional[int] = Field(None, description="LIMIT value if present")
    result_schema: Optional[List[Dict[str, str]]] = Field(
        None, 
        description="Result schema [{name, type}, ...]"
    )
    row_count: Optional[int] = Field(None, description="Number of rows returned")
    result_sample: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Sample of first few rows from result"
    )
    
    # File artifacts
    file_ids: Optional[List[str]] = Field(None, description="Files analyzed")
    extracted_chunks: Optional[List[str]] = Field(None, description="Chunk IDs extracted")
    extracted_summary: Optional[str] = Field(
        None,
        description="Short summary of derived facts from file"
    )
    
    # Chat artifacts
    chat_summary: Optional[str] = Field(
        None,
        description="Rolling summary of conversation"
    )
    confirmed_facts: Optional[List[str]] = Field(
        None,
        description="Facts confirmed by user in this turn"
    )


class TurnState(BaseModel):
    """
    Persistent state after each tool execution turn.
    Core of the semantic routing system - enables reliable follow-up detection.
    """
    session_id: str = Field(..., description="Session ID")
    turn_id: int = Field(..., description="Turn number in session (auto-incremented)")
    user_query: str = Field(..., description="The user's query text")
    assistant_summary: str = Field(
        ...,
        description="Short summary of what the assistant did/returned"
    )
    tool_used: Tool = Field(
        ...,
        description="Tool that was used to handle this turn"
    )
    artifacts: TurnStateArtifacts = Field(
        ...,
        description="Artifacts from tool execution (for follow-up detection)"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Tool execution confidence score"
    )


class RouterSignals(BaseModel):
    """
    Hard signals used by the router for deterministic routing decisions.
    These are objective, fast checks (not ML-based).
    """
    has_uploaded_files: bool = Field(
        ...,
        description="Are there files in this session?"
    )
    db_connected: bool = Field(
        default=True,
        description="Is the database available?"
    )
    last_tool_used: Optional[Tool] = Field(
        None,
        description="What tool was used in the last turn?"
    )
    last_sql_exists: bool = Field(
        default=False,
        description="Is there a previous SQL query to reference?"
    )
    last_file_context_exists: bool = Field(
        default=False,
        description="Are there file chunks from previous analysis?"
    )
    time_since_last_turn_seconds: Optional[int] = Field(
        None,
        description="Seconds since last turn (for session timeout detection)"
    )


class RouterDecision(BaseModel):
    """
    Semantic intent routing decision output.
    
    Produced by the LLM after analyzing:
    1. Hard signals (db_connected, last_tool_used, etc.)
    2. Last turn state (artifacts, summary)
    3. Current user query
    
    This is a deterministic contract between router LLM and execution engine.
    Gateway for "do we have enough info, or need clarification?"
    """
    tool: Tool = Field(
        ...,
        description="Primary tool to route to: CHAT, ANALYZE_FILE, RUN_SQL, or MIXED"
    )
    followup_type: FollowupType = Field(
        ...,
        description="Classification: NEW_QUERY or {TOOL}_FOLLOW_UP"
    )
    followup_subtype: Optional[str] = Field(
        None,
        description="Subtype of follow-up (e.g., ADD_FILTER, CHANGE_GROUPING, etc.)"
    )
    needs_clarification: bool = Field(
        default=False,
        description="Should we ask user for clarification before executing?"
    )
    clarification_questions: List[ClarificationQuestion] = Field(
        default_factory=list,
        description="Structured clarification questions if needs_clarification=True"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Router confidence 0-1"
    )
    reasoning: str = Field(
        ...,
        description="Internal reasoning (for logging/debug)"
    )
    signals_used: Dict[str, Any] = Field(
        default_factory=dict,
        description="Which signals were used in decision"
    )


class RouterInput(BaseModel):
    """
    Input context sent to the router LLM.
    Combines hard signals, last turn state, and current query.
    """
    user_query: str = Field(..., description="Current user query to classify")
    hard_signals: RouterSignals = Field(..., description="Deterministic signal checks")
    last_turn_summary: Optional[str] = Field(
        None,
        description="Summary of last turn (what was done, what was returned)"
    )
    last_turn_artifacts: Optional[TurnStateArtifacts] = Field(
        None,
        description="Artifacts from last turn (sql exists? file chunks? etc.)"
    )
    session_state: Optional[Dict[str, Any]] = Field(
        None,
        description="Full session state if needed for context"
    )


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
    id: Optional[str] = Field(None, description="Message ID from database (when persisted)")
    type: str = Field(..., description="Type of the response e.g. data_query, file_query")
    intent: str = Field(..., description="High level intent extracted from the query")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score (0.0–1.0)")
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
    
    class Config:
        extra = "allow"  # Allow debug and other additional fields


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


class ChartCustomization(BaseModel):
    """
    Advanced chart customization options for precise control.
    Enables users to override AI defaults with specific styling.
    """
    # Color customization
    colors: Optional[List[str]] = Field(None, description="Custom color palette (hex codes)")
    theme: Optional[str] = Field(None, description="Theme: 'light', 'dark', 'auto', 'high_contrast'")
    color_mapping: Optional[Dict[str, str]] = Field(None, description="Map specific values to colors")
    
    # Axes configuration
    x_axis: Optional[Dict[str, Any]] = Field(None, description="X-axis config: {label, min, max, format, grid, logarithmic}")
    y_axis: Optional[Dict[str, Any]] = Field(None, description="Y-axis config: {label, min, max, format, grid, logarithmic}")
    secondary_y_axis: Optional[Dict[str, Any]] = Field(None, description="Secondary Y-axis for dual-axis charts")
    
    # Number formatting
    number_format: Optional[str] = Field(None, description="Number format: 'compact' (1.2K, 3.4M), 'precise', 'percentage', 'currency'")
    currency: Optional[str] = Field(None, description="Currency code: 'USD', 'EUR', 'GBP', 'INR'")
    decimal_places: Optional[int] = Field(None, description="Decimal precision: 0-8")
    
    # Date formatting
    date_format: Optional[str] = Field(None, description="Date format: 'short', 'medium', 'long', 'relative', custom strftime")
    
    # Legend configuration
    legend: Optional[Dict[str, Any]] = Field(None, description="Legend config: {position: 'top'|'bottom'|'left'|'right'|'none', alignment}")
    
    # Annotations
    annotations: Optional[List[Dict[str, Any]]] = Field(None, description="Annotations: [{type: 'line'|'area'|'text', x, y, label, color}]")
    reference_lines: Optional[List[Dict[str, Any]]] = Field(None, description="Reference lines: [{axis: 'x'|'y', value, label, color, style: 'solid'|'dashed'}]")
    
    # Interaction
    tooltips: Optional[Dict[str, Any]] = Field(None, description="Tooltip config: {enabled, format, fields}")
    drill_down: Optional[Dict[str, Any]] = Field(None, description="Drill-down config: {enabled, levels, target_query}")
    cross_filter: Optional[bool] = Field(None, description="Enable cross-filtering with other charts")
    
    # Layout
    chart_height: Optional[int] = Field(None, description="Chart height in pixels")
    chart_width: Optional[int] = Field(None, description="Chart width in pixels or percentage")
    responsive: Optional[bool] = Field(True, description="Auto-resize chart to container")
    
    # Chart-specific options
    bar_spacing: Optional[float] = Field(None, description="Bar chart spacing: 0-1")
    line_style: Optional[str] = Field(None, description="Line style: 'solid', 'dashed', 'dotted'")
    point_size: Optional[int] = Field(None, description="Point size for scatter/line charts")
    
    class Config:
        extra = "allow"  # Allow additional custom options


class Visualization(BaseModel):
    """
    Dynamic, AI-centric visualization specification.
    Generated by LLM analysis of query results and schema.
    Includes intelligent aggregators, filters, and controls.
    """
    chart_id: str = Field(..., description="Unique identifier for this visualization")
    type: str = Field(..., description="Chart type: table, bar, line, pie, area, scatter, heatmap, treemap, funnel, waterfall, gantt, sankey, boxplot, multi_viz")
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
    
    # Advanced customization (NEW)
    customization: Optional[ChartCustomization] = Field(None, description="Advanced chart customization options")

    # Pre-aggregated data for immediate render (server-side) + raw row count
    pre_aggregated_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Per-view pre-aggregated data {bar:[...], pie:[...], ...} ready to render without client transform"
    )
    raw_row_count: Optional[int] = Field(
        None,
        description="Total rows before aggregation (for display e.g. 'showing top 10 of 4,832')"
    )

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
    # Allow "standard" (default) as well as "clarification" and "viz_update"
    # which share the same flexible response structure.
    type: str = "standard"


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
    session_id: Optional[str] = None  # Optional session identifier for client sync
    message_id: Optional[str] = None  # Message ID for progress tracking


class SessionSummary(BaseModel):
    session_id: str
    created_at: datetime
    last_updated: Optional[datetime] = None  # When session was last updated
    title: Optional[str] = None  # First message or session title
    message_count: Optional[int] = None  # Number of messages in session


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
    id: Optional[str] = None  # Message ID
    response_type: str  # 'user_query' or 'assistant_response'
    query: Optional[str] = None  # The user's query (for user_query messages)
    queried_at: Optional[datetime] = None  # When the query was received
    responded_at: Optional[datetime] = None  # When the response was sent (for assistant messages)
    response: Dict[str, Any]  # Full response object (LamaResponse dict for assistant messages, {} for user messages)
    created_at: datetime
    
    class Config:
        # Allow arbitrary types and ensure we serialize dicts properly
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: List[MessageSchema]
    # Optional list of files associated with this session (for UI chips, etc.)
    files: Optional[List["UploadedFileSchema"]] = None


class SessionsResponse(BaseModel):
    user_id: str
    sessions: List[SessionSummary]


class NewSessionResponse(BaseModel):
    success: bool
    session_id: str


class UpdateSessionTitleRequest(BaseModel):
    title: str


class UpdateSessionTitleResponse(BaseModel):
    success: bool
    session_id: str
    title: str


class DeleteSessionResponse(BaseModel):
    success: bool
    session_id: str


class UploadedFileSchema(BaseModel):
    """Lightweight schema for uploaded files returned with session history."""
    id: str
    filename: str
    filetype: str
    size: int
    upload_time: datetime


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
    """Single content block in a response.

    Types:
      paragraph, heading, bullets, numbered, callout, table, code — standard blocks
      metric_card — KPI highlight (big number). Uses: value, label, unit, change, change_direction
    """
    type: str = Field(..., description="Block type: paragraph, heading, bullets, numbered, callout, table, code, metric_card")
    text: Optional[str] = Field(None, description="Text content")
    items: Optional[List[str]] = Field(None, description="List items (for bullets/numbered)")
    variant: Optional[str] = Field(None, description="Callout variant: info, success, warning, next, error")
    headers: Optional[List[str]] = Field(None, description="Table headers")
    rows: Optional[List[List[str]]] = Field(None, description="Table rows")
    language: Optional[str] = Field(None, description="Code block language (python, sql, bash, etc.)")
    emoji: Optional[str] = Field(None, description="Emoji for visual appeal")

    # metric_card fields — rendered as a large highlighted KPI number by the frontend
    value: Optional[Any] = Field(None, description="Numeric value for metric_card blocks")
    label: Optional[str] = Field(None, description="Short label shown below the value (e.g. 'Total Customers')")
    unit: Optional[str] = Field(None, description="Unit or currency symbol shown next to the value (e.g. '$', 'ms')")
    change: Optional[str] = Field(None, description="Change vs prior period shown as badge (e.g. '+12%', '-3')")
    change_direction: Optional[str] = Field(None, description="Trend direction for badge colour: up | down | neutral")


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
    sql_executed: Optional[str] = None
    row_count: Optional[int] = None
    columns: Optional[List[str]] = None
    result_rows: Optional[List[Dict[str, Any]]] = None  # CRITICAL: Store actual result rows for STEP 9
    
    class Config:
        extra = "allow"


# ============================================================================
# Artifact-centric response architecture (ResponseGeneration.md design)
# ============================================================================

class AnswerType(str, Enum):
    """Semantic classification of the query result — deterministic, no LLM."""
    SINGLE_METRIC = "single_metric"          # COUNT(*) = 1 row, 1 number
    METRIC_WITH_TABLE = "metric_with_table"  # few rows, few metrics
    TABULAR_RESULT = "tabular_result"        # multi-row table, no obvious pattern
    DISTRIBUTION = "distribution"            # grouped by category + metric
    TREND = "trend"                          # grouped by time + metric
    RANKING = "ranking"                      # ordered by metric DESC
    COMPARISON = "comparison"                # multiple categories + multiple metrics
    DETAIL_RECORDS = "detail_records"        # raw row-level detail (no aggregation)
    DOCUMENT_SUMMARY = "document_summary"    # file / RAG summary
    DOCUMENT_QA = "document_qa"             # file / RAG Q&A
    CHAT_RESPONSE = "chat_response"          # conversational answer
    ERROR = "error"                          # error response


class ColumnMeta(BaseModel):
    """Rich metadata for a single result column."""
    name: str
    label: Optional[str] = None
    datatype: str                          # "number", "string", "date", "boolean"
    semantic_role: Optional[str] = None   # metric, dimension, time, category, identifier, text, currency, percentage
    format_hint: Optional[str] = None     # currency, percentage, integer, decimal, date, datetime
    nullable: bool = True


class DataPayload(BaseModel):
    """Structured data payload with semantic column metadata."""
    kind: str = "sql_result"              # sql_result | file_table | document_extract | chat_context | none
    columns: List[ColumnMeta] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    total_available_rows: Optional[int] = None
    preview_only: bool = False


class ExecutionMeta(BaseModel):
    """Execution-layer metadata."""
    sql: Optional[str] = None
    sql_safe: bool = True
    limit_applied: bool = False
    execution_time_ms: Optional[int] = None
    sources: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class LamaResponse(BaseModel):
    """
    Unified ChatGPT-like response format.

    Combines:
    - Conversational assistant message with content blocks
    - Structured artifacts and routing
    - Single visualization (multi_viz by default)
    - Follow-up suggestions
    - Debug information
    - NEW: Contextual suggested questions and actions (Question-Back Engine)
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
    variations: Optional[List[str]] = Field(None, description="Alternative response variations (ChatGPT-style)")
    
    # Pagination / dataset summary — helps frontend show "Showing 10 of 4,832 rows" and "Load more"
    total_rows: Optional[int] = Field(None, description="Total rows in the result set before any truncation/limit")
    has_more: bool = Field(False, description="True when the displayed rows are a subset of total_rows")
    column_names: Optional[List[str]] = Field(None, description="Column names of the primary dataset (for quick access without parsing the table block)")

    # NEW: Production-grade contextual suggestions (Question-Back Engine)
    suggested_questions: List[str] = Field(default_factory=list, description="Contextual follow-up questions generated by Question-Back Engine")
    suggested_actions: List[Dict[str, Any]] = Field(default_factory=list, description="Actionable UI elements (expand_limit, add_filter, create_chart, etc.)")

    # Clarification block — populated when mode=="clarification"
    clarification: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Structured clarification request when the query is ambiguous. "
            "Contains: ambiguity_types, questions (list of structured question objects), "
            "reasoning, can_proceed."
        ),
    )

    debug: Optional[DebugInfo] = Field(None, description="Debug information")

    # ── Artifact-centric fields (ResponseGeneration.md architecture) ──────────
    answer_type: Optional[str] = Field(
        None,
        description="Semantic answer type: single_metric, trend, distribution, tabular_result, etc."
    )
    data: Optional[DataPayload] = Field(
        None,
        description="Structured data payload with semantic column metadata"
    )
    render_artifacts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of renderable UI artifact blocks (stat_card, table, bar_chart, etc.)"
    )
    clarifications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured clarification requests (e.g. chart axis mapping ambiguity)"
    )
    execution_meta: Optional[Dict[str, Any]] = Field(
        None,
        description="Execution metadata: sql, limit_applied, execution_time_ms, warnings"
    )


# ============================================================================
# Feedback Schema
# ============================================================================

class FeedbackRequest(BaseModel):
    """Request body for submitting message feedback."""
    feedback: Optional[Literal['LIKED', 'DISLIKED']] = Field(
        None, 
        description="Feedback value: 'LIKED', 'DISLIKED', or null to clear feedback"
    )


# Rebuild ResponseWrapper model to resolve forward reference to LamaResponse
ResponseWrapper.model_rebuild()