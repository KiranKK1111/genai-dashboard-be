"""
Database models for the GenAI backend - DB-Agnostic Version.

These models describe tables used for user sessions, chat messages,
uploaded files and file chunks for retrieval-augmented generation.
They are defined using SQLAlchemy's declarative ORM with generic types
that work across multiple databases (PostgreSQL, MySQL, SQLite, SQL Server).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
    JSON,
    Uuid,
    Index,
)

# Try to import pgvector, fall back to JSON if not available
# IMPORTANT: Even if Python package is installed, the PostgreSQL extension
# must be enabled in your database. If you can't install pgvector extension,
# set PGVECTOR_AVAILABLE = False below to force JSON storage mode.
try:
    from pgvector.sqlalchemy import Vector
    # Force disable if you can't install pgvector extension in PostgreSQL
    # Uncomment next line to use JSON storage instead:
    # PGVECTOR_AVAILABLE = False
    
    # Auto-detect: Check if extension is enabled (this is just a flag)
    # Real detection happens at runtime in database.py
    PGVECTOR_AVAILABLE = False  # Changed to False - use JSON storage
except ImportError:
    PGVECTOR_AVAILABLE = False
    Vector = None
from sqlalchemy.orm import relationship, Mapped

from .database import Base
from .config import settings


# Get schema from .env, defaults to 'public' if not specified
def get_schema() -> str:
    """Get the database schema from settings."""
    return getattr(settings, 'postgres_schema', 'public')


def get_fk_reference(table: str, column: str = "id") -> str:
    """Build a fully qualified foreign key reference.
    
    Args:
        table: Table name (without schema)
        column: Column name (default: 'id')
        
    Returns:
        Fully qualified reference like 'schema.table.column'
    """
    schema = get_schema()
    return f"{schema}.{table}.{column}"


# Helper for UUID columns - use native UUID type for PostgreSQL
def get_uuid_column(primary_key: bool = False, **kwargs):
    """Get a UUID column that works with PostgreSQL UUID type."""
    return Column(Uuid(as_uuid=True), primary_key=primary_key, default=uuid.uuid4, **kwargs)


class User(Base):
    __tablename__ = "users"
    __table_args__ = ({"schema": get_schema()},)

    id = get_uuid_column(primary_key=True)
    username = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    sessions: Mapped[List["ChatSession"]] = relationship(
        "ChatSession", back_populates="user", cascade="all, delete-orphan"
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = ({"schema": get_schema()},)

    id = get_uuid_column(primary_key=True)
    user_id = Column(Uuid(as_uuid=True), ForeignKey(get_fk_reference("users"), ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # ===== NEW: Session state management (ChatGPT-like follow-ups) =====
    # Persisted QueryState from the last executed query
    session_state = Column(JSON, nullable=True)  # Structured query state
    # Persisted list of recent tool calls  
    tool_calls_log = Column(JSON, nullable=True)  # List of {tool_type, input, output, timestamp}
    # When session state was last updated
    state_updated_at = Column(DateTime(timezone=True), nullable=True)
    
    # ===== NEW: Result Schema Memory (for follow-up validation) =====
    # Schema of the last query result
    last_result_schema = Column(JSON, nullable=True)  # [{column, type, sample_values}, ...]
    # Number of rows in last result
    last_result_row_count = Column(Integer, nullable=True)
    # Sample values from last result (for ambiguity detection)
    last_result_samples = Column(JSON, nullable=True)

    user: Mapped["User"] = relationship("User", back_populates="sessions")
    messages: Mapped[List["Message"]] = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan"
    )
    files: Mapped[List["UploadedFile"]] = relationship(
        "UploadedFile", back_populates="session", cascade="all, delete-orphan"
    )
    tool_calls: Mapped[List["ToolCall"]] = relationship(
        "ToolCall", back_populates="session", cascade="all, delete-orphan"
    )


class Message(Base):
    """Single row containing both user query and assistant response."""
    __tablename__ = "messages"
    __table_args__ = ({"schema": get_schema()},)

    id = get_uuid_column(primary_key=True)
    session_id = Column(Uuid(as_uuid=True), ForeignKey(get_fk_reference("chat_sessions"), ondelete="CASCADE"), nullable=False)
    
    # User input
    query = Column(Text, nullable=False)  # The user's query
    queried_at = Column(DateTime(timezone=True), default=datetime.utcnow)  # When query was received
    
    # Assistant output
    response_type = Column(String(50), nullable=False)  # 'modal_response', 'confirmation_response', 'clarifying_question', 'error_response'
    response = Column(JSON, nullable=False)  # Full response object with SQL, data, metadata, etc.
    responded_at = Column(DateTime(timezone=True), nullable=True)  # When response was sent
    
    # User feedback on the response
    feedback = Column(Text, nullable=True)  # User's feedback (thumbs up/down, comments, etc.)
    
    # Timestamp tracking (updated when feedback is added/modified)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")


class UploadedFile(Base):
    __tablename__ = "uploaded_files"
    __table_args__ = ({"schema": get_schema()},)

    id = get_uuid_column(primary_key=True)
    session_id = Column(Uuid(as_uuid=True), ForeignKey(get_fk_reference("chat_sessions"), ondelete="CASCADE"), nullable=False)
    filename = Column(String(512), nullable=False)
    filetype = Column(String(255), nullable=False)
    size = Column(Integer, nullable=False)
    upload_time = Column(DateTime(timezone=True), default=datetime.utcnow)
    content_text = Column(Text)  # extracted plain text for semantic search

    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="files")
    chunks: Mapped[List["FileChunk"]] = relationship(
        "FileChunk", back_populates="file", cascade="all, delete-orphan"
    )


class FileChunk(Base):
    __tablename__ = "file_chunks"
    __table_args__ = ({"schema": get_schema()},)

    id = get_uuid_column(primary_key=True)
    file_id = Column(Uuid(as_uuid=True), ForeignKey(get_fk_reference("uploaded_files"), ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    
    # Source locator for citations (page X, sheet Y, row Z)
    source_type = Column(String(50), nullable=True)  # 'page' | 'sheet' | 'row' | 'paragraph'
    source_locator = Column(String(255), nullable=True)  # 'page 5' | 'sheet Sales'
    
    # Character offsets for precise location
    char_start = Column(Integer, nullable=True)
    char_end = Column(Integer, nullable=True)
    
    # Vector embedding for semantic search
    # For P1: pgvector_file_retriever uses this for indexed search
    # 
    # IMPORTANT: When changing embedding models, update BOTH:
    # 1. Config: EMBEDDING_DIMENSIONS in .env (e.g., 384, 768, 1536)
    # 2. Database: Vector(XXX) dimension below to match your model
    #    - all-MiniLM-L6-v2: 384 dimensions
    #    - text-embedding-3-small: 1536 dimensions
    #    - all-mpnet-base-v2: 768 dimensions
    # Then run: ALTER TABLE genai.file_chunks ALTER COLUMN embedding TYPE vector(XXX);
    if PGVECTOR_AVAILABLE:
        # Use pgvector: dimension MUST match EMBEDDING_DIMENSIONS config
        embedding = Column(Vector(384), nullable=True)  # Change 384 to match your model
    else:
        # Fallback to JSON for systems without pgvector
        embedding = Column(JSON, nullable=True)
    
    # Session and user scope (for privacy-aware retrieval)
    session_id = Column(Uuid(as_uuid=True), ForeignKey(get_fk_reference("chat_sessions"), ondelete="CASCADE"), nullable=True)
    user_id = Column(Uuid(as_uuid=True), ForeignKey(get_fk_reference("users"), ondelete="CASCADE"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=True)

    file: Mapped["UploadedFile"] = relationship("UploadedFile", back_populates="chunks")


# ============================================================================
# NEW: Tool Execution Tracking (for ChatGPT-style follow-ups)
# ============================================================================

class ToolCall(Base):
    """
    Records tool calls and their results (SQL queries, file lookups, etc.).
    
    This persists:
    - What SQL was executed and what columns/rows came back
    - What files were looked up
    - What entities were extracted
    
    This is critical for deterministic follow-ups.
    """
    __tablename__ = "tool_calls"
    __table_args__ = ({"schema": get_schema()},)

    id = get_uuid_column(primary_key=True)
    session_id = Column(Uuid(as_uuid=True), ForeignKey(get_fk_reference("chat_sessions"), ondelete="CASCADE"), nullable=False)
    
    # What tool was called
    tool_type = Column(String(50), nullable=False)  # sql_query, file_lookup, entity_extraction, etc.
    
    # What was passed in
    input_json = Column(JSON, nullable=False)  # The query/input to the tool
    
    # What came back
    output_json = Column(JSON, nullable=False)  # The result from the tool
    
    # Execution status
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    
    # Timing
    start_time = Column(DateTime(timezone=True), default=datetime.utcnow)
    end_time = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="tool_calls")


# ============================================================================
# NEW: Turn State (for semantic intent routing)
# ============================================================================

class TurnState(Base):
    """
    Persistent state after each tool execution turn.
    
    This is the core of the semantic routing system. It records:
    - What tool was used (RUN_SQL, ANALYZE_FILE, CHAT)
    - What the user asked
    - What the assistant did
    - Artifacts from execution (SQL, tables, rows, file chunks, etc.)
    
    When a follow-up question arrives, the router loads the last TurnState
    to determine if it's truly a follow-up or a new query.
    
    Example: User asks "show top 10 rows from <table>". Tool=RUN_SQL, artifacts contain SQL.
    User then asks "now group them by <column>". Router detects:
    - last_tool=RUN_SQL (hard signal)
    - last_turn has SQL (hard signal)
    - "group by" implies modification → RUN_SQL_FOLLOW_UP, subtype=CHANGE_GROUPING
    """
    __tablename__ = "turn_states"
    __table_args__ = ({"schema": get_schema()},)
    
    id = get_uuid_column(primary_key=True)
    session_id = Column(
        Uuid(as_uuid=True),
        ForeignKey(get_fk_reference("chat_sessions"), ondelete="CASCADE"),
        nullable=False
    )
    
    # Sequential turn number
    turn_id = Column(Integer, nullable=False)
    
    # User's query/request
    user_query = Column(Text, nullable=False)
    
    # What the assistant did (summary)
    assistant_summary = Column(Text, nullable=False)
    
    # Which tool was used (RUN_SQL, ANALYZE_FILE, CHAT, MIXED)
    tool_used = Column(String(50), nullable=False)
    
    # Artifacts from execution (JSON, structured)
    # Contains: sql, tables, filters, result_schema, row_count, file_ids, etc.
    artifacts = Column(JSON, nullable=False)
    
    # Router confidence for this turn 0-1
    confidence = Column(Float, default=0.5)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    session: Mapped["ChatSession"] = relationship("ChatSession")


# ============================================================================
# NEW: Query Embeddings with pgvector (RAG Vector Store)
# ============================================================================

class QueryEmbedding(Base):
    """
    Stores query embeddings and metadata for semantic search across sessions.
    
    Uses pgvector extension in PostgreSQL for efficient vector similarity search.
    Stores ALL result rows from query execution for comprehensive semantic understanding.
    Enables RAG system to find semantically similar previous queries and their complete results.
    """
    __tablename__ = "query_embeddings"
    __table_args__ = ({"schema": get_schema()},)

    id = get_uuid_column(primary_key=True)
    session_id = Column(Uuid(as_uuid=True), ForeignKey(get_fk_reference("chat_sessions"), ondelete="CASCADE"), nullable=False)
    
    # User query details
    user_query = Column(Text, nullable=False)  # Original user question
    generated_sql = Column(Text, nullable=False)  # SQL that was executed
    
    # Result metadata
    result_count = Column(Integer, nullable=False)  # Total rows returned
    column_names = Column(JSON, nullable=False)  # List of column names: ["id", "name", ...]
    
    # ALL result rows from query execution (not just first row sample)
    # Format: List of dicts, each dict is one row from the result set
    # Example: [{"id": 1, "name": "Asha"}, {"id": 2, "name": "Arjun"}, ...]
    all_result_rows = Column(JSON, nullable=False)  # ALL rows from query result
    
    # Vector embedding - optimized storage
    # IMPORTANT: Dimension MUST match EMBEDDING_DIMENSIONS in config.py
    if PGVECTOR_AVAILABLE:
        # Use pgvector for efficient similarity search and better storage
        # Dimension must match your embedding model (see file_chunks for details)
        # Default: 384 for all-MiniLM-L6-v2
        embedding = Column(Vector(384), nullable=False)  # Change to match your model
        # Add index for fast similarity search
        __table_args__ = (
            Index('idx_query_embeddings_vector', embedding, postgresql_using='ivfflat'),  # HNSW-like indexing
            {"schema": get_schema()},
        )
    else:
        # Fallback to JSON for databases without pgvector
        # This is Database-agnostic but slower for similarity search
        embedding = Column(JSON, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Metadata for tracking
    query_hash = Column(String(64), nullable=True)  # SHA256 of user_query for duplicate detection
    
    # Statistics for RAG relevance
    result_quality_score = Column(Integer, nullable=True)  # 0-100, how good were the results
    
    session: Mapped["ChatSession"] = relationship("ChatSession")


# ============================================================================
# NEW: Conversation Memory (P2 - Memory Management)
# ============================================================================

class ConversationMemory(Base):
    """
    Persistent conversation memory state for each session.
    
    Tracks:
    - All messages (user + assistant)
    - Summaries of older messages (for token budgeting)
    - Total tokens used
    - Last summarization timestamp
    
    Used by ConversationMemoryManager (P2) for:
    - Rolling summarization (keep recent N messages)
    - Token budgeting (respond to context window limits)
    - Automatic cleanup of old conversations
    
    Storage format: JSON serialized ConversationMemoryState
    """
    __tablename__ = "conversation_memory"
    __table_args__ = (
        {"schema": get_schema()},
    )
    
    id = get_uuid_column(primary_key=True)
    session_id = Column(
        Uuid(as_uuid=True),
        ForeignKey(get_fk_reference("chat_sessions"), ondelete="CASCADE"),
        nullable=False,
        unique=True  # One memory per session
    )
    
    # Complete message history (JSON)
    # Format: List[{role: "user"|"assistant", content: str, created_at: ISO8601, token_count: int}]
    messages = Column(JSON, nullable=False, default=[])
    
    # Summaries of older messages
    # Format: List[{summary_text: str, original_message_count: int, token_count: int, ...}]
    summaries = Column(JSON, nullable=False, default=[])
    
    # Total tokens used (for budget tracking)
    total_tokens_used = Column(Integer, default=0)
    
    # When was the last summarization done
    last_summarization_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    session: Mapped["ChatSession"] = relationship("ChatSession")