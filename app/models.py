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
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
    JSON,
    Uuid,
)
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
    __tablename__ = "messages"
    __table_args__ = ({"schema": get_schema()},)

    id = get_uuid_column(primary_key=True)
    session_id = Column(Uuid(as_uuid=True), ForeignKey(get_fk_reference("chat_sessions"), ondelete="CASCADE"), nullable=False)
    response_type = Column(String(50), nullable=False)  # 'modal_response', 'confirmation_response', 'clarifying_question', 'error_response'
    query = Column(Text, nullable=True)  # The user's query
    queried_at = Column(DateTime(timezone=True), default=datetime.utcnow)  # When query was received
    responded_at = Column(DateTime(timezone=True), nullable=True)  # When response was sent
    response = Column(JSON, nullable=False)  # Full response object with SQL, data, metadata, etc.
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

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
    embedding = Column(JSON)  # vector representation of the chunk

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
