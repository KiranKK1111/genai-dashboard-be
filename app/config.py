"""
Configuration management for the GenAI backend.

Settings are defined using pydantic's ``BaseSettings`` which reads
values from the environment at import time. See ``env.txt`` for a list
of variables used by this backend. When adding new configuration
options, update this class accordingly.

The ``.env`` file (if present) will be automatically loaded thanks
to pydantic. You can copy the provided ``env.txt`` to ``.env`` and
adjust values for local development.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import List, Optional

# BaseSettings has been moved to the pydantic-settings package in
# pydantic >= 2. To remain compatible with both pydantic 1.x and 2.x,
# import it from pydantic_settings. Field and validator are still
# available from pydantic.
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    # AI model configuration
    ai_factory_api: str = Field("http://localhost:11434/v1/chat/completions", env="AI_FACTORY_API")
    ai_factory_token: str = Field("", env="AI_FACTORY_TOKEN")
    ai_factory_model: str = Field("llama2", env="AI_FACTORY_MODEL")
    llm_stream: bool = Field(False, env="LLM_STREAM")

    # Embeddings (optional)
    embeddings_api: Optional[str] = Field(None, env="EMBEDDINGS_API")
    embeddings_token: Optional[str] = Field(None, env="EMBEDDINGS_TOKEN")
    embeddings_model: Optional[str] = Field(None, env="EMBEDDINGS_MODEL")
    
    # Embeddings - semantic search powered by sentence-transformers
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_model_cache_dir: Optional[str] = Field(None, env="EMBEDDING_MODEL_CACHE_DIR")
    embedding_dimensions: int = Field(384, env="EMBEDDING_DIMENSIONS")  # 384 for all-MiniLM-L6-v2, 1536 for OpenAI text-embedding-3-small

    # Server configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(5000, env="PORT")
    debug: bool = Field(False, env="DEBUG")

    # CORS
    cors_origins: List[str] = Field(default_factory=list, env="CORS_ORIGINS")

    # Database configuration (DB-agnostic)
    db_type: str = Field("postgresql", env="DB_TYPE")  # postgresql, mysql, sqlite, sqlserver, mariadb
    
    # PostgreSQL settings
    postgres_host: str = Field("localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(5432, env="POSTGRES_PORT")
    postgres_db: str = Field("postgres", env="POSTGRES_DB")
    postgres_user: str = Field("postgres", env="POSTGRES_USER")
    postgres_password: str = Field("", env="POSTGRES_PASSWORD")
    postgres_schema: str = Field("public", env="POSTGRES_SCHEMA")
    
    # MySQL/MariaDB settings
    mysql_host: str = Field("localhost", env="MYSQL_HOST")
    mysql_port: int = Field(3306, env="MYSQL_PORT")
    mysql_db: str = Field("genai", env="MYSQL_DB")
    mysql_user: str = Field("root", env="MYSQL_USER")
    mysql_password: str = Field("", env="MYSQL_PASSWORD")
    
    # SQLite settings
    sqlite_path: str = Field("./genai.db", env="SQLITE_PATH")
    
    # SQL Server settings
    sqlserver_host: str = Field("localhost", env="SQLSERVER_HOST")
    sqlserver_port: int = Field(1433, env="SQLSERVER_PORT")
    sqlserver_db: str = Field("genai", env="SQLSERVER_DB")
    sqlserver_user: str = Field("sa", env="SQLSERVER_USER")
    sqlserver_password: str = Field("", env="SQLSERVER_PASSWORD")

    # JWT settings
    jwt_secret: str = Field("CHANGE_ME_TO_A_LONG_RANDOM_SECRET", env="JWT_SECRET")
    jwt_alg: str = Field("HS256", env="JWT_ALG")
    jwt_expire_min: int = Field(60 * 24, env="JWT_EXPIRE_MIN")

    # SQL guardrails
    sql_max_rows: int = Field(500, env="SQL_MAX_ROWS")
    sql_statement_timeout_ms: int = Field(8000, env="SQL_STATEMENT_TIMEOUT_MS")
    allowed_tables: Optional[List[str]] = Field(None, env="ALLOWED_TABLES")
    
    # NLP confidence thresholds (prevents executing low-confidence queries)
    min_confidence_threshold: float = Field(0.50, env="MIN_CONFIDENCE_THRESHOLD")
    min_confidence_for_sql_generation: float = Field(0.35, env="MIN_CONFIDENCE_FOR_SQL_GENERATION")
    
    # LLM Conversational Mode Settings
    llm_system_prompt: str = Field(
        "You are a professional SDM AI Assistant helping users with data queries and file analysis. "
        "Be clear, concise, and helpful. Use emojis sparingly to enhance readability. "
        "Never use action descriptions (like *adjusts glasses*). "
        "Focus on direct, actionable assistance.",
        env="LLM_SYSTEM_PROMPT"
    )
    llm_temperature: float = Field(0.5, env="LLM_TEMPERATURE")  # 0.5 = balanced, less creative
    llm_max_tokens: int = Field(500, env="LLM_MAX_TOKENS")
    llm_conversational_confidence_threshold: float = Field(0.40, env="LLM_CONVERSATIONAL_CONFIDENCE_THRESHOLD")
    
    # Schema discovery configuration
    # Internal tables to exclude from schema discovery (comma-separated in .env)
    # Default: users, chat_sessions, messages, uploaded_files, file_chunks (GenAI framework tables)
    internal_tables: List[str] = Field(
        default_factory=lambda: ['users', 'chat_sessions', 'messages', 'uploaded_files', 'file_chunks'],
        env="INTERNAL_TABLES"
    )
    # Set to True to discover ALL tables in the database schema (ignores allowed_tables)
    discover_all_tables: bool = Field(False, env="DISCOVER_ALL_TABLES")
    
    # PHASE 1-5: Universal Schema Agnostic Feature Flags
    # Enable/disable each phase of the schema discovery system independently
    # Start with False (legacy behavior), incrementally enable for testing
    
    # PHASE 1: Schema Discovery Foundation
    # Dynamically discover FK relationships, boolean columns, enum types
    enable_schema_discovery_engine: bool = Field(False, env="ENABLE_SCHEMA_DISCOVERY_ENGINE")
    schema_discovery_cache_ttl_seconds: int = Field(3600, env="SCHEMA_DISCOVERY_CACHE_TTL_SECONDS")
    
    # PHASE 2: Intent Classification
    # Use LLM semantic analysis instead of hardcoded keyword checks
    enable_semantic_intent_classifier: bool = Field(False, env="ENABLE_SEMANTIC_INTENT_CLASSIFIER")
    intent_classification_confidence_threshold: float = Field(0.6, env="INTENT_CLASSIFICATION_CONFIDENCE_THRESHOLD")
    
    # PHASE 3: Universal Value Grounding
    # Dynamically ground user values to DB enums instead of hardcoded fallback mappings
    enable_universal_value_grounder: bool = Field(False, env="ENABLE_UNIVERSAL_VALUE_GROUNDER")
    value_grounding_confidence_threshold: float = Field(0.7, env="VALUE_GROUNDING_CONFIDENCE_THRESHOLD")
    
    # PHASE 4: Remove FK Pattern Assumptions
    # Use discovered FK relationships instead of hardcoded naming patterns
    enable_discovered_fk_joins: bool = Field(False, env="ENABLE_DISCOVERED_FK_JOINS")
    
    # PHASE 5: Remove Boolean Pattern Matching
    # Use actual column types instead of regex pattern matching
    enable_discovered_boolean_columns: bool = Field(False, env="ENABLE_DISCOVERED_BOOLEAN_COLUMNS")
    
    # PHASE 6: Semantic Concept Mapping
    # Map approval/status/verification concepts dynamically instead of hardcoded lists
    enable_semantic_concept_mapper: bool = Field(False, env="ENABLE_SEMANTIC_CONCEPT_MAPPER")
    semantic_concept_confidence_threshold: float = Field(0.7, env="SEMANTIC_CONCEPT_CONFIDENCE_THRESHOLD")

    # ── DB Connection Pool ────────────────────────────────────────────────────
    db_pool_size: int = Field(10, env="DB_POOL_SIZE")
    db_max_overflow: int = Field(20, env="DB_MAX_OVERFLOW")
    db_pool_timeout: int = Field(30, env="DB_POOL_TIMEOUT")       # seconds
    db_pool_recycle: int = Field(1800, env="DB_POOL_RECYCLE")     # seconds (30 min)
    db_pool_pre_ping: bool = Field(True, env="DB_POOL_PRE_PING")

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_per_minute: int = Field(30, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")

    # ── Vector Search ─────────────────────────────────────────────────────────
    vector_similarity_threshold: float = Field(0.5, env="VECTOR_SIMILARITY_THRESHOLD")
    vector_search_top_k: int = Field(5, env="VECTOR_SEARCH_TOP_K")
    vector_search_timeout_ms: int = Field(5000, env="VECTOR_SEARCH_TIMEOUT_MS")

    # ── Conversation Memory ───────────────────────────────────────────────────
    memory_max_context_tokens: int = Field(6000, env="MEMORY_MAX_CONTEXT_TOKENS")
    memory_keep_recent_messages: int = Field(5, env="MEMORY_KEEP_RECENT_MESSAGES")
    memory_keep_messages_hours: int = Field(24, env="MEMORY_KEEP_MESSAGES_HOURS")

    # ── Schema Intelligence ───────────────────────────────────────────────────
    schema_high_confidence_threshold: float = Field(0.85, env="SCHEMA_HIGH_CONFIDENCE_THRESHOLD")
    schema_clarification_threshold: float = Field(0.60, env="SCHEMA_CLARIFICATION_THRESHOLD")
    schema_max_join_depth: int = Field(2, env="SCHEMA_MAX_JOIN_DEPTH")
    schema_change_poll_interval_sec: int = Field(60, env="SCHEMA_CHANGE_POLL_INTERVAL_SEC")

    # ── SQL Execution ─────────────────────────────────────────────────────────
    sql_statement_timeout_sec: int = Field(30, env="SQL_STATEMENT_TIMEOUT_SEC")
    sql_max_result_rows: int = Field(100_000, env="SQL_MAX_RESULT_ROWS")

    # ── Session Archival ──────────────────────────────────────────────────────
    session_max_live_turns: int = Field(50, env="SESSION_MAX_LIVE_TURNS")
    session_archive_keep_turns: int = Field(20, env="SESSION_ARCHIVE_KEEP_TURNS")

    # ── Caching ───────────────────────────────────────────────────────────────
    query_cache_ttl_sec: int = Field(300, env="QUERY_CACHE_TTL_SEC")
    query_cache_max_items: int = Field(1000, env="QUERY_CACHE_MAX_ITEMS")

    # ── Semantic Pipeline ─────────────────────────────────────────────────────
    semantic_confidence_threshold: float = Field(0.60, env="SEMANTIC_CONFIDENCE_THRESHOLD")
    semantic_catalog_ttl_minutes: int = Field(60, env="SEMANTIC_CATALOG_TTL_MINUTES")
    smart_query_ambiguity_threshold: float = Field(0.50, env="SMART_QUERY_AMBIGUITY_THRESHOLD")

    # ── Background Tasks (arq) ────────────────────────────────────────────────
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    arq_enabled: bool = Field(False, env="ARQ_ENABLED")          # opt-in; falls back to inline
    arq_max_jobs: int = Field(10, env="ARQ_MAX_JOBS")
    heavy_query_join_threshold: int = Field(2, env="HEAVY_QUERY_JOIN_THRESHOLD")  # joins >= this → background

    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):  # type: ignore[override]
        # Accept JSON list string, comma‑separated values, or None
        if v is None or v == "" or v == "[]":
            return []
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                # Fallback to comma-separated parsing
                result = [origin.strip() for origin in v.split(",") if origin.strip()]
                return result if result else []
        if isinstance(v, list):
            return v
        return []

    @validator("allowed_tables", pre=True)
    def parse_allowed_tables(cls, v):  # type: ignore[override]
        if not v:
            return None
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return [tbl.strip() for tbl in v.split(",") if tbl.strip()]
        return v

    @validator("internal_tables", pre=True)
    def parse_internal_tables(cls, v):  # type: ignore[override]
        # Use defaults if not specified or empty
        defaults = ['users', 'chat_sessions', 'messages', 'uploaded_files', 'file_chunks']
        if v is None or v == "" or v == "[]":
            return defaults
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed if parsed else defaults
            except Exception:
                # Fallback to comma-separated parsing
                result = [tbl.strip() for tbl in v.split(",") if tbl.strip()]
                return result if result else defaults
        if isinstance(v, list):
            return v if v else defaults
        return defaults

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of Settings.

    The cache prevents reloading environment variables on every import.
    """
    return Settings()


# Instantiate settings at module import so that other modules can
# simply ``from app.config import settings``.
settings = get_settings()


def get_schema() -> str:
    """Get the database schema from settings with default fallback."""
    return getattr(settings, 'postgres_schema', 'public')


def get_search_path_sql() -> str:
    """Get the SQL command to set search_path to configured schema."""
    schema = get_schema()
    return f"SET search_path TO {schema}, public"