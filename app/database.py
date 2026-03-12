"""
Database connection utilities for the GenAI backend - DB-Agnostic Version.

This module configures SQLAlchemy engine supporting multiple databases:
- PostgreSQL (asyncpg driver)
- MySQL/MariaDB (aiomysql driver)
- SQLite (aiosqlite driver)  
- SQL Server (pyodbc driver)

It provides functions for obtaining AsyncSession to perform queries and exposes
the declarative Base which should be used to declare ORM models.
"""

from __future__ import annotations

from typing import AsyncGenerator, Optional

import anyio

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncConnection
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import text, event, inspect
from .config import settings, get_search_path_sql, get_schema


# Base class for declarative models
Base = declarative_base()


def build_database_url() -> str:
    """
    Build database URL based on configured DB type.
    
    Returns:
        Database URL string for SQLAlchemy
        
    Raises:
        ValueError: If DB type is not supported or config is invalid
    """
    db_type = settings.db_type.lower()
    
    if db_type == "postgresql":
        return (
            f"postgresql+asyncpg://{settings.postgres_user}:"
            f"{settings.postgres_password}@{settings.postgres_host}:"
            f"{settings.postgres_port}/{settings.postgres_db}"
        )
    elif db_type in ("mysql", "mariadb"):
        return (
            f"mysql+aiomysql://{settings.mysql_user}:"
            f"{settings.mysql_password}@{settings.mysql_host}:"
            f"{settings.mysql_port}/{settings.mysql_db}"
        )
    elif db_type == "sqlite":
        return f"sqlite+aiosqlite:///{settings.sqlite_path}"
    elif db_type == "sqlserver":
        # SQL Server typically uses pyodbc
        return (
            f"mssql+pyodbc://{settings.sqlserver_user}:"
            f"{settings.sqlserver_password}@{settings.sqlserver_host}:"
            f"{settings.sqlserver_port}/{settings.sqlserver_db}?driver=ODBC+Driver+17+for+SQL+Server"
        )
    else:
        raise ValueError(
            f"Unsupported DB_TYPE '{db_type}'. "
            "Supported: postgresql, mysql, mariadb, sqlite, sqlserver"
        )


DATABASE_URL = build_database_url()


def _build_engine_kwargs() -> dict:
    """Build dialect-specific engine kwargs with configured pool settings."""
    db_type = settings.db_type.lower()
    base: dict = {"echo": settings.debug, "future": True}

    if db_type == "sqlite":
        # SQLite uses StaticPool — pooling arguments are not valid
        from sqlalchemy.pool import StaticPool
        base["connect_args"] = {"check_same_thread": False}
        base["poolclass"] = StaticPool
    elif db_type == "sqlserver":
        # pyodbc connections cannot be safely shared across threads; use NullPool
        from sqlalchemy.pool import NullPool
        base["poolclass"] = NullPool
    else:
        # PostgreSQL, MySQL, MariaDB — full connection pool
        base.update(
            {
                "pool_size": settings.db_pool_size,
                "max_overflow": settings.db_max_overflow,
                "pool_timeout": settings.db_pool_timeout,
                "pool_recycle": settings.db_pool_recycle,
                "pool_pre_ping": settings.db_pool_pre_ping,
            }
        )
    return base


engine = create_async_engine(DATABASE_URL, **_build_engine_kwargs())


# Lazy-load dialect adapter to avoid circular imports
_dialect_adapter = None

def get_dialect_adapter():
    """Get dialect adapter (lazy-loaded to avoid circular imports)."""
    global _dialect_adapter
    if _dialect_adapter is None:
        from .services.dialect_adapter import get_adapter
        _dialect_adapter = get_adapter(settings.db_type)
    return _dialect_adapter



# Database-specific event listeners
def setup_connection_listeners():
    """Setup dialect-specific connection initialization."""
    db_type = settings.db_type.lower()
    
    if db_type == "postgresql":
        @event.listens_for(engine.sync_engine, "connect")
        def set_postgresql_search_path(dbapi_connection, connection_record):
            """Set search_path for PostgreSQL on each connection."""
            try:
                cursor = dbapi_connection.cursor()
                cursor.execute(get_search_path_sql())
                cursor.close()
            except Exception as e:
                print(f"[Warning] Could not set PostgreSQL search_path: {e}")
    
    elif db_type in ("mysql", "mariadb"):
        @event.listens_for(engine.sync_engine, "connect")
        def set_mysql_charset(dbapi_connection, connection_record):
            """Set charset for MySQL/MariaDB on each connection."""
            try:
                cursor = dbapi_connection.cursor()
                cursor.execute("SET NAMES utf8mb4")
                cursor.close()
            except Exception as e:
                print(f"[Warning] Could not set MySQL charset: {e}")
    
    # SQLite and SQL Server don't need special setup


setup_connection_listeners()


# Create a configured session class
async_session_factory = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Alias for compatibility with existing code
SessionLocal = async_session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an AsyncSession for dependency injection.
    
    FastAPI dependencies can use Depends(get_session) to obtain a session
    bound to the event loop. The session is automatically closed after
    the request completes.
    """
    # NOTE: request cancellations (client disconnects/timeouts) can interrupt
    # normal cleanup and cause noisy "Exception terminating connection" logs
    # from the asyncpg dialect. We shield session close from cancellation so
    # connections are returned/closed cleanly.
    session: AsyncSession = async_session_factory()
    try:
        yield session
    finally:
        with anyio.CancelScope(shield=True):
            await session.close()


async def dispose_engine() -> None:
    """Dispose the global engine, shielding from cancellation."""
    with anyio.CancelScope(shield=True):
        await engine.dispose()


async def init_db() -> None:
    """
    Initialize the database by creating all defined tables.
    
    This function can be called at application startup. It uses SQLAlchemy's
    metadata to create any tables that do not yet exist.
    """
    db_type = settings.db_type.lower()
    
    async with engine.begin() as conn:
        schema = get_schema()
        if db_type == "postgresql":
            # Create schema for PostgreSQL
            await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            await conn.execute(text(get_search_path_sql()))
            print(f"[OK] Ensured '{schema}' schema exists (PostgreSQL)")
        
        elif db_type in ("mysql", "mariadb"):
            # MySQL uses databases instead of schemas
            await conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {settings.mysql_db}"))
            await conn.execute(text(f"USE {settings.mysql_db}"))
            print(f"[OK] Ensured database '{settings.mysql_db}' exists (MySQL/MariaDB)")
        
        elif db_type == "sqlite":
            # SQLite doesn't need schema creation - file-based
            print("[OK] Using SQLite database file (no schema setup needed)")
        
        elif db_type == "sqlserver":
            # SQL Server can use schemas
            try:
                await conn.execute(text(f"CREATE SCHEMA {schema}"))
                print(f"[OK] Created '{schema}' schema (SQL Server)")
            except Exception:
                print(f"[OK] Schema '{schema}' already exists (SQL Server)")
        
        # Create all tables defined in Base.metadata
        await conn.run_sync(Base.metadata.create_all)
        print("[OK] Created all tables in configured database")
    
    # Run database migrations
    from .db_migrations import run_all_migrations
    async with async_session_factory() as session:
        await run_all_migrations(session)
