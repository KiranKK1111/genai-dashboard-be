"""
Database migrations for adding session state management columns.

This module provides functions to migrate the database schema
when adding new columns or tables to support ChatGPT-style session state.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from .config import settings


def get_schema() -> str:
    """Get the database schema from settings."""
    return getattr(settings, 'postgres_schema', 'public')


async def migrate_add_session_state_columns(session: AsyncSession) -> None:
    """
    Add session state management columns to chat_sessions table.
    
    Adds:
    - session_state (JSONB): Persisted QueryState
    - tool_calls_log (JSONB): Tool execution history
    - state_updated_at (TIMESTAMP): When state was last updated
    
    This is idempotent - will not fail if columns already exist.
    """
    schema = get_schema()
    print("\n[MIGRATION] Adding session state columns to chat_sessions...")
    
    migrations = [
        # Add session_state column if it doesn't exist
        f"""
        ALTER TABLE {schema}.chat_sessions
        ADD COLUMN IF NOT EXISTS session_state JSONB DEFAULT NULL;
        """,
        
        # Add tool_calls_log column if it doesn't exist
        f"""
        ALTER TABLE {schema}.chat_sessions
        ADD COLUMN IF NOT EXISTS tool_calls_log JSONB DEFAULT NULL;
        """,
        
        # Add state_updated_at column if it doesn't exist
        f"""
        ALTER TABLE {schema}.chat_sessions
        ADD COLUMN IF NOT EXISTS state_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NULL;
        """,
    ]
    
    for i, migration_sql in enumerate(migrations, 1):
        try:
            await session.execute(text(migration_sql))
            await session.commit()
            print(f"[OK] Migration {i} completed successfully")
        except Exception as e:
            await session.rollback()
            # Check if it's already-exists error (column already exists)
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print(f"[OK] Migration {i}: Column already exists (skipped)")
            else:
                print(f"[ERROR] Migration {i} failed: {e}")
                raise


async def migrate_add_result_schema_columns(session: AsyncSession) -> None:
    """
    Add result schema tracking columns to chat_sessions table.
    
    Adds:
    - last_result_schema (JSON): Schema of the last query result
    - last_result_row_count (INTEGER): Number of rows in last result
    - last_result_samples (JSON): Sample values from last result
    
    These columns are used for follow-up query validation and context.
    This is idempotent - will not fail if columns already exist.
    """
    schema = get_schema()
    print("\n[MIGRATION] Adding result schema columns to chat_sessions...")
    
    migrations = [
        # Add last_result_schema column if it doesn't exist
        f"""
        ALTER TABLE {schema}.chat_sessions
        ADD COLUMN IF NOT EXISTS last_result_schema JSON DEFAULT NULL;
        """,
        
        # Add last_result_row_count column if it doesn't exist
        f"""
        ALTER TABLE {schema}.chat_sessions
        ADD COLUMN IF NOT EXISTS last_result_row_count INTEGER DEFAULT NULL;
        """,
        
        # Add last_result_samples column if it doesn't exist
        f"""
        ALTER TABLE {schema}.chat_sessions
        ADD COLUMN IF NOT EXISTS last_result_samples JSON DEFAULT NULL;
        """,
    ]
    
    for i, migration_sql in enumerate(migrations, 1):
        try:
            await session.execute(text(migration_sql))
            await session.commit()
            print(f"[OK] Result schema migration {i} completed successfully")
        except Exception as e:
            await session.rollback()
            # Check if it's already-exists error (column already exists)
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print(f"[OK] Result schema migration {i}: Column already exists (skipped)")
            else:
                print(f"[ERROR] Result schema migration {i} failed: {e}")
                raise


async def migrate_create_tool_calls_table(session: AsyncSession) -> None:
    """
    Create ToolCall table if it doesn't exist.
    
    Table structure:
    - id (UUID): Primary key
    - session_id (UUID): Foreign key to chat_sessions
    - tool_type (VARCHAR): Type of tool (sql_generation, etc)
    - input_json (JSONB): Tool input parameters
    - output_json (JSONB): Tool output/results
    - executed_at (TIMESTAMP): When tool was executed
    - duration_ms (INTEGER): How long execution took
    """
    schema = get_schema()
    print("\n[MIGRATION] Creating ToolCall table if needed...")
    
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {schema}.tool_calls (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id UUID NOT NULL REFERENCES {schema}.chat_sessions(id) ON DELETE CASCADE,
        tool_type VARCHAR(50) NOT NULL,
        input_json JSONB NOT NULL,
        output_json JSONB,
        success BOOLEAN DEFAULT TRUE,
        error_message TEXT,
        start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        end_time TIMESTAMP WITH TIME ZONE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT fk_tool_calls_session
            FOREIGN KEY (session_id)
            REFERENCES {schema}.chat_sessions(id)
            ON DELETE CASCADE
    );
    """
    
    try:
        await session.execute(text(create_table_sql))
        await session.commit()
        print("[OK] ToolCall table created successfully")
    except Exception as e:
        await session.rollback()
        if "already exists" in str(e).lower() or "relation" in str(e).lower():
            print("[OK] ToolCall table already exists (skipped)")
        else:
            print(f"[ERROR] Failed to create ToolCall table: {e}")
            raise


async def migrate_add_indexes(session: AsyncSession) -> None:
    """
    Create indexes for better query performance.
    """
    schema = get_schema()
    print("\n[MIGRATION] Creating indexes for session state queries...")
    
    indexes = [
        # Index for looking up sessions by user_id
        f"""
        CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id
        ON {schema}.chat_sessions(user_id);
        """,
        
        # Index for looking up tool calls by session_id
        f"""
        CREATE INDEX IF NOT EXISTS idx_tool_calls_session_id
        ON {schema}.tool_calls(session_id);
        """,
        
        # Index for looking up recent tool calls
        f"""
        CREATE INDEX IF NOT EXISTS idx_tool_calls_start_time
        ON {schema}.tool_calls(start_time DESC);
        """,
    ]
    
    for i, index_sql in enumerate(indexes, 1):
        try:
            await session.execute(text(index_sql))
            await session.commit()
            print(f"[OK] Index {i} created successfully")
        except Exception as e:
            await session.rollback()
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print(f"[OK] Index {i} already exists (skipped)")
            else:
                print(f"[WARNING] Index {i} failed: {e}")
                # Don't raise - indexes are not critical


async def run_all_migrations(session: AsyncSession) -> None:
    """Run all pending database migrations."""
    schema = get_schema()
    print("\n" + "="*60)
    print("RUNNING DATABASE MIGRATIONS")
    print("="*60)
    
    try:
        # Set search path to configured schema
        await session.execute(text(f"SET search_path TO {schema}, public"))
        
        # Run migrations in order
        await migrate_add_session_state_columns(session)
        await migrate_add_result_schema_columns(session)
        await migrate_create_tool_calls_table(session)
        await migrate_add_indexes(session)
        
        print("\n" + "="*60)
        print("[OK] ALL MIGRATIONS COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"[ERROR] MIGRATIONS FAILED: {e}")
        print("="*60)
        raise
