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


async def migrate_setup_pgvector(session: AsyncSession) -> None:
    """
    Set up query_embeddings table for RAG embeddings persistence.
    
    This migration:
    1. Creates query_embeddings table with JSON-based vector storage
    2. Creates indexes for efficient lookups
    3. (Optional) Can enable pgvector extension if available for GPU-accelerated search
    
    This is idempotent - will not fail if table already exist.
    """
    schema = get_schema()
    print("\n[MIGRATION] Setting up query_embeddings table for RAG embeddings...")
    
    try:
        # 1. Try to create pgvector extension (optional, not required for basic functionality)
        try:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await session.commit()
            print("[OK] pgvector extension created/already exists (available for future optimization)")
        except Exception as e:
            # Rollback the transaction if pgvector creation fails
            await session.rollback()
            if "not available" in str(e).lower() or "not found" in str(e).lower():
                print(f"[INFO] pgvector extension not available - using JSON embeddings (still works)")
            else:
                # Only warn for unexpected errors
                print(f"[WARNING] pgvector setup skipped (continuing with JSON embeddings): {e}")
        
        # 2. Create query_embeddings table with JSON embeddings
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {schema}.query_embeddings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID NOT NULL REFERENCES {schema}.chat_sessions(id) ON DELETE CASCADE,
            user_query TEXT NOT NULL,
            generated_sql TEXT NOT NULL,
            result_count INTEGER NOT NULL DEFAULT 0,
            column_names JSON NOT NULL DEFAULT '[]'::json,
            first_row_sample JSON,
            embedding JSON NOT NULL,
            query_hash VARCHAR(64),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """
        await session.execute(text(create_table_sql))
        print(f"[OK] Created query_embeddings table in schema {schema}")
        
        # 3. Create indexes for efficient search
        indexes_to_create = [
            {
                "name": f"{schema}_query_embeddings_session_idx",
                "sql": f"CREATE INDEX IF NOT EXISTS {schema}_query_embeddings_session_idx ON {schema}.query_embeddings(session_id)"
            },
            {
                "name": f"{schema}_query_embeddings_created_idx",
                "sql": f"CREATE INDEX IF NOT EXISTS {schema}_query_embeddings_created_idx ON {schema}.query_embeddings(created_at DESC)"
            },
            {
                "name": f"{schema}_query_embeddings_hash_idx",
                "sql": f"CREATE INDEX IF NOT EXISTS {schema}_query_embeddings_hash_idx ON {schema}.query_embeddings(query_hash)"
            },
        ]
        
        for idx in indexes_to_create:
            try:
                await session.execute(text(idx["sql"]))
                print(f"[OK] Created index: {idx['name']}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"[OK] Index {idx['name']} already exists (skipped)")
                else:
                    print(f"[WARNING] Index creation failed for {idx['name']}: {e}")
        
        await session.commit()
        print("[OK] Query embeddings table setup completed successfully")
        print("[INFO] Note: pgvector extension can be installed later for GPU-accelerated similarity search")
    except Exception as e:
        await session.rollback()
        print(f"[ERROR] pgvector setup failed: {e}")
        raise


async def migrate_update_query_embeddings_all_result_rows(session: AsyncSession) -> None:
    """
    Add all_result_rows column to query_embeddings table.
    
    Migrates from storing just first_row_sample to storing ALL result rows.
    This is idempotent - will not fail if column already exists.
    """
    schema = get_schema()
    print("\n[MIGRATION] Adding all_result_rows column to query_embeddings table...")
    
    try:
        # Add all_result_rows column if it doesn't exist
        add_column_sql = f"""
        ALTER TABLE {schema}.query_embeddings
        ADD COLUMN IF NOT EXISTS all_result_rows JSON DEFAULT '[]'::json;
        """
        await session.execute(text(add_column_sql))
        print(f"[OK] Added all_result_rows column to query_embeddings table")
        
        await session.commit()
    except Exception as e:
        await session.rollback()
        if "already exists" in str(e).lower():
            print(f"[OK] all_result_rows column already exists (skipped)")
        else:
            print(f"[WARNING] Failed to add all_result_rows column: {e}")


async def migrate_add_result_quality_score(session: AsyncSession) -> None:
    """
    Add result_quality_score column to query_embeddings table.
    
    Tracks quality metrics for embeddings (0-100 scale).
    This is idempotent - will not fail if column already exists.
    """
    schema = get_schema()
    print("\n[MIGRATION] Adding result_quality_score column to query_embeddings table...")
    
    try:
        # Add result_quality_score column if it doesn't exist
        add_column_sql = f"""
        ALTER TABLE {schema}.query_embeddings
        ADD COLUMN IF NOT EXISTS result_quality_score INTEGER DEFAULT NULL;
        """
        await session.execute(text(add_column_sql))
        print(f"[OK] Added result_quality_score column to query_embeddings table")
        
        await session.commit()
    except Exception as e:
        await session.rollback()
        if "already exists" in str(e).lower():
            print(f"[OK] result_quality_score column already exists (skipped)")
        else:
            print(f"[WARNING] Failed to add result_quality_score column: {e}")


async def migrate_drop_first_row_sample(session: AsyncSession) -> None:
    """
    Drop first_row_sample column from query_embeddings table.
    
    Removes deprecated column that stored only first row.
    Now using all_result_rows to store all rows instead.
    This is idempotent - will not fail if column doesn't exist.
    """
    schema = get_schema()
    print("\n[MIGRATION] Removing deprecated first_row_sample column from query_embeddings table...")
    
    try:
        # Drop first_row_sample column if it exists
        drop_column_sql = f"""
        ALTER TABLE {schema}.query_embeddings
        DROP COLUMN IF EXISTS first_row_sample;
        """
        await session.execute(text(drop_column_sql))
        print(f"[OK] Dropped first_row_sample column from query_embeddings table")
        
        await session.commit()
    except Exception as e:
        await session.rollback()
        if "does not exist" in str(e).lower() or "no such column" in str(e).lower():
            print(f"[OK] first_row_sample column doesn't exist (skipped)")
        else:
            print(f"[WARNING] Failed to drop first_row_sample column: {e}")


async def migrate_update_embedding_dimensions(session: AsyncSession) -> None:
    """
    Update embedding column from 768 to 384 dimensions.
    
    Reason: Switched from hash-based sparse vectors (768 dims) to 
    sentence-transformers dense vectors (384 dims for all-MiniLM-L6-v2 model).
    
    This migration:
    1. Drops the old index (if any exists)
    2. Drops the old embedding column
    3. Creates new 384-dimensional embedding column (vector if pgvector available, JSON fallback)
    4. Tries to recreate index (skips if pgvector unavailable)
    
    Note: This clears all existing embeddings (non-breaking since they were 
    from old hash-based system and will be regenerated automatically).
    
    This is idempotent - safe to run multiple times.
    """
    schema = get_schema()
    print("\n[MIGRATION] Updating embedding column from 768 to 384 dimensions...")
    
    try:
        # Step 1: Drop old index if it exists
        try:
            drop_index_sql = f"DROP INDEX IF EXISTS {schema}.idx_query_embeddings_vector;"
            await session.execute(text(drop_index_sql))
            await session.commit()
            print(f"[OK] Dropped old index")
        except Exception as e:
            print(f"[DEBUG] Index drop (non-critical): {e}")
            await session.rollback()
        
        # Step 2: Drop old embedding column if it exists
        try:
            drop_col_sql = f"""
            ALTER TABLE {schema}.query_embeddings
            DROP COLUMN IF EXISTS embedding;
            """
            await session.execute(text(drop_col_sql))
            await session.commit()
            print(f"[OK] Dropped old embedding column")
        except Exception as e:
            await session.rollback()
            print(f"[DEBUG] Column drop (non-critical): {e}")
        
        # Step 3: Recreate embedding column with new dimensions
        # Try vector type first (if pgvector available), fall back to JSON
        pgvector_available = False
        try:
            # Try to create as vector(384) - requires pgvector
            add_vector_sql = f"""
            ALTER TABLE {schema}.query_embeddings
            ADD COLUMN IF NOT EXISTS embedding vector(384);
            """
            await session.execute(text(add_vector_sql))
            await session.commit()
            print(f"[OK] Created embedding column as vector(384) - pgvector optimized")
            pgvector_available = True
        except Exception as e:
            await session.rollback()
            if "type vector does not exist" in str(e).lower() or "does not exist" in str(e).lower():
                # pgvector not available - use JSON instead
                print(f"[INFO] pgvector not available, using JSON for embedding column")
                try:
                    add_json_sql = f"""
                    ALTER TABLE {schema}.query_embeddings
                    ADD COLUMN IF NOT EXISTS embedding JSON;
                    """
                    await session.execute(text(add_json_sql))
                    await session.commit()
                    print(f"[OK] Created embedding column as JSON (pgvector not available)")
                except Exception as e2:
                    await session.rollback()
                    print(f"[ERROR] Failed to create embedding column: {e2}")
            else:
                print(f"[DEBUG] Vector column creation: {e}")
        
        # Step 4: Try to create index for embeddings (only if pgvector available)
        if pgvector_available:
            try:
                create_index_sql = f"""
                CREATE INDEX IF NOT EXISTS idx_query_embeddings_vector 
                ON {schema}.query_embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
                """
                await session.execute(text(create_index_sql))
                await session.commit()
                print(f"[OK] Recreated ivfflat index for 384-dimensional embeddings (pgvector optimized)")
            except Exception as e:
                await session.rollback()
                if "ivfflat" in str(e).lower() or "does not exist" in str(e).lower():
                    # ivfflat not available - try simple index as fallback
                    print(f"[INFO] ivfflat not available, using simple B-tree index instead")
                    try:
                        create_simple_index_sql = f"""
                        CREATE INDEX IF NOT EXISTS idx_query_embeddings_vector 
                        ON {schema}.query_embeddings (embedding);
                        """
                        await session.execute(text(create_simple_index_sql))
                        await session.commit()
                        print(f"[OK] Created simple index for embeddings (slower but works without pgvector ivfflat)")
                    except Exception as e2:
                        await session.rollback()
                        print(f"[WARNING] Failed to create simple index: {e2}")
                else:
                    print(f"[WARNING] Failed to recreate index: {e}")
        else:
            print(f"[INFO] Skipping vector-specific index creation (using JSON embeddings)")
            
    except Exception as e:
        await session.rollback()
        print(f"[WARNING] Embedding dimension update had issues: {e}")


async def migrate_add_chat_session_title_and_updated_at(session: AsyncSession) -> None:
    """
    Add 'title' and 'updated_at' columns to chat_sessions table.

    These are used for ChatGPT-style session rename and ordering by last activity.
    Idempotent — safe to run when columns already exist.
    """
    schema = get_schema()
    print("\n[MIGRATION] Adding title/updated_at columns to chat_sessions...")

    migrations = [
        f"""
        ALTER TABLE {schema}.chat_sessions
        ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE
            DEFAULT CURRENT_TIMESTAMP;
        """,
        f"""
        ALTER TABLE {schema}.chat_sessions
        ADD COLUMN IF NOT EXISTS title VARCHAR(255) DEFAULT NULL;
        """,
        # Back-fill updated_at from created_at for existing rows
        f"""
        UPDATE {schema}.chat_sessions
        SET updated_at = created_at
        WHERE updated_at IS NULL;
        """,
    ]

    for i, sql in enumerate(migrations, 1):
        try:
            await session.execute(text(sql))
            await session.commit()
            print(f"[OK] chat_sessions migration {i} completed")
        except Exception as e:
            await session.rollback()
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print(f"[OK] chat_sessions migration {i}: already applied (skipped)")
            else:
                print(f"[ERROR] chat_sessions migration {i} failed: {e}")
                raise


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
        # Add title + updated_at columns used for session rename & ordering
        await migrate_add_chat_session_title_and_updated_at(session)
        await migrate_create_tool_calls_table(session)
        await migrate_setup_pgvector(session)  # Set up pgvector for RAG embeddings
        await migrate_update_query_embeddings_all_result_rows(session)  # Add all_result_rows column
        await migrate_add_result_quality_score(session)  # Add result_quality_score column
        await migrate_drop_first_row_sample(session)  # Remove deprecated first_row_sample column
        await migrate_update_embedding_dimensions(session)  # Update from 768 to 384 dims (sentence-transformers)
        
        # NEW: Restructure messages table (single row per conversation turn)
        from .db_migrations_messages import migrate_messages_table_restructure
        await migrate_messages_table_restructure(session)
    except Exception as e:
        print("\n" + "="*60)
        print(f"[ERROR] MIGRATIONS FAILED: {e}")
        print("="*60)
        raise
