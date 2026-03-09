"""
Migration for restructuring messages table.

Changes:
1. Add 'feedback' column for user feedback
2. Rename 'created_at' to 'updated_at'
3. Make 'query' NOT NULL (enforces single-row design)
4. Clean up logic to use single-row per conversation turn
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from .config import get_schema  # Import existing get_schema function


async def migrate_messages_table_restructure(session: AsyncSession) -> None:
    """
    Restructure messages table to store user+assistant in single row.
    
    Changes:
    - Add 'feedback' column (TEXT, nullable)
    - Rename 'created_at' to 'updated_at'
    - Make 'query' column NOT NULL
    """
    schema = get_schema()
    print("\n[MIGRATION] Restructuring messages table...")
    
    migrations = [
        # Add feedback column
        f"""
        ALTER TABLE {schema}.messages
        ADD COLUMN IF NOT EXISTS feedback TEXT DEFAULT NULL;
        """,
        
        # Rename created_at to updated_at (PostgreSQL syntax)
        f"""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = '{schema}'
                AND table_name = 'messages'
                AND column_name = 'created_at'
            ) THEN
                ALTER TABLE {schema}.messages RENAME COLUMN created_at TO updated_at;
            END IF;
        END $$;
        """,
        
        # Update query column to NOT NULL (fill NULLs first with empty string)
        f"""
        UPDATE {schema}.messages
        SET query = ''
        WHERE query IS NULL;
        """,
        
        f"""
        ALTER TABLE {schema}.messages
        ALTER COLUMN query SET NOT NULL;
        """,
    ]
    
    for i, migration_sql in enumerate(migrations, 1):
        try:
            await session.execute(text(migration_sql))
            await session.commit()
            print(f"[OK] Messages migration step {i} completed")
        except Exception as e:
            await session.rollback()
            error_str = str(e).lower()
            if "already exists" in error_str or "does not exist" in error_str:
                print(f"[OK] Step {i} already applied (skipped)")
            else:
                print(f"[ERROR] Messages migration step {i} failed: {e}")
                raise


async def run_messages_migrations(session: AsyncSession) -> None:
    """Run all message table migrations."""
    print("\n" + "="*60)
    print("RUNNING MESSAGES TABLE MIGRATIONS")
    print("="*60)
    
    try:
        schema = get_schema()
        await session.execute(text(f"SET search_path TO {schema}, public"))
        await migrate_messages_table_restructure(session)
        print("\n" + "="*60)
        print("MESSAGES MIGRATIONS COMPLETED SUCCESSFULLY")
        print("="*60 + "\n")
    except Exception as e:
        print("\n" + "="*60)
        print(f"MESSAGES MIGRATIONS FAILED: {e}")
        print("="*60 + "\n")
        raise
