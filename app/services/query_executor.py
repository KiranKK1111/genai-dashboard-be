"""Query execution service."""

from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime, date, time
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


def serialize_value(value: Any) -> Any:
    """Convert non-JSON-serializable types to JSON-serializable formats.
    
    Handles:
    - datetime, date, time objects → ISO format strings
    - Decimal objects → float
    - None, bool, int, str, list, dict → passed through as-is
    """
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    elif isinstance(value, Decimal):
        return float(value)
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    return value


async def get_record_count(session: AsyncSession, sql: str) -> int:
    """Execute a COUNT probe to check record count before applying LIMIT.
    
    This is a "cheap" operation to determine if results exceed threshold (1000).
    If they do, we'll add LIMIT 1000. If not, we show all results.
    
    Args:
        session: An async SQLAlchemy session bound to the database.
        sql: The original SELECT query.
    
    Returns:
        The count of records that would be returned by the query.
    """
    try:
        # Transform SELECT ... FROM ... to COUNT(*)
        # Strategy: Replace the SELECT clause with COUNT(*)
        import re
        
        # First, strip trailing semicolon to avoid SQL syntax errors in subqueries
        sql_cleaned = sql.rstrip(';').rstrip()
        
        # Remove existing LIMIT/OFFSET clauses
        sql_no_limit = re.sub(r'\s+LIMIT\s+\d+\s*(?:;)?\s*', ' ', sql_cleaned, flags=re.IGNORECASE)
        sql_no_limit = re.sub(r'\s+OFFSET\s+\d+\s*', ' ', sql_no_limit, flags=re.IGNORECASE)
        sql_no_limit = sql_no_limit.strip()
        
        # Find WHERE, GROUP BY, HAVING, ORDER BY positions
        where_match = re.search(r'\sWHERE\s', sql_no_limit, re.IGNORECASE)
        group_match = re.search(r'\sGROUP\s+BY\s', sql_no_limit, re.IGNORECASE)
        
        # Strategy: If there's a GROUP BY, count groups. Otherwise count all records.
        if group_match:
            # For GROUP BY queries, wrap in COUNT(*)
            count_sql = f"SELECT COUNT(*) as cnt FROM ({sql_no_limit}) as count_subquery"
        else:
            # Simple transformation: Replace SELECT ... FROM with SELECT COUNT(*) FROM
            from_match = re.search(r'\s+FROM\s+', sql_no_limit, re.IGNORECASE)
            if from_match:
                # Find everything after FROM
                from_pos = from_match.start()
                count_sql = "SELECT COUNT(*) as cnt " + sql_no_limit[from_pos:]
            else:
                # Fallback (shouldn't happen)
                count_sql = f"SELECT COUNT(*) as cnt FROM ({sql_no_limit}) as subq"
        
        print(f"[COUNT PROBE] Executing: {count_sql}")
        
        result = await session.execute(text(count_sql))
        row = result.mappings().first()
        count = row['cnt'] if row else 0
        
        print(f"[COUNT PROBE] Result: {count} records")
        return int(count)
    
    except Exception as e:
        print(f"[ERROR] COUNT probe failed: {e}")
        # If count fails, assume we should apply LIMIT for safety
        return 1001


async def run_sql(session: AsyncSession, sql: str) -> List[Dict[str, Any]]:
    """Execute an arbitrary SQL query and return rows as dictionaries.

    Args:
        session: An async SQLAlchemy session bound to the database.
        sql: The SQL statement to execute.

    Returns:
        A list of dictionaries, one per row, with datetime objects converted to ISO strings.
    
    Raises:
        Any database exceptions from the SQL query execution.
    """
    try:
        # Execute the SQL query asynchronously
        result = await session.execute(text(sql))
        rows = result.mappings().all()
        
        # Convert datetime objects to strings for JSON serialization
        return [
            {key: serialize_value(value) for key, value in dict(r).items()}
            for r in rows
        ]
    
    except Exception as e:
        error_str = str(e)
        
        # Check for greenlet-related async context errors
        if "greenlet_spawn" in error_str or "await_only" in error_str:
            print(f"[ERROR] Async context error detected: {error_str}")
            print(f"[ERROR] This typically means we're mixing sync and async operations")
            print(f"[ERROR] SQL: {sql}")
            # Try to rollback anyway
            try:
                await session.rollback()
            except Exception as rollback_error:
                print(f"[ERROR] Rollback also failed: {rollback_error}")
            raise RuntimeError(
                f"Database async context error: {error_str[:200]}. "
                "This is an internal server error - please contact support."
            ) from e
        
        # For other errors, do normal rollback and re-raise
        try:
            await session.rollback()
        except Exception:
            pass  # Rollback might also fail for non-async reasons
        
        raise


async def apply_smart_limit(session: AsyncSession, sql: str, threshold: int = 1000) -> str:
    """Intelligently apply LIMIT based on actual record count.
    
    This is the "count-first" approach:
    1. Execute COUNT(*) probe to check record count
    2. If count <= threshold: return SQL without LIMIT (show all records)
    3. If count > threshold: append LIMIT threshold
    
    Args:
        session: An async SQLAlchemy session bound to the database.
        sql: The original SELECT query (should not have LIMIT yet).
        threshold: The threshold above which to apply LIMIT (default: 1000).
    
    Returns:
        Either the original SQL (no LIMIT) or modified SQL with LIMIT appended.
    """
    try:
        # Run count probe
        count = await get_record_count(session, sql)
        
        # Decision logic
        if count <= threshold:
            print(f"[SMART LIMIT] Record count: {count} (≤ {threshold}) → NO LIMIT, showing all records")
            # Strip any existing LIMIT clause just in case
            import re
            result = re.sub(r'\s+LIMIT\s+\d+\s*', ' ', sql, flags=re.IGNORECASE)
            result = re.sub(r'\s+OFFSET\s+\d+\s*', ' ', result, flags=re.IGNORECASE)
            return result.rstrip(';') + ';' if not result.endswith(';') else result
        else:
            print(f"[SMART LIMIT] Record count: {count} (> {threshold}) → APPLYING LIMIT {threshold}")
            # Add LIMIT clause
            import re
            # First remove any existing LIMIT
            result = re.sub(r'\s+LIMIT\s+\d+.*?;?$', '', sql, flags=re.IGNORECASE)
            result = result.rstrip(';')
            return f"{result} LIMIT {threshold};"
    
    except Exception as e:
        print(f"[ERROR] Smart LIMIT application failed: {e}")
        print(f"[ERROR] Falling back to original SQL with safety LIMIT")
        # Fallback: add a safety LIMIT
        import re
        result = re.sub(r'\s+LIMIT\s+\d+.*?;?$', '', sql, flags=re.IGNORECASE)
        result = result.rstrip(';')
        return f"{result} LIMIT {threshold};"
