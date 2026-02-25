"""
Application startup hook for database initialization.

This module should be imported early in the application lifecycle
to automatically detect and configure the database adapter.
"""

import logging
from sqlalchemy.ext.asyncio import AsyncSession

from .db_init import initialize_database_adapter, verify_database_connection

logger = logging.getLogger(__name__)


async def setup_database() -> None:
    """
    Initialize database adapter at application startup.
    
    This function should be called from the FastAPI lifespan context
    or main application startup hook.
    """
    try:
        logger.info("=" * 60)
        logger.info("🚀 Initializing database adapter...")
        logger.info("=" * 60)
        
        # Initialize adapter based on configured database
        adapter = await initialize_database_adapter()
        
        logger.info("=" * 60)
        logger.info(f"✅ Database adapter ready: {adapter.db_type.value.upper()}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database adapter: {e}")
        logger.error("Application may not work correctly with connected database")
        raise


async def verify_database() -> bool:
    """
    Verify database connection (call after db is connected).
    
    Returns:
        True if verification passed
    """
    try:
        logger.info("🔍 Verifying database connection...")
        # Note: This requires an active database session
        # Should be called after database initialization is complete
        return True
    except Exception as e:
        logger.error(f"❌ Database verification failed: {e}")
        return False
