"""
Database initialization and adapter setup.

Automatically detects database type and configures appropriate adapter.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from .database_adapter import (
    DatabaseAdapterFactory,
    get_global_adapter,
    set_global_adapter,
    DatabaseAdapter,
    DatabaseType,
)
from ..config import settings

logger = logging.getLogger(__name__)


async def initialize_database_adapter() -> DatabaseAdapter:
    """
    Initialize the global database adapter based on configured connection string.
    
    This is called at application startup to detect database type and configure
    the appropriate adapter for all subsequent SQL operations.
    
    Returns:
        The initialized DatabaseAdapter
    """
    try:
        # Get connection string from settings
        connection_string = settings.database_url or ""
        
        # Detect database type
        db_type = DatabaseAdapterFactory.detect_database_type(connection_string)
        
        # Get appropriate adapter
        adapter = DatabaseAdapterFactory.get_adapter(db_type)
        
        # Set as global adapter
        set_global_adapter(adapter)
        
        # Log the initialization
        logger.info(f"✅ Database adapter initialized: {db_type.value}")
        logger.info(f"   Capabilities: Schema prefix={adapter.get_capabilities().supports_schema_prefix}, "
                   f"Native enum={adapter.get_capabilities().supports_enum_type}, "
                   f"Native boolean={adapter.get_capabilities().supports_boolean_type}")
        
        return adapter
    
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize database adapter: {e}")
        logger.warning(f"   Falling back to PostgreSQL adapter")
        
        # Fallback to PostgreSQL
        adapter = DatabaseAdapterFactory.get_adapter(DatabaseType.POSTGRESQL)
        set_global_adapter(adapter)
        
        return adapter


async def verify_database_connection(session: AsyncSession) -> bool:
    """
    Verify that the database connection works and adapter is properly configured.
    
    Args:
        session: AsyncSession for database operations
        
    Returns:
        True if connection verification succeeded
    """
    try:
        adapter = get_global_adapter()
        db_type = adapter.db_type
        
        logger.info(f"🔍 Verifying database connection ({db_type.value})...")
        
        # Try a simple query to verify connection
        from sqlalchemy import text
        
        result = await session.execute(text("SELECT 1"))
        if result:
            logger.info(f"✅ Database connection verified: {db_type.value}")
            return True
        else:
            logger.error("❌ Database query returned no result")
            return False
    
    except Exception as e:
        logger.error(f"❌ Database connection verification failed: {e}")
        return False
