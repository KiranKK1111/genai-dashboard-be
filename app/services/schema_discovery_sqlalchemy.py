"""
SQLAlchemy Inspector-based Schema Discovery Module.

Replaces raw information_schema queries with SQLAlchemy's Inspector API
for better portability across different databases.

Provides: Tables, columns, types, relationships discovery with caching.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import asyncio

from sqlalchemy import inspect, MetaData, Table as SATable
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine
from sqlalchemy.sql import text

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Information about a database column."""
    name: str
    type: str
    nullable: bool
    primary_key: bool
    foreign_keys: List[str]
    default: Optional[str] = None
    max_length: Optional[int] = None
    sample_values: Optional[List[Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TableInfo:
    """Information about a database table."""
    schema: Optional[str]
    name: str
    columns: List[ColumnInfo]
    row_count: int = 0
    sample_rows: Optional[List[Dict]] = None
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['columns'] = [col.to_dict() for col in self.columns]
        if self.last_updated:
            data['last_updated'] = self.last_updated.isoformat()
        return data


class SchemaCache:
    """
    Schema metadata cache with TTL (Time-To-Live).
    """
    
    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Cache validity period in seconds (default 5 minutes)
        """
        self.ttl_seconds = ttl_seconds
        self._tables_cache: Dict[str, Tuple[TableInfo, datetime]] = {}
        self._all_tables_cache: Optional[Tuple[List[str], datetime]] = None
        self._lock = asyncio.Lock()
    
    async def get_table(self, table_name: str) -> Optional[TableInfo]:
        """Get cached table info if valid."""
        async with self._lock:
            if table_name in self._tables_cache:
                table_info, cached_at = self._tables_cache[table_name]
                if self._is_valid(cached_at):
                    logger.debug(f"[CACHE HIT] Table {table_name}")
                    return table_info
                else:
                    # Expired
                    del self._tables_cache[table_name]
                    logger.debug(f"[CACHE EXPIRED] Table {table_name}")
            return None
    
    async def set_table(self, table_name: str, table_info: TableInfo) -> None:
        """Cache table info with timestamp."""
        async with self._lock:
            self._tables_cache[table_name] = (table_info, datetime.utcnow())
            logger.debug(f"[CACHE SET] Table {table_name}")
    
    async def get_all_tables(self) -> Optional[List[str]]:
        """Get cached list of all tables if valid."""
        async with self._lock:
            if self._all_tables_cache:
                tables, cached_at = self._all_tables_cache
                if self._is_valid(cached_at):
                    return tables
                else:
                    self._all_tables_cache = None
            return None
    
    async def set_all_tables(self, tables: List[str]) -> None:
        """Cache list of all tables."""
        async with self._lock:
            self._all_tables_cache = (tables, datetime.utcnow())
    
    def _is_valid(self, cached_at: datetime) -> bool:
        """Check if cache entry is still valid."""
        return datetime.utcnow() - cached_at < timedelta(seconds=self.ttl_seconds)
    
    async def invalidate(self, table_name: Optional[str] = None) -> None:
        """Invalidate cache entry(ies)."""
        async with self._lock:
            if table_name:
                self._tables_cache.pop(table_name, None)
                logger.debug(f"[CACHE INVALIDATED] Table {table_name}")
            else:
                self._tables_cache.clear()
                self._all_tables_cache = None
                logger.debug(f"[CACHE INVALIDATED] All")


class SQLAlchemySchemaDiscovery:
    """
    Schema discovery using SQLAlchemy's Inspector API.
    
    Works across all SQLAlchemy-supported databases without raw SQL queries.
    """
    
    def __init__(self, engine: AsyncEngine, cache_ttl_seconds: int = 300):
        """
        Initialize discovery.
        
        Args:
            engine: SQLAlchemy AsyncEngine
            cache_ttl_seconds: Schema cache TTL in seconds
        """
        self.engine = engine
        self.cache = SchemaCache(ttl_seconds=cache_ttl_seconds)
        self._metadata: Optional[MetaData] = None
    
    async def get_tables(
        self,
        schema: Optional[str] = None,
        exclude_internal: bool = True,
    ) -> List[str]:
        """
        Get list of tables in schema using SQLAlchemy Inspector.
        
        Args:
            schema: Schema name (optional, uses default if None)
            exclude_internal: Whether to exclude internal framework tables
            
        Returns:
            List of table names
        """
        # Try to get from cache
        cached = await self.cache.get_all_tables()
        if cached:
            return cached
        
        # Query using async connection and sync Inspector
        async with self.engine.connect() as conn:
            tables = await conn.run_sync(self._get_table_names, schema)
        
        if exclude_internal:
            internal_tables = {
                'users', 'chat_sessions', 'messages', 'uploaded_files',
                'file_chunks', 'tool_calls', 'sessions', 'session_state'
            }
            tables = [t for t in tables if t.lower() not in internal_tables]
        
        await self.cache.set_all_tables(tables)
        logger.info(f"[SCHEMA DISCOVERY] Found {len(tables)} tables in schema {schema}")
        return tables
    
    @staticmethod
    def _get_table_names(conn: Any, schema: Optional[str]) -> List[str]:
        """Sync method to get table names via Inspector."""
        inspector = inspect(conn)
        return inspector.get_table_names(schema=schema)
    
    async def get_table_info(
        self,
        table_name: str,
        schema: Optional[str] = None,
    ) -> Optional[TableInfo]:
        """
        Get detailed information about a table.
        
        Args:
            table_name: Table name
            schema: Schema name (optional)
            
        Returns:
            TableInfo object or None if table not found
        """
        # Check cache
        cached = await self.cache.get_table(table_name)
        if cached:
            return cached
        
        # Query using async connection and sync Inspector
        async with self.engine.connect() as conn:
            table_info = await conn.run_sync(
                self._get_table_info_sync,
                table_name,
                schema
            )
        
        if table_info:
            await self.cache.set_table(table_name, table_info)
        
        return table_info
    
    @staticmethod
    def _get_table_info_sync(
        conn: Any,
        table_name: str,
        schema: Optional[str]
    ) -> Optional[TableInfo]:
        """Sync method to get table info via Inspector."""
        inspector = inspect(conn)
        
        # Check if table exists
        if not table_name.lower() in [t.lower() for t in inspector.get_table_names(schema=schema)]:
            return None
        
        # Get columns
        columns_raw = inspector.get_columns(table_name, schema=schema)
        columns = []
        
        for col in columns_raw:
            # Get foreign keys
            fks = inspector.get_foreign_keys(table_name, schema=schema)
            col_fks = [f"{fk['referred_table']}.{fk['referred_columns'][0]}" 
                       for fk in fks if col['name'] in fk['constrained_columns']]
            
            # Get primary keys
            pk_cols = inspector.get_pk_constraint(table_name, schema=schema)
            is_pk = col['name'] in pk_cols.get('constrained_columns', []) if pk_cols else False
            
            column_info = ColumnInfo(
                name=col['name'],
                type=str(col['type']),
                nullable=col.get('nullable', True),
                primary_key=is_pk,
                foreign_keys=col_fks,
                default=col.get('default'),
                max_length=col.get('max_length'),
            )
            columns.append(column_info)
        
        # Get row count
        try:
            row_count_result = conn.execute(text(f"SELECT COUNT(*) as cnt FROM {table_name}"))
            row_count = row_count_result.scalar() or 0
        except Exception as e:
            logger.warning(f"Could not get row count for {table_name}: {e}")
            row_count = 0
        
        return TableInfo(
            schema=schema,
            name=table_name,
            columns=columns,
            row_count=row_count,
        )
    
    async def get_table_sample_values(
        self,
        table_name: str,
        schema: Optional[str] = None,
        sample_size: int = 5,
    ) -> Optional[List[Dict]]:
        """
        Get sample values from a table for value inference.
        
        Args:
            table_name: Table name
            schema: Schema name
            sample_size: Number of sample rows
            
        Returns:
            List of sample row dictionaries
        """
        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(
                    text(f"SELECT * FROM {table_name} LIMIT {sample_size}")
                )
                rows = result.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.warning(f"Could not fetch samples from {table_name}: {e}")
            return None
    
    async def get_primary_keys(
        self,
        table_name: str,
        schema: Optional[str] = None,
    ) -> List[str]:
        """Get primary key columns."""
        async with self.engine.connect() as conn:
            return await conn.run_sync(self._get_pk_sync, table_name, schema)
    
    @staticmethod
    def _get_pk_sync(conn: Any, table_name: str, schema: Optional[str]) -> List[str]:
        """Sync method to get primary keys."""
        inspector = inspect(conn)
        pk = inspector.get_pk_constraint(table_name, schema=schema)
        return pk.get('constrained_columns', []) if pk else []
    
    async def get_foreign_keys(
        self,
        table_name: str,
        schema: Optional[str] = None,
    ) -> List[Dict]:
        """Get foreign key relationships."""
        async with self.engine.connect() as conn:
            return await conn.run_sync(self._get_fk_sync, table_name, schema)
    
    @staticmethod
    def _get_fk_sync(conn: Any, table_name: str, schema: Optional[str]) -> List[Dict]:
        """Sync method to get foreign keys."""
        inspector = inspect(conn)
        return inspector.get_foreign_keys(table_name, schema=schema)


# Convenience functions
async def discover_tables(
    engine: AsyncEngine,
    schema: Optional[str] = None,
) -> List[str]:
    """Quick table discovery."""
    discovery = SQLAlchemySchemaDiscovery(engine)
    return await discovery.get_tables(schema=schema)


async def discover_table_info(
    engine: AsyncEngine,
    table_name: str,
    schema: Optional[str] = None,
) -> Optional[TableInfo]:
    """Quick table info discovery."""
    discovery = SQLAlchemySchemaDiscovery(engine)
    return await discovery.get_table_info(table_name, schema=schema)
