"""
Schema Discovery Service - Fetches and caches database metadata.

This service discovers database schema at startup/on-demand:
- Tables and columns
- Data types
- Primary/foreign keys
- Column sample values
- Column statistics

The schema is stored in memory for fast access and used by the Decision Engine
to make database-agnostic decisions about which table/column to use.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Metadata about a database column."""
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_table: Optional[str] = None
    foreign_key_column: Optional[str] = None
    sample_values: List[Any] = field(default_factory=list)
    row_count: int = 0
    distinct_count: int = 0
    enum_values: List[str] = field(default_factory=list)  # Valid enum values for USER-DEFINED types
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "nullable": self.nullable,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "foreign_key_table": self.foreign_key_table,
            "foreign_key_column": self.foreign_key_column,
            "sample_values": self.sample_values,
            "row_count": self.row_count,
            "distinct_count": self.distinct_count,
            "enum_values": self.enum_values,
        }


@dataclass
class TableInfo:
    """Metadata about a database table."""
    name: str
    schema: str
    columns: Dict[str, ColumnInfo] = field(default_factory=dict)
    row_count: int = 0
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # col -> (table, col)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "schema": self.schema,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "row_count": self.row_count,
            "primary_keys": self.primary_keys,
            "foreign_keys": self.foreign_keys,
        }


@dataclass
class SchemaDatabase:
    """Complete schema catalog for a database."""
    tables: Dict[str, TableInfo] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tables": {k: v.to_dict() for k, v in self.tables.items()},
            "last_updated": self.last_updated.isoformat(),
        }


class SchemaCatalog:
    """
    Maintains a runtime cache of database schema.
    
    Provides:
    - Fast lookup of tables/columns without DB queries
    - Schema information for LLM context
    - Candidate generation for entity matching
    """
    
    def __init__(self, schema_name: str = "public"):
        self.schema_name = schema_name
        self.database: SchemaDatabase = SchemaDatabase()
        self._initialized = False
    
    async def initialize(self, db: AsyncSession) -> None:
        """Discover and cache schema from database."""
        if self._initialized:
            return
        
        logger.info(f"Initializing schema catalog for schema '{self.schema_name}'...")
        
        try:
            # Fetch all tables
            tables_result = await db.execute(
                text(f"""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = :schema
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """),
                {"schema": self.schema_name}
            )
            table_names = [row[0] for row in tables_result.fetchall()]
            
            for table_name in table_names:
                await self._discover_table(db, table_name)
            
            self._initialized = True
            logger.info(f"✓ Schema catalog initialized: {len(table_names)} tables discovered")
            
        except Exception as e:
            logger.error(f"Failed to initialize schema catalog: {e}")
            try:
                await db.rollback()
                logger.info("[ROLLBACK] Transaction rolled back after initialization error")
            except Exception:
                pass
            raise
    
    async def _discover_table(self, db: AsyncSession, table_name: str) -> None:
        """Discover metadata for a single table."""
        try:
            # Fetch columns
            columns_result = await db.execute(
                text(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = :schema
                    AND table_name = :table
                    ORDER BY ordinal_position
                """),
                {"schema": self.schema_name, "table": table_name}
            )
            
            columns = {}
            for row in columns_result.fetchall():
                col_name, data_type, is_nullable = row
                columns[col_name] = ColumnInfo(
                    name=col_name,
                    data_type=str(data_type),
                    nullable=is_nullable == "YES"
                )
            
            # Fetch primary keys
            pk_result = await db.execute(
                text(f"""
                    SELECT column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu USING(constraint_name)
                    WHERE tc.table_schema = :schema
                    AND tc.table_name = :table
                    AND tc.constraint_type = 'PRIMARY KEY'
                """),
                {"schema": self.schema_name, "table": table_name}
            )
            primary_keys = [row[0] for row in pk_result.fetchall()]
            for pk_col in primary_keys:
                if pk_col in columns:
                    columns[pk_col].is_primary_key = True
            
            # Get row count
            count_result = await db.execute(
                text(f"SELECT COUNT(*) FROM {self.schema_name}.{table_name}")
            )
            row_count = count_result.scalar() or 0
            
            # Fetch enum values for USER-DEFINED type columns (PostgreSQL only)
            await self._fetch_enum_values(db, table_name, columns)
            
            # Fetch actual sample values from the database for semantic matching
            await self._fetch_sample_values(db, table_name, columns)
            
            # Store table info
            table_info = TableInfo(
                name=table_name,
                schema=self.schema_name,
                columns=columns,
                row_count=row_count,
                primary_keys=primary_keys,
            )
            
            self.database.tables[table_name] = table_info
            
        except Exception as e:
            logger.warning(f"Failed to discover table {table_name}: {e}")
            try:
                await db.rollback()
            except Exception:
                pass
    
    async def _fetch_enum_values(self, db: AsyncSession, table_name: str, columns: Dict[str, ColumnInfo]) -> None:
        """Fetch valid enum values for USER-DEFINED type columns (PostgreSQL)."""
        try:
            # For each column, check if it's an enum type and fetch its values
            for col_name, col_info in columns.items():
                # Check if data_type contains 'USER-DEFINED' pattern (PostgreSQL enum)
                if 'USER-DEFINED' not in str(col_info.data_type):
                    continue
                
                # Query pg_enum to get valid enum values
                enum_query = await db.execute(
                    text("""
                        SELECT e.enumlabel
                        FROM pg_type t
                        JOIN information_schema.columns c ON t.typname = c.udt_name
                        JOIN pg_enum e ON t.oid = e.enumtypid
                        WHERE c.table_schema = :schema
                        AND c.table_name = :table
                        AND c.column_name = :column
                        ORDER BY e.enumsortorder
                    """),
                    {"schema": self.schema_name, "table": table_name, "column": col_name}
                )
                
                enum_values = [row[0] for row in enum_query.fetchall()]
                if enum_values:
                    col_info.enum_values = enum_values
                    logger.info(f"Found {len(enum_values)} enum values for {table_name}.{col_name}: {enum_values}")
        
        except Exception as e:
            logger.warning(f"Failed to fetch enum values for table {table_name}: {e}")
    
    async def _fetch_sample_values(self, db: AsyncSession, table_name: str, columns: Dict[str, ColumnInfo]) -> None:
        """Fetch actual sample values from database columns for semantic matching."""
        try:
            # Get up to 5 distinct values from each column for context
            for col_name, col_info in columns.items():
                # Skip primary keys (usually just IDs) and foreign keys
                if col_info.is_primary_key or col_info.is_foreign_key:
                    continue
                
                # Skip identifier columns (just the name contains 'id')
                if 'id' in col_name.lower() and col_info.data_type.lower() in ['bigint', 'integer', 'int']:
                    continue
                
                try:
                    # Query to get distinct values from this column
                    sample_query = await db.execute(
                        text(f"""
                            SELECT DISTINCT {col_name}
                            FROM {self.schema_name}.{table_name}
                            WHERE {col_name} IS NOT NULL
                            LIMIT 5
                        """)
                    )
                    
                    sample_values = [row[0] for row in sample_query.fetchall()]
                    if sample_values:
                        col_info.sample_values = sample_values
                        logger.debug(f"Found {len(sample_values)} sample values for {table_name}.{col_name}: {sample_values}")
                
                except Exception as e:
                    # If query fails for this column, continue to next
                    logger.debug(f"Could not fetch samples for {table_name}.{col_name}: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"Failed to fetch sample values for table {table_name}: {e}")
    
    def get_all_tables(self) -> Dict[str, TableInfo]:
        """Get all discovered tables."""
        return self.database.tables
    
    def get_table(self, table_name: str) -> Optional[TableInfo]:
        """Get info for a specific table."""
        return self.database.tables.get(table_name)
    
    def get_column(self, table_name: str, column_name: str) -> Optional[ColumnInfo]:
        """Get info for a specific column."""
        table = self.get_table(table_name)
        if table:
            return table.columns.get(column_name)
        return None
    
    def list_column_names_by_type(self, data_type_pattern: str) -> Dict[str, List[str]]:
        """Get all columns matching a data type pattern (e.g., 'integer%')."""
        result = {}
        for table_name, table_info in self.database.tables.items():
            matching_cols = [
                col_name
                for col_name, col_info in table_info.columns.items()
                if col_info.data_type.lower().startswith(data_type_pattern.lower())
            ]
            if matching_cols:
                result[table_name] = matching_cols
        return result
    
    def get_tables_with_column_name(self, column_name: str) -> Dict[str, ColumnInfo]:
        """Find all columns with a given name across tables."""
        result = {}
        for table_name, table_info in self.database.tables.items():
            if column_name in table_info.columns:
                result[table_name] = table_info.columns[column_name]
        return result
    
    def get_schema_summary(self) -> str:
        """Get a text summary of the schema for LLM context."""
        lines = []
        lines.append(f"Database Schema ({self.schema_name}):")
        lines.append("-" * 60)
        
        for table_name in sorted(self.database.tables.keys()):
            table = self.database.tables[table_name]
            lines.append(f"\nTable: {table_name} ({table.row_count:,} rows)")
            
            for col_name in sorted(table.columns.keys()):
                col = table.columns[col_name]
                pk_marker = " [PK]" if col.is_primary_key else ""
                fk_marker = f" -> {col.foreign_key_table}.{col.foreign_key_column}" if col.is_foreign_key else ""
                lines.append(f"  - {col_name}: {col.data_type}{pk_marker}{fk_marker}")
        
        return "\n".join(lines)


async def create_schema_catalog(db: AsyncSession, schema_name: str = "public") -> SchemaCatalog:
    """Factory function to create and initialize a schema catalog."""
    catalog = SchemaCatalog(schema_name)
    await catalog.initialize(db)
    return catalog
