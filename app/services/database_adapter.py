"""
Database abstraction layer for multi-database support.

Provides database-agnostic interfaces for:
- Schema metadata queries
- Error pattern detection and classification
- Boolean/enum type handling
- Database-specific SQL syntax

Supports: PostgreSQL, MySQL, SQLite, SQL Server, Oracle
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any, TYPE_CHECKING
import re
import logging
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession
from ..config import get_schema

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .schema_metadata import TableMetadata, DatabaseSchema, ColumnMetadata


class DatabaseType(Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    SQL_SERVER = "sqlserver"
    ORACLE = "oracle"
    UNKNOWN = "unknown"


@dataclass
class DatabaseCapabilities:
    """Capabilities and characteristics of a database."""
    db_type: DatabaseType
    supports_schema_prefix: bool  # Can use schema.table notation
    schema_prefix_style: str  # How to format schema prefix (e.g., "schema.table" or "schema:table")
    supports_enum_type: bool  # Has native enum type
    boolean_literals: Tuple[str, str]  # (true_literal, false_literal) e.g., ("true", "false")
    supports_boolean_type: bool  # Has native boolean type
    uses_information_schema: bool  # Can query information_schema
    metadata_system: str  # "information_schema", "pragma", "sys", "all_tab_columns"
    error_patterns: Dict[str, List[str]]  # Maps error type to list of error message patterns
    default_schema: Optional[str]  # Default schema name if applicable


class DatabaseAdapter(ABC):
    """Abstract base class for database adapters."""
    
    def __init__(self, db_type: DatabaseType):
        self.db_type = db_type
    
    @property
    def dialect(self) -> str:
        """Return SQLAlchemy dialect name for this database type."""
        dialect_map = {
            DatabaseType.POSTGRESQL: 'postgresql',
            DatabaseType.MYSQL: 'mysql',
            DatabaseType.SQLITE: 'sqlite',
            DatabaseType.SQL_SERVER: 'mssql',
            DatabaseType.ORACLE: 'oracle',
            DatabaseType.UNKNOWN: 'postgresql',  # Default fallback
        }
        return dialect_map.get(self.db_type, 'postgresql')
    
    @abstractmethod
    def get_capabilities(self) -> DatabaseCapabilities:
        """Return database capabilities."""
        pass
    
    @abstractmethod
    async def get_table_columns(
        self, 
        session: AsyncSession, 
        table_name: str, 
        schema_name: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """
        Get list of (column_name, data_type) for a table.
        
        Returns:
            List of (column_name, column_type) tuples
        """
        pass
    
    @abstractmethod
    def classify_error(self, error_message: str) -> str:
        """
        Classify an error into one of:
        - "type_error": Type mismatch or operator not supported
        - "undefined_column": Column doesn't exist
        - "undefined_table": Table doesn't exist
        - "enum_error": Invalid enum value
        - "syntax_error": SQL syntax error
        - "unknown": Unknown error type
        
        Returns:
            Error classification string
        """
        pass
    
    @abstractmethod
    def normalize_boolean_value(self, value: str) -> str:
        """
        Normalize a boolean value to database-appropriate literal.
        
        Args:
            value: One of "true", "false", "1", "0", "yes", "no", "True", "False", etc.
            
        Returns:
            Normalized boolean literal for this database
        """
        pass
    
    @abstractmethod
    def normalize_enum_value(self, value: str) -> str:
        """
        Normalize an enum value to database-appropriate format.
        
        Most databases use lowercase with underscores, but some differ.
        
        Args:
            value: The enum value to normalize
            
        Returns:
            Normalized enum value
        """
        pass
    
    @abstractmethod
    def format_schema_qualified_table(self, table_name: str, schema_name: Optional[str] = None) -> str:
        """
        Format a table reference with optional schema qualification.
        
        Args:
            table_name: The table name
            schema_name: Optional schema name
            
        Returns:
            Properly formatted table reference for this database
        """
        pass
    
    @abstractmethod
    def extract_boolean_columns_from_schema(self, schema_text: str) -> List[str]:
        """
        Extract boolean column names from schema text.
        
        Returns:
            List of column names that are boolean type
        """
        pass
    
    @abstractmethod
    def extract_enum_columns_from_schema(self, schema_text: str) -> Dict[str, List[str]]:
        """
        Extract enum column definitions from schema text.
        
        Returns:
            Dict mapping column names to list of valid enum values
        """
        pass
    
    @abstractmethod
    async def reflect_table_metadata(
        self,
        session: AsyncSession,
        table_name: str,
        schema_name: Optional[str] = None
    ) -> 'TableMetadata':
        """
        Reflect complete table metadata using SQLAlchemy introspection.
        
        This replaces regex-based schema parsing with proper database introspection.
        Returns structured TableMetadata with column details, types, keys, etc.
        
        Args:
            session: Database session
            table_name: Table to reflect
            schema_name: Optional schema name
            
        Returns:
            TableMetadata object with full table structure
        """
        pass
    
    @abstractmethod
    async def reflect_database_schema(
        self,
        session: AsyncSession,
        database_name: str,
        schema_name: Optional[str] = None
    ) -> 'DatabaseSchema':
        """
        Reflect complete database schema with all tables.
        
        Args:
            session: Database session
            database_name: Database name
            schema_name: Optional schema to restrict to
            
        Returns:
            DatabaseSchema object with all tables and their metadata
        """
        pass
    
    @abstractmethod
    async def get_available_tables(
        self,
        session: AsyncSession,
        schema_name: Optional[str] = None
    ) -> List[str]:
        """
        Get list of all available table names in the database/schema.
        
        PHASE-2 REPLACEMENT for regex-based table extraction.
        Uses proper database introspection instead of schema parsing.
        
        Args:
            session: Database session
            schema_name: Optional schema to restrict to (DB-specific)
            
        Returns:
            List of table names
        """
        pass
    
    @abstractmethod
    async def get_all_tables_with_columns(
        self,
        session: AsyncSession,
        schema_name: Optional[str] = None
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Get structured metadata for all tables and their columns.
        
        PHASE-2 REPLACEMENT for regex-based column extraction.
        
        Args:
            session: Database session
            schema_name: Optional schema to restrict to
            
        Returns:
            Dict mapping table_name → [(column_name, column_type), ...]
        """
        pass
    
    @abstractmethod
    async def infer_foreign_key_relationships(
        self,
        session: AsyncSession,
        table1: str,
        table2: str,
        schema_name: Optional[str] = None
    ) -> Optional[Tuple[str, str, str, str]]:
        """
        Infer or detect foreign key relationship between two tables.
        
        PHASE-2 REPLACEMENT for hardcoded FK patterns.
        
        Args:
            session: Database session
            table1: Source table
            table2: Target table
            schema_name: Optional schema
            
        Returns:
            Tuple of (table1_fk_column, table2_fk_column) or None if no relationship found
        """
        pass


class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL-specific database adapter."""
    
    def __init__(self):
        super().__init__(DatabaseType.POSTGRESQL)
    
    def get_capabilities(self) -> DatabaseCapabilities:
        return DatabaseCapabilities(
            db_type=DatabaseType.POSTGRESQL,
            supports_schema_prefix=True,
            schema_prefix_style="schema.table",
            supports_enum_type=True,
            boolean_literals=("true", "false"),
            supports_boolean_type=True,
            uses_information_schema=True,
            metadata_system="information_schema",
            error_patterns={
                "type_error": [
                    "operator does not exist",
                    "no operator matches",
                    "type.*not supported",
                    "cannot cast"
                ],
                "undefined_column": [
                    "column.*does not exist",
                    "undefined column",
                ],
                "undefined_table": [
                    "relation.*does not exist",
                    "table.*does not exist",
                    "undefined table",
                ],
                "enum_error": [
                    "invalid input value for enum",
                    "enum value not found",
                ],
                "syntax_error": [
                    "syntax error",
                    "parse error",
                ]
            },
            default_schema=get_schema()
        )
    
    async def get_table_columns(
        self, 
        session: AsyncSession, 
        table_name: str, 
        schema_name: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """Query PostgreSQL information_schema for table columns."""
        from sqlalchemy import text
        
        # Try provided schema first, then public as fallback
        schemas_to_try = [schema_name or "public", "public"]
        # Remove duplicates while preserving order
        schemas_to_try = list(dict.fromkeys(schemas_to_try))
        
        query = text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = :schema AND table_name = :table
            ORDER BY ordinal_position
        """)
        
        for schema in schemas_to_try:
            try:
                result = await session.execute(query, {"schema": schema, "table": table_name})
                columns = result.fetchall()
                if columns:
                    return columns
            except Exception as e:
                # Try next schema
                continue
        
        # Last resort: try without schema restriction
        try:
            query_no_schema = text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = :table
                ORDER BY ordinal_position
            """)
            result = await session.execute(query_no_schema, {"table": table_name})
            return result.fetchall()
        except Exception:
            return []
    
    def classify_error(self, error_message: str) -> str:
        """Classify PostgreSQL errors."""
        msg_lower = error_message.lower()
        caps = self.get_capabilities()
        
        for error_type, patterns in caps.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, msg_lower, re.IGNORECASE):
                    return error_type
        
        return "unknown"
    
    def normalize_boolean_value(self, value: str) -> str:
        """PostgreSQL uses lowercase true/false."""
        val_lower = value.lower().strip()
        if val_lower in ("true", "1", "yes", "on"):
            return "true"
        elif val_lower in ("false", "0", "no", "off"):
            return "false"
        return val_lower
    
    def normalize_enum_value(self, value: str) -> str:
        """PostgreSQL enums typically use lowercase with underscores."""
        return value.lower() if value else value
    
    def format_schema_qualified_table(self, table_name: str, schema_name: Optional[str] = None) -> str:
        """PostgreSQL: schema.table notation."""
        schema = schema_name or "public"
        return f"{schema}.{table_name}"
    
    def extract_boolean_columns_from_schema(self, schema_text: str) -> List[str]:
        """Extract boolean columns from PostgreSQL schema."""
        # Pattern: word followed by "boolean"
        matches = re.findall(r'(\w+)\s+boolean\b', schema_text, re.IGNORECASE)
        return [m.lower() for m in matches]
    
    def extract_enum_columns_from_schema(self, schema_text: str) -> Dict[str, List[str]]:
        """Extract enum definitions from PostgreSQL schema."""
        enums = {}
        # Pattern: CREATE TYPE ... AS ENUM ('val1', 'val2', ...)
        enum_patterns = re.findall(
            r"CREATE\s+TYPE\s+(\w+)\s+AS\s+ENUM\s*\((.*?)\)",
            schema_text,
            re.IGNORECASE | re.DOTALL
        )
        
        for enum_name, values_str in enum_patterns:
            values = re.findall(r"'([^']+)'", values_str)
            enums[enum_name.lower()] = values
        
        return enums
    
    async def reflect_table_metadata(
        self,
        session: AsyncSession,
        table_name: str,
        schema_name: Optional[str] = None
    ) -> 'TableMetadata':
        """
        Reflect PostgreSQL table metadata using SQLAlchemy inspector.
        
        Returns structured TableMetadata instead of raw strings.
        """
        from .schema_metadata import TableMetadata, ColumnMetadata, ColumnType, ColumnTypeMapper
        from sqlalchemy import inspect as sa_inspect
        
        inspector = inspect(session.sync_session.get_bind())
        schema = schema_name or "public"
        
        # Get table columns
        columns_info = inspector.get_columns(table_name, schema=schema)
        primary_keys = inspector.get_pk_constraint(table_name, schema=schema)
        foreign_keys = inspector.get_foreign_keys(table_name, schema=schema)
        
        # Build metadata objects
        table_meta = TableMetadata(name=table_name, schema_name=schema)
        
        pk_names = set(primary_keys.get("constrained_columns", []))
        
        for col_info in columns_info:
            col_name = col_info["name"]
            col_type = col_info.get("type", "unknown")
            col_type_str = str(col_type)
            
            # Map to abstract type
            abstract_type = ColumnTypeMapper.map_type("postgresql", col_type_str)
            
            col_meta = ColumnMetadata(
                name=col_name,
                data_type=col_type_str,
                abstract_type=abstract_type,
                nullable=col_info.get("nullable", True),
                is_primary_key=col_name in pk_names,
                default_value=col_info.get("default"),
            )
            
            table_meta.columns[col_name] = col_meta
        
        table_meta.primary_keys = list(pk_names)
        
        # Add foreign key info
        for fk in foreign_keys:
            for col in fk.get("constrained_columns", []):
                ref_table = fk.get("referred_table", "")
                ref_cols = fk.get("referred_columns", [])
                ref_col = ref_cols[0] if ref_cols else "id"
                table_meta.foreign_keys[col] = (ref_table, ref_col)
                
                # Mark column as FK
                if col in table_meta.columns:
                    table_meta.columns[col].is_foreign_key = True
                    table_meta.columns[col].foreign_key_table = ref_table
                    table_meta.columns[col].foreign_key_column = ref_col
        
        return table_meta
    
    async def reflect_database_schema(
        self,
        session: AsyncSession,
        database_name: str,
        schema_name: Optional[str] = None
    ) -> 'DatabaseSchema':
        """
        Reflect entire PostgreSQL schema with all tables.
        """
        from .schema_metadata import DatabaseSchema
        
        inspector = inspect(session.sync_session.get_bind())
        schema = schema_name or "public"
        
        # Get all tables in schema
        table_names = inspector.get_table_names(schema=schema)
        
        db_schema = DatabaseSchema(database_name=database_name, db_type="postgresql")
        
        # Reflect each table
        for table_name in table_names:
            try:
                table_meta = await self.reflect_table_metadata(session, table_name, schema)
                db_schema.tables[table_name] = table_meta
            except Exception as e:
                print(f"[Warning] Failed to reflect table {table_name}: {e}")
                continue
        
        return db_schema
    
    async def get_available_tables(
        self,
        session: AsyncSession,
        schema_name: Optional[str] = None
    ) -> List[str]:
        """Get all table names using async-safe queries (not synchronous inspector)."""
        from sqlalchemy import text
        
        if schema_name is None:
            schema_name = self.get_capabilities().default_schema or 'public'
        
        # Use async query to get table names instead of synchronous inspector
        query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = :schema
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        
        try:
            result = await session.execute(query, {"schema": schema_name})
            tables = [row[0] for row in result.fetchall()]
            return tables
        except Exception as e:
            # Silently return empty list on error - will fall back to traditional SQL generation
            return []
    
    async def get_all_tables_with_columns(
        self,
        session: AsyncSession,
        schema_name: Optional[str] = None
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Get all tables and columns using introspection (PHASE-2)."""
        inspector = inspect(session.sync_session.get_bind())
        if schema_name is None:
            schema_name = self.get_capabilities().default_schema or 'public'
        
        result = {}
        for table_name in inspector.get_table_names(schema=schema_name):
            columns = inspector.get_columns(table_name, schema=schema_name)
            result[table_name] = [(col['name'], str(col['type'])) for col in columns]
        return result
    
    async def infer_foreign_key_relationships(
        self,
        session: AsyncSession,
        table1: str,
        table2: str,
        schema_name: Optional[str] = None
    ) -> Optional[Tuple[str, str, str, str]]:
        """Infer FK relationships using actual database constraints (PHASE-2)."""
        inspector = inspect(session.sync_session.get_bind())
        if schema_name is None:
            schema_name = self.get_capabilities().default_schema or 'public'
        
        try:
            # Get foreign keys for table1
            fks = inspector.get_foreign_keys(table1, schema=schema_name)
            for fk in fks:
                if fk['referred_table'] == table2:
                    # Found a FK from table1 to table2
                    local_col = fk['constrained_columns'][0]
                    remote_col = fk['referred_columns'][0]
                    return (table1, local_col, table2, remote_col)
            
            # Also check if table2 has FKs to table1
            fks2 = inspector.get_foreign_keys(table2, schema=schema_name)
            for fk in fks2:
                if fk['referred_table'] == table1:
                    # Found a FK from table2 to table1
                    local_col = fk['constrained_columns'][0]
                    remote_col = fk['referred_columns'][0]
                    return (table2, local_col, table1, remote_col)
        except Exception as e:
            print(f"[Warning] Failed to infer FK relationship between {table1} and {table2}: {e}")
        
        return None
    
    def get_default_schema(self) -> str:
        """Return default schema for catalog initialization."""
        return self.get_capabilities().default_schema or "public"
    
    async def get_tables_in_schema(self, schema_name: str, session: AsyncSession) -> List[str]:
        """Get all table names in a schema."""
        return await self.get_available_tables(session, schema_name)
    
    async def get_columns_for_table(self, table_name: str, schema_name: str, session: AsyncSession) -> List[Tuple[str, str]]:
        """Get columns for a specific table with types."""
        return await self.get_table_columns(session, table_name, schema_name)
    
    async def get_enum_values(self, schema_name: str, session: AsyncSession) -> Dict[str, List[str]]:
        """Get all enum type definitions and their values for a schema (PostgreSQL-specific)."""
        from sqlalchemy import text
        
        query = text("""
            SELECT t.typname, array_agg(e.enumlabel ORDER BY e.enumsortorder) as enum_values
            FROM pg_type t
            JOIN pg_enum e ON t.oid = e.enumtypid
            JOIN pg_namespace n ON n.oid = t.typnamespace
            WHERE n.nspname = :schema
            GROUP BY t.typname
        """)
        
        try:
            result = await session.execute(query, {"schema": schema_name})
            enum_map = {}
            for enum_name, enum_values in result.fetchall():
                enum_map[enum_name] = list(enum_values)
            return enum_map
        except Exception as e:
            logger.warning(f"Failed to get enum values for schema {schema_name}: {e}")
            return {}
    
    async def fetch_table_samples(self, table_name: str, schema_name: str, session: AsyncSession, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch sample rows from a table to show the LLM actual column values.
        This prevents LLM hallucination of column names and values.
        
        Args:
            table_name: Table to sample from
            schema_name: Schema name
            session: Database session
            limit: Number of rows to fetch
            
        Returns:
            List of dicts: [{column: value, ...}, ...]
        """
        from sqlalchemy import text
        
        try:
            # Build query to fetch sample rows
            query = text(f'SELECT * FROM "{schema_name}"."{table_name}" LIMIT :limit')
            result = await session.execute(query, {"limit": limit})
            
            # Convert result rows to dicts
            rows = []
            for row in result.fetchall():
                rows.append(dict(row._mapping))
            
            logger.debug(f"[ADAPTER] Fetched {len(rows)} sample rows from {schema_name}.{table_name}")
            return rows
        except Exception as e:
            logger.debug(f"[ADAPTER] Could not fetch samples from {schema_name}.{table_name}: {e}")
            return []


class MySQLAdapter(DatabaseAdapter):
    """MySQL-specific database adapter."""
    
    def __init__(self):
        super().__init__(DatabaseType.MYSQL)
    
    def get_capabilities(self) -> DatabaseCapabilities:
        return DatabaseCapabilities(
            db_type=DatabaseType.MYSQL,
            supports_schema_prefix=True,
            schema_prefix_style="database.table",
            supports_enum_type=False,  # ENUM is a string type, not true enum
            boolean_literals=("true", "false"),
            supports_boolean_type=False,  # Uses TINYINT(1)
            uses_information_schema=True,
            metadata_system="information_schema",
            error_patterns={
                "type_error": [
                    "cannot be used",
                    "incompatible types",
                    "type mismatch",
                ],
                "undefined_column": [
                    "unknown column",
                    "column.*not found",
                ],
                "undefined_table": [
                    "table.*not found",
                    "no such table",
                    "doesn't exist",
                ],
                "enum_error": [
                    "invalid enum value",
                    "data truncated",  # Often happens with enum
                ],
                "syntax_error": [
                    "syntax error",
                    "near",
                ]
            },
            default_schema=None  # MySQL uses current database
        )
    
    async def get_table_columns(
        self, 
        session: AsyncSession, 
        table_name: str, 
        schema_name: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """Query MySQL information_schema for table columns."""
        from sqlalchemy import text
        from ..config import settings
        
        database = schema_name or settings.mysql_database or "information_schema"
        query = text("""
            SELECT COLUMN_NAME, COLUMN_TYPE 
            FROM information_schema.COLUMNS 
            WHERE TABLE_SCHEMA = :database AND TABLE_NAME = :table
            ORDER BY ORDINAL_POSITION
        """)
        result = await session.execute(query, {"database": database, "table": table_name})
        return result.fetchall()
    
    def classify_error(self, error_message: str) -> str:
        """Classify MySQL errors."""
        msg_lower = error_message.lower()
        caps = self.get_capabilities()
        
        for error_type, patterns in caps.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, msg_lower, re.IGNORECASE):
                    return error_type
        
        return "unknown"
    
    def normalize_boolean_value(self, value: str) -> str:
        """MySQL: Use 1/0 for boolean (stored as TINYINT(1))."""
        val_lower = value.lower().strip()
        if val_lower in ("true", "1", "yes", "on"):
            return "1"
        elif val_lower in ("false", "0", "no", "off"):
            return "0"
        return val_lower
    
    def normalize_enum_value(self, value: str) -> str:
        """MySQL enum values are typically lowercase."""
        return value.lower() if value else value
    
    def format_schema_qualified_table(self, table_name: str, schema_name: Optional[str] = None) -> str:
        """MySQL: database.table notation."""
        database = schema_name or "DEFAULT_DB"
        return f"{database}.{table_name}"
    
    def extract_boolean_columns_from_schema(self, schema_text: str) -> List[str]:
        """Extract boolean columns from MySQL schema (TINYINT(1) or BOOLEAN)."""
        # Patterns: BOOLEAN, TINYINT(1), BIT(1)
        matches = re.findall(
            r'(\w+)\s+(?:boolean|tinyint\(1\)|bit\(1\))',
            schema_text,
            re.IGNORECASE
        )
        return [m.lower() for m in matches]
    
    def extract_enum_columns_from_schema(self, schema_text: str) -> Dict[str, List[str]]:
        """Extract ENUM column definitions from MySQL schema."""
        enums = {}
        # Pattern: column_name ENUM('val1', 'val2', ...)
        enum_patterns = re.findall(
            r"(\w+)\s+ENUM\s*\((.*?)\)",
            schema_text,
            re.IGNORECASE | re.DOTALL
        )
        
        for col_name, values_str in enum_patterns:
            values = re.findall(r"'([^']+)'", values_str)
            enums[col_name.lower()] = values
        
        return enums
    
    async def reflect_table_metadata(
        self,
        session: AsyncSession,
        table_name: str,
        schema_name: Optional[str] = None
    ) -> 'TableMetadata':
        """
        Reflect MySQL table metadata. Stub implementation for now.
        """
        from .schema_metadata import TableMetadata, ColumnMetadata, ColumnType, ColumnTypeMapper
        from sqlalchemy import inspect as sa_inspect
        
        inspector = inspect(session.sync_session.get_bind())
        schema = schema_name or "mysql"
        
        # Get table columns
        columns_info = inspector.get_columns(table_name, schema=schema)
        primary_keys = inspector.get_pk_constraint(table_name, schema=schema)
        foreign_keys = inspector.get_foreign_keys(table_name, schema=schema)
        
        # Build metadata objects
        table_meta = TableMetadata(name=table_name, schema_name=schema)
        pk_names = set(primary_keys.get("constrained_columns", []))
        
        for col_info in columns_info:
            col_name = col_info["name"]
            col_type_str = str(col_info.get("type", "unknown"))
            abstract_type = ColumnTypeMapper.map_type("mysql", col_type_str)
            
            col_meta = ColumnMetadata(
                name=col_name,
                data_type=col_type_str,
                abstract_type=abstract_type,
                nullable=col_info.get("nullable", True),
                is_primary_key=col_name in pk_names,
                default_value=col_info.get("default"),
            )
            table_meta.columns[col_name] = col_meta
        
        table_meta.primary_keys = list(pk_names)
        
        for fk in foreign_keys:
            for col in fk.get("constrained_columns", []):
                ref_table = fk.get("referred_table", "")
                ref_col = fk.get("referred_columns", ["id"])[0]
                table_meta.foreign_keys[col] = (ref_table, ref_col)
                if col in table_meta.columns:
                    table_meta.columns[col].is_foreign_key = True
                    table_meta.columns[col].foreign_key_table = ref_table
                    table_meta.columns[col].foreign_key_column = ref_col
        
        return table_meta
    
    async def reflect_database_schema(
        self,
        session: AsyncSession,
        database_name: str,
        schema_name: Optional[str] = None
    ) -> 'DatabaseSchema':
        """
        Reflect entire MySQL schema with all tables. Stub implementation.
        """
        from .schema_metadata import DatabaseSchema
        
        inspector = inspect(session.sync_session.get_bind())
        schema = schema_name or database_name or "mysql"
        
        table_names = inspector.get_table_names(schema=schema)
        db_schema = DatabaseSchema(database_name=database_name, db_type="mysql")
        
        for table_name in table_names:
            try:
                table_meta = await self.reflect_table_metadata(session, table_name, schema)
                db_schema.tables[table_name] = table_meta
            except Exception as e:
                print(f"[Warning] Failed to reflect table {table_name}: {e}")
                continue
        
        return db_schema
    
    async def get_available_tables(
        self,
        session: AsyncSession,
        schema_name: Optional[str] = None
    ) -> List[str]:
        """Get all table names using SQLAlchemy introspection (PHASE-2)."""
        inspector = inspect(session.sync_session.get_bind())
        # MySQL uses database concept, not schema - use default or passed
        if schema_name is None:
            schema_name = self.get_capabilities().default_schema
        tables = inspector.get_table_names(schema=schema_name)
        return tables
    
    async def get_all_tables_with_columns(
        self,
        session: AsyncSession,
        schema_name: Optional[str] = None
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Get all tables and columns using introspection (PHASE-2)."""
        inspector = inspect(session.sync_session.get_bind())
        if schema_name is None:
            schema_name = self.get_capabilities().default_schema
        
        result = {}
        for table_name in inspector.get_table_names(schema=schema_name):
            columns = inspector.get_columns(table_name, schema=schema_name)
            result[table_name] = [(col['name'], str(col['type'])) for col in columns]
        return result
    
    async def infer_foreign_key_relationships(
        self,
        session: AsyncSession,
        table1: str,
        table2: str,
        schema_name: Optional[str] = None
    ) -> Optional[Tuple[str, str, str, str]]:
        """Infer FK relationships using actual database constraints (PHASE-2)."""
        inspector = inspect(session.sync_session.get_bind())
        if schema_name is None:
            schema_name = self.get_capabilities().default_schema
        
        try:
            fks = inspector.get_foreign_keys(table1, schema=schema_name)
            for fk in fks:
                if fk['referred_table'] == table2:
                    local_col = fk['constrained_columns'][0]
                    remote_col = fk['referred_columns'][0]
                    return (table1, local_col, table2, remote_col)
            
            fks2 = inspector.get_foreign_keys(table2, schema=schema_name)
            for fk in fks2:
                if fk['referred_table'] == table1:
                    local_col = fk['constrained_columns'][0]
                    remote_col = fk['referred_columns'][0]
                    return (table2, local_col, table1, remote_col)
        except Exception as e:
            print(f"[Warning] Failed to infer FK relationship between {table1} and {table2}: {e}")
        
        return None
    
    def get_default_schema(self) -> str:
        """Return default schema for catalog initialization."""
        return self.get_capabilities().default_schema or "public"
    
    async def get_tables_in_schema(self, schema_name: str, session: AsyncSession) -> List[str]:
        """Get all table names in a schema."""
        return await self.get_available_tables(session, schema_name)
    
    async def get_columns_for_table(self, table_name: str, schema_name: str, session: AsyncSession) -> List[Tuple[str, str]]:
        """Get columns for a specific table with types."""
        return await self.get_table_columns(session, table_name, schema_name)
    
    async def fetch_table_samples(self, table_name: str, schema_name: str, session: AsyncSession, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch sample rows from a table to show the LLM actual column values.
        
        Args:
            table_name: Table to sample from
            schema_name: Schema/database name
            session: Database session
            limit: Number of rows to fetch
            
        Returns:
            List of dicts: [{column: value, ...}, ...]
        """
        from sqlalchemy import text
        
        try:
            # MySQL uses database concept, format accordingly
            query = text(f'SELECT * FROM `{schema_name}`.`{table_name}` LIMIT :limit')
            result = await session.execute(query, {"limit": limit})
            
            # Convert result rows to dicts
            rows = []
            for row in result.fetchall():
                rows.append(dict(row._mapping))
            
            logger.debug(f"[ADAPTER] Fetched {len(rows)} sample rows from {schema_name}.{table_name}")
            return rows
        except Exception as e:
            logger.debug(f"[ADAPTER] Could not fetch samples from {schema_name}.{table_name}: {e}")
            return []


class SQLiteAdapter(DatabaseAdapter):
    """SQLite-specific database adapter."""
    
    def __init__(self):
        super().__init__(DatabaseType.SQLITE)
    
    def get_capabilities(self) -> DatabaseCapabilities:
        return DatabaseCapabilities(
            db_type=DatabaseType.SQLITE,
            supports_schema_prefix=False,  # SQLite doesn't support schema prefix
            schema_prefix_style="table",
            supports_enum_type=False,
            boolean_literals=("1", "0"),
            supports_boolean_type=False,
            uses_information_schema=False,
            metadata_system="pragma",
            error_patterns={
                "type_error": [
                    "type mismatch",
                    "no such operator",
                ],
                "undefined_column": [
                    "no such column",
                    "table.*has no column",
                ],
                "undefined_table": [
                    "no such table",
                ],
                "enum_error": [
                    "constraint failed",  # Generic for value constraints
                ],
                "syntax_error": [
                    "syntax error",
                    "parse error",
                ]
            },
            default_schema=None
        )
    
    async def get_table_columns(
        self, 
        session: AsyncSession, 
        table_name: str, 
        schema_name: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """Query SQLite using PRAGMA table_info."""
        from sqlalchemy import text
        
        query = text(f"PRAGMA table_info({table_name})")
        result = await session.execute(query)
        # PRAGMA returns: (cid, name, type, notnull, dflt_value, pk)
        rows = result.fetchall()
        return [(row[1], row[2] or "TEXT") for row in rows]
    
    def classify_error(self, error_message: str) -> str:
        """Classify SQLite errors."""
        msg_lower = error_message.lower()
        caps = self.get_capabilities()
        
        for error_type, patterns in caps.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, msg_lower, re.IGNORECASE):
                    return error_type
        
        return "unknown"
    
    def normalize_boolean_value(self, value: str) -> str:
        """SQLite: Use 1/0 for boolean (no true boolean type)."""
        val_lower = value.lower().strip()
        if val_lower in ("true", "1", "yes", "on"):
            return "1"
        elif val_lower in ("false", "0", "no", "off"):
            return "0"
        return val_lower
    
    def normalize_enum_value(self, value: str) -> str:
        """SQLite stores enum-like values as TEXT."""
        return value.lower() if value else value
    
    def format_schema_qualified_table(self, table_name: str, schema_name: Optional[str] = None) -> str:
        """SQLite: Just table name (no schema support)."""
        return table_name
    
    def extract_boolean_columns_from_schema(self, schema_text: str) -> List[str]:
        """Extract boolean columns from SQLite schema (BOOLEAN or INTEGER)."""
        matches = re.findall(
            r'(\w+)\s+(?:boolean|integer)',
            schema_text,
            re.IGNORECASE
        )
        return [m.lower() for m in matches]
    
    def extract_enum_columns_from_schema(self, schema_text: str) -> Dict[str, List[str]]:
        """SQLite doesn't have native enums, so look for CHECK constraints."""
        enums = {}
        # Pattern: column_name TEXT CHECK (column_name IN ('val1', 'val2', ...))
        check_patterns = re.findall(
            r"(\w+)\s+TEXT\s+CHECK\s*\(\s*\1\s+IN\s*\((.*?)\)\s*\)",
            schema_text,
            re.IGNORECASE | re.DOTALL
        )
        
        for col_name, values_str in check_patterns:
            values = re.findall(r"'([^']+)'", values_str)
            enums[col_name.lower()] = values
        
        return enums
    
    async def reflect_table_metadata(
        self,
        session: AsyncSession,
        table_name: str,
        schema_name: Optional[str] = None
    ) -> 'TableMetadata':
        """Stub: Reflect SQLite table metadata."""
        from .schema_metadata import TableMetadata, ColumnMetadata, ColumnType, ColumnTypeMapper
        
        inspector = inspect(session.sync_session.get_bind())
        columns_info = inspector.get_columns(table_name)
        primary_keys = inspector.get_pk_constraint(table_name)
        
        table_meta = TableMetadata(name=table_name, schema_name=None)
        pk_names = set(primary_keys.get("constrained_columns", []))
        
        for col_info in columns_info:
            col_name = col_info["name"]
            col_type_str = str(col_info.get("type", "unknown"))
            abstract_type = ColumnTypeMapper.map_type("sqlite", col_type_str)
            
            col_meta = ColumnMetadata(
                name=col_name,
                data_type=col_type_str,
                abstract_type=abstract_type,
                nullable=col_info.get("nullable", True),
                is_primary_key=col_name in pk_names,
            )
            table_meta.columns[col_name] = col_meta
        
        table_meta.primary_keys = list(pk_names)
        return table_meta
    
    async def reflect_database_schema(
        self,
        session: AsyncSession,
        database_name: str,
        schema_name: Optional[str] = None
    ) -> 'DatabaseSchema':
        """Stub: Reflect SQLite database schema."""
        from .schema_metadata import DatabaseSchema
        
        inspector = inspect(session.sync_session.get_bind())
        table_names = inspector.get_table_names()
        
        db_schema = DatabaseSchema(database_name=database_name, db_type="sqlite")
        
        for table_name in table_names:
            try:
                table_meta = await self.reflect_table_metadata(session, table_name)
                db_schema.tables[table_name] = table_meta
            except Exception as e:
                print(f"[Warning] Failed to reflect table {table_name}: {e}")
                continue
        
        return db_schema
    
    async def get_available_tables(
        self,
        session: AsyncSession,
        schema_name: Optional[str] = None
    ) -> List[str]:
        """Get all table names using SQLAlchemy introspection (PHASE-2)."""
        inspector = inspect(session.sync_session.get_bind())
        # SQLite doesn't use schemas
        tables = inspector.get_table_names()
        return tables
    
    async def get_all_tables_with_columns(
        self,
        session: AsyncSession,
        schema_name: Optional[str] = None
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Get all tables and columns using introspection (PHASE-2)."""
        inspector = inspect(session.sync_session.get_bind())
        
        result = {}
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            result[table_name] = [(col['name'], str(col['type'])) for col in columns]
        return result
    
    async def infer_foreign_key_relationships(
        self,
        session: AsyncSession,
        table1: str,
        table2: str,
        schema_name: Optional[str] = None
    ) -> Optional[Tuple[str, str, str, str]]:
        """Infer FK relationships using actual database constraints (PHASE-2)."""
        inspector = inspect(session.sync_session.get_bind())
        
        try:
            fks = inspector.get_foreign_keys(table1)
            for fk in fks:
                if fk['referred_table'] == table2:
                    local_col = fk['constrained_columns'][0]
                    remote_col = fk['referred_columns'][0]
                    return (table1, local_col, table2, remote_col)
            
            fks2 = inspector.get_foreign_keys(table2)
            for fk in fks2:
                if fk['referred_table'] == table1:
                    local_col = fk['constrained_columns'][0]
                    remote_col = fk['referred_columns'][0]
                    return (table2, local_col, table1, remote_col)
        except Exception as e:
            print(f"[Warning] Failed to infer FK relationship between {table1} and {table2}: {e}")
        
        return None
    
    def get_default_schema(self) -> str:
        """Return default schema for catalog initialization."""
        return self.get_capabilities().default_schema or ""
    
    async def get_tables_in_schema(self, schema_name: str, session: AsyncSession) -> List[str]:
        """Get all table names in a schema."""
        return await self.get_available_tables(session, schema_name)
    
    async def get_columns_for_table(self, table_name: str, schema_name: str, session: AsyncSession) -> List[Tuple[str, str]]:
        """Get columns for a specific table with types."""
        return await self.get_table_columns(session, table_name, schema_name)
    
    async def fetch_table_samples(self, table_name: str, schema_name: str, session: AsyncSession, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch sample rows from a table to show the LLM actual column values.
        
        Args:
            table_name: Table to sample from
            schema_name: Schema name (ignored for SQLite)
            session: Database session
            limit: Number of rows to fetch
            
        Returns:
            List of dicts: [{column: value, ...}, ...]
        """
        from sqlalchemy import text
        
        try:
            # SQLite doesn't use schemas
            query = text(f'SELECT * FROM "{table_name}" LIMIT :limit')
            result = await session.execute(query, {"limit": limit})
            
            # Convert result rows to dicts
            rows = []
            for row in result.fetchall():
                rows.append(dict(row._mapping))
            
            logger.debug(f"[ADAPTER] Fetched {len(rows)} sample rows from {table_name}")
            return rows
        except Exception as e:
            logger.debug(f"[ADAPTER] Could not fetch samples from {table_name}: {e}")
            return []


class SQLServerAdapter(DatabaseAdapter):
    """SQL Server-specific database adapter."""
    
    def __init__(self):
        super().__init__(DatabaseType.SQL_SERVER)
    
    def get_capabilities(self) -> DatabaseCapabilities:
        return DatabaseCapabilities(
            db_type=DatabaseType.SQL_SERVER,
            supports_schema_prefix=True,
            schema_prefix_style="schema.table",
            supports_enum_type=False,
            boolean_literals=("1", "0"),
            supports_boolean_type=True,  # BIT type
            uses_information_schema=True,
            metadata_system="information_schema",
            error_patterns={
                "type_error": [
                    "conversion failed",
                    "cannot convert",
                    "operator.*not supported",
                ],
                "undefined_column": [
                    "invalid column name",
                    "ambiguous column name",
                ],
                "undefined_table": [
                    "table.*not found",
                    "invalid object name",
                ],
                "enum_error": [
                    "constraint failed",
                    "check violation",
                ],
                "syntax_error": [
                    "syntax error",
                    "incorrect syntax",
                ]
            },
            default_schema="dbo"
        )
    
    async def get_table_columns(
        self, 
        session: AsyncSession, 
        table_name: str, 
        schema_name: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """Query SQL Server sys.columns for table columns."""
        from sqlalchemy import text
        
        schema = schema_name or "dbo"
        query = text("""
            SELECT c.name, t.name
            FROM sys.columns c
            JOIN sys.tables tb ON c.object_id = tb.object_id
            JOIN sys.schemas s ON tb.schema_id = s.schema_id
            JOIN sys.types t ON c.user_type_id = t.user_type_id
            WHERE s.name = :schema AND tb.name = :table
            ORDER BY c.column_id
        """)
        result = await session.execute(query, {"schema": schema, "table": table_name})
        return result.fetchall()
    
    def classify_error(self, error_message: str) -> str:
        """Classify SQL Server errors."""
        msg_lower = error_message.lower()
        caps = self.get_capabilities()
        
        for error_type, patterns in caps.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, msg_lower, re.IGNORECASE):
                    return error_type
        
        return "unknown"
    
    def normalize_boolean_value(self, value: str) -> str:
        """SQL Server: Use 1/0 for boolean (BIT type)."""
        val_lower = value.lower().strip()
        if val_lower in ("true", "1", "yes", "on"):
            return "1"
        elif val_lower in ("false", "0", "no", "off"):
            return "0"
        return val_lower
    
    def normalize_enum_value(self, value: str) -> str:
        """SQL Server enum values typically use uppercase or as-is."""
        return value.lower() if value else value
    
    def format_schema_qualified_table(self, table_name: str, schema_name: Optional[str] = None) -> str:
        """SQL Server: schema.table notation with dbo as default."""
        schema = schema_name or "dbo"
        return f"[{schema}].[{table_name}]"
    
    def extract_boolean_columns_from_schema(self, schema_text: str) -> List[str]:
        """Extract boolean columns from SQL Server schema (BIT type)."""
        matches = re.findall(
            r'\[?(\w+)\]?\s+bit\b',
            schema_text,
            re.IGNORECASE
        )
        return [m.lower() for m in matches]
    
    def extract_enum_columns_from_schema(self, schema_text: str) -> Dict[str, List[str]]:
        """Extract CHECK constraint enums from SQL Server schema."""
        enums = {}
        # Pattern: CONSTRAINT ... CHECK (column IN ('val1', 'val2', ...))
        check_patterns = re.findall(
            r"CHECK\s*\(\s*\[?(\w+)\]?\s+IN\s*\((.*?)\)\s*\)",
            schema_text,
            re.IGNORECASE | re.DOTALL
        )
        
        for col_name, values_str in check_patterns:
            values = re.findall(r"'([^']+)'", values_str)
            enums[col_name.lower()] = values
        
        return enums
    
    async def reflect_table_metadata(
        self,
        session: AsyncSession,
        table_name: str,
        schema_name: Optional[str] = None
    ) -> 'TableMetadata':
        """Stub: Reflect SQL Server table metadata."""
        from .schema_metadata import TableMetadata, ColumnMetadata, ColumnType, ColumnTypeMapper
        
        inspector = inspect(session.sync_session.get_bind())
        schema = schema_name or "dbo"
        
        columns_info = inspector.get_columns(table_name, schema=schema)
        primary_keys = inspector.get_pk_constraint(table_name, schema=schema)
        
        table_meta = TableMetadata(name=table_name, schema_name=schema)
        pk_names = set(primary_keys.get("constrained_columns", []))
        
        for col_info in columns_info:
            col_name = col_info["name"]
            col_type_str = str(col_info.get("type", "unknown"))
            abstract_type = ColumnTypeMapper.map_type("sqlserver", col_type_str)
            
            col_meta = ColumnMetadata(
                name=col_name,
                data_type=col_type_str,
                abstract_type=abstract_type,
                nullable=col_info.get("nullable", True),
                is_primary_key=col_name in pk_names,
            )
            table_meta.columns[col_name] = col_meta
        
        table_meta.primary_keys = list(pk_names)
        return table_meta
    
    async def reflect_database_schema(
        self,
        session: AsyncSession,
        database_name: str,
        schema_name: Optional[str] = None
    ) -> 'DatabaseSchema':
        """Stub: Reflect SQL Server database schema."""
        from .schema_metadata import DatabaseSchema
        
        inspector = inspect(session.sync_session.get_bind())
        schema = schema_name or "dbo"
        
        table_names = inspector.get_table_names(schema=schema)
        db_schema = DatabaseSchema(database_name=database_name, db_type="sqlserver")
        
        for table_name in table_names:
            try:
                table_meta = await self.reflect_table_metadata(session, table_name, schema)
                db_schema.tables[table_name] = table_meta
            except Exception as e:
                print(f"[Warning] Failed to reflect table {table_name}: {e}")
                continue
        
        return db_schema
    
    async def get_available_tables(
        self,
        session: AsyncSession,
        schema_name: Optional[str] = None
    ) -> List[str]:
        """Get all table names using SQLAlchemy introspection (PHASE-2)."""
        inspector = inspect(session.sync_session.get_bind())
        if schema_name is None:
            schema_name = "dbo"  # SQL Server default schema
        tables = inspector.get_table_names(schema=schema_name)
        return tables
    
    async def get_all_tables_with_columns(
        self,
        session: AsyncSession,
        schema_name: Optional[str] = None
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Get all tables and columns using introspection (PHASE-2)."""
        inspector = inspect(session.sync_session.get_bind())
        if schema_name is None:
            schema_name = "dbo"
        
        result = {}
        for table_name in inspector.get_table_names(schema=schema_name):
            columns = inspector.get_columns(table_name, schema=schema_name)
            result[table_name] = [(col['name'], str(col['type'])) for col in columns]
        return result
    
    async def infer_foreign_key_relationships(
        self,
        session: AsyncSession,
        table1: str,
        table2: str,
        schema_name: Optional[str] = None
    ) -> Optional[Tuple[str, str, str, str]]:
        """Infer FK relationships using actual database constraints (PHASE-2)."""
        inspector = inspect(session.sync_session.get_bind())
        if schema_name is None:
            schema_name = "dbo"
        
        try:
            fks = inspector.get_foreign_keys(table1, schema=schema_name)
            for fk in fks:
                if fk['referred_table'] == table2:
                    local_col = fk['constrained_columns'][0]
                    remote_col = fk['referred_columns'][0]
                    return (table1, local_col, table2, remote_col)
            
            fks2 = inspector.get_foreign_keys(table2, schema=schema_name)
            for fk in fks2:
                if fk['referred_table'] == table1:
                    local_col = fk['constrained_columns'][0]
                    remote_col = fk['referred_columns'][0]
                    return (table2, local_col, table1, remote_col)
        except Exception as e:
            print(f"[Warning] Failed to infer FK relationship between {table1} and {table2}: {e}")
        
        return None
    
    def get_default_schema(self) -> str:
        """Return default schema for catalog initialization."""
        return self.get_capabilities().default_schema or "dbo"
    
    async def get_tables_in_schema(self, schema_name: str, session: AsyncSession) -> List[str]:
        """Get all table names in a schema."""
        return await self.get_available_tables(session, schema_name)
    
    async def get_columns_for_table(self, table_name: str, schema_name: str, session: AsyncSession) -> List[Tuple[str, str]]:
        """Get columns for a specific table with types."""
        return await self.get_table_columns(session, table_name, schema_name)


class OracleAdapter(DatabaseAdapter):
    """Oracle-specific database adapter."""
    
    def __init__(self):
        super().__init__(DatabaseType.ORACLE)
    
    def get_capabilities(self) -> DatabaseCapabilities:
        return DatabaseCapabilities(
            db_type=DatabaseType.ORACLE,
            supports_schema_prefix=True,
            schema_prefix_style="schema.table",
            supports_enum_type=False,
            boolean_literals=("1", "0"),
            supports_boolean_type=False,  # Uses NUMBER or CHAR
            uses_information_schema=False,
            metadata_system="all_tab_columns",
            error_patterns={
                "type_error": [
                    "ora-01722",  # Invalid number
                    "ora-00918",  # Column ambiguously defined
                ],
                "undefined_column": [
                    "ora-00904",  # Invalid identifier
                    "ora-00923",
                ],
                "undefined_table": [
                    "ora-00942",  # Table doesn't exist
                ],
                "enum_error": [
                    "ora-02291",  # Integrity constraint
                ],
                "syntax_error": [
                    "ora-00906",  # Missing left parenthesis
                ]
            },
            default_schema=None
        )
    
    async def get_table_columns(
        self, 
        session: AsyncSession, 
        table_name: str, 
        schema_name: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """Query Oracle ALL_TAB_COLUMNS for table columns."""
        from sqlalchemy import text
        
        owner = schema_name or "PUBLIC"
        query = text("""
            SELECT COLUMN_NAME, DATA_TYPE 
            FROM ALL_TAB_COLUMNS 
            WHERE OWNER = :owner AND TABLE_NAME = :table
            ORDER BY COLUMN_ID
        """)
        result = await session.execute(query, {"owner": owner, "table": table_name.upper()})
        return result.fetchall()
    
    def classify_error(self, error_message: str) -> str:
        """Classify Oracle errors."""
        msg_upper = error_message.upper()
        caps = self.get_capabilities()
        
        for error_type, patterns in caps.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, msg_upper, re.IGNORECASE):
                    return error_type
        
        return "unknown"
    
    def normalize_boolean_value(self, value: str) -> str:
        """Oracle: Use 1/0 or Y/N (no native boolean)."""
        val_lower = value.lower().strip()
        if val_lower in ("true", "1", "yes", "y", "on"):
            return "1"
        elif val_lower in ("false", "0", "no", "n", "off"):
            return "0"
        return val_lower
    
    def normalize_enum_value(self, value: str) -> str:
        """Oracle: Values are typically uppercase."""
        return value.upper() if value else value
    
    def format_schema_qualified_table(self, table_name: str, schema_name: Optional[str] = None) -> str:
        """Oracle: schema.table notation (owner.table)."""
        owner = schema_name or "PUBLIC"
        return f"{owner}.{table_name.upper()}"
    
    def extract_boolean_columns_from_schema(self, schema_text: str) -> List[str]:
        """Extract boolean columns from Oracle schema."""
        # Oracle doesn't have native boolean, look for common patterns
        matches = re.findall(
            r'(\w+)\s+(?:number\(1\)|char\(1\))',
            schema_text,
            re.IGNORECASE
        )
        return [m.lower() for m in matches]
    
    def extract_enum_columns_from_schema(self, schema_text: str) -> Dict[str, List[str]]:
        """Extract CHECK constraint enums from Oracle schema."""
        enums = {}
        # Pattern: CHECK (column IN ('val1', 'val2', ...))
        check_patterns = re.findall(
            r"CHECK\s*\(\s*(\w+)\s+IN\s*\((.*?)\)\s*\)",
            schema_text,
            re.IGNORECASE | re.DOTALL
        )
        
        for col_name, values_str in check_patterns:
            values = re.findall(r"'([^']+)'", values_str)
            enums[col_name.lower()] = values
        
        return enums
    
    async def reflect_table_metadata(
        self,
        session: AsyncSession,
        table_name: str,
        schema_name: Optional[str] = None
    ) -> 'TableMetadata':
        """Stub: Reflect Oracle table metadata."""
        from .schema_metadata import TableMetadata, ColumnMetadata, ColumnType, ColumnTypeMapper
        
        inspector = inspect(session.sync_session.get_bind())
        schema = schema_name or "PUBLIC"
        
        columns_info = inspector.get_columns(table_name, schema=schema)
        primary_keys = inspector.get_pk_constraint(table_name, schema=schema)
        
        table_meta = TableMetadata(name=table_name, schema_name=schema)
        pk_names = set(primary_keys.get("constrained_columns", []))
        
        for col_info in columns_info:
            col_name = col_info["name"]
            col_type_str = str(col_info.get("type", "unknown"))
            abstract_type = ColumnTypeMapper.map_type("oracle", col_type_str)
            
            col_meta = ColumnMetadata(
                name=col_name,
                data_type=col_type_str,
                abstract_type=abstract_type,
                nullable=col_info.get("nullable", True),
                is_primary_key=col_name in pk_names,
            )
            table_meta.columns[col_name] = col_meta
        
        table_meta.primary_keys = list(pk_names)
        return table_meta
    
    async def reflect_database_schema(
        self,
        session: AsyncSession,
        database_name: str,
        schema_name: Optional[str] = None
    ) -> 'DatabaseSchema':
        """Stub: Reflect Oracle database schema."""
        from .schema_metadata import DatabaseSchema
        
        inspector = inspect(session.sync_session.get_bind())
        schema = schema_name or "PUBLIC"
        
        table_names = inspector.get_table_names(schema=schema)
        db_schema = DatabaseSchema(database_name=database_name, db_type="oracle")
        
        for table_name in table_names:
            try:
                table_meta = await self.reflect_table_metadata(session, table_name, schema)
                db_schema.tables[table_name] = table_meta
            except Exception as e:
                print(f"[Warning] Failed to reflect table {table_name}: {e}")
                continue
        
        return db_schema
    
    async def get_available_tables(
        self,
        session: AsyncSession,
        schema_name: Optional[str] = None
    ) -> List[str]:
        """Get all table names using SQLAlchemy introspection (PHASE-2)."""
        inspector = inspect(session.sync_session.get_bind())
        if schema_name is None:
            # Oracle requires explicit schema or uses current user schema
            schema_name = None
        tables = inspector.get_table_names(schema=schema_name)
        return tables
    
    async def get_all_tables_with_columns(
        self,
        session: AsyncSession,
        schema_name: Optional[str] = None
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Get all tables and columns using introspection (PHASE-2)."""
        inspector = inspect(session.sync_session.get_bind())
        
        result = {}
        for table_name in inspector.get_table_names(schema=schema_name):
            columns = inspector.get_columns(table_name, schema=schema_name)
            result[table_name] = [(col['name'], str(col['type'])) for col in columns]
        return result
    
    async def infer_foreign_key_relationships(
        self,
        session: AsyncSession,
        table1: str,
        table2: str,
        schema_name: Optional[str] = None
    ) -> Optional[Tuple[str, str, str, str]]:
        """Infer FK relationships using actual database constraints (PHASE-2)."""
        inspector = inspect(session.sync_session.get_bind())
        
        try:
            fks = inspector.get_foreign_keys(table1, schema=schema_name)
            for fk in fks:
                if fk['referred_table'] == table2:
                    local_col = fk['constrained_columns'][0]
                    remote_col = fk['referred_columns'][0]
                    return (table1, local_col, table2, remote_col)
            
            fks2 = inspector.get_foreign_keys(table2, schema=schema_name)
            for fk in fks2:
                if fk['referred_table'] == table1:
                    local_col = fk['constrained_columns'][0]
                    remote_col = fk['referred_columns'][0]
                    return (table2, local_col, table1, remote_col)
        except Exception as e:
            print(f"[Warning] Failed to infer FK relationship between {table1} and {table2}: {e}")
        
        return None
    
    def get_default_schema(self) -> str:
        """Return default schema for catalog initialization."""
        return self.get_capabilities().default_schema or "PUBLIC"
    
    async def get_tables_in_schema(self, schema_name: str, session: AsyncSession) -> List[str]:
        """Get all table names in a schema."""
        return await self.get_available_tables(session, schema_name)
    
    async def get_columns_for_table(self, table_name: str, schema_name: str, session: AsyncSession) -> List[Tuple[str, str]]:
        """Get columns for a specific table with types."""
        return await self.get_table_columns(session, table_name, schema_name)


class DatabaseAdapterFactory:
    """Factory for creating database adapters based on connection URL."""
    
    _adapters = {
        DatabaseType.POSTGRESQL: PostgreSQLAdapter(),
        DatabaseType.MYSQL: MySQLAdapter(),
        DatabaseType.SQLITE: SQLiteAdapter(),
        DatabaseType.SQL_SERVER: SQLServerAdapter(),
        DatabaseType.ORACLE: OracleAdapter(),
    }
    
    @staticmethod
    def detect_database_type(connection_string: str) -> DatabaseType:
        """
        Detect database type from SQLAlchemy connection string.
        
        Args:
            connection_string: SQLAlchemy URL string
            
        Returns:
            Detected DatabaseType
        """
        if not connection_string:
            return DatabaseType.UNKNOWN
        
        conn_lower = connection_string.lower()
        
        if "postgresql" in conn_lower or "postgres" in conn_lower:
            return DatabaseType.POSTGRESQL
        elif "mysql" in conn_lower:
            return DatabaseType.MYSQL
        elif "sqlite" in conn_lower:
            return DatabaseType.SQLITE
        elif "mssql" in conn_lower or "sqlserver" in conn_lower or "sql+server" in conn_lower:
            return DatabaseType.SQL_SERVER
        elif "oracle" in conn_lower:
            return DatabaseType.ORACLE
        
        return DatabaseType.UNKNOWN
    
    @staticmethod
    def get_adapter(db_type: DatabaseType) -> DatabaseAdapter:
        """
        Get adapter for a database type.
        
        Args:
            db_type: The database type
            
        Returns:
            Database adapter instance
        """
        adapter = DatabaseAdapterFactory._adapters.get(db_type)
        if not adapter:
            # Default to PostgreSQL if unknown
            return DatabaseAdapterFactory._adapters[DatabaseType.POSTGRESQL]
        return adapter
    
    @staticmethod
    def get_adapter_from_connection_string(connection_string: str) -> DatabaseAdapter:
        """
        Get adapter directly from connection string.
        
        Args:
            connection_string: SQLAlchemy URL string
            
        Returns:
            Database adapter instance
        """
        db_type = DatabaseAdapterFactory.detect_database_type(connection_string)
        return DatabaseAdapterFactory.get_adapter(db_type)


# Singleton instance for global use
_global_adapter: Optional[DatabaseAdapter] = None


def set_global_adapter(adapter: DatabaseAdapter) -> None:
    """Set the global database adapter."""
    global _global_adapter
    _global_adapter = adapter


def get_global_adapter() -> DatabaseAdapter:
    """Get the global database adapter (defaults to PostgreSQL)."""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = DatabaseAdapterFactory.get_adapter(DatabaseType.POSTGRESQL)
    return _global_adapter
