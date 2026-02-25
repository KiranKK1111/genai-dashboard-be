"""
Dialect Adapter: Provides database-agnostic SQL generation and configuration.

This module abstracts differences between SQL dialects (PostgreSQL, MySQL, SQLite, SQL Server)
to enable the system to work with any SQLAlchemy-supported database.
"""

from __future__ import annotations
from typing import Literal, Dict, Any
from dataclasses import dataclass
from enum import Enum


class DatabaseDialect(str, Enum):
    """Supported database dialects."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    SQLSERVER = "sqlserver"
    MARIADB = "mariadb"


@dataclass
class DialectCapabilities:
    """
    Capabilities specific to each database dialect.
    
    Attributes:
        dialect: The dialect type
        limit_style: How to apply LIMIT - 'limit_offset' (LIMIT/OFFSET), 'fetch_first' (FETCH FIRST), 'top_n' (TOP)
        case_insensitive_like: How to do case-insensitive matching - 'ilike', 'lower_like', 'collate_like'
        quote_char: Character used for quoting identifiers - '"', '`', or '['
        bool_values: How to represent boolean values - (True, False) representation
        supports_schemas: Whether the dialect supports schema qualification
        supports_upsert: Whether the dialect supports UPSERT/INSERT...ON CONFLICT
        datetime_cast: How to cast to datetime
        null_safe_equal: How to compare with NULL - '<=>' (MySQL) or 'IS NOT DISTINCT FROM' (PostgreSQL)
    """
    dialect: DatabaseDialect
    limit_style: Literal["limit_offset", "fetch_first", "top_n"]
    case_insensitive_like: Literal["ilike", "lower_like", "collate_like"]
    quote_char: Literal['"', '`', '[']
    bool_values: tuple  # (True_repr, False_repr)
    supports_schemas: bool
    supports_upsert: bool
    datetime_cast: str  # e.g., 'CAST(col AS TIMESTAMP)'
    null_safe_equal: str  # How to do NULL-safe equality


# Dialect-specific configurations
DIALECT_CONFIG: Dict[DatabaseDialect, DialectCapabilities] = {
    DatabaseDialect.POSTGRESQL: DialectCapabilities(
        dialect=DatabaseDialect.POSTGRESQL,
        limit_style="limit_offset",
        case_insensitive_like="ilike",
        quote_char='"',
        bool_values=("true", "false"),
        supports_schemas=True,
        supports_upsert=True,
        datetime_cast="CAST(col AS TIMESTAMP)",
        null_safe_equal="IS NOT DISTINCT FROM",
    ),
    DatabaseDialect.MYSQL: DialectCapabilities(
        dialect=DatabaseDialect.MYSQL,
        limit_style="limit_offset",
        case_insensitive_like="lower_like",  # LOWER(col) = LOWER(?)
        quote_char="`",
        bool_values=("1", "0"),
        supports_schemas=False,  # Uses databases instead
        supports_upsert=True,
        datetime_cast="CAST(col AS DATETIME)",
        null_safe_equal="<=>",  # MySQL's NULL-safe equal operator
    ),
    DatabaseDialect.SQLITE: DialectCapabilities(
        dialect=DatabaseDialect.SQLITE,
        limit_style="limit_offset",
        case_insensitive_like="lower_like",
        quote_char='"',
        bool_values=("1", "0"),
        supports_schemas=False,
        supports_upsert=True,
        datetime_cast="CAST(col AS DATETIME)",
        null_safe_equal="=",  # SQLite doesn't have NULL-safe equal, must use COALESCE
    ),
    DatabaseDialect.SQLSERVER: DialectCapabilities(
        dialect=DatabaseDialect.SQLSERVER,
        limit_style="top_n",  # Uses TOP instead of LIMIT
        case_insensitive_like="collate_like",
        quote_char="[",
        bool_values=("1", "0"),
        supports_schemas=True,
        supports_upsert=True,
        datetime_cast="CAST(col AS DATETIME2)",
        null_safe_equal="=",  # Must handle NULLs manually
    ),
    DatabaseDialect.MARIADB: DialectCapabilities(
        dialect=DatabaseDialect.MARIADB,
        limit_style="limit_offset",
        case_insensitive_like="lower_like",
        quote_char="`",
        bool_values=("1", "0"),
        supports_schemas=False,
        supports_upsert=True,  # MariaDB has ON DUPLICATE KEY UPDATE
        datetime_cast="CAST(col AS DATETIME)",
        null_safe_equal="<=>",
    ),
}


class DialectAdapter:
    """
    Adapter for generating dialect-specific SQL and managing compatibility.
    """
    
    def __init__(self, dialect: DatabaseDialect):
        """
        Initialize the adapter for a specific dialect.
        
        Args:
            dialect: The target database dialect
        """
        self.dialect = dialect
        self.capabilities = DIALECT_CONFIG[dialect]
    
    def render_limit(self, limit: int, offset: int = 0) -> str:
        """
        Render LIMIT clause in dialect-specific syntax.
        
        Args:
            limit: Number of rows to limit
            offset: Number of rows to skip
            
        Returns:
            Dialect-specific LIMIT clause
        """
        if self.capabilities.limit_style == "limit_offset":
            if offset > 0:
                return f"LIMIT {limit} OFFSET {offset}"
            return f"LIMIT {limit}"
        elif self.capabilities.limit_style == "fetch_first":
            if offset > 0:
                return f"OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY"
            return f"FETCH FIRST {limit} ROWS ONLY"
        elif self.capabilities.limit_style == "top_n":
            if offset > 0:
                # SQL Server: Use OFFSET...FETCH
                return f"OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY"
            return f"TOP {limit}"
        return ""
    
    def render_case_insensitive_like(self, column: str, value: str) -> str:
        """
        Render case-insensitive LIKE clause.
        
        Args:
            column: Column name
            value: Value to match
            
        Returns:
            Dialect-specific case-insensitive comparison
        """
        safe_column = self.quote_identifier(column)
        if self.capabilities.case_insensitive_like == "ilike":
            return f"{safe_column} ILIKE %{value}%"
        elif self.capabilities.case_insensitive_like == "lower_like":
            return f"LOWER({safe_column}) LIKE LOWER('%{value}%')"
        elif self.capabilities.case_insensitive_like == "collate_like":
            return f"{safe_column} COLLATE SQL_Latin1_General_CP1_CI_AS LIKE '%{value}%'"
        return f"{safe_column} LIKE '%{value}%'"
    
    def quote_identifier(self, identifier: str) -> str:
        """
        Quote an identifier (table, column name) for this dialect.
        
        Args:
            identifier: The identifier to quote
            
        Returns:
            Quoted identifier
        """
        if self.capabilities.quote_char == "[":
            return f"[{identifier}]"
        return f"{self.capabilities.quote_char}{identifier}{self.capabilities.quote_char}"
    
    def get_bool_value(self, value: bool) -> str:
        """
        Get the dialect-specific boolean representation.
        
        Args:
            value: Boolean value
            
        Returns:
            Dialect-specific string representation
        """
        return self.capabilities.bool_values[0] if value else self.capabilities.bool_values[1]
    
    def get_schema_prefix(self, schema: str, table: str) -> str:
        """
        Get schema-prefixed table name appropriate for dialect.
        
        Args:
            schema: Schema name
            table: Table name
            
        Returns:
            Schema-prefixed table reference
        """
        if not self.capabilities.supports_schemas or not schema:
            return self.quote_identifier(table)
        return f"{self.quote_identifier(schema)}.{self.quote_identifier(table)}"
    
    def render_null_safe_equal(self, column: str, value: str) -> str:
        """
        Render NULL-safe equality comparison.
        
        Args:
            column: Column name
            value: Value to compare
            
        Returns:
            Dialect-specific NULL-safe comparison
        """
        safe_column = self.quote_identifier(column)
        if self.capabilities.null_safe_equal == "<=>":
            return f"{safe_column} <=> {value}"
        elif self.capabilities.null_safe_equal == "IS NOT DISTINCT FROM":
            return f"{safe_column} IS NOT DISTINCT FROM {value}"
        else:
            # SQLite and SQL Server: use explicit NULL check
            return f"(({safe_column} = {value}) OR ({safe_column} IS NULL AND {value} IS NULL))"
    
    def supports_feature(self, feature: str) -> bool:
        """
        Check if the dialect supports a specific feature.
        
        Args:
            feature: Feature name ('schemas', 'upsert', etc.)
            
        Returns:
            Whether the feature is supported
        """
        feature_map = {
            "schemas": self.capabilities.supports_schemas,
            "upsert": self.capabilities.supports_upsert,
        }
        return feature_map.get(feature, False)


def get_adapter(dialect: str) -> DialectAdapter:
    """
    Get a DialectAdapter for the specified dialect.
    
    Args:
        dialect: Dialect name as string (e.g., 'postgresql', 'mysql')
        
    Returns:
        Configured DialectAdapter
        
    Raises:
        ValueError: If dialect is not supported
    """
    try:
        db_dialect = DatabaseDialect(dialect.lower())
        return DialectAdapter(db_dialect)
    except ValueError:
        supported = ", ".join([d.value for d in DatabaseDialect])
        raise ValueError(f"Unsupported dialect '{dialect}'. Supported: {supported}")
