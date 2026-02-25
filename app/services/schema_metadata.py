"""
Database-agnostic schema metadata layer for Phase 2.

Replaces regex-based schema parsing with adapter-driven introspection.
Provides structural TableMetadata objects instead of raw strings.

This is the foundation for true database-agnostic query planning.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Set, List, Any
from enum import Enum


class ColumnType(Enum):
    """Abstract column types (database-agnostic)."""
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    TEXT = "text"
    DATE = "date"
    TIMESTAMP = "timestamp"
    UUID = "uuid"
    JSON = "json"
    ENUM = "enum"
    UNKNOWN = "unknown"


@dataclass
class ColumnMetadata:
    """Represents a single column with full metadata (replacing string tuples)."""
    name: str
    data_type: str  # Raw database type (e.g., "bigint", "boolean", "text")
    abstract_type: ColumnType  # Normalized type for query planning
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_table: Optional[str] = None  # Table this FK references
    foreign_key_column: Optional[str] = None  # Column this FK references
    default_value: Optional[Any] = None
    is_searchable: bool = True  # Whether this column can be searched
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, ColumnMetadata):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "data_type": self.data_type,
            "abstract_type": self.abstract_type.value,
            "nullable": self.nullable,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "foreign_key_table": self.foreign_key_table,
            "foreign_key_column": self.foreign_key_column,
        }


@dataclass
class TableMetadata:
    """Represents complete table structure (replacing regex-parsed dict)."""
    name: str
    schema_name: Optional[str] = None
    columns: Dict[str, ColumnMetadata] = field(default_factory=dict)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: Dict[str, tuple] = field(default_factory=dict)  # col -> (ref_table, ref_col)
    
    @property
    def column_names(self) -> Set[str]:
        """Get all column names."""
        return set(self.columns.keys())
    
    @property
    def boolean_columns(self) -> Set[str]:
        """Get all boolean-type columns."""
        return {
            name for name, col in self.columns.items()
            if col.abstract_type == ColumnType.BOOLEAN
        }
    
    @property
    def searchable_columns(self) -> Set[str]:
        """Get all searchable columns."""
        return {
            name for name, col in self.columns.items()
            if col.is_searchable
        }
    
    def get_column(self, name: str) -> Optional[ColumnMetadata]:
        """Get column by name (case-insensitive)."""
        # Try exact match first
        if name in self.columns:
            return self.columns[name]
        
        # Try case-insensitive match
        name_lower = name.lower()
        for col_name, col_meta in self.columns.items():
            if col_name.lower() == name_lower:
                return col_meta
        
        return None
    
    def has_column(self, name: str) -> bool:
        """Check if column exists (case-insensitive)."""
        return self.get_column(name) is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "schema_name": self.schema_name,
            "columns": {name: col.to_dict() for name, col in self.columns.items()},
            "primary_keys": self.primary_keys,
            "foreign_keys": self.foreign_keys,
        }


@dataclass
class DatabaseSchema:
    """Complete database schema with all tables (replacing multiple regex calls)."""
    database_name: str
    tables: Dict[str, TableMetadata] = field(default_factory=dict)
    db_type: str = "unknown"  # The database type (postgresql, mysql, etc.)
    
    @property
    def table_names(self) -> Set[str]:
        """Get all table names."""
        return set(self.tables.keys())
    
    def get_table(self, name: str) -> Optional[TableMetadata]:
        """Get table by name (case-insensitive)."""
        # Try exact match first
        if name in self.tables:
            return self.tables[name]
        
        # Try case-insensitive match
        name_lower = name.lower()
        for table_name, table_meta in self.tables.items():
            if table_name.lower() == name_lower:
                return table_meta
        
        return None
    
    def has_table(self, name: str) -> bool:
        """Check if table exists (case-insensitive)."""
        return self.get_table(name) is not None
    
    def find_column_across_tables(self, column_name: str) -> Optional[tuple]:
        """
        Search for a column across all joinable tables.
        
        Returns:
            Tuple of (table_name, column_metadata) or None if not found
        """
        column_lower = column_name.lower()
        
        # Find all tables containing this column
        candidates = []
        for table_name, table_meta in self.tables.items():
            col = table_meta.get_column(column_name)
            if col:
                candidates.append((table_name, col))
        
        # Return first match (prefer exact case match)
        if candidates:
            # Sort to prefer exact case match
            candidates.sort(key=lambda x: x[1].name != column_name)
            return candidates[0]
        
        return None
    
    def find_join_path(self, table1: str, table2: str) -> Optional[Dict[str, str]]:
        """
        Find join conditions between two tables using foreign keys.
        
        Returns:
            Dict with join metadata or None if tables can't be joined
        """
        t1 = self.get_table(table1)
        t2 = self.get_table(table2)
        
        if not t1 or not t2:
            return None
        
        # Check if t1 has FK to t2
        for col_name, (ref_table, ref_col) in t1.foreign_keys.items():
            if ref_table.lower() == table2.lower():
                return {
                    "join_type": "INNER",
                    "left_table": table1,
                    "left_column": col_name,
                    "right_table": table2,
                    "right_column": ref_col,
                }
        
        # Check if t2 has FK to t1
        for col_name, (ref_table, ref_col) in t2.foreign_keys.items():
            if ref_table.lower() == table1.lower():
                return {
                    "join_type": "INNER",
                    "left_table": table2,
                    "left_column": col_name,
                    "right_table": table1,
                    "right_column": ref_col,
                }
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "database_name": self.database_name,
            "db_type": self.db_type,
            "tables": {name: table.to_dict() for name, table in self.tables.items()},
        }


class ColumnTypeMapper:
    """Maps database-specific types to abstract column types."""
    
    # PostgreSQL type mappings
    POSTGRESQL_MAPPINGS = {
        # Boolean
        "boolean": ColumnType.BOOLEAN,
        "bool": ColumnType.BOOLEAN,
        
        # Integer
        "integer": ColumnType.INTEGER,
        "int": ColumnType.INTEGER,
        "int2": ColumnType.INTEGER,
        "int4": ColumnType.INTEGER,
        "int8": ColumnType.INTEGER,
        "bigint": ColumnType.INTEGER,
        "smallint": ColumnType.INTEGER,
        "serial": ColumnType.INTEGER,
        "bigserial": ColumnType.INTEGER,
        
        # Float
        "real": ColumnType.FLOAT,
        "double precision": ColumnType.FLOAT,
        "numeric": ColumnType.FLOAT,
        "decimal": ColumnType.FLOAT,
        
        # Text
        "text": ColumnType.TEXT,
        "character varying": ColumnType.TEXT,
        "varchar": ColumnType.TEXT,
        "char": ColumnType.TEXT,
        
        # Date/Time
        "date": ColumnType.DATE,
        "timestamp": ColumnType.TIMESTAMP,
        "timestamp without time zone": ColumnType.TIMESTAMP,
        "timestamp with time zone": ColumnType.TIMESTAMP,
        "time": ColumnType.TIMESTAMP,
        
        # UUID
        "uuid": ColumnType.UUID,
        
        # JSON
        "json": ColumnType.JSON,
        "jsonb": ColumnType.JSON,
    }
    
    # MySQL type mappings
    MYSQL_MAPPINGS = {
        # Boolean
        "boolean": ColumnType.BOOLEAN,
        "bool": ColumnType.BOOLEAN,
        "tinyint": ColumnType.INTEGER,  # Often used for bool in MySQL
        
        # Integer
        "int": ColumnType.INTEGER,
        "integer": ColumnType.INTEGER,
        "int2": ColumnType.INTEGER,
        "int4": ColumnType.INTEGER,
        "int8": ColumnType.INTEGER,
        "bigint": ColumnType.INTEGER,
        "smallint": ColumnType.INTEGER,
        "mediumint": ColumnType.INTEGER,
        
        # Float
        "float": ColumnType.FLOAT,
        "double": ColumnType.FLOAT,
        "decimal": ColumnType.FLOAT,
        "numeric": ColumnType.FLOAT,
        
        # Text
        "char": ColumnType.TEXT,
        "varchar": ColumnType.TEXT,
        "text": ColumnType.TEXT,
        "tinytext": ColumnType.TEXT,
        "mediumtext": ColumnType.TEXT,
        "longtext": ColumnType.TEXT,
        
        # Date/Time
        "date": ColumnType.DATE,
        "datetime": ColumnType.TIMESTAMP,
        "timestamp": ColumnType.TIMESTAMP,
        "time": ColumnType.TIMESTAMP,
        
        # JSON
        "json": ColumnType.JSON,
    }
    
    # SQL Server type mappings
    SQLSERVER_MAPPINGS = {
        # Boolean
        "bit": ColumnType.BOOLEAN,
        
        # Integer
        "int": ColumnType.INTEGER,
        "bigint": ColumnType.INTEGER,
        "smallint": ColumnType.INTEGER,
        "tinyint": ColumnType.INTEGER,
        
        # Float
        "float": ColumnType.FLOAT,
        "real": ColumnType.FLOAT,
        "numeric": ColumnType.FLOAT,
        "decimal": ColumnType.FLOAT,
        
        # Text
        "char": ColumnType.TEXT,
        "varchar": ColumnType.TEXT,
        "text": ColumnType.TEXT,
        "nchar": ColumnType.TEXT,
        "nvarchar": ColumnType.TEXT,
        "ntext": ColumnType.TEXT,
        
        # Date/Time
        "date": ColumnType.DATE,
        "datetime": ColumnType.TIMESTAMP,
        "datetime2": ColumnType.TIMESTAMP,
        "smalldatetime": ColumnType.TIMESTAMP,
        "time": ColumnType.TIMESTAMP,
        
        # UUID
        "uniqueidentifier": ColumnType.UUID,
    }
    
    # Oracle type mappings
    ORACLE_MAPPINGS = {
        # Boolean - Oracle doesn't have native boolean, uses CHAR(1) or NUMBER
        "char": ColumnType.TEXT,
        
        # Integer
        "number": ColumnType.INTEGER,
        "integer": ColumnType.INTEGER,
        
        # Float
        "binary_float": ColumnType.FLOAT,
        "binary_double": ColumnType.FLOAT,
        
        # Text
        "varchar2": ColumnType.TEXT,
        "varchar": ColumnType.TEXT,
        "char": ColumnType.TEXT,
        "clob": ColumnType.TEXT,
        
        # Date/Time
        "date": ColumnType.DATE,
        "timestamp": ColumnType.TIMESTAMP,
        
        # JSON
        "json": ColumnType.JSON,
    }
    
    @classmethod
    def map_type(cls, db_type: str, raw_column_type: str) -> ColumnType:
        """Map database-specific type to abstract type."""
        type_lower = raw_column_type.lower().strip()
        
        # Get appropriate mapping for database type
        if db_type.lower() in ["postgresql", "postgres"]:
            mapping = cls.POSTGRESQL_MAPPINGS
        elif db_type.lower() in ["mysql"]:
            mapping = cls.MYSQL_MAPPINGS
        elif db_type.lower() in ["sqlserver", "sql_server", "mssql"]:
            mapping = cls.SQLSERVER_MAPPINGS
        elif db_type.lower() in ["oracle"]:
            mapping = cls.ORACLE_MAPPINGS
        else:
            # Default/fallback
            mapping = cls.POSTGRESQL_MAPPINGS
        
        # Try exact match
        if type_lower in mapping:
            return mapping[type_lower]
        
        # Try substring match (e.g., "character varying" -> just take the type)
        for db_pattern, abstract_type in mapping.items():
            if db_pattern in type_lower or type_lower in db_pattern:
                return abstract_type
        
        # Default to unknown
        return ColumnType.UNKNOWN
