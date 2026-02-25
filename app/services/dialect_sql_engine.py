"""
Dialect-aware SQL engine - handles all database-specific SQL generation and transformation.

This module isolates all database differences so sql_generator.py can be fully dialect-neutral.

Supported databases:
- PostgreSQL (schema.table, LIMIT, ||, current_date, double quotes)
- MySQL (database.table, LIMIT, CONCAT(), NOW(), backticks)
- SQLite (no schemas, LIMIT, ||, CURRENT_DATE, unquoted)
- SQL Server (schema.table, TOP (N), +, GETDATE(), square brackets)
- Oracle (schema.table, FETCH FIRST N ROWS ONLY, ||, SYSDATE, quotes)
"""

from typing import Optional, List, Tuple, Dict, Any
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.engine import Engine
import re

from .database_adapter import DatabaseType


class DialectSqlEngine:
    """Handles database-specific SQL generation and transformation."""
    
    def __init__(self, dialect_name: str):
        """
        Initialize dialect engine.
        
        Args:
            dialect_name: SQLAlchemy dialect name (e.g., 'postgresql', 'mysql', 'sqlite', 'mssql', 'oracle')
        """
        self.dialect = dialect_name.lower()
        self.dialect_type = self._map_dialect_type(self.dialect)
    
    @staticmethod
    def _map_dialect_type(dialect: str) -> DatabaseType:
        """Map SQLAlchemy dialect name to our DatabaseType."""
        dialect_lower = dialect.lower()
        
        if 'postgres' in dialect_lower or 'psycopg' in dialect_lower:
            return DatabaseType.POSTGRESQL
        elif 'mysql' in dialect_lower or 'pymysql' in dialect_lower:
            return DatabaseType.MYSQL
        elif 'sqlite' in dialect_lower:
            return DatabaseType.SQLITE
        elif 'mssql' in dialect_lower or 'sqlserver' in dialect_lower:
            return DatabaseType.SQL_SERVER
        elif 'oracle' in dialect_lower:
            return DatabaseType.ORACLE
        else:
            return DatabaseType.UNKNOWN
    
    # ==================== ROW LIMITING - Dialect-Specific ====================
    
    def enforce_row_limit(self, sql: str, max_rows: int = 500) -> str:
        """
        Enforce a row limit on a SQL query using dialect-appropriate syntax.
        
        Args:
            sql: The SQL query (assumed to be a single SELECT statement)
            max_rows: Maximum rows to return
            
        Returns:
            SQL with row limit enforced (or original SQL if already has limit)
        """
        sql_upper = sql.upper()
        
        # Check if query already has a limit/top/fetch clause
        if self._has_row_limit(sql_upper):
            return sql
        
        # Apply dialect-specific limiting
        if self.dialect_type == DatabaseType.POSTGRESQL:
            return self._limit_postgresql(sql, max_rows)
        elif self.dialect_type == DatabaseType.MYSQL:
            return self._limit_mysql(sql, max_rows)
        elif self.dialect_type == DatabaseType.SQLITE:
            return self._limit_sqlite(sql, max_rows)
        elif self.dialect_type == DatabaseType.SQL_SERVER:
            return self._limit_sqlserver(sql, max_rows)
        elif self.dialect_type == DatabaseType.ORACLE:
            return self._limit_oracle(sql, max_rows)
        else:
            # Fallback: try LIMIT syntax (works for most DBs)
            return f"{sql.rstrip(';')} LIMIT {max_rows};"
    
    def _has_row_limit(self, sql_upper: str) -> bool:
        """Check if SQL already has a row limiting clause."""
        return (
            'LIMIT' in sql_upper or  # PostgreSQL, MySQL, SQLite
            'TOP (' in sql_upper or  # SQL Server
            'FETCH FIRST' in sql_upper or  # Oracle, DB2
            'FETCH NEXT' in sql_upper  # SQL Server (OFFSET ... FETCH NEXT)
        )
    
    @staticmethod
    def _limit_postgresql(sql: str, max_rows: int) -> str:
        """PostgreSQL: append LIMIT clause."""
        return f"{sql.rstrip(';')} LIMIT {max_rows};"
    
    @staticmethod
    def _limit_mysql(sql: str, max_rows: int) -> str:
        """MySQL: append LIMIT clause."""
        return f"{sql.rstrip(';')} LIMIT {max_rows};"
    
    @staticmethod
    def _limit_sqlite(sql: str, max_rows: int) -> str:
        """SQLite: append LIMIT clause."""
        return f"{sql.rstrip(';')} LIMIT {max_rows};"
    
    @staticmethod
    def _limit_sqlserver(sql: str, max_rows: int) -> str:
        """SQL Server: inject TOP (N) after SELECT keyword."""
        # Match SELECT or SELECT DISTINCT
        pattern = r'(\bSELECT)(\s+DISTINCT)?(\s+)'
        replacement = rf'\1\2 TOP ({max_rows})\3'
        result = re.sub(pattern, replacement, sql, flags=re.IGNORECASE, count=1)
        return result if result != sql else f"{sql.rstrip(';')} LIMIT {max_rows};"  # Fallback
    
    @staticmethod
    def _limit_oracle(sql: str, max_rows: int) -> str:
        """Oracle: append FETCH FIRST ... ROWS ONLY (12c+) or use ROWNUM (earlier versions)."""
        # Try FETCH FIRST syntax (Oracle 12c+)
        return f"{sql.rstrip(';')} FETCH FIRST {max_rows} ROWS ONLY;"
    
    # ==================== IDENTIFIER QUOTING - Dialect-Specific ====================
    
    def quote_identifier(self, name: str) -> str:
        """
        Quote an identifier (table name, column name) using dialect-appropriate quoting.
        Properly handles qualified names (table.column) by quoting each part separately.
        
        Args:
            name: The unquoted identifier (e.g., 'table' or 'table.column')
            
        Returns:
            Quoted identifier safe for use in SQL
        """
        if not name:
            return name
        
        # Avoid double-quoting
        if (name.startswith('"') and name.endswith('"')) or \
           (name.startswith('`') and name.endswith('`')) or \
           (name.startswith('[') and name.endswith(']')):
            return name
        
        # Handle qualified identifiers (table.column) by quoting parts separately
        if '.' in name and not name.startswith('.'):
            parts = name.split('.')
            quoted_parts = [self.quote_identifier(part) for part in parts]
            return '.'.join(quoted_parts)
        
        if self.dialect_type == DatabaseType.POSTGRESQL:
            return f'"{name}"'
        elif self.dialect_type == DatabaseType.MYSQL:
            return f'`{name}`'
        elif self.dialect_type == DatabaseType.SQLITE:
            return f'"{name}"'
        elif self.dialect_type == DatabaseType.SQL_SERVER:
            return f'[{name}]'
        elif self.dialect_type == DatabaseType.ORACLE:
            return f'"{name}"'
        else:
            return f'"{name}"'  # Fallback
    
    # ==================== TABLE/SCHEMA QUALIFICATION - Dialect-Specific ====================
    
    def qualify_table(self, table_name: str, schema_name: Optional[str] = None, 
                     catalog_name: Optional[str] = None) -> str:
        """
        Qualify a table name with schema and/or catalog based on dialect.
        
        Args:
            table_name: The table name
            schema_name: Optional schema name
            catalog_name: Optional catalog/database name
            
        Returns:
            Properly qualified table reference for this database
        """
        # Quote identifiers if they contain special characters or are reserved words
        quoted_table = self.quote_identifier(table_name) if needs_quoting(table_name) else table_name
        
        if self.dialect_type == DatabaseType.POSTGRESQL:
            # PostgreSQL: schema.table
            if schema_name:
                quoted_schema = self.quote_identifier(schema_name) if needs_quoting(schema_name) else schema_name
                return f"{quoted_schema}.{quoted_table}"
            return quoted_table
        
        elif self.dialect_type == DatabaseType.MYSQL:
            # MySQL: database.table (no separate schema concept, database ==  schema)
            if schema_name:
                quoted_schema = self.quote_identifier(schema_name) if needs_quoting(schema_name) else schema_name
                return f"{quoted_schema}.{quoted_table}"
            return quoted_table
        
        elif self.dialect_type == DatabaseType.SQLITE:
            # SQLite: no schemas, just table name
            return quoted_table
        
        elif self.dialect_type == DatabaseType.SQL_SERVER:
            # SQL Server: [catalog].[schema].[table]
            parts = []
            if catalog_name:
                parts.append(self.quote_identifier(catalog_name) if needs_quoting(catalog_name) else catalog_name)
            if schema_name:
                parts.append(self.quote_identifier(schema_name) if needs_quoting(schema_name) else schema_name)
            parts.append(quoted_table)
            return '.'.join(parts)
        
        elif self.dialect_type == DatabaseType.ORACLE:
            # Oracle: schema.table (catalog doesn't work the same way)
            if schema_name:
                quoted_schema = self.quote_identifier(schema_name) if needs_quoting(schema_name) else schema_name
                return f"{quoted_schema}.{quoted_table}"
            return quoted_table
        
        else:
            # Fallback
            if schema_name:
                return f"{schema_name}.{quoted_table}"
            return quoted_table
    
    # ==================== STRING CONCATENATION - Dialect-Specific ====================
    
    def concat_function(self, *parts: str) -> str:
        """
        Generate string concatenation SQL for this dialect.
        
        Args:
            parts: Expressions/values to concatenate
            
        Returns:
            SQL string concatenation
        """
        if not parts:
            return "''"
        
        if self.dialect_type in (DatabaseType.POSTGRESQL, DatabaseType.SQLITE, DatabaseType.ORACLE):
            # PostgreSQL, SQLite, Oracle: use ||
            return ' || '.join(parts)
        elif self.dialect_type == DatabaseType.MYSQL:
            # MySQL: use CONCAT()
            return f"CONCAT({', '.join(parts)})"
        elif self.dialect_type == DatabaseType.SQL_SERVER:
            # SQL Server: use +
            return ' + '.join(parts)
        else:
            # Fallback to ||
            return ' || '.join(parts)
    
    # ==================== DATE/TIME FUNCTIONS - Dialect-Specific ====================
    
    def current_date_function(self) -> str:
        """Return SQL for getting current date."""
        if self.dialect_type == DatabaseType.SQL_SERVER:
            return "CAST(GETDATE() AS DATE)"
        elif self.dialect_type == DatabaseType.MYSQL:
            return "CURDATE()"  # or "DATE(NOW())"
        elif self.dialect_type == DatabaseType.ORACLE:
            return "TRUNC(SYSDATE)"
        else:
            # PostgreSQL, SQLite, and most others
            return "CURRENT_DATE"
    
    def current_timestamp_function(self) -> str:
        """Return SQL for getting current timestamp."""
        if self.dialect_type == DatabaseType.SQL_SERVER:
            return "GETDATE()"
        elif self.dialect_type == DatabaseType.MYSQL:
            return "NOW()"
        elif self.dialect_type == DatabaseType.ORACLE:
            return "SYSDATE"
        else:
            # PostgreSQL, SQLite, and most others
            return "CURRENT_TIMESTAMP"
    
    # ==================== SCHEMA INTROSPECTION - Using SQLAlchemy ====================
    
    @staticmethod
    async def introspect_tables(engine: Engine, schema_name: Optional[str] = None) -> List[str]:
        """
        Get list of all table names in a schema using SQLAlchemy Inspector.
        
        Args:
            engine: SQLAlchemy engine
            schema_name: Optional schema name (required for some DBs, ignored for others)
            
        Returns:
            List of table names
        """
        inspector = inspect(engine)
        return inspector.get_table_names(schema=schema_name)
    
    @staticmethod
    async def introspect_columns(engine: Engine, table_name: str, 
                                 schema_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of columns for a table using SQLAlchemy Inspector.
        
        Args:
            engine: SQLAlchemy engine
            table_name: The table name
            schema_name: Optional schema name
            
        Returns:
            List of column info dicts with keys: name, type, nullable, default, etc.
        """
        inspector = inspect(engine)
        return inspector.get_columns(table_name, schema=schema_name)
    
    @staticmethod
    async def introspect_pk(engine: Engine, table_name: str, 
                           schema_name: Optional[str] = None) -> List[str]:
        """Get primary key column names using SQLAlchemy Inspector."""
        inspector = inspect(engine)
        pk = inspector.get_pk_constraint(table_name, schema=schema_name)
        return pk.get('constrained_columns', []) if pk else []
    
    @staticmethod
    async def introspect_fks(engine: Engine, table_name: str, 
                            schema_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get foreign key information using SQLAlchemy Inspector."""
        inspector = inspect(engine)
        return inspector.get_foreign_keys(table_name, schema=schema_name)
    
    # ==================== LLM PROMPT HINTS - Dialect-Specific Guidance ====================
    
    def get_dialect_prompt_hint(self) -> str:
        """
        Return a dialect-specific prompt hint to include in the LLM system prompt.
        
        This tells the LLM what SQL syntax to use for this specific database.
        """
        if self.dialect_type == DatabaseType.POSTGRESQL:
            return """
DIALECT: PostgreSQL
- Table references: schema_name.table_name (use double quotes for case sensitivity)
- Row limiting: append LIMIT {number} at the end
- String concat: use || operator
- Current date: CURRENT_DATE
- Boolean: true/false or 't'/'f'
- Quoting: double quotes for identifiers
            """
        
        elif self.dialect_type == DatabaseType.MYSQL:
            return """
DIALECT: MySQL
- Table references: database_name.table_name 
- Row limiting: append LIMIT {number} at the end
- String concat: use CONCAT(str1, str2, ...)
- Current date: CURDATE() or DATE(NOW())
- Boolean: true/false or 1/0
- Quoting: backticks ` for identifiers
            """
        
        elif self.dialect_type == DatabaseType.SQLITE:
            return """
DIALECT: SQLite
- Table references: table_name only (no schemas)
- Row limiting: append LIMIT {number} at the end
- String concat: use || operator
- Current date: DATE('now') or CURRENT_DATE
- Boolean: no native boolean; use 1/0 or true/false
- Quoting: double quotes for identifiers
            """
        
        elif self.dialect_type == DatabaseType.SQL_SERVER:
            return """
DIALECT: SQL Server
- Table references: [schema_name].[table_name] (bracket notation)
- Row limiting: use TOP ({number}) immediately after SELECT keyword
- String concat: use + operator
- Current date: CAST(GETDATE() AS DATE)
- Boolean: no native type; use 1/0 or bit type
- Quoting: square brackets [identifier] for case sensitivity
            """
        
        elif self.dialect_type == DatabaseType.ORACLE:
            return """
DIALECT: Oracle
- Table references: schema_name.table_name (use double quotes for case sensitivity)
- Row limiting: append FETCH FIRST {number} ROWS ONLY (12c+) or use ROWNUM
- String concat: use || operator
- Current date: TRUNC(SYSDATE)
- Boolean: no native type; use 1/0 or char(1) Y/N
- Quoting: double quotes for identifiers
            """
        
        else:
            return "DIALECT: Unknown—use standard SQL syntax (LIMIT, double quotes, ||, CURRENT_DATE)"


def needs_quoting(identifier: str) -> bool:
    """
    Check if an identifier needs quoting (contains special chars, reserved words, mixed case).
    
    Args:
        identifier: The identifier to check
        
    Returns:
        True if quoting is recommended
    """
    if not identifier:
        return False
    
    # Check for special characters or spaces
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
        return True
    
    # Check for mixed case (suggests user wants case-preserving)
    if identifier != identifier.lower() and identifier != identifier.upper():
        return True
    
    # Common SQL reserved words that might need quoting
    reserved_words = {
        'select', 'from', 'where', 'join', 'inner', 'left', 'right', 'full', 'on',
        'group', 'by', 'order', 'having', 'limit', 'offset', 'union', 'all', 'distinct',
        'insert', 'update', 'delete', 'drop', 'create', 'alter', 'table', 'schema',
        'database', 'index', 'view', 'procedure', 'function', 'trigger',
        'and', 'or', 'not', 'in', 'is', 'null', 'true', 'false', 'between', 'like',
        'as', 'with', 'case', 'when', 'then', 'else', 'end', 'cast'
    }
    
    return identifier.lower() in reserved_words
