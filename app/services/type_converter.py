"""
Dynamic SQL type converter - fixes type mismatches based on actual schema.

Instead of hardcoding column names, this reads the actual database schema
to detect boolean columns and automatically converts comparisons.

Works across different databases (PostgreSQL, MySQL, SQLite, etc.)
"""

import re
from typing import Dict, List, Optional, Set
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from ..config import settings


class DynamicTypeConverter:
    """Converts SQL based on actual database schema types."""
    
    def __init__(self):
        self.schema_cache: Dict[str, Dict[str, str]] = {}  # {table: {col: type}}
        self.db_type: Optional[str] = None  # postgresql, mysql, sqlite, etc.
    
    async def initialize(self, session: AsyncSession) -> None:
        """Detect database type from connection."""
        try:
            # Default to PostgreSQL (most common for this system)
            # Try to get from engine if available
            if hasattr(session, 'engine') and session.engine:
                db_url = str(session.engine.url)
            else:
                db_url = ""
            
            # Determine database type from URL
            if 'mysql' in db_url.lower():
                self.db_type = "mysql"
            elif 'sqlite' in db_url.lower():
                self.db_type = "sqlite"
            else:
                self.db_type = "postgresql"  # Default
            
            print(f"[TYPE_CONVERTER] Using database type: {self.db_type}")
        except Exception as e:
            print(f"[WARNING] Could not detect database type: {e}")
            self.db_type = "postgresql"  # Default fallback
    
    async def get_table_schema(
        self, 
        session: AsyncSession, 
        table_name: str,
        schema_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Fetch actual column types for a table from database schema.
        
        Returns: {column_name: data_type, ...}
        """
        # Use cache if available
        cache_key = f"{schema_name or settings.postgres_schema}.{table_name}"
        if cache_key in self.schema_cache:
            return self.schema_cache[cache_key]
        
        schema = schema_name or settings.postgres_schema
        
        try:
            # PostgreSQL
            if self.db_type == "postgresql":
                query = text("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_schema = :schema AND table_name = :table
                    ORDER BY ordinal_position
                """)
                result = await session.execute(query, {"schema": schema, "table": table_name})
            
            # MySQL
            elif self.db_type == "mysql":
                query = text("""
                    SELECT COLUMN_NAME, COLUMN_TYPE
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
                    ORDER BY ORDINAL_POSITION
                """)
                result = await session.execute(query, {"schema": schema, "table": table_name})
            
            # SQLite
            elif self.db_type == "sqlite":
                query = text(f"PRAGMA table_info({table_name})")
                result = await session.execute(query)
                # SQLite returns: cid, name, type, notnull, dflt_value, pk
                columns = {}
                for row in result:
                    columns[row[1]] = row[2]  # name, type
                self.schema_cache[cache_key] = columns
                return columns
            
            else:
                print(f"[WARNING] Unsupported database type: {self.db_type}")
                return {}
            
            # Build column type dict
            columns = {}
            for col_name, col_type in result.fetchall():
                columns[col_name] = str(col_type).lower()
            
            # Cache it
            self.schema_cache[cache_key] = columns
            return columns
            
        except Exception as e:
            print(f"[ERROR] Failed to get schema for {table_name}: {e}")
            return {}
    
    def detect_boolean_columns(self, schema: Dict[str, str]) -> Set[str]:
        """Detect which columns are boolean based on actual types."""
        boolean_types = {
            'boolean', 'bool',  # PostgreSQL
            'tinyint',  # MySQL (sometimes used for booleans)
            'bit',  # SQL Server
        }
        
        boolean_cols = set()
        for col_name, col_type in schema.items():
            col_type_lower = col_type.lower()
            
            # Check for exact matches
            if any(btype in col_type_lower for btype in boolean_types):
                boolean_cols.add(col_name)
        
        return boolean_cols
    
    def get_boolean_literals(self) -> tuple[str, str]:
        """Get TRUE/FALSE literals for the database type."""
        if self.db_type == "postgresql":
            return ("true", "false")
        elif self.db_type == "mysql":
            return ("TRUE", "FALSE")
        elif self.db_type == "sqlite":
            return ("1", "0")  # SQLite uses 1/0 for boolean
        else:
            return ("true", "false")  # Default PostgreSQL style
    
    async def convert_sql(
        self,
        sql: str,
        session: AsyncSession,
        schema_name: Optional[str] = None
    ) -> str:
        """
        Convert SQL to fix type mismatches based on actual schema.
        
        Example:
            Input:  SELECT * FROM customers WHERE kyc_verified = 1
            Output: SELECT * FROM customers WHERE kyc_verified = true
        
        Works by:
        1. Extracting table names from SQL
        2. Fetching their schemas
        3. Detecting boolean columns
        4. Converting integer/string comparisons to boolean literals
        """
        schema = schema_name or settings.postgres_schema
        
        # Extract table names from SQL
        table_pattern = r'(?:FROM|JOIN)\s+(?:' + schema.replace('.', r'\.') + r'\.)?(\w+)'
        tables = set(re.findall(table_pattern, sql, re.IGNORECASE))
        
        if not tables:
            print(f"[TYPE_CONVERTER] No tables found in SQL, skipping conversion")
            return sql
        
        print(f"[TYPE_CONVERTER] Detected tables: {tables}")
        
        # Fetch schemas and collect all boolean columns
        all_bool_cols: Dict[str, Set[str]] = {}  # {table_name: {bool_cols}}
        
        for table in tables:
            table_schema = await self.get_table_schema(session, table, schema)
            bool_cols = self.detect_boolean_columns(table_schema)
            if bool_cols:
                all_bool_cols[table] = bool_cols
                print(f"[TYPE_CONVERTER] {table} boolean columns: {bool_cols}")
        
        if not all_bool_cols:
            print(f"[TYPE_CONVERTER] No boolean columns found, skipping conversion")
            return sql
        
        # Get database-specific boolean literals
        true_lit, false_lit = self.get_boolean_literals()
        
        # Convert boolean comparisons for each column
        converted_sql = sql
        for table, bool_cols in all_bool_cols.items():
            for bool_col in bool_cols:
                # Pattern: column = 1 → column = true
                converted_sql = re.sub(
                    rf'\b{bool_col}\s*=\s*1\b',
                    f'{bool_col} = {true_lit}',
                    converted_sql,
                    flags=re.IGNORECASE
                )
                
                # Pattern: column = 0 → column = false
                converted_sql = re.sub(
                    rf'\b{bool_col}\s*=\s*0\b',
                    f'{bool_col} = {false_lit}',
                    converted_sql,
                    flags=re.IGNORECASE
                )
                
                # Pattern: column = '1' → column = true
                converted_sql = re.sub(
                    rf"{bool_col}\s*=\s*['\"]1['\"]",
                    f"{bool_col} = {true_lit}",
                    converted_sql,
                    flags=re.IGNORECASE
                )
                
                # Pattern: column = '0' → column = false
                converted_sql = re.sub(
                    rf"{bool_col}\s*=\s*['\"]0['\"]",
                    f"{bool_col} = {false_lit}",
                    converted_sql,
                    flags=re.IGNORECASE
                )
        
        # Make LIKE clauses case-insensitive (helps with city names, etc.)
        converted_sql = self._make_like_case_insensitive(converted_sql)
        
        if converted_sql != sql:
            print(f"[TYPE_CONVERTER] SQL converted:\n  Before: {sql[:100]}\n  After: {converted_sql[:100]}")
        
        return converted_sql
    
    def _make_like_case_insensitive(self, sql: str) -> str:
        """Convert LIKE to case-insensitive version for the database type."""
        if self.db_type == "postgresql":
            # PostgreSQL: LIKE → ILIKE
            return re.sub(r'\bLIKE\b', 'ILIKE', sql, flags=re.IGNORECASE)
        elif self.db_type == "mysql":
            # MySQL: LIKE is already case-insensitive by default
            return sql
        elif self.db_type == "sqlite":
            # SQLite: Use COLLATE NOCASE or lower()
            # Pattern: field LIKE 'value%' → field LIKE 'value%' COLLATE NOCASE
            return re.sub(
                r"(\bLIKE\s+'[^']*')",
                r"\1 COLLATE NOCASE",
                sql,
                flags=re.IGNORECASE
            )
        else:
            # Default: convert to ILIKE (PostgreSQL style)
            return re.sub(r'\bLIKE\b', 'ILIKE', sql, flags=re.IGNORECASE)


# Global instance
_type_converter: Optional[DynamicTypeConverter] = None


async def get_type_converter() -> DynamicTypeConverter:
    """Get or create global type converter."""
    global _type_converter
    if _type_converter is None:
        _type_converter = DynamicTypeConverter()
    return _type_converter


async def initialize_type_converter(session: AsyncSession) -> None:
    """Initialize the type converter (detect database type)."""
    converter = await get_type_converter()
    await converter.initialize(session)
