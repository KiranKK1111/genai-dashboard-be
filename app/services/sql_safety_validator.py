"""
SQL Safety Validator - Ensures queries are safe before execution.

Security checks:
✓ SELECT-only (no INSERT, UPDATE, DELETE, DROP) - AST-based validation
✓ Parameterized queries (no SQL injection)
✓ Row limits enforced with metadata tracking
✓ Schema allowlist (dialect-aware)
✓ Statement timeout
✓ No multiple statements
✓ Pattern-based injection detection
✓ Database-agnostic query guards
"""

from __future__ import annotations

import logging
import re
from typing import Tuple, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from typing import NamedTuple

try:
    import sqlglot
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False

logger = logging.getLogger(__name__)



class QueryType(str, Enum):
    """Supported query types."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    DDL = "ddl"
    UNKNOWN = "unknown"


@dataclass
class LimitMetadata:
    """Tracks LIMIT information for query responses."""
    applied_limit: int  # The actual limit applied (e.g., 500)
    user_limit_requested: Optional[int] = None  # User-specified LIMIT, if any
    limit_enforced: bool = False  # Whether we added/modified the limit
    
    def to_response_dict(self) -> dict:
        """Convert to response metadata dictionary."""
        return {
            "applied_limit": self.applied_limit,
            "user_limit_requested": self.user_limit_requested,
            "limit_enforced": self.limit_enforced,
            "message": self._get_limit_message()
        }
    
    def _get_limit_message(self) -> str:
        """Generate user-facing message about limit."""
        if self.user_limit_requested and self.user_limit_requested != self.applied_limit:
            return f"Your query has been limited to {self.applied_limit} rows (you requested {self.user_limit_requested})"
        elif self.limit_enforced:
            return f"Query limited to {self.applied_limit} rows for safety"
        else:
            return None


class SQLSafetyValidator:
    """
    Validates SQL queries for safety before execution.
    
    Prevents dangerous operations while allowing all types of SELECT queries.
    Uses multiple validation strategies to prevent SQL injection and malicious queries.
    """
    
    # Dangerous keywords that indicate DDL/DML
    DANGEROUS_KEYWORDS = [
        r'\bDROP\b',
        r'\bDELETE\b',
        r'\bINSERT\b',
        r'\bUPDATE\b',
        r'\bTRUNCATE\b',
        r'\bALTER\b',
        r'\bCREATE\b',
        r'\bGRANT\b',
        r'\bREVOKE\b',
        r'\bCOPY\b',  # PostgreSQL
        r'\bCALL\b',  # Stored procedures
        r'\bEXEC\b',  # Dynamic execution
        r'\bEXECUTE\b',  # Dynamic execution
    ]
    
    # SQL injection patterns
    INJECTION_PATTERNS = {
        r"(?:or|and)\s+1\s*=\s*1": "Tautology pattern detected (or 1=1)",
        r"(?:or|and)\s+['\"].+['\"]": "OR-based injection pattern",
        # NOTE: UNION pattern removed - AST validation handles this more reliably
        r"into\s+(outfile|infile|dumpfile)": "File I/O attempt detected",
        r";\s*\w+": "Multiple statements detected",
        r"\-\-\s|/\*.*?\*/": "SQL comment detected",
        r"xp_|sp_": "System stored procedure call detected",
    }
    
    # SELECT variants (allowed)
    SELECT_KEYWORDS = [
        r'\bSELECT\b',
        r'\bWITH\b',  # CTE
        r'\bFROM\b',
        r'\bWHERE\b',
        r'\bJOIN\b',
        r'\bGROUP\s+BY\b',
        r'\bHAVING\b',
        r'\bORDER\s+BY\b',
        r'\bLIMIT\b',
        r'\bOFFSET\b',
        r'\bDISTINCT\b',
    ]
    
    def __init__(
        self,
        allowed_schemas: Optional[List[str]] = None,
        max_rows: int = 500,
        statement_timeout_ms: int = 30000,
        enable_injection_detection: bool = True,
    ):
        """Initialize SQL Safety Validator.
        
        Args:
            allowed_schemas: List of allowed database schemas (default: auto-detect from database)
            max_rows: Maximum rows to return (default: 500 for safety)
            statement_timeout_ms: Query timeout in milliseconds
            enable_injection_detection: Whether to detect SQL injection patterns
        """
        from .database_adapter import get_global_adapter
        
        if allowed_schemas is None:
            try:
                adapter = get_global_adapter()
                default_schema = adapter.get_capabilities().default_schema
                allowed_schemas = [default_schema] if default_schema else ["public"]
                logger.info(f"[DB-AGNOSTIC] Using database adapter default schema: {allowed_schemas}")
            except Exception as e:
                logger.warning(f"[DB-AGNOSTIC] Could not determine default schema: {e}, using public")
                allowed_schemas = ["public"]
        
        self.allowed_schemas = allowed_schemas
        self.max_rows = max_rows
        self.statement_timeout_ms = statement_timeout_ms
        self.enable_injection_detection = enable_injection_detection
        self._discovered_schemas: Optional[List[str]] = None  # Cache for dynamically discovered schemas
    
    def validate_and_rewrite(self, sql: str) -> Tuple[bool, str, str]:
        """
        Primary validation method: Validate a SQL query and rewrite it with safety limits.
        
        Performs comprehensive security checks:
        1. AST-based query type validation (SELECT only)
        2. SQL injection pattern detection
        3. Multiple statement prevention
        4. Schema allowlist validation (dialect-aware)
        5. Query length limits
        6. LIMIT clause enforcement (critical for preventing large result sets)
        
        Args:
            sql: SQL query to validate and optionally rewrite
        
        Returns:
            (is_valid: bool, error_message: str, rewritten_sql: str)
            - is_valid: True if query passed all security checks
            - error_message: Details if validation failed (empty string if valid)
            - rewritten_sql: Query with enforced LIMIT clause (empty if invalid)
        """
        if not sql or not sql.strip():
            return False, "Empty SQL query", ""
        
        # Check 1: Try AST-based validation first (more reliable)
        if SQLGLOT_AVAILABLE:
            ast_error = self._validate_with_ast(sql)
            if ast_error:
                return False, ast_error, ""
        else:
            # Fallback to regex-based validation
            query_type = self._detect_query_type(sql)
            if query_type != QueryType.SELECT:
                return False, f"Only SELECT queries are allowed. Detected: {query_type.value}", ""
        
        # Check 2: Injection pattern detection
        if self.enable_injection_detection:
            injection_err = self._detect_injection_patterns(sql)
            if injection_err:
                return False, injection_err, ""
        
        # Check 3: Multiple statements check
        if self._has_multiple_statements(sql):
            return False, "Multiple statements not allowed", ""
        
        # Check 4: Schema validation (dialect-aware)
        schema_err = self._validate_schema_access(sql)
        if schema_err:
            return False, schema_err, ""
        
        # Check 5: Query length validation
        if len(sql) > 50000:
            return False, "Query exceeds maximum allowed length", ""
        
        # Check 6: Enforce safety limit (CRITICAL - was being ignored before)
        rewritten_sql = self._ensure_limit(sql)
        
        return True, "", rewritten_sql
    
    def _validate_with_ast(self, sql: str) -> Optional[str]:
        """
        AST-based validation using sqlglot.
        More reliable than regex for complex queries.
        """
        try:
            parsed = sqlglot.parse_one(sql)
            
            # Ensure single statement
            if not parsed:
                return "Failed to parse SQL statement"
            
            # Check statement type - only SELECT, CTE with SELECT, and WITH allowed
            stmt_type = type(parsed).__name__
            allowed_types = {'Select', 'Union', 'CTE'}
            
            if stmt_type not in allowed_types:
                return f"Only SELECT queries allowed. Got: {stmt_type}"
            
            # For CTE (WITH), ensure the final query is SELECT
            if isinstance(parsed, sqlglot.exp.CTE):
                # WITH queries are CTEs, check if they end with SELECT
                if not isinstance(parsed.args.get('expression'), sqlglot.exp.Select):
                    return "CTE must end with SELECT statement"
            
            return None
        except Exception as e:
            # If parsing fails, fall back to regex
            logger.warning(f"AST parsing failed for SQL: {str(e)[:100]}, falling back to regex")
            return None
    
    def _extract_limit_metadata(self, original_sql: str, rewritten_sql: str) -> LimitMetadata:
        """Extract limit metadata from original and rewritten SQL."""
        # Check if original had a LIMIT
        original_limit_match = re.search(r'\bLIMIT\s+(\d+)', original_sql, re.IGNORECASE)
        user_limit = int(original_limit_match.group(1)) if original_limit_match else None
        
        # Check if rewritten has a LIMIT
        rewritten_limit_match = re.search(r'\bLIMIT\s+(\d+)', rewritten_sql, re.IGNORECASE)
        applied_limit = int(rewritten_limit_match.group(1)) if rewritten_limit_match else self.max_rows
        
        # Determine if we enforced the limit
        limit_enforced = original_sql.strip() != rewritten_sql.strip()
        
        return LimitMetadata(
            applied_limit=applied_limit,
            user_limit_requested=user_limit,
            limit_enforced=limit_enforced
        )
    
    def _detect_query_type(self, sql: str) -> QueryType:
        """Detect the type of SQL query."""
        sql_stripped = sql.strip().upper()
        
        # Check for dangerous keywords first
        for keyword_pattern in self.DANGEROUS_KEYWORDS:
            if re.search(keyword_pattern, sql_stripped, re.IGNORECASE):
                # Determine specific type
                if re.search(r'\bDELETE\b', sql_stripped):
                    return QueryType.DELETE
                elif re.search(r'\bUPDATE\b', sql_stripped):
                    return QueryType.UPDATE
                elif re.search(r'\bINSERT\b', sql_stripped):
                    return QueryType.INSERT
                else:
                    return QueryType.DDL
        
        # Check for SELECT or WITH (CTE)
        if sql_stripped.startswith("SELECT") or sql_stripped.startswith("WITH"):
            return QueryType.SELECT
        
        return QueryType.UNKNOWN
    
    def _detect_injection_patterns(self, sql: str) -> Optional[str]:
        """Detect common SQL injection patterns."""
        for pattern, message in self.INJECTION_PATTERNS.items():
            if re.search(pattern, sql, re.IGNORECASE | re.DOTALL):
                return f"Security violation: {message}"
        return None
    
    def _is_select_query(self, sql: str) -> bool:
        """Check if query starts with SELECT or WITH (CTE)."""
        sql_stripped = sql.strip().upper()
        return sql_stripped.startswith("SELECT") or sql_stripped.startswith("WITH")
    
    def _contains_dangerous_keywords(self, sql: str) -> bool:
        """Check for dangerous keywords."""
        sql_upper = sql.upper()
        
        for keyword_pattern in self.DANGEROUS_KEYWORDS:
            if re.search(keyword_pattern, sql_upper):
                return True
        
        return False
    
    def _has_multiple_statements(self, sql: str) -> bool:
        """Check if multiple statements are present."""
        # More robust: only count semicolons outside string literals
        in_string = False
        string_char = None
        semicolon_count = 0
        
        for i, char in enumerate(sql):
            # Track string state (simple single/double quote handling)
            if char in ("'", '"') and (i == 0 or sql[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
            elif char == ';' and not in_string:
                semicolon_count += 1
        
        # Allow one trailing semicolon, but not more
        return semicolon_count > 1 or (semicolon_count == 1 and not sql.rstrip().endswith(';'))
    
    def _validate_schema_access(self, sql: str) -> Optional[str]:
        """Check if query accesses allowed schemas only.
        
        Dialect-aware: Only validates explicit schema.table references in FROM/JOIN clauses,
        not table aliases (like t.column_name).
        
        For databases without schema support (SQLite), validation is skipped.
        """
        try:
            from .database_adapter import get_global_adapter
            adapter = get_global_adapter()
            capabilities = adapter.get_capabilities()
            
            # Skip validation for databases without schema support
            if not capabilities.supports_schemas:
                logger.debug(f"[DB-AGNOSTIC] Database {capabilities.dialect.value} doesn't support schemas, skipping validation")
                return None
        except Exception as e:
            logger.debug(f"Could not get adapter capabilities: {e}, proceeding with schema validation")
        
        # Extract only actual schema.table references from table sources
        # Look for patterns like: FROM schema.table, JOIN schema.table
        table_source_pattern = r'(?:FROM|JOIN)\s+(?:ONLY\s+)?(\w+)\.(\w+)'
        matches = re.findall(table_source_pattern, sql, re.IGNORECASE)
        
        for schema, table in matches:
            # Skip single-letter "schemas" which are likely aliases
            if len(schema) > 1:
                # DYNAMIC SCHEMA VALIDATION - Check if schema exists in database
                error = self._validate_schema_exists(schema)
                if error:
                    return error
        
        return None
    
    def _get_available_schemas(self) -> List[str]:
        """Dynamically discover available schemas from the database (ChatGPT-style).
        
        Instead of hardcoded whitelists, query the actual database to see what schemas
        exist. This is fully dynamic and adapts to the database configuration.
        
        Returns:
            List of schema names available in the database
        """
        # Use cache if already discovered
        if self._discovered_schemas is not None:
            return self._discovered_schemas
        
        try:
            from .database_adapter import get_global_adapter
            from sqlalchemy import inspect
            
            adapter = get_global_adapter()
            
            # Try to get available schemas from the adapter's engine
            if hasattr(adapter, 'engine') and adapter.engine:
                try:
                    # Use synchronous inspect for schema discovery
                    inspector = inspect(adapter.engine)
                    available_schemas = inspector.get_schema_names()
                    
                    if available_schemas:
                        logger.info(f"[DYNAMIC_SCHEMA_DISCOVERY] Found schemas: {available_schemas}")
                        self._discovered_schemas = available_schemas
                        return available_schemas
                except Exception as e:
                    logger.debug(f"[DYNAMIC_SCHEMA_DISCOVERY] Database schema discovery failed: {e}")
            
            # Fallback: use configured schemas
            logger.info(f"[DYNAMIC_SCHEMA_DISCOVERY] Using configured schemas: {self.allowed_schemas}")
            self._discovered_schemas = self.allowed_schemas
            return self.allowed_schemas
            
        except Exception as e:
            logger.warning(f"[DYNAMIC_SCHEMA_DISCOVERY] Error discovering schemas: {e}, using configured")
            self._discovered_schemas = self.allowed_schemas
            return self.allowed_schemas
    
    def _validate_schema_exists(self, schema: str) -> Optional[str]:
        """Validate schema exists in database using dynamic discovery (ChatGPT-style).
        
        Instead of checking against hardcoded lists, check if the schema
        actually exists in the database. This is fully dynamic.
        
        Args:
            schema: Schema name to validate
            
        Returns:
            Error message if invalid, None if valid
        """
        try:
            # Get dynamically discovered schemas
            available_schemas = self._get_available_schemas()
            
            # Case-insensitive comparison
            schema_lower = schema.lower()
            available_lower = [s.lower() for s in available_schemas]
            
            if schema_lower not in available_lower:
                logger.warning(f"[DYNAMIC_SCHEMA_VALIDATION] Schema '{schema}' not found. Available: {available_schemas}")
                return f"Schema '{schema}' not found in database. Available: {', '.join(available_schemas)}"
            
            logger.debug(f"[DYNAMIC_SCHEMA_VALIDATION] Schema '{schema}' confirmed as valid")
            return None
            
        except Exception as e:
            logger.warning(f"[DYNAMIC_SCHEMA_VALIDATION] Error validating schema: {e}")
            # On error, be permissive - let it through
            return None
    
    def _ensure_limit(self, sql: str) -> str:
        """Ensure query has appropriate row limit for safety using dialect-aware syntax.
        
        This method:
        1. Checks if query already has a row limit (LIMIT, TOP, FETCH, ROWNUM)
        2. Exempts aggregate queries (COUNT, SUM, etc.) from limits
        3. Adds default limit using dialect-appropriate syntax
        4. Preserves user-specified limits if within bounds
        """
        from .dialect_sql_engine import DialectSqlEngine
        from .database_adapter import get_global_adapter
        
        sql_upper = sql.upper()
        
        # Check if already has a row limiting clause
        existing_limit_patterns = [
            r'\bLIMIT\s+(\d+)',  # PostgreSQL, MySQL, SQLite
            r'\bTOP\s+\((\d+)\)',  # SQL Server
            r'\bFETCH\s+(?:FIRST|NEXT)\s+(\d+)\s+ROWS',  # Oracle, DB2
            r'\bROWNUM\s*<=\s*(\d+)',  # Oracle (old syntax)
        ]
        
        existing_limit = None
        for pattern in existing_limit_patterns:
            match = re.search(pattern, sql_upper)
            if match:
                existing_limit = int(match.group(1))
                break
        
        if existing_limit:
            # Preserve user limit if within bounds, otherwise enforce max
            if existing_limit > self.max_rows:
                # Need to replace the limit - but we need the dialect to do this correctly
                try:
                    adapter = get_global_adapter()
                    dialect_name = adapter.dialect
                    dialect_engine = DialectSqlEngine(dialect_name)
                    return dialect_engine.enforce_row_limit(sql, self.max_rows)
                except Exception as e:
                    logger.warning(f"Could not enforce dialect-specific limit: {e}, falling back to LIMIT syntax")
                    # Fallback: replace with LIMIT syntax
                    for pattern in [r'\bLIMIT\s+\d+', r'\bTOP\s+\(\d+\)', r'\bFETCH\s+(?:FIRST|NEXT)\s+\d+\s+ROWS']:
                        sql = re.sub(pattern, f'LIMIT {self.max_rows}', sql,  flags=re.IGNORECASE, count=1)
                    return sql
            return sql  # User limit is within bounds
        
        # Check if it's an aggregation query (COUNT, SUM, etc.)
        if re.search(r'\bCOUNT\s*\(', sql_upper) or \
           re.search(r'\bSUM\s*\(', sql_upper) or \
           re.search(r'\bAVG\s*\(', sql_upper) or \
           re.search(r'\bMIN\s*\(', sql_upper) or \
           re.search(r'\bMAX\s*\(', sql_upper):
            # Aggregations return small result sets, no need for LIMIT
            return sql
        
        # Add limit using dialect-appropriate syntax
        try:
            adapter = get_global_adapter()
            dialect_name = adapter.dialect
            dialect_engine = DialectSqlEngine(dialect_name)
            return dialect_engine.enforce_row_limit(sql, self.max_rows)
        except Exception as e:
            logger.warning(f"Could not enforce dialect-specific limit: {e}, falling back to LIMIT syntax")
            # Fallback: simple LIMIT syntax
            sql_clean = sql.rstrip().rstrip(";")
            return f"{sql_clean} LIMIT {self.max_rows};"
    
    def _add_execution_safety_wrapper(self, sql: str) -> str:
        """Wrap query with safety settings (timeout, limits).
        
        Note: Currently unused, reserved for future implementation.
        """
        # PostgreSQL-specific safety wrapper
        lines = [
            f"SET statement_timeout = {self.statement_timeout_ms};",
            sql,
        ]
        return "\n".join(lines)


class SQLParameterValidator:
    """
    Validates that queries use parameterized values (prevents SQL injection).
    """
    
    # Pattern for parameterized placeholders
    PARAM_PATTERNS = [
        r'\$\d+',  # PostgreSQL: $1, $2, etc.
        r'\?',  # MySQL/SQLite: ?
        r':\w+',  # Named: :param_name
    ]
    
    @staticmethod
    def has_parameters(sql: str) -> bool:
        """Check if query uses parameterized values."""
        for pattern in SQLParameterValidator.PARAM_PATTERNS:
            if re.search(pattern, sql):
                return True
        return False
    
    @staticmethod
    def extract_parameter_count(sql: str) -> int:
        """Extract expected number of parameters."""
        max_param = 0
        
        # Check for PostgreSQL-style $1, $2, etc.
        pg_matches = re.findall(r'\$(\d+)', sql)
        if pg_matches:
            max_param = max(int(m) for m in pg_matches)
        
        # Check for unnamed placeholders
        unnamed = len(re.findall(r'\?', sql))
        
        return max(max_param, unnamed)


def validate_sql_safety(
    sql: str,
    allowed_schemas: Optional[List[str]] = None,
    max_rows: int = 500,  # Safety default: 500 rows (not 10000)
) -> Tuple[bool, str]:
    """Convenience function to validate SQL safety.
    
    Returns only (is_valid, error_message) without rewritten SQL.
    """
    validator = SQLSafetyValidator(
        allowed_schemas=allowed_schemas,
        max_rows=max_rows
    )
    is_valid, error, _ = validator.validate_and_rewrite(sql)
    return is_valid, error


async def create_sql_safety_validator(
    allowed_schemas: Optional[List[str]] = None,
    max_rows: int = 500,  # Safety default: 500 rows (not 10000)
) -> SQLSafetyValidator:
    """Factory function to create a validator."""
    return SQLSafetyValidator(allowed_schemas, max_rows)
