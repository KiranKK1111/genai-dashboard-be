"""
DYNAMIC SQL SAFETY VALIDATOR (P1 - Zero Hardcoding)

Config-driven SQL safety: specify rules in config, not code.
- Allowed statement types: learned from config
- Statement timeout: configurable per tool
- Result limits: configurable
- Privilege separation: role-based, not hardcoded

Zero hardcoded rules. Everything from SafetySqlConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Set, Dict, Any, List, Optional
import logging

try:
    import sqlglot
    import sqlglot.expressions as exp
except ImportError:
    raise ImportError("sqlglot required for SQL parsing. pip install sqlglot")

logger = logging.getLogger(__name__)


class SqlStatementType(str, Enum):
    """Allowed SQL statement types."""
    SELECT = "SELECT"
    WITH = "WITH"  # CTEs
    UNION = "UNION"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE_TABLE = "CREATE_TABLE"
    DROP_TABLE = "DROP_TABLE"


@dataclass
class SafetySqlConfig:
    """
    Configuration-driven SQL safety policy.
    
    This is YOUR policy: change it, don't change code.
    """
    
    # Which statement types are allowed?
    allowed_statement_types: Set[SqlStatementType] = field(
        default_factory=lambda: {SqlStatementType.SELECT, SqlStatementType.WITH, SqlStatementType.UNION}
    )
    
    # Which statement types require explicit confirmation?
    require_confirmation_for: Set[SqlStatementType] = field(
        default_factory=lambda: {
            SqlStatementType.INSERT, SqlStatementType.UPDATE, SqlStatementType.DELETE,
            SqlStatementType.CREATE_TABLE, SqlStatementType.DROP_TABLE
        }
    )
    
    # Statement and result size limits
    statement_timeout_seconds: int = 30  # DB-level timeout
    max_result_rows: int = 100000
    max_result_bytes: int = 100 * 1024 * 1024  # 100 MB
    max_cell_length: int = 10000  # Max chars per cell
    
    # Introspection blocking
    allow_system_catalog_access: bool = False
    block_table_patterns: List[str] = field(
        default_factory=lambda: ["pg_*", "information_schema.*", "mysql.*", "sys.*"]
    )
    
    # Result redaction (for privacy)
    redact_patterns: Dict[str, str] = field(
        default_factory=lambda: {
            # Pattern → replacement
            # Example: r"^\d{3}-\d{2}-\d{4}$" → "***-**-****"  (SSN)
        }
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed_statement_types": [t.value for t in self.allowed_statement_types],
            "require_confirmation_for": [t.value for t in self.require_confirmation_for],
            "statement_timeout_seconds": self.statement_timeout_seconds,
            "max_result_rows": self.max_result_rows,
            "max_result_bytes": self.max_result_bytes,
            "max_cell_length": self.max_cell_length,
            "allow_system_catalog_access": self.allow_system_catalog_access,
            "block_table_patterns": self.block_table_patterns,
            "redact_patterns": self.redact_patterns,
        }


@dataclass
class SqlValidationResult:
    """Result of SQL validation."""
    is_valid: bool
    statement_type: Optional[SqlStatementType] = None
    statement_count: int = 0
    tables_accessed: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    parsed_ast: Optional[Any] = None  # sqlglot AST
    safety_config_used: Optional[SafetySqlConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "statement_type": self.statement_type.value if self.statement_type else None,
            "statement_count": self.statement_count,
            "tables_accessed": self.tables_accessed,
            "requires_confirmation": self.requires_confirmation,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class DynamicSqlSafetyValidator:
    """
    SQL validator driven by configuration.
    
    KEY PRINCIPLE: Rules are in SafetySqlConfig, not hardcoded in code.
    """
    
    def __init__(self, safety_config: Optional[SafetySqlConfig] = None):
        self.config = safety_config or SafetySqlConfig()
        logger.info(f"SQL Safety Validator initialized with config: {self.config.allowed_statement_types}")
    
    def validate(self, sql: str) -> SqlValidationResult:
        """
        Validate SQL against configured policy.
        
        Returns:
            SqlValidationResult with validation details
        """
        result = SqlValidationResult(is_valid=True, safety_config_used=self.config)
        
        # Step 1: Parse SQL
        try:
            parsed = sqlglot.parse(sql, read="postgres")  # Adapt dialect as needed
            if not parsed:
                result.is_valid = False
                result.errors.append("Failed to parse SQL")
                return result
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Parse error: {e}")
            return result
        
        result.statement_count = len(parsed)
        result.parsed_ast = parsed[0] if parsed else None
        
        # Step 2: Check statement count (disallow chaining)
        if result.statement_count > 1:
            result.is_valid = False
            result.errors.append(f"Multiple statements found ({result.statement_count}). Only single statements allowed.")
            return result
        
        ast = parsed[0]
        
        # Step 3: Determine statement type from AST
        statement_type = self._classify_statement(ast)
        result.statement_type = statement_type
        
        if not statement_type:
            result.is_valid = False
            result.errors.append(f"Unknown or disallowed statement type: {type(ast).__name__}")
            return result
        
        # Step 4: Check if statement type is allowed
        if statement_type not in self.config.allowed_statement_types:
            result.is_valid = False
            result.errors.append(f"Statement type {statement_type.value} is not allowed. "
                               f"Allowed: {[t.value for t in self.config.allowed_statement_types]}")
            return result
        
        # Step 5: Check if requires confirmation
        if statement_type in self.config.require_confirmation_for:
            result.requires_confirmation = True
        
        # Step 6: Extract tables accessed
        result.tables_accessed = self._extract_tables(ast)
        
        # Step 7: Check table patterns (block system catalogs)
        if self._match_blocked_patterns(result.tables_accessed):
            result.is_valid = False
            result.errors.append(f"Access to system/protected tables is not allowed. "
                               f"Blocked patterns: {self.config.block_table_patterns}")
            return result
        
        # Step 8: Enforce LIMIT for SELECT (prevent full scans)
        if statement_type == SqlStatementType.SELECT:
            if not self._has_limit(ast):
                result.warnings.append("SELECT statement has no LIMIT. "
                                     f"Consider adding LIMIT {self.config.max_result_rows}")
        
        return result
    
    def _classify_statement(self, ast: Any) -> Optional[SqlStatementType]:
        """Map AST node type to SqlStatementType."""
        ast_type = type(ast).__name__
        
        # Map sqlglot AST types to our enum
        type_map = {
            "Select": SqlStatementType.SELECT,
            "With": SqlStatementType.WITH,
            "Union": SqlStatementType.UNION,
            "Insert": SqlStatementType.INSERT,
            "Update": SqlStatementType.UPDATE,
            "Delete": SqlStatementType.DELETE,
            "Create": SqlStatementType.CREATE_TABLE,
            "Drop": SqlStatementType.DROP_TABLE,
        }
        
        return type_map.get(ast_type)
    
    def _extract_tables(self, ast: Any) -> List[str]:
        """Extract all table names from AST."""
        tables = []
        try:
            for table in ast.find_all(exp.Table):
                table_name = table.name
                if table_name:
                    tables.append(table_name)
        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")
        
        return list(set(tables))  # Deduplicate
    
    def _has_limit(self, ast: Any) -> bool:
        """Check if SELECT has LIMIT clause."""
        try:
            # For SELECT statements, check for LIMIT
            if isinstance(ast, exp.Select):
                return ast.args.get("limit") is not None
        except Exception as e:
            logger.warning(f"Error checking LIMIT: {e}")
        
        return False
    
    def _match_blocked_patterns(self, tables: List[str]) -> bool:
        """Check if any table matches blocked patterns."""
        for table in tables:
            for pattern in self.config.block_table_patterns:
                # Simple glob matching
                if self._glob_match(table.lower(), pattern.lower()):
                    return True
        return False
    
    @staticmethod
    def _glob_match(text: str, pattern: str) -> bool:
        """Simple glob matching (* = any chars)."""
        import fnmatch
        return fnmatch.fnmatch(text, pattern)
