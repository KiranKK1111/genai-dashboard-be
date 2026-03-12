"""app.services.sql — SQL generation, safety validation and dialect support."""

from ..advanced_sql_generator import AdvancedSQLGenerator
from ..dynamic_sql_safety import DynamicSqlSafetyValidator, SafetySqlConfig, SqlStatementType
from ..sql_safety_validator import SQLSafetyValidator
from ..plan_validator import PlanValidator
from ..dialect_adapter import DialectAdapter
from ..database_adapter import DatabaseAdapter

# Correct names from actual modules
from ..sql_generator import generate_sql                  # module exposes generate_sql(), not a class
from ..auto_retry_logic import AutoRetryExecutor
from ..dialect_sql_engine import DialectSqlEngine

__all__ = [
    "AdvancedSQLGenerator", "DynamicSqlSafetyValidator", "SafetySqlConfig",
    "SqlStatementType", "SQLSafetyValidator", "PlanValidator", "DialectAdapter",
    "DatabaseAdapter", "generate_sql", "AutoRetryExecutor", "DialectSqlEngine",
]
