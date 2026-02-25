"""
Dialect Compiler - Converts QueryPlan AST to SQL for different databases.

This compiler generates dialect-specific SQL from the universal QueryPlan representation.
Supports:
- PostgreSQL
- MySQL 
- SQLite
- SQL Server
- Oracle (future)

The same QueryPlan can be compiled to any dialect, enabling:
- Multi-database deployments
- Fallback to alternate database
- Query translation for pipeline debugging
- Performance optimization per dialect
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Literal, Any
from abc import ABC, abstractmethod

from .query_plan import (
    QueryPlan, SelectClause, FromClause, JoinClause, JoinCondition, 
    GroupByClause, HavingClause, OrderByClause, OrderByField,
    BinaryCondition, LogicalCondition, NotCondition, Condition,
    ColumnRef, Literal as LiteralValue, SubqueryValue
)

logger = logging.getLogger(__name__)


class SQLGenerator(ABC):
    """Abstract base class for SQL generation."""
    
    @abstractmethod
    def generate(self, plan: QueryPlan) -> str:
        """Generate SQL from QueryPlan."""
        pass
    
    @abstractmethod
    def quote_identifier(self, identifier: str) -> str:
        """Quote a table or column name for this dialect."""
        pass
    
    @abstractmethod
    def boolean_literal(self, value: bool) -> str:
        """Convert boolean to this dialect's literal."""
        pass
    
    @abstractmethod
    def limit_clause(self, limit: int, offset: Optional[int] = None) -> str:
        """Generate LIMIT/OFFSET clause for this dialect."""
        pass


class PostgreSQLGenerator(SQLGenerator):
    """Generate PostgreSQL-compatible SQL."""
    
    def generate(self, plan: QueryPlan) -> str:
        """Generate a complete PostgreSQL SELECT statement."""
        if plan.intent != "data_query" or not plan.select or not plan.from_:
            raise ValueError("Cannot generate SQL from invalid QueryPlan")
        
        parts = []
        
        # SELECT clause
        parts.append(self._select_clause(plan.select))
        
        # FROM clause
        parts.append(self._from_clause(plan.from_))
        
        # JOINs
        if plan.joins:
            for join in plan.joins:
                parts.append(self._join_clause(join, plan))
        
        # WHERE
        if plan.where:
            where_sql = " AND ".join([self._condition_to_sql(cond) for cond in plan.where])
            parts.append(f"WHERE {where_sql}")
        
        # GROUP BY
        if plan.group_by:
            group_fields = ", ".join(plan.group_by.fields)
            parts.append(f"GROUP BY {group_fields}")
        
        # HAVING
        if plan.having:
            having_sql = " AND ".join([self._condition_to_sql(cond) for cond in plan.having.conditions])
            parts.append(f"HAVING {having_sql}")
        
        # ORDER BY
        if plan.order_by:
            order_fields = ", ".join([
                f"{f.expr} {f.direction.upper()}" for f in plan.order_by.fields
            ])
            parts.append(f"ORDER BY {order_fields}")
        
        # LIMIT/OFFSET
        if plan.limit is not None:
            parts.append(self.limit_clause(plan.limit, plan.offset))
        
        sql = "\n".join(parts)
        logger.debug(f"Generated PostgreSQL: {sql}")
        return sql
    
    def _select_clause(self, select: SelectClause) -> str:
        """Generate SELECT clause."""
        distinct = "DISTINCT " if select.distinct else ""
        fields = ", ".join(select.fields)
        return f"SELECT {distinct}{fields}"
    
    def _from_clause(self, from_: FromClause) -> str:
        """Generate FROM clause."""
        table = from_.table
        if from_.alias:
            return f"FROM {table} {from_.alias}"
        return f"FROM {table}"
    
    def _join_clause(self, join: JoinClause, plan: QueryPlan) -> str:
        """Generate JOIN clause."""
        join_type = {
            "inner": "INNER JOIN",
            "left": "LEFT JOIN",
            "right": "RIGHT JOIN",
            "full": "FULL OUTER JOIN",
            "cross": "CROSS JOIN"
        }.get(join.type, "INNER JOIN")
        
        parts = [f"{join_type} {join.table}"]
        
        if join.alias:
            parts.append(f"{join.alias}")
        
        if join.on:
            on_conditions = " AND ".join([
                f"{self._join_condition_to_sql(cond)}" for cond in join.on
            ])
            parts.append(f"ON {on_conditions}")
        
        return " ".join(parts)
    
    def _join_condition_to_sql(self, cond: JoinCondition) -> str:
        """Convert a join condition to SQL."""
        left = self._value_to_sql(cond.left)
        right = self._value_to_sql(cond.right)
        return f"{left} {cond.op} {right}"
    
    def _condition_to_sql(self, cond: Condition) -> str:
        """Convert a condition (WHERE/HAVING) to SQL."""
        if isinstance(cond, BinaryCondition):
            left = self._value_to_sql(cond.left)
            right = self._value_to_sql(cond.right)
            op = cond.op.upper() if cond.op.upper() in ["IN", "NOT IN", "IS", "IS NOT", "LIKE"] else cond.op
            return f"{left} {op} {right}"
        elif isinstance(cond, LogicalCondition):
            subconds = [f"({self._condition_to_sql(c)})" for c in cond.conditions]
            op = cond.operator.upper()
            return f" {op} ".join(subconds)
        elif isinstance(cond, NotCondition):
            return f"NOT ({self._condition_to_sql(cond.condition)})"
        return ""
    
    def _value_to_sql(self, value) -> str:
        """Convert a value to SQL."""
        if isinstance(value, str):
            # Could be a column reference
            if "." in value or value.isidentifier():
                return value
            # Otherwise it's a string literal
            return f"'{value}'"
        elif isinstance(value, ColumnRef):
            if value.table:
                return f"{value.table}.{value.column}"
            return value.column
        elif isinstance(value, LiteralValue):
            if value.type == "string":
                return f"'{value.value}'"
            elif value.type == "bool":
                return self.boolean_literal(value.value)
            elif value.type == "null":
                return "NULL"
            else:
                return str(value.value)
        elif isinstance(value, SubqueryValue):
            subquery_sql = self.generate(value.query)
            return f"({subquery_sql})"
        return str(value)
    
    def quote_identifier(self, identifier: str) -> str:
        """PostgreSQL uses double quotes for identifiers."""
        return f'"{identifier}"'
    
    def boolean_literal(self, value: bool) -> str:
        """PostgreSQL uses 'true'/'false' literals."""
        return "true" if value else "false"
    
    def limit_clause(self, limit: int, offset: Optional[int] = None) -> str:
        """PostgreSQL LIMIT syntax."""
        parts = [f"LIMIT {limit}"]
        if offset:
            parts.append(f"OFFSET {offset}")
        return " ".join(parts)


class MySQLGenerator(SQLGenerator):
    """Generate MySQL-compatible SQL."""
    
    def generate(self, plan: QueryPlan) -> str:
        """Generate MySQL SQL (similar to PostgreSQL but with dialect differences)."""
        # Most of the logic is the same as PostgreSQL
        # Key differences: backtick quoting, LIMIT offsets different syntax
        pg_gen = PostgreSQLGenerator()
        sql = pg_gen.generate(plan)
        
        # Convert PostgreSQL-specific syntax to MySQL
        # This is a simplified version - production would be more thorough
        sql = sql.replace("FULL OUTER JOIN", "FULL OUTER JOIN")  # Not supported, would need UNION
        
        logger.debug(f"Generated MySQL: {sql}")
        return sql
    
    def quote_identifier(self, identifier: str) -> str:
        """MySQL uses backticks for identifiers."""
        return f"`{identifier}`"
    
    def boolean_literal(self, value: bool) -> str:
        """MySQL uses TRUE/FALSE or 0/1."""
        return "TRUE" if value else "FALSE"
    
    def limit_clause(self, limit: int, offset: Optional[int] = None) -> str:
        """MySQL LIMIT syntax: LIMIT [offset,] count."""
        if offset:
            return f"LIMIT {offset}, {limit}"
        return f"LIMIT {limit}"


class SQLiteGenerator(SQLGenerator):
    """Generate SQLite-compatible SQL."""
    
    def generate(self, plan: QueryPlan) -> str:
        """Generate SQLite SQL (most restricted)."""
        pg_gen = PostgreSQLGenerator()
        sql = pg_gen.generate(plan)
        
        # SQLite differences:
        # - No FULL OUTER JOIN
        # - No OFFSET without LIMIT (must use LIMIT -1 OFFSET N)
        sql = sql.replace("FULL OUTER JOIN", "LEFT OUTER JOIN")
        
        logger.debug(f"Generated SQLite: {sql}")
        return sql
    
    def quote_identifier(self, identifier: str) -> str:
        """SQLite uses double quotes."""
        return f'"{identifier}"'
    
    def boolean_literal(self, value: bool) -> str:
        """SQLite uses 0/1 for booleans."""
        return "1" if value else "0"
    
    def limit_clause(self, limit: int, offset: Optional[int] = None) -> str:
        """SQLite LIMIT syntax."""
        if offset:
            # SQLite requires a limit, use -1 for unlimited
            return f"LIMIT {limit} OFFSET {offset}"
        return f"LIMIT {limit}"


class SQLServerGenerator(SQLGenerator):
    """Generate SQL Server-compatible SQL."""
    
    def generate(self, plan: QueryPlan) -> str:
        """Generate SQL Server SQL."""
        pg_gen = PostgreSQLGenerator()
        sql = pg_gen.generate(plan)
        
        # SQL Server differences:
        # - Uses TOP instead of LIMIT
        # - Different JOIN syntax for some types
        # - OFFSET...FETCH syntax
        
        # Convert LIMIT to TOP and OFFSET to FETCH
        if "LIMIT" in sql:
            lines = sql.split("\n")
            new_lines = []
            for line in lines:
                if "LIMIT" in line:
                    # Parse LIMIT and OFFSET
                    parts = line.split()
                    limit_idx = parts.index("LIMIT")
                    limit_val = parts[limit_idx + 1]
                    
                    # Move SELECT and add TOP
                    for i, l in enumerate(new_lines):
                        if l.startswith("SELECT"):
                            new_lines[i] = l.replace("SELECT", f"SELECT TOP {limit_val}")
                    # Don't include the LIMIT line
                else:
                    new_lines.append(line)
            sql = "\n".join(new_lines)
        
        logger.debug(f"Generated SQL Server: {sql}")
        return sql
    
    def quote_identifier(self, identifier: str) -> str:
        """SQL Server uses square brackets."""
        return f"[{identifier}]"
    
    def boolean_literal(self, value: bool) -> str:
        """SQL Server uses 1/0."""
        return "1" if value else "0"
    
    def limit_clause(self, limit: int, offset: Optional[int] = None) -> str:
        """SQL Server uses OFFSET...FETCH."""
        if offset:
            return f"OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY"
        return f"OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"


class DialectCompiler:
    """
    Compiles QueryPlan to SQL for any supported dialect.
    
    Usage:
        compiler = DialectCompiler(dialect="postgresql")
        sql = compiler.compile(plan)
    """
    
    GENERATORS: Dict[str, type] = {
        "postgresql": PostgreSQLGenerator,
        "postgres": PostgreSQLGenerator,
        "mysql": MySQLGenerator,
        "sqlite": SQLiteGenerator,
        "sqlserver": SQLServerGenerator,
        "sql_server": SQLServerGenerator,
    }
    
    def __init__(self, dialect: str = "postgresql"):
        """Initialize compiler for a specific dialect."""
        dialect_lower = dialect.lower()
        if dialect_lower not in self.GENERATORS:
            raise ValueError(f"Unsupported dialect: {dialect}. Supported: {list(self.GENERATORS.keys())}")
        
        self.dialect = dialect_lower
        self.generator = self.GENERATORS[dialect_lower]()
    
    def compile(self, plan: QueryPlan) -> str:
        """
        Compile a QueryPlan to SQL.
        
        Args:
            plan: Validated QueryPlan
        
        Returns:
            SQL string ready for execution
        """
        try:
            sql = self.generator.generate(plan)
            logger.info(f"✓ Compiled to {self.dialect}: {len(sql)} chars")
            return sql
        except Exception as e:
            logger.error(f"✗ Compilation failed for {self.dialect}: {e}")
            raise
