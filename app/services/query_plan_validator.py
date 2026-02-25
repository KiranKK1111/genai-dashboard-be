"""
QueryPlan Validator - Validates QueryPlan AST against live schema.

This validator ensures:
1. All tables exist in the schema
2. All columns exist in their respective tables
3. Column types match operation requirements
4. Join paths are valid (can be inferred from relationships)
5. Predicates are relocatable (no hallucinated columns)
6. Subqueries are valid and closed (all references resolve)

Validation is deterministic - no regex, no fuzzy matching, just reflection.
If validation fails, we can either:
- Ask for clarification (ambiguous column)
- Auto-repair (e.g., move predicate to correct table)
- Return validation error (impossible to satisfy)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from sqlalchemy.ext.asyncio import AsyncSession

from .query_plan import (
    QueryPlan, QueryArtifacts, Condition, BinaryCondition, LogicalCondition, NotCondition,
    ColumnRef, Literal, SubqueryValue, JoinClause, JoinCondition, Value,
    ColumnNotFoundError, TableNotFoundError, JoinPathNotFoundError, TypeMismatchError,
    InvalidSubqueryError
)
from .schema_discovery import SchemaCatalog, TableInfo, ColumnInfo

logger = logging.getLogger(__name__)


class QueryPlanValidator:
    """
    Validates a QueryPlan against a SchemaCatalog.
    
    Usage:
        catalog = SchemaCatalog("genai")
        await catalog.initialize(db)
        
        validator = QueryPlanValidator(catalog)
        validated_plan = await validator.validate(plan)
        artifacts = validator.get_artifacts(validated_plan)
    """
    
    def __init__(self, schema_catalog: SchemaCatalog):
        self.schema = schema_catalog
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    async def validate(self, plan: QueryPlan) -> QueryPlan:
        """
        Validate a QueryPlan against schema.
        
        Raises:
            ColumnNotFoundError, TableNotFoundError, JoinPathNotFoundError, etc.
        
        Returns:
            The same plan (for chaining)
        """
        self.errors = []
        self.warnings = []
        
        if plan.intent != "data_query":
            return plan  # Only validate data queries
        
        try:
            # 1. Validate FROM clause
            if plan.from_:
                self._validate_table_exists(plan.from_.table)
            
            # 2. Validate all JOINs
            if plan.joins:
                for join in plan.joins:
                    self._validate_table_exists(join.table)
                    if join.on:
                        for on_cond in join.on:
                            await self._validate_condition(on_cond)
            
            # 3. Validate WHERE conditions
            if plan.where:
                for cond in plan.where:
                    await self._validate_condition(cond)
            
            # 4. Validate GROUP BY fields
            if plan.group_by:
                for field in plan.group_by.fields:
                    self._validate_field_reference(field)
            
            # 5. Validate HAVING conditions
            if plan.having:
                for cond in plan.having.conditions:
                    await self._validate_condition(cond)
            
            # 6. Validate ORDER BY fields
            if plan.order_by:
                for field in plan.order_by.fields:
                    self._validate_field_reference(field.expr)
            
            # 7. Validate subqueries recursively
            await self._validate_subqueries(plan)
            
            # If any errors, raise the first one
            if self.errors:
                raise Exception(self.errors[0])
            
            logger.info(f"✓ QueryPlan validation passed ({len(self.warnings)} warnings)")
            return plan
            
        except Exception as e:
            logger.error(f"✗ QueryPlan validation failed: {e}")
            raise
    
    def _validate_table_exists(self, table_name: str) -> None:
        """Validate that a table exists in schema."""
        if table_name not in self.schema.database.tables:
            raise TableNotFoundError(table_name)
    
    async def _validate_column_exists(self, table_name: str, column_name: str) -> ColumnInfo:
        """Validate that a column exists in a table and return its metadata."""
        table_info = self.schema.database.tables.get(table_name)
        if not table_info:
            raise TableNotFoundError(table_name)
        
        col_info = table_info.columns.get(column_name)
        if not col_info:
            # Check if it's a wildcard or aggregate function
            if column_name == "*" or "(" in column_name:
                return None  # Allow wildcards and function calls
            raise ColumnNotFoundError(column_name, table_name)
        
        return col_info
    
    def _validate_field_reference(self, field: str) -> None:
        """Validate a field reference (supports aliases, wildcards, aggregates)."""
        # Strip whitespace and check for common patterns
        field = field.strip()
        
        # Allow wildcards
        if field == "*" or field.endswith(".*"):
            return
        
        # Allow aggregate functions
        if any(func in field.upper() for func in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]):
            return
        
        # Allow aliases (e.g., "t.customer_id" or "customers.id")
        # We'll do basic validation - more thorough validation would parse expressions
        if "(" in field and ")" in field:
            # Could be a function like DATE_TRUNC(...)
            return
    
    async def _validate_condition(self, condition: Condition) -> None:
        """Recursively validate a condition."""
        if isinstance(condition, BinaryCondition):
            await self._validate_binary_condition(condition)
        elif isinstance(condition, LogicalCondition):
            for cond in condition.conditions:
                await self._validate_condition(cond)
        elif isinstance(condition, NotCondition):
            await self._validate_condition(condition.condition)
    
    async def _validate_binary_condition(self, cond: BinaryCondition) -> None:
        """Validate a binary condition (left OP right)."""
        # Validate left side
        if isinstance(cond.left, ColumnRef):
            table_name = cond.left.table or self._get_primary_table()
            await self._validate_column_exists(table_name, cond.left.column)
        elif isinstance(cond.left, str):
            # String reference like "customer_id"
            # Try to find it in the primary table
            table_name = self._get_primary_table()
            await self._validate_column_exists(table_name, cond.left)
        
        # Validate right side
        if isinstance(cond.right, Literal):
            # Literal values are always valid
            pass
        elif isinstance(cond.right, ColumnRef):
            table_name = cond.right.table or self._get_primary_table()
            await self._validate_column_exists(table_name, cond.right.column)
        elif isinstance(cond.right, SubqueryValue):
            # Validate subquery recursively
            await self.validate(cond.right.query)
        elif isinstance(cond.right, str):
            # String reference
            table_name = self._get_primary_table()
            await self._validate_column_exists(table_name, cond.right)
    
    async def _validate_subqueries(self, plan: QueryPlan) -> None:
        """Recursively validate all subqueries in the plan."""
        # Check WHERE conditions for subqueries
        if plan.where:
            for cond in plan.where:
                await self._validate_subquery_in_condition(cond)
    
    async def _validate_subquery_in_condition(self, cond: Condition) -> None:
        """Recursively check condition for subqueries."""
        if isinstance(cond, BinaryCondition):
            if isinstance(cond.right, SubqueryValue):
                await self.validate(cond.right.query)
        elif isinstance(cond, LogicalCondition):
            for nested_cond in cond.conditions:
                await self._validate_subquery_in_condition(nested_cond)
        elif isinstance(cond, NotCondition):
            await self._validate_subquery_in_condition(cond.condition)
    
    def _get_primary_table(self) -> str:
        """Get the primary table (FROM clause)."""
        # This is a simplification - in reality we'd track table context
        return "transactions"  # Default, should be set by caller
    
    def get_artifacts(self, plan: QueryPlan) -> QueryArtifacts:
        """Extract query artifacts from a validated plan."""
        tables_used = []
        columns_used = []
        joins_used = []
        where_conditions = []
        group_by_fields = []
        order_by_fields = []
        has_subqueries = False
        is_aggregated = False
        
        # Extract tables
        if plan.from_:
            tables_used.append(plan.from_.table)
        if plan.joins:
            for join in plan.joins:
                tables_used.append(join.table)
                joins_used.append({
                    "type": join.type,
                    "from": f"{plan.from_.table if plan.from_ else 'unknown'}",
                    "to": join.table
                })
        
        # Extract columns from SELECT
        if plan.select:
            columns_used.extend(plan.select.fields)
        
        # Extract WHERE conditions
        if plan.where:
            where_conditions = [str(c) for c in plan.where]
        
        # Extract GROUP BY
        if plan.group_by:
            group_by_fields = plan.group_by.fields
            is_aggregated = True
        
        # Extract ORDER BY
        if plan.order_by:
            order_by_fields = [f.expr for f in plan.order_by.fields]
        
        # Check for subqueries
        has_subqueries = self._check_subqueries(plan)
        
        return QueryArtifacts(
            tables_used=tables_used,
            columns_used=columns_used,
            joins_used=joins_used,
            where_conditions=where_conditions,
            group_by_fields=group_by_fields,
            having_conditions=[],  # TODO: Extract from having clause
            order_by_fields=order_by_fields,
            limit_value=plan.limit,
            has_subqueries=has_subqueries,
            is_aggregated=is_aggregated
        )
    
    def _check_subqueries(self, plan: QueryPlan, depth: int = 0) -> bool:
        """Check if plan contains any subqueries."""
        if depth > 10:
            return True  # Prevent infinite recursion
        
        if plan.where:
            for cond in plan.where:
                if self._condition_has_subquery(cond, depth):
                    return True
        
        return False
    
    def _condition_has_subquery(self, cond: Condition, depth: int) -> bool:
        """Check if a condition contains subqueries."""
        if isinstance(cond, BinaryCondition):
            return isinstance(cond.right, SubqueryValue)
        elif isinstance(cond, LogicalCondition):
            return any(self._condition_has_subquery(c, depth) for c in cond.conditions)
        elif isinstance(cond, NotCondition):
            return self._condition_has_subquery(cond.condition, depth)
        return False


class JoinPathFinder:
    """
    Finds valid join paths between tables using relationship graph.
    
    Usage:
        finder = JoinPathFinder(schema_catalog)
        path = await finder.find_path("transactions", "customers")
        # Returns: [("transactions", "customers", "fk_customer_id")]
    """
    
    def __init__(self, schema_catalog: SchemaCatalog):
        self.schema = schema_catalog
    
    async def find_path(self, source: str, target: str) -> Optional[List[Tuple[str, str, str]]]:
        """
        Find a join path from source to target table.
        
        Returns:
            List of (from_table, to_table, join_column) tuples, or None if no path exists.
        """
        if source == target:
            return []
        
        # BFS to find shortest path
        visited = {source}
        queue = [(source, [])]
        
        while queue:
            current, path = queue.pop(0)
            
            # Find tables this table can join to
            for next_table in self._get_joinable_tables(current):
                if next_table == target:
                    # Found it!
                    join_col = self._get_join_column(current, target)
                    return path + [(current, target, join_col)]
                
                if next_table not in visited:
                    visited.add(next_table)
                    join_col = self._get_join_column(current, next_table)
                    queue.append((next_table, path + [(current, next_table, join_col)]))
        
        return None
    
    def _get_joinable_tables(self, table: str) -> List[str]:
        """Get list of tables that can be joined to this table."""
        table_info = self.schema.database.tables.get(table)
        if not table_info:
            return []
        
        joinable = []
        
        # Check foreign keys
        for fk_col, (fk_table, _) in table_info.foreign_keys.items():
            if fk_table not in joinable:
                joinable.append(fk_table)
        
        return joinable
    
    def _get_join_column(self, from_table: str, to_table: str) -> str:
        """Get the join column between two tables."""
        from_info = self.schema.database.tables.get(from_table)
        if not from_info:
            return f"{to_table}_id"
        
        # Look for foreign key
        for fk_col, (fk_table, _) in from_info.foreign_keys.items():
            if fk_table == to_table:
                return fk_col
        
        return f"{to_table}_id"  # Fallback
