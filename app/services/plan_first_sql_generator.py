"""
Plan-First SQL Generator - Stage 3 & 4 of Query Understanding Pipeline

Converts semantic concepts into structured query plans, then renders SQL.
This implements the critical missing piece: structured plan generation BEFORE SQL.

Architecture:
Semantic Intent → Grounded Plan → SQL Rendering

This fixes the core issue where the system was generating SQL directly without
a structured understanding of what should be included.

Example flow:
1. Semantic Intent: {"intent": "count", "filters": [{"concept": "birth_month", "value": 1}]}
2. Grounded Plan: {"table": "customers", "where": [{"column": "dob", "operator": "MONTH_EQUALS", "value": 1}]}
3. Rendered SQL: "SELECT COUNT(*) FROM customers WHERE EXTRACT(MONTH FROM dob) = 1"

This ensures ALL semantic concepts get converted to SQL constraints.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from sqlalchemy.ext.asyncio import AsyncSession

from .semantic_concept_extractor import SemanticIntent, FilterConcept, OperatorType, IntentType

logger = logging.getLogger(__name__)


@dataclass
class ColumnMapping:
    """Maps semantic concepts to actual database columns"""
    concept: str           # Semantic concept (e.g., 'birth_month')
    table: str            # Database table
    column: str           # Database column 
    data_type: str        # Column data type
    confidence: float = 1.0


@dataclass
class QueryPlan:
    """Structured query plan before SQL generation"""
    intent: IntentType
    primary_table: str
    select_clauses: List[Dict[str, Any]] = field(default_factory=list) 
    where_conditions: List[Dict[str, Any]] = field(default_factory=list)
    joins: List[Dict[str, Any]] = field(default_factory=list)
    group_by: Optional[List[str]] = None
    order_by: Optional[List[Dict[str, str]]] = None
    limit: Optional[int] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for debugging/logging"""
        return {
            "intent": self.intent.value,
            "primary_table": self.primary_table,
            "select_clauses": self.select_clauses,
            "where_conditions": self.where_conditions,
            "joins": self.joins,
            "group_by": self.group_by,
            "order_by": self.order_by,
            "limit": self.limit,
            "confidence": self.confidence
        }


class SemanticGrounder:
    """
    Maps semantic concepts to actual database schema.
    Stage 2 of the pipeline - grounding abstract concepts to concrete columns.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        
        # Semantic concept mappings (schema-aware, not configuration dependent)
        self.concept_mappings = {
            # Temporal mappings
            "birth_month": ["dob", "birth_date", "birthdate", "date_of_birth"],
            "birth_year": ["dob", "birth_date", "birthdate", "date_of_birth"],
            
            # Identity mappings
            "gender": ["gender", "sex"],
            
            # Business entity mappings (semantic)
            "customers": ["customers", "clients", "customer"],
            "accounts": ["accounts", "account"],
            "transactions": ["transactions", "transaction", "payments"],
            "employees": ["employees", "employee", "staff"]
        }
        
        logger.info(f"[SEMANTIC_GROUNDER] Loaded {len(self.concept_mappings)} concept mappings")
    
    async def ground_semantic_intent(self, intent: SemanticIntent) -> Tuple[QueryPlan, List[ColumnMapping]]:
        """
        Ground semantic intent to concrete query plan.
        
        This is the CRITICAL step that maps:
        - "clients" → "customers" table
        - "birth_month" → "dob" column with MONTH extraction
        - "gender" → "gender" column
        """
        # Find primary table
        primary_table = await self._find_primary_table(intent.entity)
        if not primary_table:
            raise ValueError(f"Could not find table for entity: {intent.entity}")
        
        # Ground all filter concepts to columns
        mappings = []
        where_conditions = []
        
        for filter_concept in intent.filters:
            column_mapping = await self._ground_filter_concept(filter_concept, primary_table)
            if column_mapping:
                mappings.append(column_mapping)
                
                # Convert to WHERE condition
                where_condition = self._create_where_condition(filter_concept, column_mapping)
                if where_condition:
                    where_conditions.append(where_condition)
        
        # Create SELECT clauses based on intent
        select_clauses = self._create_select_clauses(intent, primary_table)
        
        plan = QueryPlan(
            intent=intent.intent,
            primary_table=primary_table,
            select_clauses=select_clauses,
            where_conditions=where_conditions,
            limit=1000  # Default safety limit
        )
        
        logger.info(f"[SEMANTIC_GROUNDER] Grounded plan: {len(where_conditions)} filters, table={primary_table}")
        return plan, mappings
    
    async def _find_primary_table(self, entity: Optional[str]) -> Optional[str]:
        """Find the primary table for a semantic entity"""
        if not entity:
            return None
            
        # Get all table names from database
        from sqlalchemy import text
        result = await self.db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'genai'
            ORDER BY table_name
        """))
        tables = [row[0] for row in result]
        
        # Map entity to table using concept mappings
        if entity in self.concept_mappings:
            candidates = self.concept_mappings[entity]
            for candidate in candidates:
                if candidate in tables:
                    return candidate
        
        # Direct match
        if entity in tables:
            return entity
            
        # Fuzzy match (customers ≈ customer)
        for table in tables:
            if entity.rstrip('s') == table.rstrip('s'):
                return table
                
        return None
    
    async def _ground_filter_concept(self, filter_concept: FilterConcept, table: str) -> Optional[ColumnMapping]:
        """Ground a semantic filter concept to an actual column"""
        
        # Get column info for table
        from sqlalchemy import text
        result = await self.db.execute(text(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'genai' AND table_name = '{table}'
            ORDER BY column_name
        """))
        columns = {row[0]: row[1] for row in result}
        
        # Map concept to column using concept mappings
        if filter_concept.concept in self.concept_mappings:
            candidates = self.concept_mappings[filter_concept.concept]
            for candidate in candidates:
                if candidate in columns:
                    return ColumnMapping(
                        concept=filter_concept.concept,
                        table=table,
                        column=candidate,
                        data_type=columns[candidate]
                    )
        
        # Direct column name match
        if filter_concept.concept in columns:
            return ColumnMapping(
                concept=filter_concept.concept,
                table=table, 
                column=filter_concept.concept,
                data_type=columns[filter_concept.concept]
            )
            
        return None
    
    def _create_where_condition(self, filter_concept: FilterConcept, mapping: ColumnMapping) -> Optional[Dict[str, Any]]:
        """Create WHERE condition from semantic filter and column mapping"""
        
        if filter_concept.operator == OperatorType.EQUALS:
            return {
                "column": mapping.column,
                "operator": "=",
                "value": filter_concept.value,
                "type": "simple"
            }
        elif filter_concept.operator == OperatorType.IN:
            return {
                "column": mapping.column, 
                "operator": "IN",
                "values": filter_concept.values,
                "type": "simple"
            }
        elif filter_concept.operator == OperatorType.MONTH_EQUALS:
            # THE KEY FIX: Handle temporal month extraction
            return {
                "column": mapping.column,
                "operator": "MONTH_EQUALS", 
                "value": filter_concept.value,
                "type": "temporal",
                "expression": f"EXTRACT(MONTH FROM {mapping.column}) = {filter_concept.value}"
            }
        
        return None
    
    def _create_select_clauses(self, intent: SemanticIntent, table: str) -> List[Dict[str, Any]]:
        """Create SELECT clauses based on intent"""
        if intent.intent == IntentType.COUNT:
            return [{"type": "aggregate", "function": "COUNT", "column": "*"}]
        elif intent.intent == IntentType.LIST:
            return [{"type": "wildcard", "column": "*"}]
        else:
            return [{"type": "wildcard", "column": "*"}]


class PlanFirstSQLGenerator:
    """
    Generates SQL from structured query plans.
    Stage 4 - deterministic SQL rendering from plans.
    """
    
    def render_sql_from_plan(self, plan: QueryPlan) -> str:
        """
        Render SQL from structured query plan.
        
        This is deterministic - same plan always produces same SQL.
        No LLM involved in this step, purely rule-based.
        """
        # Build SELECT clause
        select_parts = []
        for clause in plan.select_clauses:
            if clause["type"] == "aggregate":
                select_parts.append(f"{clause['function']}({clause['column']})")
            elif clause["type"] == "wildcard":
                select_parts.append("*")
            else:
                select_parts.append(clause["column"])
        
        select_clause = "SELECT " + ", ".join(select_parts)
        
        # Build FROM clause
        from_clause = f"FROM genai.{plan.primary_table}"
        
        # Build WHERE clause - THE CRITICAL PART
        where_parts = []
        for condition in plan.where_conditions:
            if condition["type"] == "temporal":
                # Use the pre-built expression for temporal logic
                where_parts.append(condition["expression"])
            elif condition["operator"] == "=":
                where_parts.append(f"{condition['column']} = '{condition['value']}'")
            elif condition["operator"] == "IN":
                values_str = ", ".join([f"'{v}'" for v in condition["values"]])
                where_parts.append(f"{condition['column']} IN ({values_str})")
        
        where_clause = ""
        if where_parts:
            where_clause = "WHERE " + " AND ".join(where_parts)
        
        # Build complete SQL
        sql_parts = [select_clause, from_clause]
        if where_clause:
            sql_parts.append(where_clause)
            
        # Add safety limit
        if plan.limit:
            sql_parts.append(f"LIMIT {plan.limit}")
        
        sql = "\n".join(sql_parts)
        
        logger.info(f"[PLAN_SQL_GENERATOR] Rendered SQL from plan: {len(plan.where_conditions)} conditions")
        return sql


class PlanFirstQueryHandler:
    """
    Complete plan-first query handler.
    Orchestrates the full pipeline: SemanticIntent → GroundedPlan → SQL
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.grounder = SemanticGrounder(db)
        self.sql_generator = PlanFirstSQLGenerator()
    
    async def handle_semantic_intent(self, intent: SemanticIntent) -> Tuple[str, Dict[str, Any]]:
        """
        Handle semantic intent end-to-end: ground to plan, render SQL.
        
        Returns:
            Tuple of (generated_sql, debug_info)
        """
        try:
            # Ground semantic intent to query plan
            plan, mappings = await self.grounder.ground_semantic_intent(intent)
            
            # Render SQL from plan
            sql = self.sql_generator.render_sql_from_plan(plan)
            
            debug_info = {
                "semantic_intent": intent.to_dict(),
                "query_plan": plan.to_dict(),
                "column_mappings": [
                    {"concept": m.concept, "table": m.table, "column": m.column}
                    for m in mappings
                ],
                "pipeline_stage": "plan_first_generation"
            }
            
            logger.info(f"[PLAN_FIRST_HANDLER] Generated SQL with {len(mappings)} concept mappings")
            return sql, debug_info
            
        except Exception as e:
            logger.error(f"[PLAN_FIRST_HANDLER] Error in plan-first generation: {e}")
            raise


async def get_plan_first_handler(db: AsyncSession) -> PlanFirstQueryHandler:
    """Get plan-first query handler instance"""
    return PlanFirstQueryHandler(db)