"""
Intelligent Query Orchestrator
===============================

This module orchestrates the complete query understanding and SQL generation pipeline:

1. SEMANTIC ANALYSIS: UniversalQueryAnalyzer understands query intent
2. VALUE DISCOVERY: IntelligentFollowupValueMapper finds value->column mappings
3. SQL GENERATION: AdvancedSQLGenerator creates SQL for any pattern
4. EXECUTION: Runs SQL and returns results

ARCHITECTURE:
    User Prompt → Semantic Analysis → Value Discovery → SQL Generation → Execution
    
KEY PRINCIPLES:
- ZERO HARDCODING: All behavior driven by schema, LLM, and semantic analysis
- UNIVERSAL: Handles ANY query for ANY table with ANY pattern
- INTELLIGENT: LLM-guided semantic understanding
- DYNAMIC: Discovers tables, columns, values, patterns at runtime
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
import sqlalchemy as sa

from .universal_query_analyzer import UniversalQueryAnalyzer, SemanticAnalysis
from .intelligent_followup_value_mapper import IntelligentFollowupValueMapper, FollowUpContext
from .advanced_sql_generator import AdvancedSQLGenerator, SQLGenerationContext
from .database_adapter import DatabaseAdapter

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Result from complete orchestration pipeline."""
    user_prompt: str
    semantic_analysis: SemanticAnalysis
    value_mappings: Dict[str, Any]
    generated_sql: str
    sql_patterns: List[str]
    sql_complexity: str
    query_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    reasoning: str = ""


class IntelligentQueryOrchestrator:
    """
    Orchestrates complete query flow from prompt to results.
    
    Pipeline:
    1. ANALYZE: Semantic analysis of user prompt
    2. DISCOVER: Discover values->columns mappings
    3. GENERATE: Create SQL based on analysis
    4. EXECUTE: Run SQL and return results
    """
    
    def __init__(self, db_adapter: DatabaseAdapter):
        """Initialize orchestrator with database adapter."""
        self.db_adapter = db_adapter
        self.analyzer = UniversalQueryAnalyzer()
        self.mapper = IntelligentFollowupValueMapper()
        self.generator = AdvancedSQLGenerator()
        
        logger.info("[ORCHESTRATOR] Initialized with semantic analysis pipeline")
    
    async def orchestrate_query(
        self,
        user_prompt: str,
        db: AsyncSession,
        previous_query_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[str]] = None
    ) -> OrchestrationResult:
        """
        Execute complete query orchestration pipeline.
        
        Steps:
        1. Semantic analysis (understand intent, discover tables)
        2. Value discovery (map user values to columns)
        3. SQL generation (create appropriate SQL)
        4. Execution (run and return results)
        
        Args:
            user_prompt: User's natural language query
            db: Async database session
            previous_query_context: Previous query details for follow-ups
            conversation_history: Previous queries in conversation
            
        Returns:
            OrchestrationResult with complete analysis, SQL, and results
        """
        
        result = OrchestrationResult(
            user_prompt=user_prompt,
            semantic_analysis=None,
            value_mappings={},
            generated_sql="",
            sql_patterns=[],
            sql_complexity="",
            reasoning="Starting intelligent query orchestration..."
        )
        
        try:
            import time
            start_time = time.time()
            
            # STEP 1: SEMANTIC ANALYSIS
            logger.info("[ORCH-STEP1] 🔍 SEMANTIC ANALYSIS")
            semantic_analysis = await self._analyze_semantics(
                user_prompt, db, previous_query_context, conversation_history
            )
            result.semantic_analysis = semantic_analysis
            
            # Check if this is a conversational query (not database-related)
            from .universal_query_analyzer import QueryType
            if semantic_analysis.query_type == QueryType.UNRELATED:
                logger.info("[ORCH-STEP1] ✅ Conversational query detected - no database access needed")
                result.reasoning += f"\n✅ Conversational query (no database tables needed)"
                result.error = None  # Not an error, just conversational
                return result  # Return early - this should be handled by conversational handler
            
            # VALIDATION: Check if any tables were identified
            if not semantic_analysis.relevant_tables:
                error_msg = f"Could not identify any relevant tables for query: '{user_prompt}'"
                logger.error(f"[ORCH-STEP1] ❌ {error_msg}")
                result.reasoning += f"\n❌ No relevant tables found"
                result.error = error_msg
                return result  # Return early instead of generating "FROM unknown" SQL
            
            result.reasoning += f"\n✅ Identified {len(semantic_analysis.relevant_tables)} relevant tables"
            logger.info(f"[ORCH-STEP1] ✅ Found tables: {[t.table_name for t in semantic_analysis.relevant_tables]}")
            
            # STEP 2: VALUE DISCOVERY
            logger.info("[ORCH-STEP2] 🎯 VALUE DISCOVERY")
            value_mappings = await self._discover_values(
                semantic_analysis, user_prompt, db, previous_query_context
            )
            result.value_mappings = value_mappings
            result.reasoning += f"\n✅ Discovered {len(value_mappings)} value->column mappings"
            logger.info(f"[ORCH-STEP2] ✅ Mappings: {list(value_mappings.keys())}")
            
            # STEP 3: SQL GENERATION
            logger.info("[ORCH-STEP3] ⚙️ SQL GENERATION")
            sql_query, sql_metadata = await self._generate_sql(
                semantic_analysis, value_mappings, db, user_prompt
            )
            result.generated_sql = sql_query
            result.sql_patterns = sql_metadata.get('patterns', [])
            result.sql_complexity = sql_metadata.get('complexity', 'unknown')
            result.reasoning += f"\n✅ Generated {result.sql_complexity} SQL using patterns: {', '.join(result.sql_patterns)}"
            logger.info(f"[ORCH-STEP3] ✅ SQL: {sql_query[:100]}...")
            
            # STEP 4: EXECUTION
            logger.info("[ORCH-STEP4] ⚡ EXECUTION")
            query_result = await self._execute_query(sql_query, db)
            result.query_result = query_result
            result.reasoning += f"\n✅ Query executed successfully, returned results"
            logger.info(f"[ORCH-STEP4] ✅ Results: {query_result.get('row_count', 0)} rows")
            
            # Calculate execution time
            result.execution_time_ms = (time.time() - start_time) * 1000
            
            logger.info(f"[ORCHESTRATOR] ✅ COMPLETE (took {result.execution_time_ms:.1f}ms)")
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] ❌ Error: {str(e)}", exc_info=True)
            result.error = str(e)
            result.reasoning += f"\n❌ Error: {str(e)}"
        
        return result
    
    async def _analyze_semantics(
        self,
        user_prompt: str,
        db: AsyncSession,
        previous_query_context: Optional[Dict[str, Any]],
        conversation_history: Optional[List[str]]
    ) -> SemanticAnalysis:
        """STEP 1: Perform semantic analysis."""
        
        try:
            logger.debug(f"[ANALYSIS] Analyzing: '{user_prompt}'")
            
            # Use universal analyzer for semantic understanding
            semantic_analysis = await self.analyzer.analyze_query(
                user_prompt,
                db,
                previous_query_context,
                conversation_history
            )
            
            logger.debug(f"[ANALYSIS] Query type: {semantic_analysis.query_type}")
            logger.debug(f"[ANALYSIS] Tables: {[t.table_name for t in semantic_analysis.relevant_tables]}")
            logger.debug(f"[ANALYSIS] Joins: {len(semantic_analysis.required_joins)}")
            logger.debug(f"[ANALYSIS] Values: {semantic_analysis.user_values}")
            
            return semantic_analysis
            
        except Exception as e:
            logger.error(f"[ANALYSIS] Semantic analysis failed: {e}")
            raise
    
    async def _discover_values(
        self,
        semantic_analysis: SemanticAnalysis,
        user_prompt: str,
        db: AsyncSession,
        previous_query_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """STEP 2: Discover value->column mappings."""
        
        value_mappings = {}
        
        try:
            logger.debug(f"[VALUE_DISCOVERY] Discovering values: {semantic_analysis.user_values}")
            
            # For each discovered value, find which column it belongs to
            for user_value in semantic_analysis.user_values:
                
                # Try to discover column for this value
                for table in semantic_analysis.relevant_tables:
                    
                    # Create discovery context
                    discovery_context = FollowUpContext(
                        table_name=table.name,
                        table_schema=table.columns,
                        table_sample_data=table.sample_data,
                        previous_columns_used=previous_query_context.get('columns_used') if previous_query_context else None,
                        previous_filters=previous_query_context.get('filters') if previous_query_context else None,
                        previous_query=previous_query_context.get('query') if previous_query_context else None,
                        is_followup=bool(previous_query_context)
                    )
                    
                    # Discover column for this value
                    matches = await self.mapper.discover_column_for_value(
                        user_value,
                        discovery_context,
                        db
                    )
                    
                    if matches:
                        # Took best match
                        best_match = matches[0]
                        value_mappings[user_value] = {
                            'column': best_match.column_name,
                            'table': table.table_name,
                            'confidence': best_match.confidence,
                            'strategy': best_match.discovery_strategy,
                            'reasoning': best_match.reasoning
                        }
                        logger.debug(f"[VALUE_DISCOVERY] '{user_value}' -> table.{best_match.column_name} ({best_match.confidence:.1%})")
                        break
            
            return value_mappings
            
        except Exception as e:
            logger.warning(f"[VALUE_DISCOVERY] Value discovery failed (non-fatal): {e}")
            # Non-fatal: Continue even if value discovery fails
            return value_mappings
    
    async def _generate_sql(
        self,
        semantic_analysis: SemanticAnalysis,
        value_mappings: Dict[str, Any],
        db: AsyncSession,
        user_prompt: str
    ) -> Tuple[str, Dict[str, Any]]:
        """STEP 3: Generate SQL query."""
        
        try:
            logger.debug(f"[SQL_GENERATION] Query type: {semantic_analysis.query_type}")
            
            # Build generation context from semantic analysis
            gen_context = self._build_generation_context(
                semantic_analysis,
                value_mappings
            )
            
            # Generate SQL
            sql, metadata = await self.generator.generate_sql(
                gen_context,
                db,
                user_prompt
            )
            
            logger.debug(f"[SQL_GENERATION] Generated: {sql}")
            
            return sql, metadata
            
        except Exception as e:
            logger.error(f"[SQL_GENERATION] SQL generation failed: {e}")
            raise
    
    def _build_generation_context(
        self,
        semantic_analysis: SemanticAnalysis,
        value_mappings: Dict[str, Any]
    ) -> SQLGenerationContext:
        """Convert semantic analysis to SQL generation context."""
        
        # Get primary table (first/most relevant)
        # SAFETY: relevant_tables guaranteed to be non-empty due to validation in orchestrate()
        if not semantic_analysis.relevant_tables:
            raise ValueError("No relevant tables found in semantic analysis")
        primary_table = semantic_analysis.relevant_tables[0].table_name
        
        # Get related tables
        related_tables = [t.table_name for t in semantic_analysis.relevant_tables[1:]]
        
        # Get joins from semantic analysis
        joins = []
        for table1, table2, on_clause in semantic_analysis.required_joins:
            joins.append((table1, table2, on_clause))
        
        # Build filters from value mappings
        filters = []
        for user_value, mapping in value_mappings.items():
            filters.append({
                'column': f"{mapping['table']}.{mapping['column']}",
                'operator': '=',
                'value': user_value
            })
        
        # Get selected columns (default to all)
        selected_columns = ["*"]
        
        # Create context
        context = SQLGenerationContext(
            primary_table=primary_table,
            related_tables=related_tables,
            joins=joins,
            filters=filters,
            selected_columns=selected_columns,
            aggregation=None,  # Not used by SemanticAnalysis
            group_by=[],  # Not available in SemanticAnalysis
            order_by=[],  # Not available in SemanticAnalysis
            limit=None,  # Not available in SemanticAnalysis
            offset=None,  # Not available in SemanticAnalysis
            distinct=semantic_analysis.distinct_requested,
        )
        
        return context
    
    async def _execute_query(self, sql_query: str, db: AsyncSession) -> Dict[str, Any]:
        """STEP 4: Execute generated SQL and return results."""
        
        try:
            logger.debug(f"[EXECUTION] Executing: {sql_query[:100]}...")
            
            # Execute query
            result = await db.execute(sa.text(sql_query))
            rows = result.fetchall()
            
            # Convert to dicts
            row_dicts = [dict(row._mapping) for row in rows]
            
            logger.info(f"[EXECUTION] ✅ Returned {len(row_dicts)} rows")
            
            return {
                'row_count': len(row_dicts),
                'rows': row_dicts,
                'columns': list(row_dicts[0].keys()) if row_dicts else [],
                'success': True
            }
            
        except Exception as e:
            logger.error(f"[EXECUTION] Query execution failed: {e}")
            
            # CRITICAL: Rollback transaction to prevent cascade failures
            # When a query fails in PostgreSQL, the transaction is marked "aborted"
            # All subsequent queries fail until transaction is rolled back
            try:
                await db.rollback()
                await db.commit()  # Commit to complete rollback transaction
                logger.debug("[EXECUTION] Transaction rolled back and committed after error")
            except Exception as rb_error:
                logger.debug(f"[EXECUTION] Rollback failed (non-critical): {rb_error}")
            
            return {
                'row_count': 0,
                'rows': [],
                'columns': [],
                'success': False,
                'error': str(e)
            }


# Singleton instance
_orchestrator: Optional[IntelligentQueryOrchestrator] = None

def get_query_orchestrator(db_adapter: DatabaseAdapter = None) -> IntelligentQueryOrchestrator:
    """Get or create singleton orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        if db_adapter is None:
            from .database_adapter import get_global_adapter
            db_adapter = get_global_adapter()
        _orchestrator = IntelligentQueryOrchestrator(db_adapter)
    return _orchestrator
