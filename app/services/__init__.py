from .classification import classify_query
from .file_handler import add_file, process_file_upload, retrieve_relevant_chunks
from .file_processor import DynamicFileProcessor, create_file_processor
from .embedding_service import EmbeddingService, create_embedding_service
from .nlp_processor import AdvancedNLPProcessor, create_nlp_processor
from .query_executor import run_sql
from .query_handler import (
    build_config_update_response,
    build_data_query_response,
    build_file_lookup_response,
    build_file_query_response,
    build_standard_response,
)
from .query_orchestrator import handle_dynamic_query
from .session_query_handler import execute_with_session_state
from .response_generator import (
    ConversationState,
    DynamicResponseGenerator,
    create_conversation_state,
)
from .sql_generator import generate_sql, generate_sql_with_analysis
from .response_formatter import should_return_as_message, determine_visualization_type
from .database_analyzer import DatabaseAnalyzer, create_database_analyzer
from .followup_context_manager import FollowupContextManager, FollowupContext, ContextSnapshot

# New architecture services
from .schema_discovery import SchemaCatalog, create_schema_catalog
from .schema_normalizer import SchemaNormalizer, DomainEntity
from .schema_rag import SchemaRAG, create_schema_rag
from .hybrid_matcher import HybridMatcher, MatchResult, create_hybrid_matcher
from .decision_engine import DecisionEngine, Action, DecisionOutput, create_decision_engine
from .entity_parser import EntityParser, ParsedQuery, QueryIntent, create_entity_parser
from .sql_safety_validator import SQLSafetyValidator, SQLParameterValidator, validate_sql_safety, create_sql_safety_validator
from .followup_manager import FollowUpAnalyzer, FollowUpContext, FollowUpType, PreviousQueryContext, get_followup_analyzer
from .adaptive_schema_analyzer import AdaptiveSchemaAnalyzer, ColumnPurpose
from .response_composer import ResponseComposer, ContentBlock, AssistantMessage, FollowUp, ArtifactsMetadata

# QueryPlan architecture (NEW)
from .query_plan import (
    QueryPlan, SelectClause, FromClause, JoinClause, GroupByClause, 
    HavingClause, OrderByClause, OrderByField, BinaryCondition, LogicalCondition,
    NotCondition, Condition, ColumnRef, Literal as LiteralValue, SubqueryValue,
    QueryArtifacts, ValueType, QueryPlanValidationError, ColumnNotFoundError,
    TableNotFoundError, JoinPathNotFoundError, TypeMismatchError, InvalidSubqueryError
)
from .query_plan_validator import QueryPlanValidator, JoinPathFinder
from .query_plan_compiler import DialectCompiler, SQLGenerator, PostgreSQLGenerator, MySQLGenerator, SQLiteGenerator, SQLServerGenerator

# New validation and grounding services
from .plan_validator import PlanValidator, validate_and_fix_plan
from .value_grounding import ValueGrounder, FilterValueMapper

# DB-Agnostic infrastructure (NEW - Phase 2+)
from .dialect_adapter import (
    DialectAdapter, DialectCapabilities, DatabaseDialect, 
    DIALECT_CONFIG, get_adapter
)
from .schema_discovery_sqlalchemy import (
    SQLAlchemySchemaDiscovery, SchemaCache, ColumnInfo, TableInfo,
    discover_tables, discover_table_info
)
from .evaluation_harness import (
    EvaluationHarness, TestCase, TestResult, TestStatus,
    create_evaluation_harness
)

# Query classifier - Plan-first JSON output (NEW - AI-centric)


# Semantic query pipeline (NEW - WIRED FOR INTEGRATION)
from .semantic_query_orchestrator import (
    SemanticQueryOrchestrator, PipelineStage, RetrievalContext, SemanticQueryResult,
    create_semantic_orchestrator
)

__all__ = [
    # Core existing services
    "add_file",
    "AdvancedNLPProcessor",
    "build_config_update_response",
    "build_data_query_response",
    "build_file_lookup_response",
    "build_file_query_response",
    "build_standard_response",
    "classify_query",
    "ConversationState",
    "ContextSnapshot",
    "create_conversation_state",
    "create_database_analyzer",
    "create_file_processor",
    "create_embedding_service",
    "create_nlp_processor",
    "DatabaseAnalyzer",
    "DynamicFileProcessor",
    "DynamicResponseGenerator",
    "determine_visualization_type",
    "EmbeddingService",
    "FollowupContext",
    "FollowupContextManager",
    "generate_sql",
    "generate_sql_with_analysis",
    "handle_dynamic_query",
    "process_file_upload",
    "retrieve_relevant_chunks",
    "run_sql",
    "should_return_as_message",
    
    # Follow-up query management
    "FollowUpAnalyzer",
    "FollowUpContext",
    "FollowUpType",
    "PreviousQueryContext",
    "get_followup_analyzer",
    
    # New architecture services
    "SchemaCatalog",
    "create_schema_catalog",
    "SchemaNormalizer",
    "DomainEntity",
    "SchemaRAG",
    "create_schema_rag",
    "HybridMatcher",
    "MatchResult",
    "create_hybrid_matcher",
    "DecisionEngine",
    "Action",
    "DecisionOutput",
    "create_decision_engine",
    "EntityParser",
    "ParsedQuery",
    "QueryIntent",
    "create_entity_parser",
    "SQLSafetyValidator",
    "SQLParameterValidator",
    "validate_sql_safety",
    "create_sql_safety_validator",
    "AdaptiveSchemaAnalyzer",
    "ColumnPurpose",
    "ResponseComposer",
    "ContentBlock",
    "AssistantMessage",
    "FollowUp",
    "ArtifactsMetadata",
    
    # QueryPlan architecture (NEW)
    "QueryPlan",
    "SelectClause",
    "FromClause",
    "JoinClause",
    "GroupByClause",
    "HavingClause",
    "OrderByClause",
    "OrderByField",
    "BinaryCondition",
    "LogicalCondition",
    "NotCondition",
    "Condition",
    "ColumnRef",
    "LiteralValue",
    "SubqueryValue",
    "QueryArtifacts",
    "ValueType",
    "QueryPlanValidationError",
    "ColumnNotFoundError",
    "TableNotFoundError",
    "JoinPathNotFoundError",
    "TypeMismatchError",
    "InvalidSubqueryError",
    "QueryPlanValidator",
    "JoinPathFinder",
    "DialectCompiler",
    "SQLGenerator",
    "PostgreSQLGenerator",
    "MySQLGenerator",
    "SQLiteGenerator",
    "SQLServerGenerator",
    
    # DB-Agnostic infrastructure (NEW)
    "DialectAdapter",
    "DialectCapabilities",
    "DatabaseDialect",
    "DIALECT_CONFIG",
    "get_adapter",
    
    # SQLAlchemy-based schema discovery (NEW)
    "SQLAlchemySchemaDiscovery",
    "SchemaCache",
    "ColumnInfo",
    "TableInfo",
    "discover_tables",
    "discover_table_info",
    
    # Evaluation harness (NEW)
    "EvaluationHarness",
    "TestCase",
    "TestResult",
    "TestStatus",
    "create_evaluation_harness",
    
    # Semantic query pipeline orchestrator (NEW - WIRED FOR INTEGRATION)
    "SemanticQueryOrchestrator",
    "PipelineStage",
    "RetrievalContext",
    "SemanticQueryResult",
    "create_semantic_orchestrator",
    
    # Session state management (ChatGPT-style)
    "execute_with_session_state",
]