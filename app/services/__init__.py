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
from .followup_query_rewriter import SemanticFollowUpRewriter, SemanticFollowUpContext, get_followup_rewriter
from .advanced_semantic_analyzer import AdvancedSemanticAnalyzer, SemanticContext, IntentType, ConversationState, get_advanced_semantic_analyzer
from .chatgpt_query_rewriter import ChatGPTLevelQueryRewriter, QueryRewriteResult, get_chatgpt_query_rewriter
from .intelligent_conversation_manager import IntelligentConversationManager, ConversationTurn, ConversationMemory, get_conversation_manager

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

# Plan-First Query Understanding Pipeline (NEW - ChatGPT-style semantic processing)
from .semantic_concept_extractor import (
    SemanticConceptExtractor, SemanticIntent, FilterConcept, IntentType, OperatorType,
    get_concept_extractor
)
from .plan_first_sql_generator import (
    PlanFirstQueryHandler, SemanticGrounder, QueryPlan, ColumnMapping,
    get_plan_first_handler
)
from .query_coverage_verifier import (
    QueryCoverageVerifier, CoverageReport, ConceptCoverage, VerificationResult,
    get_coverage_verifier
)

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

# P0-P3: Comprehensive Improvements (Zero Hardcoding, DB-Agnostic)
# P0: Unified Semantic Router (Core routing schema + decision engine)
from .router_decision import (
    RouterDecision, Tool, RequestType, FollowupTool, 
    RunSQLFollowupSubtype, AnalyzeFileFollowupSubtype, ChatFollowupSubtype,
    ClarificationOption, ROUTER_DECISION_JSON_SCHEMA
)
from .hard_signals import (
    HardSignals, HardSignalsExtractor
)
from .unified_semantic_router import (
    UnifiedSemanticRouter
)

# P1: SQL Safety Validation & pgVector File Retrieval
from .dynamic_sql_safety import (
    DynamicSqlSafetyValidator, SafetySqlConfig, 
    SqlStatementType, SqlValidationResult
)
from .pgvector_file_retriever import (
    PgVectorFileRetriever, VectorSearchConfig, VectorIndexStrategy,
    Vector, FileChunkMetadata, RetrievedChunk
)

# P2: Conversation Memory Management & Privacy/Audit Layer

# AGENTIC ARCHITECTURE - 100% Implementation (NEW)
# Master Orchestrator & Tool Registry
from .master_orchestrator import (
    MasterOrchestrator, get_master_orchestrator,
    TaskType, ToolType, ExecutionStatus,
    TaskIntent, ExecutionStep, ExecutionPlan, VerificationResult
)
from .tool_registry import (
    ToolRegistry, get_tool_registry,
    BaseTool, ToolMetadata,
    SQLExecutorTool, FileReaderTool, PythonExecutorTool,
    ChartGeneratorTool, VectorSearchTool
)

# Python Sandbox Execution
from .python_sandbox import (
    execute_python_code, generate_python_code_for_task,
    ExecutionResult, ExecutionStatus as PythonExecutionStatus
)

# Knowledge Graph
from .knowledge_graph import (
    KnowledgeGraph, get_knowledge_graph,
    EntityNode, RelationshipEdge, EntityPath,
    EntityType, RelationshipType
)

# Clarification Engine - 100% Implementation
from .clarification_engine import (
    ClarificationEngine, get_clarification_engine,
    AmbiguityType, ClarificationStrategy,
    ClarificationRequest, ClarificationResponse, ClarificationOption,
    AmbiguityAnalysis
)

# Enhanced Observability
from .observability import (
    ObservabilityLogger, get_observability_logger,
    ExecutionTracer, get_execution_tracer,
    EventType, LogLevel, ObservabilityEvent, PerformanceMetrics,
    log_query_start, log_query_complete, log_llm_call
)

# Agentic Query Handler - NEW Unified Entry Point
from .agentic_query_handler import (
    AgenticQueryHandler, create_agentic_handler
)
from .conversation_memory_manager import (
    ConversationMemoryManager, ConversationMemoryConfig,
    ConversationMemoryState, ConversationMessage, ConversationSummary,
    MessageRole
)
from .privacy_audit_layer import (
    PiiDetector, AuditLogger, AuditLogEntry, PrivacyConfig,
    PiiType, PiiDetection
)

# P3: Routing Evaluation & Testing Harness
from .routing_evaluation_harness import (
    RoutingEvaluationHarness, GroundTruth, EvaluationResult,
    EvaluationReport, EvaluationMetric,
    HOSPITAL_DOMAIN_TESTS, EDUCATION_DOMAIN_TESTS,
    COMMERCE_DOMAIN_TESTS, HR_DOMAIN_TESTS, ADVERSARIAL_TESTS,
    ALL_TEST_CASES
)

# Real-time query control and monitoring
from .cancellation_manager import (
    CancellationManager, CancellationToken, CancellationException,
    cancellation_manager
)
from .progress_tracker import (
    ProgressTracker, ProgressTrackerManager, ProgressStep, ProgressUpdate,
    progress_tracker_manager
)

# NEW: Efficient Semantic Routing System
from .semantic_routing_integration import SemanticRoutingIntegration, create_routing_integration
from .semantic_intent_router import SemanticIntentRouter, create_router
from .efficient_query_router import EfficientQueryRouter, create_efficient_router
from .turn_state_manager import TurnStateManager, create_turn_state_manager

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
    
    # P0-P3: Comprehensive Improvements (ZERO HARDCODING)
    # P0: Unified Router
    "RouterDecision",
    "Tool",
    "RequestType",
    "FollowupTool",
    "HardSignals",
    "HardSignalsExtractor",
    "UnifiedSemanticRouter",
    
    # P1: SQL Safety & File Retrieval
    "DynamicSqlSafetyValidator",
    "SafetySqlConfig",
    "SqlStatementType",
    "SqlValidationResult",
    "PgVectorFileRetriever",
    "VectorSearchConfig",
    "VectorIndexStrategy",
    "FileChunkMetadata",
    "RetrievedChunk",
    
    # P2: Memory & Privacy
    "ConversationMemoryManager",
    "ConversationMemoryConfig",
    "ConversationMemoryState",
    "ConversationMessage",
    "ConversationSummary",
    "MessageRole",
    "PiiDetector",
    "AuditLogger",
    "AuditLogEntry",
    "PrivacyConfig",
    "PiiType",
    "PiiDetection",
    
    # P3: Evaluation & Testing
    "RoutingEvaluationHarness",
    "GroundTruth",
    "EvaluationResult",
    "EvaluationReport",
    "ALL_TEST_CASES",
    
    # Real-time query control and monitoring
    "CancellationManager",
    "CancellationToken", 
    "CancellationException",
    "cancellation_manager",
    "ProgressTracker",
    "ProgressTrackerManager",
    "ProgressStep",
    "ProgressUpdate",
    "progress_tracker_manager",
    
    # NEW: Efficient Semantic Routing System
    "SemanticRoutingIntegration",
    "create_routing_integration",
    "SemanticIntentRouter", 
    "create_router",
    "EfficientQueryRouter",
    "create_efficient_router",
    "TurnStateManager",
    "create_turn_state_manager",
    
    # NEW: Plan-First Query Understanding Pipeline (ChatGPT-style)
    "SemanticConceptExtractor",
    "SemanticIntent", 
    "FilterConcept",
    "IntentType",
    "OperatorType",
    "get_concept_extractor",
    "PlanFirstQueryHandler",
    "SemanticGrounder",
    "QueryPlan",
    "ColumnMapping", 
    "get_plan_first_handler",
    "QueryCoverageVerifier",
    "CoverageReport",
    "ConceptCoverage",
    "VerificationResult",
    "get_coverage_verifier",
    
    # NEW: Advanced Semantic Intelligence (ChatGPT-Level)
    "AdvancedSemanticAnalyzer",
    "SemanticContext", 
    "IntentType",
    "ConversationState",
    "get_advanced_semantic_analyzer",
    
    # NEW: ChatGPT-Level Query Rewriting
    "ChatGPTLevelQueryRewriter",
    "QueryRewriteResult",
    "get_chatgpt_query_rewriter",
    
    # NEW: Intelligent Conversation Management
    "IntelligentConversationManager", 
    "ConversationTurn",
    "ConversationMemory",
    "get_conversation_manager",
    
    # NEW: Follow-Up Query Rewriting (Semantic ChatGPT-style context expansion)
    "SemanticFollowUpRewriter",
    "SemanticFollowUpContext",
    "get_followup_rewriter",
]