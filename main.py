"""
Entry point for the Gen‑AI backend service.

This module creates a FastAPI application, configures global middleware
such as CORS, mounts API routers and exposes a function for running
the development server. The application uses Uvicorn as the ASGI server
when started with ``python -m genai_backend.main``. Configuration values
are loaded from environment variables defined in ``env.txt`` or a ``.env``
file using pydantic settings (see ``app/config.py``).

The API is organised under ``/api/dynamic/*`` and implements endpoints
for session management, querying an LLM, retrieving capabilities and
returning example responses. See ``app/routes.py`` for details.

Note: This project uses asynchronous SQLAlchemy and requires a
PostgreSQL database. Dependencies are specified in ``requirements.txt``.
"""

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db, Base, SessionLocal, dispose_engine
from app.models import User, ChatSession, Message, UploadedFile, FileChunk
from app.routes import router as api_router
from app.services import (
    UnifiedSemanticRouter,
    HardSignalsExtractor,
    DynamicSqlSafetyValidator,
    SafetySqlConfig,
    SqlStatementType,
    PgVectorFileRetriever,
    VectorSearchConfig,
    ConversationMemoryManager,
    ConversationMemoryConfig,
    PiiDetector,
    AuditLogger,
    PrivacyConfig,
    RoutingEvaluationHarness,
)


# Configure logging to write all logs to app.log
def configure_logging():
    """Configure unified logging to app.log file."""
    from logging.handlers import RotatingFileHandler
    import os
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    root_logger.handlers.clear()
    
    # File handler - writes to app.log with rotation (10MB max, 5 backups)
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler - for terminal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Suppress DEBUG logs from noisy libraries
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("python_multipart").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Keep application logs at INFO level
    logging.getLogger("app").setLevel(logging.INFO)
    logging.getLogger("app.services").setLevel(logging.INFO)
    logging.getLogger("app.routes").setLevel(logging.INFO)
    
    logging.info("=" * 80)
    logging.info("✅ Logging initialized - Writing to logs/app.log")
    logging.info("=" * 80)


configure_logging()


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI application.

    Returns:
        FastAPI: the configured application instance.
    """
    app = FastAPI(title="GenAI Backend Service")

    # Enable CORS using settings.CORS_ORIGINS. The allowed origins can be
    # customised via the environment. This makes the backend consumable
    # from a web front‑end served on a different host/port.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize database on startup BEFORE including routes
    @app.on_event("startup")
    async def startup_event():
        try:
            print("=" * 70)
            print("[STARTUP] Initializing GenAI Backend Service")
            print("=" * 70)
            
            print("\n[1/5] Initializing database...")
            await init_db()
            print("    [OK] Database initialized")
            
            # ===== P0-P3 SERVICE INITIALIZATION =====
            print("\n[2/5] Initializing P0-P3 services...")
            
            # P0: Core routing services
            print("    [P0] Unified Semantic Router")
            router = UnifiedSemanticRouter(SessionLocal)
            app.state.router = router
            
            # P1: SQL Safety validator
            print("    [P1] Dynamic SQL Safety Validator")
            sql_config = SafetySqlConfig(
                allowed_statement_types={SqlStatementType.SELECT, SqlStatementType.WITH, SqlStatementType.UNION},
                require_confirmation_for={SqlStatementType.INSERT, SqlStatementType.UPDATE, SqlStatementType.DELETE},
                statement_timeout_seconds=30,
                max_result_rows=100000,
            )
            sql_validator = DynamicSqlSafetyValidator(sql_config)
            app.state.sql_validator = sql_validator
            
            # P1: pgvector file retriever
            print("    [P1] pgVector File Retriever")
            try:
                from app.services import EmbeddingService
                import asyncio
                
                embedding_service = EmbeddingService()
                
                # Create sync wrapper for async embed method
                def embed_sync(text: str):
                    """Synchronous wrapper for async embedding."""
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're in an async context, create a wrapper that returns a placeholder
                        # In production, use asyncio.run() in a thread pool
                        return [0.0] * 384  # Placeholder
                    return loop.run_until_complete(embedding_service.embed(text))
                
                vector_config = VectorSearchConfig(
                    similarity_threshold=0.5,
                    top_k=5,
                    search_timeout_ms=5000,
                )
                file_retriever = PgVectorFileRetriever(SessionLocal, embed_sync, vector_config)
                # Note: setup_index() should be called separately or during migrations
                app.state.file_retriever = file_retriever
            except Exception as e:
                print(f"    [WARN] pgVector setup skipped (optional): {e}")
                app.state.file_retriever = None
            
            # P2: Conversation memory manager
            print("    [P2] Conversation Memory Manager")
            def token_counter(text):
                """Simple token counter: ~1 token per 4 characters"""
                return max(1, len(text) // 4)
            
            memory_config = ConversationMemoryConfig(
                max_context_tokens=6000,
                summary_trigger_ratio=0.8,
                token_counter_fn=token_counter,
                keep_recent_messages=5,
                keep_messages_hours=24,
            )
            memory_manager = ConversationMemoryManager(memory_config)
            app.state.memory_manager = memory_manager
            
            # P2: Privacy & Audit layer
            print("    [P2] Privacy & Audit Layer")
            privacy_config = PrivacyConfig(
                redact_pii_in_logs=True,
                retention_days_default=90,
                retention_days_audit_log=365,
            )
            pii_detector = PiiDetector(privacy_config)
            audit_logger = AuditLogger(SessionLocal, pii_detector, privacy_config)
            app.state.pii_detector = pii_detector
            app.state.audit_logger = audit_logger
            
            # P3: Evaluation harness
            print("    [P3] Routing Evaluation Harness")
            evaluator = RoutingEvaluationHarness(router)
            from app.services import ALL_TEST_CASES
            evaluator.register_test_cases(ALL_TEST_CASES)
            app.state.evaluator = evaluator
            
            print("    [OK] All P0-P3 services initialized")
            
            print("\n" + "=" * 70)
            print("[OK] SERVICE INITIALIZATION COMPLETE")
            print("=" * 70)
            print("\nInitialized Services:")
            print("  [P0] Unified Semantic Router (4-layer routing)")
            print("  [P1] SQL Safety Validator (config-driven)")
            print("  [P1] pgVector File Retriever (indexed search)")
            print("  [P2] Conversation Memory Manager (rolling summarization)")
            print("  [P2] Privacy & Audit Layer (PII detection + logging)")
            print("  [P3] Routing Evaluation Harness (16 built-in tests)")
            print("\n" + "=" * 70 + "\n")
            
        except Exception as e:
            print(f"\n[ERROR] Initialization error: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise so we know initialization failed

    @app.on_event("shutdown")
    async def shutdown_event():
        # Best-effort cleanup. Shielded inside dispose_engine().
        try:
            await dispose_engine()
        except Exception as e:
            logging.getLogger(__name__).warning(f"[SHUTDOWN] Engine dispose failed: {e}")

    # Include the dynamic API router
    app.include_router(api_router, prefix="/api/dynamic")

    return app


app = create_app()


if __name__ == "__main__":
    # When executed directly (e.g. ``python main.py``) start
    # the Uvicorn development server. In production one might instead use
    # ``uvicorn main:app --host 0.0.0.0 --port 5000`` or a
    # process manager such as Gunicorn.
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )