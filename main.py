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

import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException, RequestValidationError
from app.middleware.exception_handler import (
    global_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from app.middleware.rate_limiter import limiter, rate_limit_exceeded_handler

from app.config import settings
from app.database import init_db, SessionLocal, dispose_engine
import app.models as _models; _ = _models  # registers ORM models with SQLAlchemy metadata
from app.routes import router as api_router
from app.services import (
    UnifiedSemanticRouter,
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
    """Configure unified logging to app.log file (idempotent - safe to call multiple times)."""
    from logging.handlers import RotatingFileHandler
    import os
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    
    # GUARD: Check if already configured by looking for existing handlers
    # with our specific formatter pattern to avoid duplicate initialization
    for handler in root_logger.handlers:
        if isinstance(handler, RotatingFileHandler) and 'app.log' in str(handler.baseFilename):
            return  # Already configured with our handler, skip
    
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to prevent duplicates
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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """
    FastAPI lifespan handler — replaces deprecated @app.on_event.
    Everything before `yield` runs at startup; everything after at shutdown.
    """
    # ── STARTUP ──────────────────────────────────────────────────────────────
    try:
        print("=" * 70)
        print("[STARTUP] Initializing GenAI Backend Service")
        print("=" * 70)

        print("\n[1/5] Initializing database...")
        await init_db()
        print("    [OK] Database initialized")

        print("\n[2/5] Initializing P0-P3 services...")

        # P0: Core routing
        print("    [P0] Unified Semantic Router")
        app.state.router = UnifiedSemanticRouter(SessionLocal)

        # P1: SQL Safety Validator (magic numbers replaced with settings)
        print("    [P1] Dynamic SQL Safety Validator")
        sql_config = SafetySqlConfig(
            allowed_statement_types={SqlStatementType.SELECT, SqlStatementType.WITH, SqlStatementType.UNION},
            require_confirmation_for={SqlStatementType.INSERT, SqlStatementType.UPDATE, SqlStatementType.DELETE},
            statement_timeout_seconds=settings.sql_statement_timeout_sec,
            max_result_rows=settings.sql_max_result_rows,
        )
        app.state.sql_validator = DynamicSqlSafetyValidator(sql_config)

        # P1: pgVector file retriever
        print("    [P1] pgVector File Retriever")
        try:
            _dims = settings.embedding_dimensions

            def embed_sync(_: str) -> list:  # real embedding is async; stub used for retriever init
                return [0.0] * _dims

            vector_config = VectorSearchConfig(
                similarity_threshold=settings.vector_similarity_threshold,
                top_k=settings.vector_search_top_k,
                search_timeout_ms=settings.vector_search_timeout_ms,
            )
            app.state.file_retriever = PgVectorFileRetriever(SessionLocal, embed_sync, vector_config)
        except Exception as e:
            print(f"    [WARN] pgVector setup skipped (optional): {e}")
            app.state.file_retriever = None

        # P2: Conversation memory manager
        print("    [P2] Conversation Memory Manager")
        memory_config = ConversationMemoryConfig(
            max_context_tokens=settings.memory_max_context_tokens,
            summary_trigger_ratio=0.8,
            token_counter_fn=lambda text: max(1, len(text) // 4),
            keep_recent_messages=settings.memory_keep_recent_messages,
            keep_messages_hours=settings.memory_keep_messages_hours,
        )
        app.state.memory_manager = ConversationMemoryManager(memory_config)

        # P2: Semantic Value Grounder
        print("    [P2] Semantic Value Grounder")
        try:
            from app.services.semantic_value_grounding_enhanced import get_semantic_value_grounder_enhanced
            grounder = get_semantic_value_grounder_enhanced()
            if not grounder.initialized:
                print("    [P2] Warming grounder (profiling database schema)...")
                async with SessionLocal() as db:
                    await grounder.initialize_for_tables(db, sample_size=100)
                print("    [P2] Grounder warmed and ready")
            app.state.grounder = grounder
        except Exception as e:
            print(f"    [WARN] Grounder initialization failed (non-critical): {e}")
            app.state.grounder = None

        # P2: Privacy & Audit layer
        print("    [P2] Privacy & Audit Layer")
        privacy_config = PrivacyConfig(
            redact_pii_in_logs=True,
            retention_days_default=90,
            retention_days_audit_log=365,
        )
        pii_detector = PiiDetector(privacy_config)
        app.state.pii_detector = pii_detector
        app.state.audit_logger = AuditLogger(SessionLocal, pii_detector, privacy_config)

        # P3: Evaluation harness
        print("    [P3] Routing Evaluation Harness")
        from app.services import ALL_TEST_CASES
        evaluator = RoutingEvaluationHarness(app.state.router)
        evaluator.register_test_cases(ALL_TEST_CASES)
        app.state.evaluator = evaluator

        # Production-grade components
        print("\n    [PROD] Production-Grade Architecture Components")
        print("    " + "=" * 60)

        print("    [PROD] Schema Intelligence Service (weighted resolution)")
        try:
            from app.services import get_schema_intelligence
            schema_intel = get_schema_intelligence()
            async with SessionLocal() as schema_db:
                await schema_intel.initialize(schema_db)
            app.state.schema_intelligence = schema_intel
            print("    [PROD] Schema Intelligence initialized")
        except Exception as e:
            print(f"    [WARN] Schema Intelligence init failed (non-critical): {e}")
            app.state.schema_intelligence = None

        print("    [PROD] Execution Policy Engine (fast path routing)")
        from app.services import get_execution_policy_engine
        app.state.policy_engine = get_execution_policy_engine()

        print("    [PROD] Question-Back Engine (contextual suggestions)")
        from app.services import get_question_back_engine
        app.state.question_engine = get_question_back_engine()

        print("    [PROD] Schema Change Detector (invalidates cache on DDL)")
        try:
            from app.services.schema_change_detector import start_schema_change_detector
            _schema_name = getattr(settings, "postgres_schema", "public")
            app.state.schema_detector_task = asyncio.create_task(
                start_schema_change_detector(SessionLocal, schema_name=_schema_name)
            )
            print("    [PROD] Schema Change Detector running")
        except Exception as e:
            print(f"    [WARN] Schema Change Detector failed (non-critical): {e}")
            app.state.schema_detector_task = None

        print("    [PROD] Background Task Queue (arq)")
        try:
            from app.services.task_queue import start_task_queue
            await start_task_queue()
            print("    [PROD] Background Task Queue ready")
        except Exception as e:
            print(f"    [WARN] Background Task Queue unavailable (non-critical): {e}")

        print("    " + "=" * 60)
        print("\n" + "=" * 70)
        print("[OK] SERVICE INITIALIZATION COMPLETE")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Initialization error: {e}")
        import traceback
        traceback.print_exc()
        raise

    yield  # ── application runs here ─────────────────────────────────────────

    # ── SHUTDOWN ─────────────────────────────────────────────────────────────
    # Cancel background tasks
    task = getattr(app.state, "schema_detector_task", None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    try:
        from app.services.task_queue import stop_task_queue
        await stop_task_queue()
    except Exception as e:
        logging.getLogger(__name__).debug("[SHUTDOWN] Task queue stop: %s", e)

    try:
        await dispose_engine()
    except Exception as e:
        logging.getLogger(__name__).warning("[SHUTDOWN] Engine dispose failed: %s", e)


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI application."""
    app = FastAPI(title="GenAI Backend Service", lifespan=_lifespan)

    # ── Exception handlers ────────────────────────────────────────────────────
    app.add_exception_handler(Exception, global_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # ── Rate limiting (slowapi) ───────────────────────────────────────────────
    if settings.rate_limit_enabled:
        try:
            from slowapi import SlowAPIMiddleware
            from slowapi.errors import RateLimitExceeded
            app.state.limiter = limiter
            app.add_middleware(SlowAPIMiddleware)
            app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
        except ImportError:
            print("[WARN] slowapi not installed — rate limiting disabled (pip install slowapi)")

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/dynamic")
    return app


app = create_app()


if __name__ == "__main__":
    # When executed directly (e.g. ``python main.py``) start
    # the Uvicorn development server. In production one might instead use
    # ``uvicorn main:app --host 0.0.0.0 --port 5000`` or a
    # process manager such as Gunicorn.
    # PERFORMANCE FIX: Disable reload and use single worker to prevent multiple initializations
    import os
    port = int(os.getenv("PORT", settings.port))
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=port,
        reload=False,  # Disable auto-reload to prevent multiple initializations
        workers=1,     # Single worker mode
    )