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

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db, Base
from app.models import User, ChatSession, Message, UploadedFile, FileChunk
from app.routes import router as api_router


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
            print("Initializing database...")
            await init_db()
            print("[OK] Database initialized successfully")
        except Exception as e:
            print(f"[ERROR] Database initialization error: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise so we know initialization failed

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