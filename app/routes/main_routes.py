"""
GenAI Backend API Routes

All API endpoints live here.  The main /query endpoint is a thin dispatcher
that delegates to one of the five mode handlers in this package.

ENDPOINTS:
  /register          - Register new user
  /login             - Authenticate and get JWT token
  /new_session                  - Create new chat session
  /sessions                     - List sessions (paginated, ?page=&page_size=)
  /sessions/{id}/title          - PATCH rename a session
  /sessions/{id}                - DELETE a session and all its data
  /history/{id}                 - Get messages in a session
  /query             - Main unified query endpoint (handles SQL/FILES/CHAT)
  /capabilities      - Get supported visualization types
  /examples          - Get example queries
  /health            - Health check
"""

from __future__ import annotations

import json
import logging
import uuid as _uuid
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Union

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from .. import auth, models, schemas, schemas_enterprise
from ..config import get_search_path_sql
from ..database import get_session
from ..helpers import current_timestamp
from ..services.orchestrator import Orchestrator

from ..services.prompt_injection_guardian import (
    GuardianConfig,
    InjectionRiskLevel,
    PromptInjectionGuardian,
)
from ..services.file_security_scanner import (
    FileSecurityScanner,
    SecurityConfig,
    ThreatLevel,
)
from pathlib import Path

from ._context import QueryContext
from . import DISPATCH_TABLE

logger = logging.getLogger(__name__)

# ============================================================================
# FILE FORMAT RESTRICTIONS
# ============================================================================
ALLOWED_FILE_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls", ".json", ".txt"}
STRUCTURED_FILE_EXTENSIONS = {".csv", ".xlsx", ".xls"}
UNSTRUCTURED_FILE_EXTENSIONS = {".pdf", ".json", ".txt"}


def get_file_category(filename: str) -> str:
    """Classify a file as 'structured' or 'unstructured'."""
    ext = Path(filename).suffix.lower()
    return "structured" if ext in STRUCTURED_FILE_EXTENSIONS else "unstructured"


def validate_file_extension(filename: str) -> None:
    """Raise HTTPException if file extension is not in the allowed set."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                "Allowed formats: PDF, CSV, XLSX, XLS, JSON, TXT"
            ),
        )


# ============================================================================
# Auth dependency
# ============================================================================
bearer_scheme = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_session),
) -> models.User:
    """Verify JWT token and return current user."""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = auth.verify_access_token(credentials.credentials)
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except Exception:
        raise credentials_exception
    user = (
        await db.execute(select(models.User).where(models.User.username == username))
    ).scalars().first()
    if user is None:
        raise credentials_exception
    return user


router = APIRouter()


# ============================================================================
# Session Management
# ============================================================================

@router.post("/new_session", response_model=schemas.NewSessionResponse)
async def new_session(
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """Create a new chat session for the current user."""
    session = models.ChatSession(user_id=current_user.id)
    db.add(session)
    try:
        await db.execute(text(get_search_path_sql()))
    except Exception:
        pass
    await db.commit()
    return schemas.NewSessionResponse(success=True, session_id=str(session.id))


@router.get("/sessions", response_model=schemas.SessionsResponse)
async def list_sessions(
    page: int = 1,
    page_size: int = 20,
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """List chat sessions for the current user (paginated, newest first)."""
    page = max(1, page)
    page_size = min(max(1, page_size), 100)
    offset = (page - 1) * page_size

    # 1 query: sessions (ordered + paginated)
    sessions = (
        await db.execute(
            select(models.ChatSession)
            .where(models.ChatSession.user_id == current_user.id)
            .order_by(models.ChatSession.updated_at.desc().nulls_last(),
                      models.ChatSession.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
    ).scalars().all()

    if not sessions:
        return schemas.SessionsResponse(user_id=current_user.username, sessions=[])

    session_ids = [s.id for s in sessions]

    # 1 query: message count + latest updated_at per session
    stats_rows = (
        await db.execute(
            select(
                models.Message.session_id,
                func.count(models.Message.id).label("msg_count"),
                func.max(models.Message.updated_at).label("last_updated"),
            )
            .where(models.Message.session_id.in_(session_ids))
            .group_by(models.Message.session_id)
        )
    ).all()
    stats = {row.session_id: row for row in stats_rows}

    # 1 query: first message (earliest queried_at) per session for auto-title
    # Use a subquery to get min queried_at per session, then join
    min_qat_sub = (
        select(
            models.Message.session_id,
            func.min(models.Message.queried_at).label("min_qat"),
        )
        .where(models.Message.session_id.in_(session_ids))
        .group_by(models.Message.session_id)
        .subquery()
    )
    first_msgs_rows = (
        await db.execute(
            select(models.Message)
            .join(
                min_qat_sub,
                (models.Message.session_id == min_qat_sub.c.session_id)
                & (models.Message.queried_at == min_qat_sub.c.min_qat),
            )
        )
    ).scalars().all()
    first_msgs = {m.session_id: m for m in first_msgs_rows}

    session_summaries = []
    for s in sessions:
        st = stats.get(s.id)
        fm = first_msgs.get(s.id)

        # Prefer explicit title, fall back to first message snippet
        if s.title:
            title = s.title
        elif fm and fm.query:
            q = fm.query
            title = q[:50] + ("..." if len(q) > 50 else "")
        else:
            title = "New Chat"

        session_summaries.append(
            schemas.SessionSummary(
                session_id=str(s.id),
                created_at=s.created_at,
                last_updated=st.last_updated if st else s.updated_at or s.created_at,
                title=title,
                message_count=st.msg_count if st else 0,
            )
        )

    return schemas.SessionsResponse(
        user_id=current_user.username, sessions=session_summaries
    )


@router.patch("/sessions/{session_id}/title", response_model=schemas.UpdateSessionTitleResponse)
async def update_session_title(
    session_id: str,
    body: schemas.UpdateSessionTitleRequest,
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """Rename a session (ChatGPT-style)."""
    session = (
        await db.execute(
            select(models.ChatSession).where(
                models.ChatSession.id == session_id,
                models.ChatSession.user_id == current_user.id,
            )
        )
    ).scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    title = body.title.strip()[:255]
    session.title = title
    await db.commit()
    return schemas.UpdateSessionTitleResponse(
        success=True, session_id=session_id, title=title
    )


@router.delete("/sessions/{session_id}", response_model=schemas.DeleteSessionResponse)
async def delete_session(
    session_id: str,
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """Delete a session and all its messages, files, and tool calls."""
    session = (
        await db.execute(
            select(models.ChatSession).where(
                models.ChatSession.id == session_id,
                models.ChatSession.user_id == current_user.id,
            )
        )
    ).scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await db.delete(session)
    await db.commit()
    return schemas.DeleteSessionResponse(success=True, session_id=session_id)


@router.get("/history/{session_id}", response_model=schemas.SessionHistoryResponse)
async def get_history(
    session_id: str,
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """Retrieve chat history for a particular session."""
    session = (
        await db.execute(
            select(models.ChatSession)
            .where(models.ChatSession.id == session_id)
            .where(models.ChatSession.user_id == current_user.id)
        )
    ).scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = (
        await db.execute(
            select(models.Message)
            .where(models.Message.session_id == session_id)
            .order_by(models.Message.updated_at)
        )
    ).scalars().all()

    message_schemas = []
    for m in messages:
        response_data = m.response
        if isinstance(response_data, dict):
            response_data = json.loads(json.dumps(response_data, default=str))
            response_data["id"] = str(m.id)
        else:
            response_data = {}

        message_schemas.append(
            schemas.MessageSchema(
                id=str(m.id),
                response_type=m.response_type,
                query=m.query,
                queried_at=m.queried_at,
                responded_at=m.responded_at,
                response=response_data,
                created_at=m.updated_at,
            )
        )

    uploaded_files = (
        await db.execute(
            select(models.UploadedFile)
            .where(models.UploadedFile.session_id == session_id)
            .order_by(models.UploadedFile.upload_time)
        )
    ).scalars().all()

    file_schemas = [
        schemas.UploadedFileSchema(
            id=str(f.id),
            filename=f.filename,
            filetype=f.filetype,
            size=f.size,
            upload_time=f.upload_time,
        )
        for f in uploaded_files
    ]

    return schemas.SessionHistoryResponse(
        session_id=session_id,
        messages=message_schemas,
        files=file_schemas or None,
    )


# ============================================================================
# Authentication
# ============================================================================

@router.post("/register", response_model=schemas.UserResponse)
async def register(
    request: schemas.RegisterRequest,
    db: AsyncSession = Depends(get_session),
):
    """Register a new user."""
    existing = (
        await db.execute(
            select(models.User).where(models.User.username == request.username)
        )
    ).scalars().first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")

    user = models.User(
        username=request.username,
        password_hash=auth.hash_password(request.password),
    )
    db.add(user)
    try:
        await db.execute(text(get_search_path_sql()))
    except Exception:
        pass
    await db.commit()
    await db.refresh(user)
    return schemas.UserResponse(
        id=str(user.id), username=user.username, created_at=user.created_at
    )


@router.post("/login", response_model=schemas.TokenResponse)
async def login(
    request: schemas.LoginRequest,
    db: AsyncSession = Depends(get_session),
):
    """Authenticate a user and return a JWT token."""
    user = (
        await db.execute(
            select(models.User).where(models.User.username == request.username)
        )
    ).scalars().first()
    if not user or not auth.verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    return schemas.TokenResponse(
        access_token=auth.create_access_token(data={"sub": user.username})
    )


# ============================================================================
# SSE streaming helper (used by StreamHandler)
# ============================================================================

async def stream_events(
    orchestrator: Orchestrator,
    user_message: str,
    user_id: str,
    session_id: str,
    conversation_history: str,
    db: AsyncSession,
    files: Optional[List] = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE events for the stream query mode."""
    try:
        event = schemas_enterprise.StreamingEvent(
            event_type="message_start", data={"turn_start": True}
        )
        yield f"data: {json.dumps(event.dict())}\n\n"

        response = await orchestrator.process_turn(
            user_message=user_message,
            user_id=user_id,
            session_id=session_id,
            conversation_history=conversation_history,
            db=db,
            files=files,
        )

        event = schemas_enterprise.StreamingEvent(
            event_type="intent_detected",
            data={
                "intent": (
                    response.intent.get("domain")
                    if isinstance(response.intent, dict)
                    else response.intent
                ),
                "confidence": (
                    response.intent.get("confidence")
                    if isinstance(response.intent, dict)
                    else getattr(response.intent, "confidence", 0.0)
                ),
                "reasoning": (
                    response.intent.get("reasoning", "")
                    if isinstance(response.intent, dict)
                    else getattr(response.intent, "reasoning", "")
                ),
            },
        )
        yield f"data: {json.dumps(event.dict())}\n\n"

        if response.tool_execution:
            yield f"data: {json.dumps(schemas_enterprise.StreamingEvent(event_type='tool_call', data={'tool': response.tool_execution.tool_name, 'sql': response.tool_execution.sql or ''}).dict())}\n\n"
            yield f"data: {json.dumps(schemas_enterprise.StreamingEvent(event_type='tool_result', data={'rows_returned': response.tool_execution.row_count or 0, 'execution_time_ms': response.tool_execution.execution_time_ms, 'success': not response.tool_execution.error}).dict())}\n\n"

        for char in response.response_text:
            yield f"data: {json.dumps(schemas_enterprise.StreamingEvent(event_type='text_delta', data={'delta': char}).dict())}\n\n"

        yield f"data: {json.dumps(schemas_enterprise.StreamingEvent(event_type='message_end', data={'message_id': response.message_id, 'stop_reason': 'end_turn'}).dict())}\n\n"

    except Exception as exc:
        yield f"data: {json.dumps(schemas_enterprise.StreamingEvent(event_type='message_end', data={'stop_reason': 'error', 'error': str(exc)}).dict())}\n\n"


# ============================================================================
# Main Query Endpoint — thin dispatcher
# ============================================================================

@router.post("/query")
async def query(
    session_id: Optional[str] = Form(None),
    query: str = Form(...),
    message_id: Optional[str] = Form(None),
    mode: str = Form("standard"),
    num_variations: int = Form(5),
    files: Union[List[UploadFile], UploadFile, None] = File(None),
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    Unified query endpoint.  Delegates to one of five mode handlers:
    standard | agentic | stream | viz_update | variations
    """
    # ── Security: prompt injection ────────────────────────────────────────────
    guardian = PromptInjectionGuardian(
        config=GuardianConfig(
            block_threshold=InjectionRiskLevel.HIGH,
            sanitize_threshold=InjectionRiskLevel.MEDIUM,
            enable_pattern_detection=True,
            enable_encoding_detection=True,
            enable_semantic_analysis=True,
        )
    )
    injection_result = await guardian.detect_injection(query)
    if injection_result.is_injection and injection_result.risk_level >= InjectionRiskLevel.HIGH:
        raise HTTPException(
            status_code=400,
            detail=f"Potential prompt injection detected: {injection_result.explanation}",
        )
    if (
        injection_result.risk_level == InjectionRiskLevel.MEDIUM
        and injection_result.sanitized_input
    ):
        query = injection_result.sanitized_input

    # ── Validate mode ─────────────────────────────────────────────────────────
    if mode not in DISPATCH_TABLE:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Must be one of: {', '.join(DISPATCH_TABLE)}",
        )

    # ── Clamp num_variations ──────────────────────────────────────────────────
    num_variations = max(3, min(6, num_variations))

    # ── Normalise file list ───────────────────────────────────────────────────
    def _is_upload(f) -> bool:
        return hasattr(f, "filename") and hasattr(f, "file")

    if _is_upload(files):
        safe_files: list = [files]
    elif isinstance(files, list):
        safe_files = [f for f in files if _is_upload(f)]
    else:
        safe_files = []

    # ── File format validation ────────────────────────────────────────────────
    for f in safe_files:
        if f.filename:
            validate_file_extension(f.filename)

    # ── Session resolution ────────────────────────────────────────────────────
    if not session_id:
        new_sess = models.ChatSession(user_id=current_user.id)
        db.add(new_sess)
        try:
            await db.execute(text(get_search_path_sql()))
        except Exception:
            pass
        await db.commit()
        await db.refresh(new_sess)
        session_id = str(new_sess.id)

    session_uuid = None
    try:
        session_uuid = _uuid.UUID(session_id) if isinstance(session_id, str) else session_id
    except ValueError:
        new_sess = models.ChatSession(user_id=current_user.id)
        db.add(new_sess)
        try:
            await db.execute(text(get_search_path_sql()))
        except Exception:
            pass
        await db.commit()
        await db.refresh(new_sess)
        session_id = str(new_sess.id)
        session_uuid = new_sess.id

    session = (
        await db.execute(
            select(models.ChatSession)
            .where(models.ChatSession.id == session_uuid)
            .where(models.ChatSession.user_id == current_user.id)
        )
    ).scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # ── Security: file scanning ───────────────────────────────────────────────
    if safe_files:
        file_scanner = FileSecurityScanner(
            config=SecurityConfig(
                max_file_size_mb=50,
                scan_for_scripts=True,
                scan_for_macros=True,
                block_threshold=ThreatLevel.HIGH,
            )
        )
        for f in safe_files:
            content = await f.read()
            await f.seek(0)
            result = await file_scanner.scan_file(content=content, filename=f.filename)
            if not result.is_safe:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"File '{f.filename}' blocked (threat: {result.threat_level.value}). "
                        f"Issues: {', '.join(result.threats_found[:3])}"
                    ),
                )
            if result.threat_level in (ThreatLevel.LOW, ThreatLevel.MEDIUM):
                logger.warning(
                    "[FILE_SECURITY] Warning for '%s': %s",
                    f.filename,
                    ", ".join(result.threats_found),
                )

    # ── Streaming mode: return immediately, no message placeholder needed ─────
    if mode == "stream":
        ctx = QueryContext(
            query=query,
            mode=mode,
            session_id=session_id,
            message_id=message_id or "",
            num_variations=num_variations,
            safe_files=safe_files,
            db=db,
            current_user=current_user,
            session=session,
        )
        return await DISPATCH_TABLE["stream"]().execute(ctx)

    # ── Non-streaming modes: create message placeholder for progress tracking ─
    try:
        from ..config import get_search_path_sql
        await db.execute(text(get_search_path_sql()))
    except Exception:
        pass

    from ..services import cancellation_manager, progress_tracker_manager

    if message_id:
        try:
            msg_uuid = _uuid.UUID(message_id)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid message_id format. Must be valid UUID."
            )
    else:
        msg_uuid = _uuid.uuid4()
        message_id = str(msg_uuid)

    try:
        message = models.Message(
            id=msg_uuid,
            session_id=session_id,
            query=query,
            queried_at=datetime.utcnow(),
            response_type="pending",
            response={"status": "processing", "id": message_id},
            responded_at=None,
            feedback=None,
        )
        db.add(message)
        await db.commit()
        await db.refresh(message)

        tracker = progress_tracker_manager.start_tracking(
            message_id, initial_label="Starting query processing"
        )
        token = cancellation_manager.create_token(message_id)
    except Exception as exc:
        logger.error("[QUERY] Error creating placeholder message: %s", exc)
        raise

    try:
        ctx = QueryContext(
            query=query,
            mode=mode,
            session_id=session_id,
            message_id=message_id,
            num_variations=num_variations,
            safe_files=safe_files,
            db=db,
            current_user=current_user,
            session=session,
            message=message,
            tracker=tracker,
            cancellation_manager=cancellation_manager,
        )

        wrapper = await DISPATCH_TABLE[mode]().execute(ctx)

        # ── Ensure session_id echoed ──────────────────────────────────────────
        wrapper.session_id = session_id

        # ── Post-query: question-back engine (skip for chat mode) ────────────
        try:
            from ..services import get_question_back_engine as _get_qb

            _qb_engine = _get_qb()
            _resp = wrapper.response

            # Skip question-back for conversational chat — user wants no follow-ups
            _resp_mode = getattr(_resp, "mode", None) or (
                _resp.get("mode") if isinstance(_resp, dict) else None
            )
            _is_chat_response = _resp_mode in ("chat", "general_chat", "conversational")
            if _is_chat_response:
                logger.debug("[QUESTION-BACK] Skipping for chat response (mode=%s)", _resp_mode)
            else:
                _row_count = 0
                _columns: list = []
                _sql_executed = None
                if hasattr(_resp, "debug") and _resp.debug:
                    _row_count = getattr(_resp.debug, "row_count", 0) or 0
                    _columns = getattr(_resp.debug, "columns", []) or []
                    _sql_executed = getattr(_resp.debug, "sql_executed", None)
                elif hasattr(_resp, "artifacts"):
                    _sql_executed = getattr(_resp.artifacts, "sql", None)
                elif isinstance(_resp, dict):
                    _row_count = _resp.get("debug", {}).get("row_count", 0) or 0
                    _columns = _resp.get("debug", {}).get("columns", []) or []

                _qb_result = await _qb_engine.generate_questions(
                    query_context={
                        "user_query": query,
                        "tool": mode,
                        "sql": _sql_executed,
                    },
                    result_context={
                        "row_count": _row_count,
                        "columns": _columns,
                        "has_filters": bool(
                            _sql_executed and "WHERE" in (_sql_executed or "").upper()
                        ),
                        "is_aggregated": bool(
                            _sql_executed
                            and any(
                                k in (_sql_executed or "").upper()
                                for k in ("GROUP BY", "COUNT(", "SUM(", "AVG(")
                            )
                        ),
                        "limit_applied": bool(
                            _sql_executed and "LIMIT" in (_sql_executed or "").upper()
                        ),
                    },
                )

                if hasattr(_resp, "suggested_questions") and not _resp.suggested_questions:
                    _resp.suggested_questions = _qb_result.questions
                if hasattr(_resp, "suggested_actions") and not _resp.suggested_actions:
                    _resp.suggested_actions = [
                        {
                            "id": a.action_type,
                            "label": a.label,
                            "action_type": a.action_type,
                            "payload": a.parameters,
                        }
                        for a in _qb_result.actions
                    ]
                wrapper.intent = {
                    **(wrapper.intent or {}),
                    "suggested_questions": _qb_result.questions,
                }
        except Exception as qbe:
            logger.debug("[QUESTION-BACK] Generation failed (non-critical): %s", qbe)

        # ── Persist response into placeholder message ──────────────────────────
        response_dict = (
            wrapper.response.model_dump(mode="json")
            if hasattr(wrapper.response, "model_dump")
            else {}
        )
        response_type = getattr(wrapper.response, "type", None) or getattr(
            wrapper.response, "mode", "standard"
        )
        response_dict["id"] = message_id

        message.response_type = response_type
        message.response = response_dict
        message.responded_at = datetime.utcnow()
        flag_modified(message, "response")
        await db.commit()
        await db.refresh(message)

        tracker.complete("Query completed successfully")

        if hasattr(wrapper.response, "id"):
            wrapper.response.id = message_id

        return wrapper

    except Exception as exc:
        if "tracker" in dir() or tracker is not None:
            try:
                tracker.error(f"Query failed: {str(exc)[:100]}")
            except Exception:
                pass

        try:
            await db.rollback()
        except Exception:
            pass

        error_msg = str(exc)[:200]
        if "syntax error" in error_msg.lower():
            error_msg = "The generated SQL had syntax errors. Please rephrase your question."
        elif "does not exist" in error_msg.lower():
            error_msg = "Could not find the requested data. Please try a different query."

        error_response = schemas.StandardResponse(
            intent=query,
            confidence=0.0,
            message=error_msg,
            related_queries=[],
            metadata={"error": True, "type": "error", "mode": mode},
        )

        if message is not None:
            error_dict = error_response.model_dump(mode="json")
            error_dict["id"] = message_id
            message.response_type = "error_response"
            message.response = error_dict
            message.responded_at = datetime.utcnow()
            flag_modified(message, "response")
            try:
                await db.commit()
                await db.refresh(message)
            except Exception:
                try:
                    await db.rollback()
                except Exception:
                    pass
        else:
            try:
                await db.execute(text(get_search_path_sql()))
            except Exception:
                pass
            err_message = models.Message(
                session_id=session_id,
                query=query,
                queried_at=datetime.utcnow(),
                response_type="error_response",
                response=error_response.model_dump(mode="json"),
                responded_at=datetime.utcnow(),
                feedback=None,
            )
            db.add(err_message)
            try:
                await db.commit()
                await db.refresh(err_message)
                message_id = str(err_message.id)
            except Exception:
                try:
                    await db.rollback()
                except Exception:
                    pass

        return schemas.ResponseWrapper(
            success=False,
            response=error_response,
            timestamp=current_timestamp(),
            original_query=query,
            session_id=session_id,
            message_id=message_id,
        )

    finally:
        try:
            cancellation_manager.remove_token(message_id)
        except Exception:
            pass


# ============================================================================
# System / Admin Endpoints
# ============================================================================

@router.post("/admin/fix-message-ids")
async def fix_message_ids(
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """Fix all messages where response.id doesn't match the database message ID."""
    try:
        all_messages = (await db.execute(select(models.Message))).scalars().all()
        updated_count = 0
        for msg in all_messages:
            if isinstance(msg.response, dict) and msg.response.get("id") != str(msg.id):
                updated = dict(msg.response)
                updated["id"] = str(msg.id)
                msg.response = updated
                flag_modified(msg, "response")
                updated_count += 1
        await db.commit()
        return {
            "success": True,
            "message": f"Fixed {updated_count} messages",
            "total_messages": len(all_messages),
            "updated_count": updated_count,
        }
    except Exception as exc:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed: {exc}")


@router.get("/admin/debug-session/{session_id}")
async def debug_session(
    session_id: str,
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """Debug endpoint: show raw DB data for a session."""
    messages = (
        await db.execute(
            select(models.Message)
            .where(models.Message.session_id == session_id)
            .order_by(models.Message.updated_at)
        )
    ).scalars().all()
    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": [
            {
                "db_message_id": str(m.id),
                "response_id_in_db": m.response.get("id") if isinstance(m.response, dict) else None,
                "query": m.query,
                "response_type": m.response_type,
                "id_matches": str(m.id) == m.response.get("id") if isinstance(m.response, dict) else False,
            }
            for m in messages
        ],
    }


@router.get("/capabilities")
async def capabilities():
    """Return supported visualization types and response types."""
    from ..helpers import build_capabilities
    return build_capabilities()


@router.get("/examples")
async def examples():
    """Return example queries for each supported response type."""
    return {
        "standard": "What are your business hours?",
        "data_query": "Show me all transactions from last month",
        "file_query": "Upload and analyze a CSV file",
        "chat": "Tell me about blockchain",
    }


@router.put("/message/{message_id}/feedback")
async def update_message_feedback(
    message_id: str,
    feedback: str = Body(..., embed=True),
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """Update feedback for a specific message (PUT)."""
    try:
        from uuid import UUID
        try:
            msg_uuid = UUID(message_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid message ID format")

        msg = (
            await db.execute(select(models.Message).where(models.Message.id == msg_uuid))
        ).scalars().first()
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found")

        sess = (
            await db.execute(
                select(models.ChatSession)
                .where(models.ChatSession.id == msg.session_id)
                .where(models.ChatSession.user_id == current_user.id)
            )
        ).scalars().first()
        if not sess:
            raise HTTPException(status_code=403, detail="Not authorized")

        msg.feedback = feedback
        await db.commit()
        await db.refresh(msg)
        return {
            "success": True,
            "message_id": str(msg.id),
            "feedback": msg.feedback,
            "updated_at": msg.updated_at.isoformat() if msg.updated_at else None,
        }
    except HTTPException:
        raise
    except Exception as exc:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/messages/{message_id}/progress")
async def stream_message_progress(
    message_id: str,
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """Stream real-time SSE progress updates for a query."""
    import asyncio
    from uuid import UUID
    from ..services import progress_tracker_manager

    async def _events():
        try:
            try:
                msg_uuid = UUID(message_id)
            except ValueError:
                yield f'data: {json.dumps({"step": "error", "label": "Invalid message ID"})}\n\n'
                return

            msg = (
                await db.execute(select(models.Message).where(models.Message.id == msg_uuid))
            ).scalars().first()
            if not msg:
                yield f'data: {json.dumps({"step": "error", "label": "Message not found"})}\n\n'
                return

            sess = (
                await db.execute(
                    select(models.ChatSession)
                    .where(models.ChatSession.id == msg.session_id)
                    .where(models.ChatSession.user_id == current_user.id)
                )
            ).scalars().first()
            if not sess:
                yield f'data: {json.dumps({"step": "error", "label": "Not authorized"})}\n\n'
                return

            if msg.responded_at:
                yield f'data: {json.dumps({"step": "complete", "label": "Complete!", "timestamp": msg.responded_at.isoformat()})}\n\n'
                return

            tracker = progress_tracker_manager.get_tracker(message_id)
            if not tracker:
                await asyncio.sleep(0.5)
                tracker = progress_tracker_manager.get_tracker(message_id)
                if not tracker:
                    yield f'data: {json.dumps({"step": "starting", "label": "Initializing..."})}\n\n'
                    for _ in range(20):
                        await asyncio.sleep(0.5)
                        tracker = progress_tracker_manager.get_tracker(message_id)
                        if tracker:
                            break
                        await db.refresh(msg)
                        if msg.responded_at:
                            yield f'data: {json.dumps({"step": "complete", "label": "Complete!"})}\n\n'
                            return
                    if not tracker:
                        yield f'data: {json.dumps({"step": "complete", "label": "Query processed"})}\n\n'
                        return

            for update in tracker.get_all_updates():
                yield f"data: {json.dumps(update.to_dict())}\n\n"

            update_queue = await progress_tracker_manager.subscribe(message_id)
            try:
                while True:
                    try:
                        update = await asyncio.wait_for(update_queue.get(), timeout=30.0)
                        yield f"data: {json.dumps(update.to_dict())}\n\n"
                        if update.step in ("complete", "error", "cancelled"):
                            break
                    except asyncio.TimeoutError:
                        await db.refresh(msg)
                        if msg.responded_at:
                            yield f'data: {json.dumps({"step": "complete", "label": "Complete!"})}\n\n'
                            break
                        yield ": keepalive\n\n"
            finally:
                progress_tracker_manager.unsubscribe(message_id, update_queue)

        except Exception as exc:
            yield f'data: {json.dumps({"step": "error", "label": str(exc)})}\n\n'

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/messages/{message_id}/stop")
async def stop_message_execution(
    message_id: str,
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """Stop an ongoing query execution."""
    try:
        from uuid import UUID
        from ..services import cancellation_manager, progress_tracker_manager

        try:
            msg_uuid = UUID(message_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid message ID format")

        msg = (
            await db.execute(select(models.Message).where(models.Message.id == msg_uuid))
        ).scalars().first()
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found")

        sess = (
            await db.execute(
                select(models.ChatSession)
                .where(models.ChatSession.id == msg.session_id)
                .where(models.ChatSession.user_id == current_user.id)
            )
        ).scalars().first()
        if not sess:
            raise HTTPException(status_code=403, detail="Not authorized")

        if msg.responded_at:
            return {
                "success": True,
                "stopped": False,
                "message_id": str(msg.id),
                "message": "Query already completed",
                "completed_at": msg.responded_at.isoformat(),
            }

        cancellation_manager.cancel(message_id, reason="User requested cancellation")
        tracker = progress_tracker_manager.get_tracker(message_id)
        if tracker:
            tracker.cancelled("Query execution stopped by user")

        msg.response = {
            "type": "standard",
            "intent": msg.query,
            "confidence": 0.0,
            "message": "Query execution was stopped by user.",
            "metadata": {"cancelled": True, "cancelled_at": datetime.utcnow().isoformat()},
        }
        msg.response_type = "cancelled"
        msg.responded_at = datetime.utcnow()
        flag_modified(msg, "response")
        await db.commit()
        await db.refresh(msg)

        return {"success": True, "stopped": True, "message_id": str(msg.id), "message": "Stopped"}
    except HTTPException:
        raise
    except Exception as exc:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(exc))


@router.patch("/messages/{message_id}/feedback")
async def submit_message_feedback(
    message_id: str,
    feedback_request: schemas.FeedbackRequest,
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """Submit LIKED/DISLIKED feedback for a message."""
    try:
        from uuid import UUID
        try:
            msg_uuid = UUID(message_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid message ID format")

        msg = (
            await db.execute(select(models.Message).where(models.Message.id == msg_uuid))
        ).scalars().first()
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found")

        sess = (
            await db.execute(
                select(models.ChatSession)
                .where(models.ChatSession.id == msg.session_id)
                .where(models.ChatSession.user_id == current_user.id)
            )
        ).scalars().first()
        if not sess:
            raise HTTPException(status_code=403, detail="Not authorized")

        msg.feedback = feedback_request.feedback
        await db.commit()
        await db.refresh(msg)
        return {"success": True, "message_id": str(msg.id), "feedback": msg.feedback}
    except HTTPException:
        raise
    except Exception as exc:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/health", response_model=schemas.HealthResponse)
async def health():
    """Simple health check endpoint."""
    return schemas.HealthResponse(status="ok")
