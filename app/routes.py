"""
GenAI Backend API Routes - NEW ARCHITECTURE

This module defines all API endpoints using the new modular architecture:

ENDPOINTS:
  /register          - Register new user
  /login             - Authenticate and get JWT token
  /new_session       - Create new chat session
  /sessions          - List all sessions for user
  /history/{id}      - Get messages in a session
  /query             - Main unified query endpoint (handles SQL/FILES/CHAT)
  /capabilities      - Get supported visualization types
  /examples          - Get example queries
  /health            - Health check

MAIN QUERY FLOW (/query):
  1. DecisionEngine decides: SQL | FILES | CHAT
  2. Routes to appropriate handler
  3. Returns consistent ResponseWrapper

DATABASE-AGNOSTIC:
  - No hardcoded table/column names
  - Automatic schema discovery
  - Domain concept mapping
  - Semantic table matching
  - Safety validation

All endpoints require JWT authentication (Bearer token).
All responses follow app/schemas.py definitions.
"""

from __future__ import annotations

import copy
import json
from typing import List, Optional, Union, AsyncGenerator, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Body
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from . import auth, models, schemas
from . import schemas_enterprise
from .database import get_session
from .config import get_search_path_sql
from .helpers import current_timestamp, format_conversation_context, make_json_serializable
from .services import handle_dynamic_query
from .services.session_query_handler import execute_with_session_state
from .services.query_handler import build_data_query_response, build_file_query_response
from .services.response_generator import DynamicResponseGenerator
from .services.orchestrator import Orchestrator

# Phase 1: Security integrations
from .services.prompt_injection_guardian import (
    PromptInjectionGuardian,
    InjectionRiskLevel,
    GuardianConfig,
)
from .services.file_security_scanner import (
    FileSecurityScanner,
    ThreatLevel,
    SecurityConfig,
)


# Use HTTPBearer for JWT tokens (improves Swagger UI experience)
bearer_scheme = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_session),
):
    """Verify JWT token and return current user."""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = auth.verify_access_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except Exception:
        raise credentials_exception
    user = await db.execute(
        select(models.User).where(models.User.username == username)
    )
    user = user.scalars().first()
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
    db: AsyncSession = Depends(get_session)
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
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    """List all chat sessions for the current user with session metadata."""
    sessions = (
        await db.execute(
            select(models.ChatSession)
            .where(models.ChatSession.user_id == current_user.id)
            .order_by(models.ChatSession.created_at.desc())  # Most recent first
        )
    ).scalars().all()
    
    session_summaries = []
    for s in sessions:
        # Get first message for title
        first_message = (
            await db.execute(
                select(models.Message)
                .where(models.Message.session_id == s.id)
                .order_by(models.Message.queried_at)
                .limit(1)
            )
        ).scalars().first()
        
        # Get message count
        message_count_result = (
            await db.execute(
                select(models.Message)
                .where(models.Message.session_id == s.id)
            )
        ).scalars().all()
        message_count = len(message_count_result)
        
        # Get last updated time
        last_message = (
            await db.execute(
                select(models.Message)
                .where(models.Message.session_id == s.id)
                .order_by(models.Message.updated_at.desc())
                .limit(1)
            )
        ).scalars().first()
        
        # Generate title from first message
        title = None
        if first_message and first_message.query:
            title = first_message.query[:50] + ("..." if len(first_message.query) > 50 else "")
        
        session_summaries.append(
            schemas.SessionSummary(
                session_id=str(s.id),
                created_at=s.created_at,
                last_updated=last_message.updated_at if last_message else s.created_at,
                title=title or "New Chat",
                message_count=message_count
            )
        )
    
    return schemas.SessionsResponse(user_id=current_user.username, sessions=session_summaries)


@router.get("/history/{session_id}", response_model=schemas.SessionHistoryResponse)
async def get_history(
    session_id: str,
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    """Retrieve chat history for a particular session.
    
    Returns all messages in chronological order, with full response data including
    visualizations, artifacts, and follow-ups for complete ChatGPT-like responses.
    """
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
            # Order by updated_at to preserve actual conversation order
            .order_by(models.Message.updated_at)
        )
    ).scalars().all()

    message_schemas = []
    for m in messages:
        # Get the response data from the database
        response_data = m.response
        
        # Log the original ID for debugging
        original_id = response_data.get('id', 'N/A') if isinstance(response_data, dict) else 'N/A'
        print(f"[HISTORY] Processing message {m.id}, original response.id: {original_id}")
        
        # Create a NEW dict (not just a reference) and update the id field with database message ID
        if isinstance(response_data, dict):
            # Deep copy to ensure we don't modify the cached ORM object
            response_data = json.loads(json.dumps(response_data, default=str))
            # Force update the ID to match the database message ID
            response_data['id'] = str(m.id)
            print(f"[HISTORY] Updated response.id to: {response_data['id']}")
            print(f"[HISTORY] Verification - response_data['id'] is now: {response_data.get('id')}")
        else:
            response_data = {}
        
        msg_schema = schemas.MessageSchema(
            id=str(m.id),  # Include message ID at root level
            response_type=m.response_type,
            query=m.query,
            queried_at=m.queried_at,
            responded_at=m.responded_at,
            response=response_data,  # Full LamaResponse with corrected ID
            created_at=m.updated_at
        )
        print(f"[HISTORY] MessageSchema created, response.id in schema: {msg_schema.response.get('id', 'N/A')}")
        message_schemas.append(msg_schema)
    # Also fetch uploaded files for this session so UI can render file chips
    uploaded_files = (
        await db.execute(
            select(models.UploadedFile)
            .where(models.UploadedFile.session_id == session_id)
            .order_by(models.UploadedFile.upload_time)
        )
    ).scalars().all()

    file_schemas: list[schemas.UploadedFileSchema] = [
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
    db: AsyncSession = Depends(get_session)
):
    """Register a new user."""
    existing_user = await db.execute(
        select(models.User).where(models.User.username == request.username)
    )
    if existing_user.scalars().first():
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_password = auth.hash_password(request.password)
    user = models.User(username=request.username, password_hash=hashed_password)
    db.add(user)
    try:
        await db.execute(text(get_search_path_sql()))
    except Exception:
        pass
    await db.commit()
    await db.refresh(user)

    return schemas.UserResponse(id=str(user.id), username=user.username, created_at=user.created_at)


@router.post("/login", response_model=schemas.TokenResponse)
async def login(
    request: schemas.LoginRequest,
    db: AsyncSession = Depends(get_session)
):
    """Authenticate a user and return a JWT token."""
    user = await db.execute(
        select(models.User).where(models.User.username == request.username)
    )
    user = user.scalars().first()
    if not user or not auth.verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    access_token = auth.create_access_token(data={"sub": user.username})
    return schemas.TokenResponse(access_token=access_token)


# ============================================================================
# Main Query Endpoint (NEW ARCHITECTURE)
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
    """
    Stream response for /query/stream endpoint.
    
    Yields Server-Sent Events for: intent → tool_call → tool_result → text_delta → message_end
    """
    try:
        # Yield message_start event
        event = schemas_enterprise.StreamingEvent(
            event_type="message_start",
            data={"turn_start": True}
        )
        yield f"data: {json.dumps(event.dict())}\n\n"
        
        # Process turn
        response = await orchestrator.process_turn(
            user_message=user_message,
            user_id=user_id,
            session_id=session_id,
            conversation_history=conversation_history,
            db=db,
            files=files,
        )
        
        # Yield intent_detected event
        event = schemas_enterprise.StreamingEvent(
            event_type="intent_detected",
            data={
                "intent": response.intent.get("domain") if isinstance(response.intent, dict) else response.intent,
                "confidence": response.intent.get("confidence") if isinstance(response.intent, dict) else getattr(response.intent, "confidence", 0.0),
                "reasoning": response.intent.get("reasoning", "") if isinstance(response.intent, dict) else getattr(response.intent, "reasoning", ""),
            }
        )
        yield f"data: {json.dumps(event.dict())}\n\n"
        
        # If tool was executed, yield events
        if response.tool_execution:
            event = schemas_enterprise.StreamingEvent(
                event_type="tool_call",
                data={
                    "tool": response.tool_execution.tool_name,
                    "sql": response.tool_execution.sql or "",
                }
            )
            yield f"data: {json.dumps(event.dict())}\n\n"
            
            event = schemas_enterprise.StreamingEvent(
                event_type="tool_result",
                data={
                    "rows_returned": response.tool_execution.row_count or 0,
                    "execution_time_ms": response.tool_execution.execution_time_ms,
                    "success": not response.tool_execution.error,
                }
            )
            yield f"data: {json.dumps(event.dict())}\n\n"
        
        # Yield response text as deltas
        for char in response.response_text:
            event = schemas_enterprise.StreamingEvent(
                event_type="text_delta",
                data={"delta": char}
            )
            yield f"data: {json.dumps(event.dict())}\n\n"
        
        # Yield message_end event
        event = schemas_enterprise.StreamingEvent(
            event_type="message_end",
            data={
                "message_id": response.message_id,
                "stop_reason": "end_turn",
            }
        )
        yield f"data: {json.dumps(event.dict())}\n\n"
    
    except Exception as e:
        error_event = schemas_enterprise.StreamingEvent(
            event_type="message_end",
            data={
                "stop_reason": "error",
                "error": str(e),
            }
        )
        yield f"data: {json.dumps(error_event.dict())}\n\n"

@router.post("/query")
async def query(
    session_id: Optional[str] = Form(None, description="Session ID for the chat session (if absent, a new one will be created)"),
    query: str = Form(..., description="User query or JSON string for viz_update mode"),
    message_id: Optional[str] = Form(None, description="Optional client-generated message ID for progress tracking"),
    mode: str = Form("standard", description="Query mode: 'standard' (pipeline), 'agentic' (multi-tool AI), 'variations' (multiple response styles), 'stream' (real-time SSE), 'viz_update' (update visualization)"),
    num_variations: int = Form(5, description="Number of response variations (3-6, only used in 'variations' mode)"),
    files: Union[List[UploadFile], UploadFile, None] = File(
        None, description="Optional files to upload"
    ),
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    🚀 **UNIFIED QUERY ENDPOINT** - One API for ALL query modes
    
    **MODE OPTIONS:**
    
    1️⃣ **mode="standard"** (Default - Pipeline-based)
       - Uses decision engine routing (SQL/FILES/CHAT)
       - Fast, optimized for simple queries
       - Returns: ResponseWrapper with StandardResponse
    
    2️⃣ **mode="agentic"** (Advanced AI Orchestrator - NEW 100% Implementation)
       - ✅ Master AI Orchestrator with autonomous reasoning
       - ✅ Multi-tool planning: SQL + Files + Python + Charts
       - ✅ Knowledge Graph for relationship discovery
       - ✅ 100% Clarification Engine (10 ambiguity types)
       - ✅ Self-correction loops and retry logic
       - ✅ Python sandbox for computations
       - Use for: Complex multi-source queries like "Compare sales in file with database"
       - Returns: ResponseWrapper with agentic execution plan
    
    3️⃣ **mode="variations"** (Multiple Response Styles)
       - Generates 3-6 different response variations with different tones:
         • Friendly and casual
         • Professional and concise
         • Enthusiastic and engaging
         • Thoughtful and detailed
         • Witty and creative
         • Direct and to-the-point
       - Use for: Chat scenarios where users want style options
       - Returns: ResponseWrapper with variations array
    
    4️⃣ **mode="stream"** (Real-time Streaming - ChatGPT Style)
       - Server-Sent Events (SSE) streaming
       - Events: message_start → intent_detected → tool_call → text_delta → message_end
       - Character-by-character typing effect
       - Use for: Real-time UI streaming like ChatGPT
       - Returns: StreamingResponse (SSE format)
    
    5️⃣ **mode="viz_update"** (Update Visualization - Lazy Render)
       - Update existing visualization with new dimension/metric selections
       - Re-aggregates data on demand without re-running SQL query
       - Query param should be JSON string with: message_id, view, selection
       - Use for: Interactive chart updates (bar → line → pie switching)
       - Returns: ResponseWrapper with updated visualization data
    
    **ARCHITECTURE:**
    - Standard: DecisionEngine → EntityParser → HybridMatcher → SQLValidator
    - Agentic: TaskInterpreter → ClarificationEngine → KnowledgeGraph → ToolRegistry → MasterOrchestrator
    - Variations: DynamicResponseGenerator with multiple tone profiles
    - Stream: Real-time event generation with SSE protocol
    
    Works with ANY database schema - no hardcoding!
    
    Args:
        session_id: Chat session ID (creates new if not provided)
        query: User's natural language question
        mode: Query processing mode (standard/agentic/variations/stream)
        num_variations: Number of variations for 'variations' mode (default 5)
        files: Optional files to analyze
        current_user: Authenticated user
        db: Database session

    Returns:
        - ResponseWrapper (for standard/agentic/variations/viz_update modes)
        - StreamingResponse (for stream mode with SSE)
    """
    # ============================================================================
    # PHASE 1: SECURITY - Prompt Injection Detection
    # ============================================================================
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
            detail=f"Potential prompt injection detected: {injection_result.explanation}. Risk level: {injection_result.risk_level.value}"
        )
    
    # Sanitize if medium risk
    if injection_result.risk_level == InjectionRiskLevel.MEDIUM and injection_result.sanitized_input:
        query = injection_result.sanitized_input
        print(f"[SECURITY] Query sanitized due to medium risk injection patterns")
    
    # Validate mode
    valid_modes = ["standard", "agentic", "variations", "stream", "viz_update"]
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}"
        )
    
    # Clamp num_variations
    num_variations = max(3, min(6, num_variations))
    
    # Filter files to only include UploadFile instances
    def is_upload_file(f):
        return hasattr(f, 'filename') and hasattr(f, 'file')

    if is_upload_file(files):
        safe_files = [files]
    elif isinstance(files, list):
        safe_files = [f for f in files if is_upload_file(f)]
    else:
        safe_files = []
    
    # If no session_id is provided, create a new session silently (first-time prompt UX)
    if not session_id:
        new_session = models.ChatSession(user_id=current_user.id)
        db.add(new_session)
        try:
            await db.execute(text(get_search_path_sql()))
        except Exception:
            pass
        await db.commit()
        await db.refresh(new_session)
        session_id = str(new_session.id)

    # Convert session_id to UUID if it's a string (handle both UUID strings and create new for invalid)
    session_uuid = None
    if session_id:
        try:
            import uuid
            session_uuid = uuid.UUID(session_id) if isinstance(session_id, str) else session_id
        except ValueError:
            # Invalid UUID format - create new session
            new_session = models.ChatSession(user_id=current_user.id)
            db.add(new_session)
            try:
                await db.execute(text(get_search_path_sql()))
            except Exception:
                pass
            await db.commit()
            await db.refresh(new_session)
            session_id = str(new_session.id)
            session_uuid = new_session.id

    # Validate session ownership
    session = (
        await db.execute(
            select(models.ChatSession)
            .where(models.ChatSession.id == session_uuid)
            .where(models.ChatSession.user_id == current_user.id)
        )
    ).scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # ============================================================================
    # PHASE 1: SECURITY - File Security Scanning
    # ============================================================================
    if safe_files:
        file_scanner = FileSecurityScanner(
            config=SecurityConfig(
                max_file_size_mb=50,
                scan_for_scripts=True,
                scan_for_macros=True,
                block_threshold=ThreatLevel.HIGH,
            )
        )
        
        for file in safe_files:
            # Read file content for scanning
            file_content = await file.read()
            await file.seek(0)  # Reset file pointer
            
            # Scan file
            scan_result = await file_scanner.scan_file(
                content=file_content,
                filename=file.filename,
            )
            
            if not scan_result.is_safe:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{file.filename}' blocked due to security threat. "
                           f"Threat level: {scan_result.threat_level.value}. "
                           f"Issues: {', '.join(scan_result.threats_found[:3])}"
                )
            
            # Log warnings for low/medium threats
            if scan_result.threat_level in [ThreatLevel.LOW, ThreatLevel.MEDIUM]:
                print(f"[FILE_SECURITY] Warning for file '{file.filename}': "
                      f"{', '.join(scan_result.threats_found)}")

    # ============================================================================
    # VIZ UPDATE MODE - Update visualization without re-querying
    # ============================================================================
    if mode == "viz_update":
        from .services.aggregation_resolver import AggregationResolver
        
        try:
            # Parse viz update request from query parameter
            viz_request = json.loads(query) if isinstance(query, str) else query
            
            message_id = viz_request.get("message_id")
            view = viz_request.get("view", "bar")
            selection = viz_request.get("selection", {})
            
            if not message_id:
                raise HTTPException(status_code=400, detail="message_id required for viz_update mode")
            
            # Fetch message
            msg_result = await db.execute(
                select(models.Message).where(
                    models.Message.id == message_id,
                    models.Message.session_id == session_id
                )
            )
            message = msg_result.scalars().first()
            
            if not message:
                raise HTTPException(status_code=404, detail="Message not found")
            
            # Parse message response to get visualization and data
            response_data = message.response
            if isinstance(response_data, str):
                response_data = json.loads(response_data)
            
            visualization = response_data.get("visualizations")
            table_data = response_data.get("table_preview", []) or response_data.get("data", [])
            
            if not visualization:
                raise HTTPException(status_code=400, detail="No visualization data found for this message")
            
            # Get transform spec for the selected view
            view_aggregators = visualization.get(view, {})
            transform_spec = view_aggregators.get("transform_template", {})
            
            if not transform_spec:
                raise HTTPException(status_code=400, detail=f"No transform template for view: {view}")
            
            if not table_data:
                raise HTTPException(status_code=400, detail="Insufficient data for aggregation")
            
            # Resolve aggregation
            result = await AggregationResolver.resolve(
                transform_spec=transform_spec,
                selection=selection,
                query_results=table_data,
                db=db
            )
            
            # Wrap in StandardResponse
            viz_response = schemas.StandardResponse(
                type="viz_update",
                intent="Visualization Update",
                confidence=1.0,
                message=result.get("message", "Visualization updated"),
                metadata={
                    "viz_update": True,
                    "view": view,
                    "dataset": result.get("dataset", []),
                    "echarts_option": result.get("echarts_option"),
                    "selection_applied": result.get("selection_applied"),
                    "success": result.get("success", True)
                }
            )
            
            return schemas.ResponseWrapper(
                success=True,
                response=viz_response,
                timestamp=current_timestamp(),
                original_query=query,
                session_id=session_id,
                message_id=message_id,
            )
        
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in query parameter for viz_update mode")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Visualization update failed: {str(e)}")
    
    # ============================================================================
    # STREAMING MODE - Return SSE immediately
    # ============================================================================
    if mode == "stream":
        # Note: For streaming, we'll need to update message after response completes
        # For now, skip message persistence in streaming mode (to be enhanced later)
        
        # Get conversation history for context
        previous_messages = (
            await db.execute(
                select(models.Message)
                .where(models.Message.session_id == session_id)
                .order_by(models.Message.updated_at.desc())
                .limit(5)
            )
        ).scalars().all()
        
        conversation_history = "\n".join([
            f"User: {m.query}\nAssistant: {m.response.get('message', '')[:100]}"
            for m in reversed(previous_messages)
        ])
        
        # Return streaming response
        orchestrator = Orchestrator(db=db)
        return StreamingResponse(
            stream_events(
                orchestrator=orchestrator,
                user_message=query,
                user_id=str(current_user.id),
                session_id=session_id,
                conversation_history=conversation_history,
                db=db,
                files=safe_files,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ============================================================================
    # NON-STREAMING MODES - Process query then persist single message
    # ============================================================================
    try:
        await db.execute(text("SET search_path TO genai, public"))
    except Exception:
        pass

    # ============================================================================
    # CREATE MESSAGE PLACEHOLDER FIRST - For progress tracking
    # ============================================================================
    from uuid import UUID
    from .services import cancellation_manager, progress_tracker_manager, ProgressStep
    
    # Use client-provided message_id or generate new one
    if message_id:
        try:
            # Validate UUID format
            msg_uuid = UUID(message_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid message_id format. Must be valid UUID.")
    else:
        # Generate new UUID if not provided
        import uuid
        msg_uuid = uuid.uuid4()
        message_id = str(msg_uuid)
    
    try:
        # Create message BEFORE processing so progress endpoint can find it
        message = models.Message(
            id=msg_uuid,  # Use the UUID (client-provided or generated)
            session_id=session_id,
            query=query,
            queried_at=datetime.utcnow(),
            response_type="pending",  # Will be updated after processing
            response={"status": "processing", "id": message_id},  # Placeholder with ID
            responded_at=None,  # Will be set after processing
            feedback=None,
        )
        db.add(message)
        await db.commit()
        await db.refresh(message)
        
        print(f"[MESSAGE] Created placeholder message with ID: {message_id}")
        
        # Initialize progress tracking and cancellation token
        tracker = progress_tracker_manager.start_tracking(
            message_id,
            initial_label="Starting query processing..."
        )
        print(f"\n🚀 [PROGRESS] Starting query processing for message {message_id}")
        token = cancellation_manager.create_token(message_id)
        
    except Exception as e:
        print(f"[MESSAGE] Error creating placeholder: {e}")
        raise

    try:
        # ========================================================================
        # ROUTE BY MODE
        # ========================================================================
        
        if mode == "standard":
            # Unified semantic routing (single source of truth)
            wrapper = await execute_with_session_state(
                session_id=session_id,
                user_id=str(current_user.id),
                user_query=query,
                db=db,
                current_user=current_user,
                handler_func=build_data_query_response,
                files=safe_files,
                message_id=message_id,
            )
        
        elif mode == "agentic":
            # NEW: Agentic multi-tool orchestrator
            from .services import create_agentic_handler
            
            # Get conversation history
            from .helpers import extract_assistant_message_text

            previous_messages = (
                await db.execute(
                    select(models.Message)
                    .where(models.Message.session_id == session_id)
                    .where(models.Message.responded_at.is_not(None))
                    .order_by(models.Message.updated_at.desc())
                    .limit(5)
                )
            ).scalars().all()

            conversation_lines = []
            for m in reversed(previous_messages):
                if m.query:
                    conversation_lines.append(f"User: {m.query}")
                assistant_text = extract_assistant_message_text(m.response)
                if assistant_text:
                    conversation_lines.append(f"Assistant: {assistant_text[:500]}")

            conversation_history = "\n".join(conversation_lines)
            
            handler = await create_agentic_handler(db)
            wrapper = await handler.handle_query(
                user_query=query,
                user_id=str(current_user.id),
                session_id=session_id,
                conversation_history=conversation_history,
                uploaded_files=safe_files,
            )
        
        elif mode == "variations":
            # Generate multiple response variations
            from .services.response_generator import create_conversation_state

            state = await create_conversation_state(
                session_id=session_id,
                user_id=str(current_user.id),
                db=db,
                exclude_message_id=message_id,
            )
            response_generator = DynamicResponseGenerator()
            
            variations = await response_generator.generate_multiple_responses(
                query=query,
                query_type="chat",
                db=db,
                session_id=session_id,
                user_id=str(current_user.id),
                conversation_state=state,
                num_responses=num_variations,
            )
            
            response_obj = schemas.StandardResponse(
                type="standard",
                intent=query,
                confidence=0.95,
                message=variations[0] if variations else "I'm here to help!",
                variations=variations if len(variations) > 1 else None,
                related_queries=["Tell me more", "Give me examples", "How does this compare?"],
                metadata={
                    "type": "chat_variations",
                    "mode": "variations",
                    "variations_count": len(variations),
                    "num_requested": num_variations,
                },
            )
            
            wrapper = schemas.ResponseWrapper(
                success=True,
                response=response_obj,
                timestamp=current_timestamp(),
                original_query=query,
                session_id=session_id,
                message_id=message_id,
            )

        # Ensure session_id is present on wrapper
        wrapper.session_id = session_id

        # UPDATE the existing message (created earlier) with the response
        response_dict = wrapper.response.model_dump(mode='json') if hasattr(wrapper.response, 'model_dump') else {}
        response_type = getattr(wrapper.response, 'type', None) or getattr(wrapper.response, 'mode', 'standard')
        
        # Add message_id to response
        if 'id' not in response_dict:
            response_dict['id'] = message_id
        else:
            response_dict['id'] = message_id
        
        # Update the placeholder message with actual response
        message.response_type = response_type
        message.response = response_dict
        message.responded_at = datetime.utcnow()
        flag_modified(message, 'response')  # Mark JSON field as modified
        
        await db.commit()
        await db.refresh(message)
        print(f"[MESSAGE] Updated message {message_id} with response")
        
        # Mark progress as complete
        tracker.complete("Query completed successfully")
        print(f"\n✅ [PROGRESS] Query completed successfully\n")
        
        # Also update the wrapper response for the return value
        if hasattr(wrapper.response, 'id'):
            wrapper.response.id = message_id

        return wrapper

    except Exception as e:
        # Mark progress as error
        if 'tracker' in locals():
            tracker.error(f"Query failed: {str(e)[:100]}")
            print(f"\n❌ [PROGRESS] Query failed: {str(e)[:100]}\n")
        
        try:
            await db.rollback()
        except Exception:
            pass

        error_msg = str(e)[:200]
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
        
        # Update message with error response
        if 'message' in locals() and message:
            error_response_dict = error_response.model_dump(mode="json")
            error_response_dict['id'] = message_id
            
            message.response_type = "error_response"
            message.response = error_response_dict
            message.responded_at = datetime.utcnow()
            flag_modified(message, 'response')
            
            try:
                await db.commit()
                await db.refresh(message)
                print(f"[MESSAGE] Updated message {message_id} with error response")
            except Exception as commit_error:
                print(f"[MESSAGE] Error updating message: {commit_error}")
                try:
                    await db.rollback()
                except Exception:
                    pass
        else:
            # Fallback: message wasn't created, create error message now
            try:
                await db.execute(text(get_search_path_sql()))
            except Exception:
                pass
            
            error_response_dict = error_response.model_dump(mode="json")
            error_message = models.Message(
                session_id=session_id,
                query=query,
                queried_at=datetime.utcnow(),
                response_type="error_response",
                response=error_response_dict,
                responded_at=datetime.utcnow(),
                feedback=None,
            )
            db.add(error_message)
            try:
                await db.commit()
                await db.refresh(error_message)
                message_id = str(error_message.id)
                error_response_dict['id'] = message_id
                error_message.response = error_response_dict
                flag_modified(error_message, 'response')
                await db.commit()
                print(f"[MESSAGE] Created fallback error message with ID: {message_id}")
            except Exception:
                try:
                    await db.rollback()
                except Exception:
                    pass

        wrapper = schemas.ResponseWrapper(
            success=False,
            response=error_response,
            timestamp=current_timestamp(),
            original_query=query,
            session_id=session_id,
            message_id=message_id if 'message_id' in locals() else None,
        )
        
        # Update error response id with database message ID if available
        if 'message_id' in locals() and hasattr(wrapper.response, 'id'):
            wrapper.response.id = message_id
        
        return wrapper
    
    finally:
        # Cleanup cancellation token
        if 'message_id' in locals() and 'cancellation_manager' in locals():
            cancellation_manager.remove_token(message_id)


# ============================================================================
# System Endpoints
# ============================================================================

@router.post("/admin/fix-message-ids")
async def fix_message_ids(
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    """
    Admin endpoint to fix all message IDs in the database.
    This updates all messages where response.id doesn't match the database message ID.
    """
    try:
        # Get all messages
        messages_result = await db.execute(select(models.Message))
        all_messages = messages_result.scalars().all()
        
        updated_count = 0
        for message in all_messages:
            if isinstance(message.response, dict):
                response_id = message.response.get('id')
                db_id = str(message.id)
                
                # If IDs don't match, update the response
                if response_id != db_id:
                    # Create a copy and update the ID
                    updated_response = dict(message.response)
                    updated_response['id'] = db_id
                    message.response = updated_response
                    flag_modified(message, 'response')  # Mark JSON field as modified
                    updated_count += 1
                    print(f"[FIX] Updated message {db_id}: {response_id} -> {db_id}")
        
        # Commit all changes
        await db.commit()
        
        return {
            "success": True,
            "message": f"Fixed {updated_count} messages",
            "total_messages": len(all_messages),
            "updated_count": updated_count
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to fix message IDs: {str(e)}")


@router.get("/admin/debug-session/{session_id}")
async def debug_session(
    session_id: str,
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    """
    Debug endpoint to see raw database data for a session.
    """
    messages = (
        await db.execute(
            select(models.Message)
            .where(models.Message.session_id == session_id)
            .order_by(models.Message.updated_at)
        )
    ).scalars().all()
    
    debug_data = []
    for m in messages:
        debug_data.append({
            "db_message_id": str(m.id),
            "response_id_in_db": m.response.get('id') if isinstance(m.response, dict) else None,
            "query": m.query,
            "response_type": m.response_type,
            "id_matches": str(m.id) == m.response.get('id') if isinstance(m.response, dict) else False
        })
    
    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": debug_data
    }


@router.get("/capabilities")
async def capabilities():
    """Return supported visualization types and response types."""
    from .helpers import build_capabilities
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
    """
    Update feedback for a specific message.
    
    Args:
        message_id: UUID of the message
        feedback: Feedback text (e.g., "thumbs_up", "thumbs_down", or custom text)
        
    Returns:
        Updated message with feedback
    """
    try:
        from uuid import UUID
        # Convert string to UUID for database query
        try:
            msg_uuid = UUID(message_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid message ID format")
        
        # Get the message
        result = await db.execute(
            select(models.Message).where(models.Message.id == msg_uuid)
        )
        message = result.scalars().first()
        
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Verify the message belongs to a session owned by this user
        session_result = await db.execute(
            select(models.ChatSession)
            .where(models.ChatSession.id == message.session_id)
            .where(models.ChatSession.user_id == current_user.id)
        )
        session = session_result.scalars().first()
        
        if not session:
            raise HTTPException(status_code=403, detail="Not authorized to update this message")
        
        # Update feedback
        message.feedback = feedback
        # updated_at will be automatically updated by onupdate trigger
        
        await db.commit()
        await db.refresh(message)
        
        return {
            "success": True,
            "message_id": str(message.id),
            "feedback": message.feedback,
            "updated_at": message.updated_at.isoformat() if message.updated_at else None,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update feedback: {str(e)}")


# ============================================================================
# Message Progress, Stop, and Feedback Endpoints (NEW)
# ============================================================================

@router.get("/messages/{message_id}/progress")
async def stream_message_progress(
    message_id: str,
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    Stream real-time progress updates for a query using Server-Sent Events (SSE).
    
    This endpoint streams live progress updates as the query is being processed.
    Updates come from the ProgressTracker that's updated by the query handler.
    
    Args:
        message_id: UUID of the message to track progress for
        current_user: Authenticated user
        db: Database session
        
    Returns:
        StreamingResponse with SSE events containing progress updates
        
    SSE Event Format:
        data: {"step": "routing", "label": "Determining query type...", "timestamp": "..."}
        data: {"step": "discovering_schema", "label": "Finding relevant tables...", "timestamp": "..."}
        data: {"step": "generating_sql", "label": "Writing SQL query...", "timestamp": "..."}
        data: {"step": "executing_query", "label": "Executing query...", "timestamp": "..."}
        data: {"step": "complete", "label": "Complete!", "timestamp": "..."}
    """
    import asyncio
    from uuid import UUID
    from .services import progress_tracker_manager
    
    async def generate_progress_events():
        """Stream real progress updates from the progress tracker."""
        try:
            # Convert string to UUID for database query
            try:
                msg_uuid = UUID(message_id)
            except ValueError:
                yield f'data: {json.dumps({"step": "error", "label": "Invalid message ID format"})}\n\n'
                return
            
            # Verify message exists and belongs to user's session
            result = await db.execute(
                select(models.Message).where(models.Message.id == msg_uuid)
            )
            message = result.scalars().first()
            
            if not message:
                yield f'data: {json.dumps({"step": "error", "label": "Message not found"})}\n\n'
                return
            
            # Verify session ownership
            session_result = await db.execute(
                select(models.ChatSession)
                .where(models.ChatSession.id == message.session_id)
                .where(models.ChatSession.user_id == current_user.id)
            )
            session = session_result.scalars().first()
            
            if not session:
                yield f'data: {json.dumps({"step": "error", "label": "Not authorized"})}\n\n'
                return
            
            # Check if message already completed
            if message.responded_at:
                yield f'data: {json.dumps({"step": "complete", "label": "Complete!", "timestamp": message.responded_at.isoformat()})}\n\n'
                return
            
            # **Get or subscribe to progress tracker**
            tracker = progress_tracker_manager.get_tracker(message_id)
            
            if not tracker:
                # No tracker yet - query might not have started or already completed
                # Wait a bit and check again
                await asyncio.sleep(0.5)
                tracker = progress_tracker_manager.get_tracker(message_id)
                
                if not tracker:
                    # Still no tracker - send a waiting message
                    yield f'data: {json.dumps({"step": "starting", "label": "Initializing query..."})}\n\n'
                    
                    # Poll for tracker creation (max 10 seconds)
                    for _ in range(20):
                        await asyncio.sleep(0.5)
                        tracker = progress_tracker_manager.get_tracker(message_id)
                        if tracker:
                            break
                        
                        # Check if completed while waiting
                        await db.refresh(message)
                        if message.responded_at:
                            yield f'data: {json.dumps({"step": "complete", "label": "Complete!"})}\n\n'
                            return
                    
                    if not tracker:
                        # Query completed before tracker was created or error occurred
                        yield f'data: {json.dumps({"step": "complete", "label": "Query processed"})}\n\n'
                        return
            
            # **Stream all existing updates first**
            for update in tracker.get_all_updates():
                yield f'data: {json.dumps(update.to_dict())}\n\n'
            
            # **Subscribe to new updates**
            update_queue = await progress_tracker_manager.subscribe(message_id)
            
            try:
                # Stream updates as they arrive
                while True:
                    try:
                        # Wait for next update with timeout
                        update = await asyncio.wait_for(update_queue.get(), timeout=30.0)
                        yield f'data: {json.dumps(update.to_dict())}\n\n'
                        
                        # Check if complete or error
                        if update.step in ["complete", "error", "cancelled"]:
                            break
                    except asyncio.TimeoutError:
                        # Send keepalive or check if completed
                        await db.refresh(message)
                        if message.responded_at:
                            yield f'data: {json.dumps({"step": "complete", "label": "Complete!"})}\n\n'
                            break
                        # Send keepalive ping
                        yield f': keepalive\n\n'
                        continue
            finally:
                # Cleanup subscription
                progress_tracker_manager.unsubscribe(message_id, update_queue)
                
        except Exception as e:
            print(f"[PROGRESS] Error in stream: {e}")
            import traceback
            traceback.print_exc()
            yield f'data: {json.dumps({"step": "error", "label": str(e)})}\n\n'
    
    return StreamingResponse(
        generate_progress_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/messages/{message_id}/stop")
async def stop_message_execution(
    message_id: str,
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    Stop an ongoing message/query execution (like ChatGPT stop button).
    
    This endpoint immediately cancels ongoing query execution by:
    1. Setting a cancellation flag that the query handler checks
    2. Marking progress as cancelled
    3. Updating the message with cancellation response
    
    Args:
        message_id: UUID of the message to stop
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Success status and message details
    """
    try:
        from uuid import UUID
        from .services import cancellation_manager, progress_tracker_manager
        
        # Convert string to UUID for database query
        try:
            msg_uuid = UUID(message_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid message ID format")
        
        # Get the message
        result = await db.execute(
            select(models.Message).where(models.Message.id == msg_uuid)
        )
        message = result.scalars().first()
        
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Verify the message belongs to a session owned by this user
        session_result = await db.execute(
            select(models.ChatSession)
            .where(models.ChatSession.id == message.session_id)
            .where(models.ChatSession.user_id == current_user.id)
        )
        session = session_result.scalars().first()
        
        if not session:
            raise HTTPException(status_code=403, detail="Not authorized to stop this message")
        
        # Check if message already has a response (already completed)
        if message.responded_at:
            return {
                "success": True,
                "stopped": False,
                "message_id": str(message.id),
                "message": "Query already completed",
                "completed_at": message.responded_at.isoformat()
            }
        
        # **1. Signal cancellation to the active query handler**
        cancelled = cancellation_manager.cancel(message_id, reason="User requested cancellation")
        
        # **2. Update progress tracker**
        tracker = progress_tracker_manager.get_tracker(message_id)
        if tracker:
            tracker.cancelled("Query execution stopped by user")
        
        # **3. Update message in database with cancellation response**
        cancellation_response = {
            "type": "standard",
            "intent": message.query,
            "confidence": 0.0,
            "message": "⏹️ Query execution was stopped by user.",
            "metadata": {
                "cancelled": True,
                "cancelled_at": datetime.utcnow().isoformat(),
                "cancellation_reason": "User requested stop"
            }
        }
        
        message.response = cancellation_response
        message.response_type = "cancelled"
        message.responded_at = datetime.utcnow()
        flag_modified(message, 'response')
        
        await db.commit()
        await db.refresh(message)
        
        print(f"[STOP] Message {message_id} cancelled by user")
        
        return {
            "success": True,
            "stopped": True,
            "message_id": str(message.id),
            "message": "Query execution stopped successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        print(f"[STOP] Error stopping message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop message: {str(e)}")


@router.patch("/messages/{message_id}/feedback")
async def submit_message_feedback(
    message_id: str,
    feedback_request: schemas.FeedbackRequest,
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    Submit feedback (LIKED/DISLIKED) for a specific message.
    
    This endpoint matches the frontend API call and uses PATCH method with 
    plural 'messages' path. It updates the feedback field on a message.
    
    Args:
        message_id: UUID of the message
        feedback_request: Feedback value ('LIKED', 'DISLIKED', or null)
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Success status confirmation
    """
    try:
        from uuid import UUID
        # Convert string to UUID for database query
        try:
            msg_uuid = UUID(message_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid message ID format")
        
        # Get the message
        result = await db.execute(
            select(models.Message).where(models.Message.id == msg_uuid)
        )
        message = result.scalars().first()
        
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Verify the message belongs to a session owned by this user
        session_result = await db.execute(
            select(models.ChatSession)
            .where(models.ChatSession.id == message.session_id)
            .where(models.ChatSession.user_id == current_user.id)
        )
        session = session_result.scalars().first()
        
        if not session:
            raise HTTPException(status_code=403, detail="Not authorized to update this message")
        
        # Update feedback
        message.feedback = feedback_request.feedback
        
        await db.commit()
        await db.refresh(message)
        
        print(f"[FEEDBACK] Message {message_id} feedback: {feedback_request.feedback}")
        
        # Return minimal response (frontend expects void)
        return {
            "success": True,
            "message_id": str(message.id),
            "feedback": message.feedback
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        print(f"[FEEDBACK] Error updating feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update feedback: {str(e)}")


@router.get("/health", response_model=schemas.HealthResponse)
async def health():
    """Simple health check endpoint."""
    return schemas.HealthResponse(status="ok")
