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

import json
from typing import List, Optional, Union, AsyncGenerator, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Body
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from . import auth, models, schemas
from . import schemas_enterprise
from .database import get_session
from .config import get_search_path_sql
from .helpers import current_timestamp, format_conversation_context, make_json_serializable
from .services import handle_dynamic_query
from .services.session_query_handler import execute_with_session_state
from .services.query_handler import build_data_query_response
from .services.response_generator import DynamicResponseGenerator
from .services.orchestrator import Orchestrator


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
    """List all chat sessions for the current user."""
    sessions = (
        await db.execute(
            select(models.ChatSession).where(models.ChatSession.user_id == current_user.id)
        )
    ).scalars().all()
    session_summaries = [
        schemas.SessionSummary(session_id=str(s.id), created_at=s.created_at) for s in sessions
    ]
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
            .order_by(models.Message.queried_at)
        )
    ).scalars().all()

    message_schemas = [
        schemas.MessageSchema(
            response_type=m.response_type,
            query=m.query,
            queried_at=m.queried_at,
            responded_at=m.responded_at,
            response=m.response,  # Full LamaResponse stored as JSON
            created_at=m.created_at
        ) for m in messages
    ]
    return schemas.SessionHistoryResponse(session_id=session_id, messages=message_schemas)


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
            session_id=session_id,
            conversation_history=conversation_history,
            db=db,
            files=files,
        )
        
        # Yield intent_detected event
        event = schemas_enterprise.StreamingEvent(
            event_type="intent_detected",
            data={
                "intent": response.intent.name.value,
                "confidence": response.intent.confidence,
                "reasoning": response.intent.reasoning or "",
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

@router.post("/query", response_model=schemas.ResponseWrapper)
async def query(
    session_id: str = Form(..., description="Session ID for the chat session"),
    query: str = Form(..., description="User query"),
    files: Union[List[UploadFile], UploadFile, None] = File(
        None, description="Optional files to upload"
    ),
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    🚀 Unified Query Endpoint - Handles ALL query types intelligently

    This endpoint uses the new modular architecture to:
    1. Decide what to do (SQL / FILE / CHAT) using DecisionEngine
    2. Parse user intent using EntityParser
    3. Discover schema using SchemaCatalog
    4. Match tables/columns using HybridMatcher
    5. Validate safety using SQLSafetyValidator
    6. Execute and format results

    Works with ANY database schema - no hardcoding!

    Args:
        session_id: Chat session ID
        query: User's natural language question
        files: Optional files to analyze
        current_user: Authenticated user
        db: Database session

    Returns:
        ResponseWrapper with intelligent decision routing
    """
    # Validate session ownership
    session = (
        await db.execute(
            select(models.ChatSession)
            .where(models.ChatSession.id == session_id)
            .where(models.ChatSession.user_id == current_user.id)
        )
    ).scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Filter files to only include UploadFile instances
    def is_upload_file(f):
        return hasattr(f, 'filename') and hasattr(f, 'file')

    if is_upload_file(files):
        safe_files = [files]
    elif isinstance(files, list):
        safe_files = [f for f in files if is_upload_file(f)]
    else:
        safe_files = []

    # Persist user message
    user_message = models.Message(
        session_id=session_id,
        response_type="user_query",
        query=query,
        response={},
        queried_at=datetime.utcnow()
    )
    db.add(user_message)
    try:
        await db.execute(text("SET search_path TO genai, public"))
    except Exception:
        pass
    await db.commit()

    try:
        # ===== ChatGPT-STYLE SESSION STATE MANAGEMENT =====
        # This wrapper ensures:
        # 1. State is loaded from DB before processing
        # 2. Follow-up type is classified BEFORE SQL generation
        # 3. State is merged based on classification
        # 4. Updated state is persisted back to DB
        wrapper = await execute_with_session_state(
            session_id=session_id,
            user_id=str(current_user.id),
            user_query=query,
            db=db,
            current_user=current_user,
            handler_func=build_data_query_response,
            files=safe_files,  # Pass uploaded files to decision engine
        )

        # Persist assistant message - Store the complete LamaResponse
        # Convert wrapper.response to dict for JSON storage
        response_dict = wrapper.response.model_dump(mode='json') if hasattr(wrapper.response, 'model_dump') else {
            "type": wrapper.response.type if hasattr(wrapper.response, 'type') else "standard",
            "message": wrapper.response.message if hasattr(wrapper.response, 'message') else "",
        }

        try:
            await db.execute(text(get_search_path_sql()))
        except Exception:
            pass

        assistant_message = models.Message(
            session_id=session_id,
            response_type=wrapper.response.type if hasattr(wrapper.response, 'type') else "standard",
            query=None,  # Response doesn't have a query
            response=response_dict,  # Store full response as JSON
            responded_at=datetime.utcnow()
        )
        db.add(assistant_message)
        await db.commit()

        return wrapper

    except Exception as e:
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
            metadata={"error": True, "type": "error"},
        )
        return schemas.ResponseWrapper(
            success=False,
            response=error_response,
            timestamp=current_timestamp(),
            original_query=query,
        )


@router.post("/query/variations")
async def query_with_variations(
    session_id: str = Form(..., description="Session ID for the chat session"),
    query: str = Form(..., description="User query"),
    num_variations: int = Form(5, description="Number of response variations (3-6)"),
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    ✨ Get Multiple Response Variations - Returns 5-6 different response options

    This endpoint generates multiple distinct response variations for the same query,
    allowing users to choose which response style they prefer.

    Each response has a different tone, perspective, or approach:
    - Friendly and casual
    - Professional and concise
    - Enthusiastic and engaging
    - Thoughtful and detailed
    - Witty and creative
    - Direct and to-the-point

    Args:
        session_id: Chat session ID
        query: User's natural language question
        num_variations: Number of variations to generate (default 5, allowed 3-6)
        current_user: Authenticated user
        db: Database session

    Returns:
        ResponseWrapper with list of variations 
    """
    from .services.session_state_manager import SessionStateManager
    
    # Clamp num_variations
    num_variations = max(3, min(6, num_variations))
    
    # Validate session ownership
    session = (
        await db.execute(
            select(models.ChatSession)
            .where(models.ChatSession.id == session_id)
            .where(models.ChatSession.user_id == current_user.id)
        )
    ).scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Persist user message
    user_message = models.Message(
        session_id=session_id,
        response_type="user_query",
        query=query,
        response={},
        queried_at=datetime.utcnow()
    )
    db.add(user_message)
    try:
        await db.execute(text(get_search_path_sql()))
    except Exception:
        pass
    await db.commit()

    try:
        # Import ConversationState from response_generator
        from .services.response_generator import ConversationState
        
        # Create a simple conversation state
        state = ConversationState(session_id=session_id, user_id=str(current_user.id))
        
        # Generate multiple response variations using response generator
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
        
        # Create response with variations for StandardResponse
        response_obj = schemas.StandardResponse(
            type="standard",
            intent=query,
            confidence=0.95,
            message=variations[0] if variations else "I'm here to help!",
            variations=variations if len(variations) > 1 else None,
            related_queries=["Tell me more", "Give me examples", "How does this compare?"],
            metadata={
                "type": "chat_variations",
                "variations_count": len(variations),
                "num_requested": num_variations,
            },
        )
        
        # Persist assistant message with variations
        try:
            await db.execute(text(get_search_path_sql()))
        except Exception:
            pass

        response_dict = {
            "type": "chat_variations",
            "intent": query,
            "confidence": 0.95,
            "message": variations[0] if variations else "",
            "variations": variations,
            "metadata": {
                "type": "chat_variations",
                "variations_count": len(variations),
                "num_requested": num_variations,
            }
        }
        
        assistant_message = models.Message(
            session_id=session_id,
            response_type="assistant_chat_variations",
            query=None,
            response=response_dict,
            responded_at=datetime.utcnow()
        )
        db.add(assistant_message)
        await db.commit()

        return schemas.ResponseWrapper(
            success=True,
            response=response_obj,
            timestamp=int(datetime.utcnow().timestamp() * 1000),
            original_query=query,
        )

    except Exception as e:
        try:
            await db.rollback()
        except Exception:
            pass

        error_msg = f"Error generating variations: {str(e)[:200]}"
        error_response = schemas.StandardResponse(
            type="standard",
            intent=query,
            confidence=0.0,
            message=error_msg,
            related_queries=[],
            metadata={"error": True, "type": "error"},
        )
        return schemas.ResponseWrapper(
            success=False,
            response=error_response,
            timestamp=int(datetime.utcnow().timestamp() * 1000),
            original_query=query,
        )


@router.post("/query/stream")
async def query_stream(
    session_id: str = Form(..., description="Session ID for the chat session"),
    query: str = Form(..., description="User query"),
    files: Union[List[UploadFile], UploadFile, None] = File(
        None, description="Optional files to upload"
    ),
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    🚀 Streaming Query Endpoint - ChatGPT-style streaming responses

    Like OpenAI's ChatGPT, this endpoint does NOT wait for the full response.
    Instead, it streams events to the client as they complete:

    1. message_start - Turn started
    2. intent_detected - User intent recognized
    3. tool_call (optional) - SQL query being executed
    4. tool_result (optional) - Query results received
    5. text_delta - Individual characters of response (real-time typing effect)
    6. message_end - Response complete

    Client receives events via Server-Sent Events (SSE).
    Each line is JSON formatted as: data: {...}

    Args:
        session_id: Chat session ID
        query: User's natural language question
        files: Optional files to analyze
        current_user: Authenticated user
        db: Database session

    Returns:
        StreamingResponse with Server-Sent Events
    """
    # Validate session ownership
    session = (
        await db.execute(
            select(models.ChatSession)
            .where(models.ChatSession.id == session_id)
            .where(models.ChatSession.user_id == current_user.id)
        )
    ).scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Filter files
    def is_upload_file(f):
        return hasattr(f, 'filename') and hasattr(f, 'file')

    if is_upload_file(files):
        safe_files = [files]
    elif isinstance(files, list):
        safe_files = [f for f in files if is_upload_file(f)]
    else:
        safe_files = []

    # Persist user message
    user_message = models.Message(
        session_id=session_id,
        response_type="user_query",
        query=query,
        response={},
        queried_at=datetime.utcnow()
    )
    db.add(user_message)
    try:
        await db.execute(text("SET search_path TO genai, public"))
    except Exception:
        pass
    await db.commit()

    # Retrieve conversation history
    previous_messages = (
        await db.execute(
            select(models.Message)
            .where(models.Message.session_id == session_id)
            .order_by(models.Message.created_at)
        )
    ).scalars().all()
    if previous_messages and previous_messages[-1].id == user_message.id:
        previous_messages = previous_messages[:-1]

    conversation_history = format_conversation_context(previous_messages)

    # Create orchestrator and stream
    orchestrator = Orchestrator()
    
    return StreamingResponse(
        stream_events(
            orchestrator=orchestrator,
            user_message=query,
            session_id=session_id,
            conversation_history=conversation_history,
            db=db,
            files=safe_files if safe_files else None,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ============================================================================
# On-Demand Visualization Updates (Lazy-Render Pattern)
# ============================================================================

@router.post("/viz/update")
async def update_visualization(
    request: Dict[str, Any] = Body(...),
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    Update visualization by applying user selections and re-aggregating data.
    
    Used for lazy-render pattern: when user selects bar/line/pie chart,
    backend computes aggregation on demand with user's selected dimensions/metrics.
    
    Request:
    {
        "session_id": "session-uuid",
        "message_id": "message-uuid",
        "chart_id": "bar|line|pie",
        "view": "bar|line|pie|table",
        "selection": {
            "x_dimension": {"field": "gender"},
            "y_metric": {"op": "count", "field": "*", "as": "count"},
            "series_dimension": null,
            "top_k": 10,
            "sort": "metric_desc"
        }
    }
    
    Response:
    {
        "type": "viz_update",
        "dataset": [...],
        "echarts_option": {...},
        "selection_applied": {...},
        "success": true,
        "message": "Aggregated 10 groups"
    }
    """
    from .services.aggregation_resolver import AggregationResolver
    from sqlalchemy import select, and_
    
    try:
        # Extract request params
        session_id = request.get("session_id")
        message_id = request.get("message_id")
        view = request.get("view", "bar")
        selection = request.get("selection", {})
        
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required")
        
        # Fetch session
        session_record = await db.execute(
            select(models.ChatSession).where(
                and_(
                    models.ChatSession.session_id == session_id,
                    models.ChatSession.user_id == current_user.id
                )
            )
        )
        session_record = session_record.scalars().first()
        
        if not session_record:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Fetch message if message_id provided
        message = None
        visualization = None
        dataset = []
        
        if message_id:
            msg_result = await db.execute(
                select(models.Message).where(
                    and_(
                        models.Message.message_id == message_id,
                        models.Message.session_id == session_record.id
                    )
                )
            )
            message = msg_result.scalars().first()
            
            if message and message.response:
                # Parse response to get visualization and original data
                response_data = message.response
                if isinstance(response_data, str):
                    response_data = json.loads(response_data)
                
                # Extract visualization spec
                visualization = response_data.get("visualizations")
                
                # Extract original query results if available
                table_data = response_data.get("table_preview", [])
                if not table_data and "data" in response_data:
                    table_data = response_data.get("data", [])
        
        # If no visualization found, fail gracefully
        if not visualization:
            return {
                "type": "viz_update",
                "success": False,
                "message": "No visualization data found for this message"
            }
        
        # Get aggregators for the view
        view_aggregators = visualization.get(view, {})
        transform_spec = view_aggregators.get("transform_template", {})
        
        if not transform_spec:
            return {
                "type": "viz_update",
                "success": False,
                "message": f"No transform template for view: {view}"
            }
        
        # If no table data, we can't aggregate
        if not table_data:
            return {
                "type": "viz_update",
                "success": False,
                "message": "Insufficient data for aggregation"
            }
        
        # Resolve aggregation
        result = await AggregationResolver.resolve(
            transform_spec=transform_spec,
            selection=selection,
            query_results=table_data,
            db=db
        )
        
        return {
            "type": "viz_update",
            **result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /viz/update: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization update failed: {str(e)}")


# ============================================================================
# System Endpoints
# ============================================================================

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


@router.get("/health", response_model=schemas.HealthResponse)
async def health():
    """Simple health check endpoint."""
    return schemas.HealthResponse(status="ok")
