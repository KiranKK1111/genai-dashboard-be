"""
Orchestrator for streaming responses.

Provides the Orchestrator class used by /query/stream endpoint for
Server-Sent Events streaming. Wraps the existing query handlers.
"""

from __future__ import annotations

from typing import Optional, List, Any, Dict
from dataclasses import dataclass
import json

from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import UploadFile

from .query_orchestrator import handle_dynamic_query
from .. import schemas


@dataclass
class ToolExecution:
    """Tool execution result for streaming."""
    tool_name: str
    sql: Optional[str] = None
    row_count: Optional[int] = None
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class IntentInfo:
    """Intent information."""
    name: Any  # schemas.QueryType
    confidence: float
    reasoning: str


@dataclass
class ProcessTurnResult:
    """Result of orchestrator.process_turn()."""
    intent: Dict[str, Any]
    tool_execution: Optional[ToolExecution]
    response_text: str
    message_id: str


class Orchestrator:
    """
    Orchestrator for streaming responses.
    
    Provides structured output for /query/stream endpoint.
    """
    
    def __init__(self, db: Optional[AsyncSession] = None):
        self._db = db

    async def process_turn(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
        conversation_history: str,
        db: Optional[AsyncSession] = None,
        files: Optional[List[UploadFile]] = None,
    ) -> ProcessTurnResult:
        """
        Process a single turn of conversation.
        
        Args:
            user_message: User's query
            session_id: Session ID
            conversation_history: Previous conversation context
            db: Database session
            files: Optional uploaded files
        
        Returns:
            ProcessTurnResult with intent, tool execution, and response text
        """
        try:
            active_db = db or self._db
            if active_db is None:
                raise ValueError("Database session is required")

            # Use existing query handler
            response_wrapper = await handle_dynamic_query(
                user_message=user_message,
                db=active_db,
                user_id=user_id,
                session_id=session_id,
                uploaded_files=files,
                conversation_history=conversation_history,
            )

            # Intent for streaming events
            intent = response_wrapper.intent or {
                "domain": "unknown",
                "confidence": 0.0,
                "reasoning": "",
            }

            response = response_wrapper.response

            # Extract tool execution if SQL was generated (best-effort)
            tool_execution = None

            try:
                if getattr(response, "type", None) == "data_query":
                    artifacts = getattr(response, "artifacts", None)
                    if artifacts:
                        tool_execution = ToolExecution(
                            tool_name="sql",
                            sql=getattr(artifacts, "sql_generated", None),
                            row_count=getattr(artifacts, "row_count", None),
                            execution_time_ms=getattr(artifacts, "execution_time_ms", None),
                        )
            except Exception:
                tool_execution = None
            
            # Build response text
            response_text = getattr(response, "message", "") or ""
            message_id = str(response_wrapper.message_id or response_wrapper.timestamp or "")
            
            return ProcessTurnResult(
                intent=intent,
                tool_execution=tool_execution,
                response_text=response_text,
                message_id=str(message_id),
            )
            
        except Exception as e:
            # Return error response
            print(f"[ERROR] Orchestrator.process_turn failed: {e}")
            
            class Intent:
                def __init__(self):
                    self.confidence = 0.0
                    self.reasoning = str(e)
            
            return ProcessTurnResult(
                intent={"domain": "error", "confidence": 0.0, "reasoning": str(e)},
                tool_execution=None,
                response_text=f"Error: {str(e)}",
                message_id="error",
            )
