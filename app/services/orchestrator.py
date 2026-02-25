"""
Orchestrator for streaming responses.

Provides the Orchestrator class used by /query/stream endpoint for
Server-Sent Events streaming. Wraps the existing query handlers.
"""

from __future__ import annotations

from typing import Optional, List, Any
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
    intent: IntentInfo
    tool_execution: Optional[ToolExecution]
    response_text: str
    message_id: str


class Orchestrator:
    """
    Orchestrator for streaming responses.
    
    Provides structured output for /query/stream endpoint.
    """
    
    async def process_turn(
        self,
        user_message: str,
        session_id: str,
        conversation_history: str,
        db: AsyncSession,
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
            # Use existing query handler
            response_wrapper = await handle_dynamic_query(
                user_message=user_message,
                db=db,
                user_id="streaming-user",  # Placeholder
                session_id=session_id,
                uploaded_files=files,
                conversation_history=conversation_history,
            )
            
            # Extract intent from response
            response = response_wrapper.response
            intent_str = getattr(response, 'intent', 'unknown')
            
            # Create intent info
            from .. import schemas
            try:
                query_type = schemas.QueryType(intent_str.lower())
            except (ValueError, AttributeError):
                query_type = schemas.QueryType.chat
            
            # Create intent with proper structure
            class Intent:
                def __init__(self, query_type):
                    self.name = type('obj', (object,), {
                        'value': query_type.value
                    })()
                    self.value = query_type.value
            
            intent = Intent(query_type)
            intent.confidence = getattr(response, 'confidence', 0.8)
            intent.reasoning = getattr(response, 'message', '')
            
            # Extract tool execution if SQL was generated
            tool_execution = None
            if intent_str == 'sql' and hasattr(response, 'datasets'):
                metadata = getattr(response, 'metadata', {})
                if isinstance(metadata, dict):
                    tool_execution = ToolExecution(
                        tool_name="sql",
                        sql=metadata.get('sql', ''),
                        row_count=metadata.get('row_count', 0),
                        execution_time_ms=metadata.get('execution_time_ms'),
                    )
            
            # Build response text
            response_text = getattr(response, 'message', '')
            if not response_text and hasattr(response, 'intent'):
                response_text = f"Processed {intent_str} query"
            
            message_id = getattr(response_wrapper, 'timestamp', '')
            
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
                    self.name = type('obj', (object,), {'value': 'chat'})()
                    self.confidence = 0.0
                    self.reasoning = str(e)
            
            return ProcessTurnResult(
                intent=Intent(),
                tool_execution=None,
                response_text=f"Error: {str(e)}",
                message_id="error",
            )
