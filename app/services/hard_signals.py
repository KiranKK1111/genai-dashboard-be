"""
HARD SIGNALS EXTRACTOR (Zero Hardcoding)

Extracts deterministic, objective signals from session state.
These signals are schema-agnostic and domain-neutral facts.

No heuristics. No keyword matching. Just facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .. import models
from .router_decision import Tool

logger = logging.getLogger(__name__)


@dataclass
class HardSignals:
    """
    Objective, deterministic facts about session state.
    
    These are NOT heuristics or inferences—just binary/numeric facts.
    """
    
    # File context
    files_uploaded_this_turn: bool = False
    has_files_in_session: bool = False
    file_count: int = 0
    
    # Database context  
    db_available: bool = True  # Could check connectivity; default True
    
    # Prior execution context
    last_tool_used: Optional[Tool] = None
    last_tool_used_at: Optional[datetime] = None
    has_last_sql: bool = False
    has_last_file_context: bool = False
    
    # Time context
    seconds_since_last_turn: Optional[int] = None
    
    # Session lifecycle
    is_new_session: bool = True
    prior_turns_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_uploaded_this_turn": self.files_uploaded_this_turn,
            "has_files_in_session": self.has_files_in_session,
            "file_count": self.file_count,
            "db_available": self.db_available,
            "last_tool_used": self.last_tool_used.value if self.last_tool_used else None,
            "last_tool_used_at": self.last_tool_used_at.isoformat() if self.last_tool_used_at else None,
            "has_last_sql": self.has_last_sql,
            "has_last_file_context": self.has_last_file_context,
            "seconds_since_last_turn": self.seconds_since_last_turn,
            "is_new_session": self.is_new_session,
            "prior_turns_count": self.prior_turns_count,
        }


class HardSignalsExtractor:
    """
    Extract deterministic signals from session state and database.
    
    KEY PRINCIPLE: No domain assumptions. No hardcoding. Just facts.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def extract(
        self,
        session: models.ChatSession,
        current_request_has_files: bool = False,
    ) -> HardSignals:
        """
        Extract hard signals from a session.
        
        Args:
            session: ChatSession object
            current_request_has_files: Whether user uploaded files in this turn
            
        Returns:
            HardSignals with deterministic facts
        """
        
        # File context (deterministic)
        files_uploaded_this_turn = current_request_has_files
        has_files_in_session = bool(session.files) if session.files else False
        file_count = len(session.files) if session.files else 0
        
        # Database availability (could check here if you have a health check)
        db_available = True
        
        # Prior execution context (from session_state JSON or tool_calls_log)
        last_tool_used = None
        last_tool_used_at = None
        has_last_sql = False
        has_last_file_context = False
        
        if session.session_state:
            # Extract from stored session state
            tool_str = session.session_state.get("tool_used")
            if tool_str:
                try:
                    last_tool_used = Tool(tool_str)
                except ValueError:
                    pass
            
            # Check for SQL artifacts
            artifacts = session.session_state.get("artifacts", {})
            has_last_sql = bool(artifacts.get("sql"))
            has_last_file_context = bool(artifacts.get("file_ids"))
            
            # Timestamp of last state update
            if session.state_updated_at:
                last_tool_used_at = session.state_updated_at
        
        # Time since last turn
        seconds_since_last_turn = None
        if session.state_updated_at:
            delta = datetime.utcnow() - session.state_updated_at
            seconds_since_last_turn = int(delta.total_seconds())
        
        # Session lifecycle
        is_new_session = (session.session_state is None) or (len(session.messages or []) == 0)
        prior_turns_count = len(session.messages or []) if session.messages else 0
        
        return HardSignals(
            files_uploaded_this_turn=files_uploaded_this_turn,
            has_files_in_session=has_files_in_session,
            file_count=file_count,
            db_available=db_available,
            last_tool_used=last_tool_used,
            last_tool_used_at=last_tool_used_at,
            has_last_sql=has_last_sql,
            has_last_file_context=has_last_file_context,
            seconds_since_last_turn=seconds_since_last_turn,
            is_new_session=is_new_session,
            prior_turns_count=prior_turns_count,
        )
