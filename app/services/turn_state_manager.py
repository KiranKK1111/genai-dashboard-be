"""
Turn State Manager - Persists and retrieves structured session state.

This service is responsible for:
1. Saving TurnState after each tool execution
2. Loading last TurnState for a session
3. Managing turn_id counter
4. Converting between Pydantic and database models

Together with SemanticIntentRouter, this enables:
- Reliable follow-up detection (not regex hacks)
- Complete session memory
- Deterministic query modifications (using QueryPlan AST not string manipulation)
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Dict, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, text

from .. import models, schemas

logger = logging.getLogger(__name__)


class TurnStateManager:
    """Manages persistent turn state for sessions."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def save_turn_state(
        self,
        session_id: str,
        user_query: str,
        assistant_summary: str,
        tool_used: schemas.Tool,
        artifacts: Dict[str, Any],
        confidence: float,
    ) -> models.TurnState:
        """
        Save a turn state after tool execution.
        
        Args:
            session_id: Session ID
            user_query: User's query text
            assistant_summary: Short summary of what was done
            tool_used: Tool that was used (RUN_SQL, ANALYZE_FILE, CHAT, MIXED)
            artifacts: Tool execution artifacts (sql, tables, filters, etc.)
            confidence: Router confidence 0-1
            
        Returns:
            Created TurnState model
        """
        # Get next turn_id
        turn_id = await self._get_next_turn_id(session_id)
        
        # Create TurnState record
        turn_state = models.TurnState(
            session_id=UUID(session_id),
            turn_id=turn_id,
            user_query=user_query,
            assistant_summary=assistant_summary,
            tool_used=tool_used.value,
            artifacts=artifacts,
            confidence=confidence,
        )
        
        self.db.add(turn_state)
        await self.db.flush()  # Ensure ID is generated
        
        logger.info(
            f"Saved TurnState {turn_id} for session {session_id}: "
            f"tool={tool_used.value}, confidence={confidence:.2f}"
        )
        
        return turn_state
    
    async def get_last_turn_state(
        self,
        session_id: str,
    ) -> Optional[models.TurnState]:
        """
        Get the most recent TurnState for a session.
        
        Returns None if no turn states exist.
        """
        stmt = (
            select(models.TurnState)
            .where(models.TurnState.session_id == UUID(session_id))
            .order_by(desc(models.TurnState.turn_id))
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalars().first()
    
    async def get_turn_state(
        self,
        session_id: str,
        turn_id: int,
    ) -> Optional[models.TurnState]:
        """Get a specific TurnState by turn_id."""
        stmt = select(models.TurnState).where(
            models.TurnState.session_id == UUID(session_id),
            models.TurnState.turn_id == turn_id,
        )
        result = await self.db.execute(stmt)
        return result.scalars().first()
    
    async def _get_next_turn_id(self, session_id: str) -> int:
        """Calculate the next turn_id for a session."""
        stmt = select(models.TurnState).where(
            models.TurnState.session_id == UUID(session_id)
        ).order_by(desc(models.TurnState.turn_id)).limit(1)
        
        result = await self.db.execute(stmt)
        last_turn = result.scalars().first()
        
        return (last_turn.turn_id + 1) if last_turn else 1
    
    def turn_state_to_dict(self, turn_state: models.TurnState) -> Dict[str, Any]:
        """Convert TurnState model to dict for session_state storage."""
        return {
            "turn_id": turn_state.turn_id,
            "user_query": turn_state.user_query,
            "assistant_summary": turn_state.assistant_summary,
            "tool_used": turn_state.tool_used,
            "artifacts": turn_state.artifacts,
            "confidence": turn_state.confidence,
            "created_at": turn_state.created_at.isoformat() if turn_state.created_at else None,
        }
    
    def artifacts_to_pydantic(
        self,
        artifacts_dict: Dict[str, Any],
    ) -> schemas.TurnStateArtifacts:
        """Convert raw artifacts dict to Pydantic schema."""
        return schemas.TurnStateArtifacts(
            tool_used=artifacts_dict.get("tool_used", "UNKNOWN"),
            sql=artifacts_dict.get("sql"),
            sql_plan_json=artifacts_dict.get("sql_plan_json"),
            tables=artifacts_dict.get("tables"),
            filters=artifacts_dict.get("filters"),
            grouping=artifacts_dict.get("grouping"),
            having=artifacts_dict.get("having"),
            order_by=artifacts_dict.get("order_by"),
            limit=artifacts_dict.get("limit"),
            result_schema=artifacts_dict.get("result_schema"),
            row_count=artifacts_dict.get("row_count"),
            result_sample=artifacts_dict.get("result_sample"),
            file_ids=artifacts_dict.get("file_ids"),
            extracted_chunks=artifacts_dict.get("extracted_chunks"),
            extracted_summary=artifacts_dict.get("extracted_summary"),
            chat_summary=artifacts_dict.get("chat_summary"),
            confirmed_facts=artifacts_dict.get("confirmed_facts"),
        )
    
    def pydantic_to_artifacts_dict(
        self,
        artifacts: schemas.TurnStateArtifacts,
    ) -> Dict[str, Any]:
        """Convert Pydantic schema to dict for storage."""
        result = {}
        for key, value in artifacts.model_dump().items():
            if value is not None:
                result[key] = value
        return result


async def create_turn_state_manager(db: AsyncSession) -> TurnStateManager:
    """Factory function."""
    return TurnStateManager(db)
