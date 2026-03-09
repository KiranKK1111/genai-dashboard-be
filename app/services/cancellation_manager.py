"""
Cancellation Manager - Tracks and manages query cancellation tokens.

This service allows stopping ongoing query execution, similar to ChatGPT's stop button.
In production, this should use Redis for distributed cancellation across workers.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


class CancellationToken:
    """Represents a cancellation token for a query."""
    
    def __init__(self, message_id: str):
        self.message_id = message_id
        self.is_cancelled = False
        self.cancelled_at: Optional[datetime] = None
        self.reason: Optional[str] = None
    
    def cancel(self, reason: str = "User requested cancellation"):
        """Mark this token as cancelled."""
        self.is_cancelled = True
        self.cancelled_at = datetime.utcnow()
        self.reason = reason
        logger.info(f"[CANCEL] Message {self.message_id} cancelled: {reason}")
    
    def check_cancelled(self):
        """Check if cancelled and raise exception if so."""
        if self.is_cancelled:
            raise CancellationException(
                f"Query cancelled: {self.reason}",
                message_id=self.message_id
            )


class CancellationException(Exception):
    """Exception raised when a query is cancelled."""
    
    def __init__(self, message: str, message_id: str):
        super().__init__(message)
        self.message_id = message_id


class CancellationManager:
    """
    Manages cancellation tokens for ongoing queries.
    
    Usage:
        # In query handler:
        token = cancellation_manager.create_token(message_id)
        try:
            # At critical points:
            token.check_cancelled()
            # ... do work ...
        except CancellationException:
            return cancellation_response()
        
        # In stop endpoint:
        cancellation_manager.cancel(message_id)
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tokens = {}  # Dict[str, CancellationToken]
            cls._instance._cleanup_task = None
        return cls._instance
    
    def create_token(self, message_id: str) -> CancellationToken:
        """Create a new cancellation token for a message."""
        token = CancellationToken(message_id)
        self._tokens[message_id] = token
        logger.debug(f"[CANCEL] Created token for message {message_id}")
        return token
    
    def get_token(self, message_id: str) -> Optional[CancellationToken]:
        """Get existing cancellation token."""
        return self._tokens.get(message_id)
    
    def cancel(self, message_id: str, reason: str = "User requested cancellation") -> bool:
        """
        Cancel a query by message ID.
        
        Returns:
            True if token was found and cancelled, False if not found
        """
        token = self._tokens.get(message_id)
        if token:
            token.cancel(reason)
            return True
        logger.warning(f"[CANCEL] No active token found for message {message_id}")
        return False
    
    def remove_token(self, message_id: str):
        """Remove a token after query completion."""
        if message_id in self._tokens:
            del self._tokens[message_id]
            logger.debug(f"[CANCEL] Removed token for message {message_id}")
    
    def cleanup_old_tokens(self, max_age_minutes: int = 60):
        """Remove tokens older than max_age_minutes."""
        cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        to_remove = []
        
        for message_id, token in self._tokens.items():
            # If cancelled and old, or very old regardless
            if token.cancelled_at and token.cancelled_at < cutoff:
                to_remove.append(message_id)
        
        for message_id in to_remove:
            del self._tokens[message_id]
        
        if to_remove:
            logger.info(f"[CANCEL] Cleaned up {len(to_remove)} old tokens")
    
    async def start_cleanup_task(self, interval_minutes: int = 10):
        """Start background task to cleanup old tokens."""
        while True:
            await asyncio.sleep(interval_minutes * 60)
            self.cleanup_old_tokens()
    
    def get_active_count(self) -> int:
        """Get count of active (non-cancelled) tokens."""
        return sum(1 for t in self._tokens.values() if not t.is_cancelled)


# Global singleton instance
cancellation_manager = CancellationManager()
