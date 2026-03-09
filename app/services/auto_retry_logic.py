"""
AUTO RETRY LOGIC - Automatic retry with error analysis.

Provides:
- Intelligent error categorization
- Automatic retry strategies
- Error analysis and feedback
- Exponential backoff
"""

from __future__ import annotations

import logging
import asyncio
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Categories of errors."""
    SYNTAX_ERROR = "syntax_error"
    SCHEMA_ERROR = "schema_error"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT_ERROR = "timeout_error"
    PERMISSION_ERROR = "permission_error"
    DATA_ERROR = "data_error"
    UNKNOWN_ERROR = "unknown"


@dataclass
class ErrorAnalysis:
    """Analysis of an error."""
    category: ErrorCategory
    error_message: str
    is_retryable: bool
    suggested_fix: Optional[str] = None
    retry_strategy: str = "exponential_backoff"


@dataclass
class RetryResult:
    """Result of retry execution."""
    success: bool
    result: Any
    attempts: int
    errors: list[str]
    final_error: Optional[str] = None


class AutoRetryExecutor:
    """Executes operations with automatic retry logic."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0
    ):
        """
        Initialize auto retry executor.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            backoff_factor: Multiplier for exponential backoff
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
    
    async def execute_with_retry(
        self,
        operation: Callable,
        *args,
        error_analyzer: Optional[Callable[[Exception], ErrorAnalysis]] = None,
        **kwargs
    ) -> RetryResult:
        """
        Execute operation with automatic retry on retriable errors.
        
        Args:
            operation: Async function to execute
            *args: Arguments for operation
            error_analyzer: Optional function to analyze errors
            **kwargs: Keyword arguments for operation
            
        Returns:
            RetryResult with execution details
        """
        attempts = 0
        errors = []
        delay = self.initial_delay
        
        while attempts < self.max_retries:
            attempts += 1
            
            try:
                result = await operation(*args, **kwargs)
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    errors=errors
                )
            
            except Exception as e:
                error_msg = str(e)
                errors.append(error_msg)
                
                # Analyze error
                if error_analyzer:
                    analysis = error_analyzer(e)
                else:
                    analysis = self._default_error_analysis(e)
                
                logger.warning(
                    f"[AUTO_RETRY] Attempt {attempts}/{self.max_retries} failed: {error_msg[:100]}"
                )
                
                # Check if retryable
                if not analysis.is_retryable or attempts >= self.max_retries:
                    logger.error(
                        f"[AUTO_RETRY] Giving up after {attempts} attempts. "
                        f"Error category: {analysis.category.value}"
                    )
                    return RetryResult(
                        success=False,
                        result=None,
                        attempts=attempts,
                        errors=errors,
                        final_error=error_msg
                    )
                
                # Wait before retry (exponential backoff)
                logger.info(f"[AUTO_RETRY] Waiting {delay:.2f}s before retry...")
                await asyncio.sleep(delay)
                delay *= self.backoff_factor
        
        # Max retries exhausted
        return RetryResult(
            success=False,
            result=None,
            attempts=attempts,
            errors=errors,
            final_error=errors[-1] if errors else "Unknown error"
        )
    
    def _default_error_analysis(self, error: Exception) -> ErrorAnalysis:
        """Default error analysis."""
        error_str = str(error).lower()
        
        # Check for syntax errors
        if 'syntax' in error_str or 'parse' in error_str:
            return ErrorAnalysis(
                category=ErrorCategory.SYNTAX_ERROR,
                error_message=str(error),
                is_retryable=False,
                suggested_fix="Check SQL syntax"
            )
        
        # Check for schema errors
        if 'column' in error_str or 'table' in error_str or 'does not exist' in error_str:
            return ErrorAnalysis(
                category=ErrorCategory.SCHEMA_ERROR,
                error_message=str(error),
                is_retryable=False,
                suggested_fix="Verify table and column names"
            )
        
        # Check for connection errors
        if 'connection' in error_str or 'timeout' in error_str:
            return ErrorAnalysis(
                category=ErrorCategory.CONNECTION_ERROR,
                error_message=str(error),
                is_retryable=True,
                suggested_fix="Retry connection"
            )
        
        # Default to unknown
        return ErrorAnalysis(
            category=ErrorCategory.UNKNOWN_ERROR,
            error_message=str(error),
            is_retryable=True,
            suggested_fix=None
        )


# Global instance
_auto_retry_executor: Optional[AutoRetryExecutor] = None


def get_auto_retry_executor() -> AutoRetryExecutor:
    """Get or create auto retry executor."""
    global _auto_retry_executor
    if _auto_retry_executor is None:
        _auto_retry_executor = AutoRetryExecutor()
    return _auto_retry_executor
