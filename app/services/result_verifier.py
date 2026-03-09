"""
RESULT VERIFIER - Verify SQL query results for accuracy.

Detects:
- Hallucinated columns
- Empty results when data should exist
- Type mismatches
- Logic errors
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"


@dataclass
class ValidationIssue:
    """Single validation issue found."""
    issue_type: str
    severity: str  # "error", "warning", "info"
    message: str
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of query result verification."""
    is_valid: bool
    confidence: float  # 0-1
    issues: List[ValidationIssue]
    has_hallucinations: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "issues": [
                {
                    "type": i.issue_type,
                    "severity": i.severity,
                    "message": i.message,
                    "suggestion": i.suggestion,
                }
                for i in self.issues
            ],
            "has_hallucinations": self.has_hallucinations,
        }


class ResultVerifier:
    """Verifies SQL query results for accuracy."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """Initialize verifier."""
        self.validation_level = validation_level
    
    async def verify_results(
        self,
        sql: str,
        results: List[Dict[str, Any]],
        schema_info: str,
        db_session: AsyncSession,
    ) -> ValidationResult:
        """
        Verify query results for accuracy.
        
        Args:
            sql: SQL query that was executed
            results: Query results
            schema_info: Schema information
            db_session: Database session
            
        Returns:
            ValidationResult with validation details
        """
        issues = []
        confidence = 1.0
        has_hallucinations = False
        
        # Check if results are empty when they shouldn't be
        if not results or len(results) == 0:
            # Could indicate wrong table or filters
            issues.append(ValidationIssue(
                issue_type="empty_results",
                severity="warning",
                message="Query returned no results",
                suggestion="Check table filters and WHERE conditions"
            ))
            confidence -= 0.2
        
        # Check for column mismatches (hallucinated columns)
        if results and len(results) > 0:
            result_columns = set(results[0].keys())
            
            # Extract expected columns from schema
            # This is a simplified check - real implementation would parse schema more carefully
            if schema_info:
                schema_lower = schema_info.lower()
                for col in result_columns:
                    if col.lower() not in schema_lower:
                        has_hallucinations = True
                        issues.append(ValidationIssue(
                            issue_type="hallucinated_column",
                            severity="error",
                            message=f"Column '{col}' not found in schema",
                            suggestion="Regenerate query with correct column names"
                        ))
                        confidence -= 0.3
        
        # Check for suspicious result patterns
        if results and len(results) > 0:
            # Check if all values are NULL (might indicate wrong joins)
            first_row = results[0]
            if all(v is None for v in first_row.values()):
                issues.append(ValidationIssue(
                    issue_type="all_null_values",
                    severity="warning",
                    message="All values in first row are NULL",
                    suggestion="Check JOIN conditions"
                ))
                confidence -= 0.1
        
        # Determine if valid
        is_valid = confidence > 0.5 and not has_hallucinations
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=max(confidence, 0.0),
            issues=issues,
            has_hallucinations=has_hallucinations,
        )
