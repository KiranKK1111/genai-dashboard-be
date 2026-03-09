"""
QUERY OPTIMIZER - Analyze and optimize SQL queries.

Provides:
- Query complexity analysis
- Index usage suggestions
- Join optimization recommendations
- Performance predictions
"""

from __future__ import annotations

import re
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class OptimizationLevel(str, Enum):
    """Optimization levels."""
    NONE = "none"
    BASIC = "basic"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class OptimizationSuggestion:
    """Single optimization suggestion."""
    category: str
    suggestion: str
    impact: str  # "high", "medium", "low"
    explanation: str


@dataclass
class QueryOptimizationResult:
    """Result of query optimization analysis."""
    complexity_score: float  # 0-1, higher = more complex
    estimated_cost: int
    suggestions: List[OptimizationSuggestion]
    optimized_query: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity_score": self.complexity_score,
            "estimated_cost": self.estimated_cost,
            "suggestions": [
                {
                    "category": s.category,
                    "suggestion": s.suggestion,
                    "impact": s.impact,
                    "explanation": s.explanation
                }
                for s in self.suggestions
            ],
            "optimized_query": self.optimized_query,
        }


class QueryOptimizer:
    """Analyzes and optimizes SQL queries."""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.MODERATE):
        """Initialize optimizer."""
        self.optimization_level = optimization_level
    
    async def analyze_query(
        self,
        sql: str,
        db_session: Optional[AsyncSession] = None
    ) -> QueryOptimizationResult:
        """
        Analyze query and provide optimization suggestions.
        
        Args:
            sql: SQL query to analyze
            db_session: Optional database session for EXPLAIN analysis
            
        Returns:
            QueryOptimizationResult with suggestions
        """
        suggestions = []
        complexity_score = 0.0
        
        # Count joins
        join_count = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
        if join_count > 3:
            complexity_score += 0.3
            suggestions.append(OptimizationSuggestion(
                category="joins",
                suggestion=f"Query has {join_count} joins which may impact performance",
                impact="medium",
                explanation="Consider breaking into smaller queries or adding indexes on join columns"
            ))
        
        # Check for SELECT *
        if re.search(r'SELECT\s+\*', sql, re.IGNORECASE):
            suggestions.append(OptimizationSuggestion(
                category="columns",
                suggestion="Avoid SELECT * - specify only needed columns",
                impact="low",
                explanation="Reduces data transfer and improves performance"
            ))
        
        # Check for subqueries
        subquery_count = len(re.findall(r'SELECT\s+', sql, re.IGNORECASE)) - 1
        if subquery_count > 0:
            complexity_score += 0.2 * subquery_count
            suggestions.append(OptimizationSuggestion(
                category="subqueries",
                suggestion=f"Query has {subquery_count} subqueries",
                impact="medium",
                explanation="Consider using JOINs or CTEs for better performance"
            ))
        
        # Check for WHERE clause
        if not re.search(r'\bWHERE\b', sql, re.IGNORECASE):
            suggestions.append(OptimizationSuggestion(
                category="filtering",
                suggestion="Query has no WHERE clause",
                impact="high",
                explanation="Add filters to reduce result set size"
            ))
        
        # Check for LIMIT
        if not re.search(r'\bLIMIT\b', sql, re.IGNORECASE):
            suggestions.append(OptimizationSuggestion(
                category="pagination",
                suggestion="Consider adding LIMIT clause",
                impact="low",
                explanation="Prevents accidentally returning entire table"
            ))
        
        # Estimate cost (simplified)
        estimated_cost = int(complexity_score * 100) + join_count * 10 + subquery_count * 20
        
        return QueryOptimizationResult(
            complexity_score=min(complexity_score, 1.0),
            estimated_cost=estimated_cost,
            suggestions=suggestions,
            optimized_query=None,  # Could implement query rewriting here
        )
