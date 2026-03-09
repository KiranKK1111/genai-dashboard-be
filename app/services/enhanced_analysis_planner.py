"""
ENHANCED ANALYSIS PLANNER - Plan complex file analysis tasks.

Provides:
- Task complexity assessment
- Multi-step analysis planning
- Resource estimation
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TaskComplexity(str, Enum):
    """Complexity levels for analysis tasks."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class AnalysisStep:
    """Single step in analysis plan."""
    step_id: int
    description: str
    operation: str
    estimated_time_seconds: float


@dataclass
class AnalysisPlan:
    """Plan for file analysis."""
    complexity: TaskComplexity
    steps: List[AnalysisStep]
    estimated_total_time: float
    requires_computation: bool = False
    requires_visualization: bool = False


class EnhancedAnalysisPlanner:
    """Plans complex file analysis tasks."""
    
    def __init__(self):
        """Initialize analysis planner."""
        pass
    
    async def create_plan(
        self,
        query: str,
        file_info: Dict[str, Any],
        context: Optional[str] = None
    ) -> AnalysisPlan:
        """
        Create analysis plan for file query.
        
        Args:
            query: User's analysis query
            file_info: Information about the file
            context: Optional context
            
        Returns:
            AnalysisPlan with steps
        """
        query_lower = query.lower()
        steps = []
        complexity = TaskComplexity.SIMPLE
        requires_computation = False
        requires_visualization = False
        
        # Step 1: Load file
        steps.append(AnalysisStep(
            step_id=1,
            description="Load file content",
            operation="file_load",
            estimated_time_seconds=1.0
        ))
        
        # Check for statistical analysis
        if any(word in query_lower for word in ['average', 'mean', 'sum', 'count', 'max', 'min']):
            steps.append(AnalysisStep(
                step_id=2,
                description="Calculate statistics",
                operation="statistics",
                estimated_time_seconds=0.5
            ))
            complexity = TaskComplexity.MODERATE
            requires_computation = True
        
        # Check for filtering/grouping
        if any(word in query_lower for word in ['filter', 'where', 'group by', 'by']):
            steps.append(AnalysisStep(
                step_id=len(steps) + 1,
                description="Filter and group data",
                operation="filter_group",
                estimated_time_seconds=1.0
            ))
            complexity = TaskComplexity.MODERATE
        
        # Check for visualization
        if any(word in query_lower for word in ['chart', 'plot', 'graph', 'visualize', 'show']):
            steps.append(AnalysisStep(
                step_id=len(steps) + 1,
                description="Generate visualization",
                operation="visualization",
                estimated_time_seconds=2.0
            ))
            requires_visualization = True
        
        # Check for complex operations
        if any(word in query_lower for word in ['join', 'merge', 'combine', 'compare']):
            complexity = TaskComplexity.COMPLEX
        
        # Calculate total time
        total_time = sum(step.estimated_time_seconds for step in steps)
        
        return AnalysisPlan(
            complexity=complexity,
            steps=steps,
            estimated_total_time=total_time,
            requires_computation=requires_computation,
            requires_visualization=requires_visualization
        )
