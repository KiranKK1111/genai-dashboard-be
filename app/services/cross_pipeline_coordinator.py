"""
CROSS-PIPELINE COORDINATOR - Coordinate multi-pipeline queries.

Handles queries that require multiple pipelines:
- Compare database data with uploaded files
- Combine SQL results with file analysis
- Multi-source aggregation
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PipelineType(str, Enum):
    """Types of pipelines."""
    SQL = "sql"
    FILE_ANALYSIS = "file_analysis"
    CHAT = "chat"
    PYTHON_COMPUTATION = "python_computation"


@dataclass
class ExecutionTask:
    """Single task in cross-pipeline execution."""
    task_id: str
    pipeline: PipelineType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """Plan for cross-pipeline execution."""
    tasks: List[ExecutionTask] = field(default_factory=list)
    pipelines_involved: List[PipelineType] = field(default_factory=list)
    description: str = ""


class CrossPipelineCoordinator:
    """Coordinates execution across multiple pipelines."""
    
    def __init__(self):
        """Initialize coordinator."""
        pass
    
    async def analyze_and_plan(
        self,
        user_query: str,
        available_files: List[Dict[str, Any]],
        conversation_context: str = "",
    ) -> ExecutionPlan:
        """
        Analyze query and create execution plan.
        
        Args:
            user_query: User's natural language query
            available_files: List of available files
            conversation_context: Previous conversation
            
        Returns:
            ExecutionPlan with tasks
        """
        plan = ExecutionPlan(description=user_query)
        query_lower = user_query.lower()
        
        # Detect if SQL is needed
        if any(word in query_lower for word in ['database', 'table', 'query', 'select']):
            plan.tasks.append(ExecutionTask(
                task_id="task_1",
                pipeline=PipelineType.SQL,
                description="Query database",
                parameters={"query": user_query}
            ))
            plan.pipelines_involved.append(PipelineType.SQL)
        
        # Detect if file analysis is needed
        if available_files and any(word in query_lower for word in ['file', 'upload', 'document']):
            plan.tasks.append(ExecutionTask(
                task_id="task_2",
                pipeline=PipelineType.FILE_ANALYSIS,
                description="Analyze uploaded files",
                parameters={"files": available_files}
            ))
            plan.pipelines_involved.append(PipelineType.FILE_ANALYSIS)
        
        # Detect if comparison/computation is needed
        if any(word in query_lower for word in ['compare', 'combine', 'merge', 'join']):
            if len(plan.tasks) > 1:
                plan.tasks.append(ExecutionTask(
                    task_id="task_3",
                    pipeline=PipelineType.PYTHON_COMPUTATION,
                    description="Compare and combine results",
                    parameters={},
                    depends_on=[t.task_id for t in plan.tasks]
                ))
                plan.pipelines_involved.append(PipelineType.PYTHON_COMPUTATION)
        
        return plan


# Global instance
_coordinator: Optional[CrossPipelineCoordinator] = None


def get_cross_pipeline_coordinator() -> CrossPipelineCoordinator:
    """Get or create cross-pipeline coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = CrossPipelineCoordinator()
    return _coordinator
