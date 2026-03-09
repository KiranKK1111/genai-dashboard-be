"""
RESULT INTERPRETER - Interpret and explain query results.

Provides:
- Natural language explanations of results
- Trend detection
- Insight generation
- Visualization recommendations
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InterpretationResult:
    """Result of interpreting query results."""
    summary: str
    insights: List[str]
    trends: List[str]
    recommended_viz: Optional[str] = None
    key_findings: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "insights": self.insights,
            "trends": self.trends,
            "recommended_viz": self.recommended_viz,
            "key_findings": self.key_findings or [],
        }


class ResultInterpreter:
    """Interprets query results and generates insights."""
    
    def __init__(self):
        """Initialize result interpreter."""
        pass
    
    async def interpret_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> InterpretationResult:
        """
        Interpret query results and generate insights.
        
        Args:
            results: Query results
            query: Original user query
            metadata: Optional metadata about the query
            
        Returns:
            InterpretationResult with insights
        """
        insights = []
        trends = []
        key_findings = []
        
        # Handle empty results
        if not results or len(results) == 0:
            return InterpretationResult(
                summary="No results found for your query.",
                insights=["No data matches your criteria"],
                trends=[],
                recommended_viz=None,
                key_findings=[]
            )
        
        # Generate summary
        row_count = len(results)
        col_count = len(results[0].keys()) if results else 0
        summary = f"Found {row_count} record{'s' if row_count != 1 else ''} with {col_count} column{'s' if col_count != 1 else ''}."
        
        # Detect numeric columns for trend analysis
        if row_count > 0:
            first_row = results[0]
            numeric_cols = [k for k, v in first_row.items() if isinstance(v, (int, float))]
            
            if numeric_cols:
                # Calculate basic statistics
                for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    values = [r[col] for r in results if r.get(col) is not None]
                    if values:
                        avg = sum(values) / len(values)
                        max_val = max(values)
                        min_val = min(values)
                        insights.append(
                            f"{col}: avg={avg:.2f}, max={max_val}, min={min_val}"
                        )
                        
                        # Detect trends
                        if len(values) > 1:
                            if values[-1] > values[0] * 1.1:
                                trends.append(f"{col} is increasing")
                            elif values[-1] < values[0] * 0.9:
                                trends.append(f"{col} is decreasing")
        
        # Recommend visualization
        recommended_viz = self._recommend_visualization(results, query)
        
        # Key findings
        if row_count > 0:
            key_findings.append(f"Dataset contains {row_count} records")
            if numeric_cols:
                key_findings.append(f"Found {len(numeric_cols)} numeric measures")
        
        return InterpretationResult(
            summary=summary,
            insights=insights,
            trends=trends,
            recommended_viz=recommended_viz,
            key_findings=key_findings
        )
    
    def _recommend_visualization(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> Optional[str]:
        """Recommend visualization type based on data."""
        if not results:
            return None
        
        row_count = len(results)
        query_lower = query.lower()
        
        # Time series patterns
        if any(word in query_lower for word in ['trend', 'over time', 'timeline', 'date']):
            return "line"
        
        # Comparison patterns
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'between']):
            return "bar"
        
        # Distribution patterns
        if any(word in query_lower for word in ['distribution', 'breakdown', 'proportion']):
            return "pie"
        
        # Default based on row count
        if row_count <= 10:
            return "table"
        elif row_count <= 50:
            return "bar"
        else:
            return "line"


# Global instance
_result_interpreter: Optional[ResultInterpreter] = None


def get_result_interpreter() -> ResultInterpreter:
    """Get or create result interpreter."""
    global _result_interpreter
    if _result_interpreter is None:
        _result_interpreter = ResultInterpreter()
    return _result_interpreter
