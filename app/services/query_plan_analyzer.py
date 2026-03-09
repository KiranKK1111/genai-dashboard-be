"""
Query Plan Analyzer - Recommend optimizations for slow queries.

Analyzes query execution and suggests:
- Indexes on WHERE columns
- Indexes on JOIN columns (critical)
- LIMIT clauses
- Query complexity warnings
"""

from __future__ import annotations

import logging
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class QueryPlanAnalyzer:
    """
    Analyze query execution plans and recommend optimizations.
    """
    
    def __init__(self, slow_threshold_ms: float = 500):
        """Initialize analyzer with slow query threshold."""
        self.slow_threshold_ms = slow_threshold_ms
    
    async def analyze_slow_query(
        self,
        sql: str,
        execution_time_ms: float,
    ) -> Dict:
        """
        Analyze why a query is slow and recommend fixes.
        
        Args:
            sql: The SQL query that was slow
            execution_time_ms: How long it took to execute
        
        Returns:
            Dict with analysis and recommendations
        """
        
        analysis = {
            "sql": sql[:200],  # Truncate for logging
            "execution_time_ms": execution_time_ms,
            "is_slow": execution_time_ms > self.slow_threshold_ms,
            "recommendations": [],
            "severity": self._calculate_severity(execution_time_ms),
        }
        
        if not analysis["is_slow"]:
            return analysis
        
        # Extract WHERE columns (potential index candidates)
        where_columns = self._extract_where_columns(sql)
        analysis["where_columns"] = where_columns
        
        # Extract JOIN columns (critical for performance)
        join_columns = self._extract_join_columns(sql)
        analysis["join_columns"] = join_columns
        
        # Extract GROUP BY columns
        group_columns = self._extract_group_by_columns(sql)
        analysis["group_columns"] = group_columns
        
        # Extract SELECT columns for complexity
        select_columns = self._extract_select_columns(sql)
        analysis["select_column_count"] = len(select_columns)
        
        # Make recommendations
        if join_columns:
            analysis["recommendations"].append({
                "type": "index",
                "priority": "CRITICAL",
                "suggestion": f"Index JOIN columns: {', '.join(join_columns)}",
                "example": f"CREATE INDEX idx_join ON table({', '.join(join_columns)})",
                "expected_improvement": "50-70% speedup",
            })
        
        if where_columns:
            analysis["recommendations"].append({
                "type": "index",
                "priority": "HIGH",
                "suggestion": f"Index WHERE columns: {', '.join(where_columns)}",
                "example": f"CREATE INDEX idx_where ON table({', '.join(where_columns)})",
                "expected_improvement": "30-50% speedup",
            })
        
        if group_columns:
            analysis["recommendations"].append({
                "type": "index",
                "priority": "HIGH",
                "suggestion": f"Index GROUP BY columns: {', '.join(group_columns)}",
                "example": f"CREATE INDEX idx_group ON table({', '.join(group_columns)})",
                "expected_improvement": "20-40% speedup",
            })
        
        # Check query complexity
        if len(sql) > 1000:
            analysis["recommendations"].append({
                "type": "complexity",
                "priority": "MEDIUM",
                "suggestion": "Query is complex (>1000 chars). Consider breaking into smaller queries or using materialized views",
                "expected_improvement": "Easier to optimize later",
            })
        
        # Check for missing LIMIT
        if "LIMIT" not in sql.upper():
            analysis["recommendations"].append({
                "type": "limit",
                "priority": "MEDIUM",
                "suggestion": "Add LIMIT clause to prevent scanning too many rows",
                "example": "SELECT ... LIMIT 1000",
                "expected_improvement": "10-20% speedup for large result sets",
            })
        
        # Check for SELECT *
        if re.search(r"SELECT\s+\*", sql, re.IGNORECASE):
            analysis["recommendations"].append({
                "type": "columns",
                "priority": "LOW",
                "suggestion": "Replace SELECT * with specific columns to reduce I/O",
                "expected_improvement": "5-10% speedup",
            })
        
        logger.info(
            f"[QUERY-PLAN] Analyzed slow query ({execution_time_ms:.0f}ms), "
            f"severity={analysis['severity']}, "
            f"recommendations={len(analysis['recommendations'])}"
        )
        
        return analysis
    
    async def analyze_batch_queries(
        self,
        queries: List[Dict],
    ) -> Dict:
        """
        Analyze batch of queries and aggregate recommendations.
        
        Args:
            queries: List of {sql, execution_time_ms} dicts
        
        Returns:
            Aggregated analysis and top recommendations
        """
        
        all_analyses = []
        critical_recommendations = []
        column_frequency = {}
        
        for query in queries:
            analysis = await self.analyze_slow_query(
                query["sql"],
                query["execution_time_ms"]
            )
            all_analyses.append(analysis)
            
            # Collect critical recommendations
            for rec in analysis.get("recommendations", []):
                if rec.get("priority") == "CRITICAL":
                    critical_recommendations.append(rec)
                
                # Track column frequency
                for col in analysis.get("join_columns", []):
                    column_frequency[f"{col}_join"] = column_frequency.get(f"{col}_join", 0) + 1
                for col in analysis.get("where_columns", []):
                    column_frequency[f"{col}_where"] = column_frequency.get(f"{col}_where", 0) + 1
        
        # Top columns to index (by frequency)
        top_columns = sorted(
            column_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "total_queries": len(queries),
            "slow_queries": sum(1 for a in all_analyses if a["is_slow"]),
            "avg_execution_time_ms": sum(
                a["execution_time_ms"] for a in all_analyses
            ) / len(all_analyses) if all_analyses else 0,
            "critical_recommendations": critical_recommendations,
            "top_columns_to_index": top_columns,
            "all_analyses": all_analyses,
        }
    
    def _calculate_severity(self, execution_time_ms: float) -> str:
        """Calculate severity based on execution time."""
        
        if execution_time_ms < 500:
            return "LOW"
        elif execution_time_ms < 1000:
            return "MEDIUM"
        elif execution_time_ms < 5000:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _extract_where_columns(self, sql: str) -> List[str]:
        """Extract column names from WHERE clause."""
        
        # Find WHERE clause up to next major keyword
        where_match = re.search(
            r"WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)",
            sql,
            re.IGNORECASE | re.DOTALL
        )
        
        if not where_match:
            return []
        
        where_clause = where_match.group(1)
        
        # Extract column names (simplified: word before comparison operator)
        columns = re.findall(r"(\w+)\s*[=<>!]", where_clause)
        
        # Filter out common keywords
        excluded = {"and", "or", "not", "between", "in", "like"}
        columns = [c for c in columns if c.lower() not in excluded]
        
        return list(set(columns))
    
    def _extract_join_columns(self, sql: str) -> List[str]:
        """Extract column names from JOIN ON clause."""
        
        # Find JOIN ... ON clauses
        join_pattern = r"ON\s+(.+?)(?:WHERE|GROUP|ORDER|JOIN|$)"
        join_matches = re.finditer(join_pattern, sql, re.IGNORECASE | re.DOTALL)
        
        columns = []
        for match in join_matches:
            on_clause = match.group(1)
            # Extract column names (word before = sign)
            cols = re.findall(r"(\w+)\s*[=<>]", on_clause)
            columns.extend(cols)
        
        # Filter out keywords
        excluded = {"and", "or", "not", "between", "in"}
        columns = [c for c in columns if c.lower() not in excluded]
        
        return list(set(columns))
    
    def _extract_group_by_columns(self, sql: str) -> List[str]:
        """Extract column names from GROUP BY clause."""
        
        # Find GROUP BY clause
        group_match = re.search(
            r"GROUP\s+BY\s+(.+?)(?:ORDER|LIMIT|HAVING|$)",
            sql,
            re.IGNORECASE | re.DOTALL
        )
        
        if not group_match:
            return []
        
        group_clause = group_match.group(1)
        
        # Extract column names
        columns = re.findall(r"(\w+)", group_clause)
        
        # Filter out numbers and keywords
        columns = [c for c in columns if not c.isdigit()]
        
        return list(set(columns))
    
    def _extract_select_columns(self, sql: str) -> List[str]:
        """Extract selected columns."""
        
        # Find SELECT clause
        select_match = re.search(
            r"SELECT\s+(.+?)\s+FROM",
            sql,
            re.IGNORECASE | re.DOTALL
        )
        
        if not select_match:
            return []
        
        select_clause = select_match.group(1)
        
        # Extract column names
        columns = re.findall(r"(\w+)", select_clause)
        
        return columns


# Singleton instance
_query_analyzer: Optional[QueryPlanAnalyzer] = None


async def get_query_plan_analyzer(
    slow_threshold_ms: float = 500
) -> QueryPlanAnalyzer:
    """Get or create query plan analyzer."""
    global _query_analyzer
    
    if _query_analyzer is None:
        _query_analyzer = QueryPlanAnalyzer(slow_threshold_ms=slow_threshold_ms)
    
    return _query_analyzer
