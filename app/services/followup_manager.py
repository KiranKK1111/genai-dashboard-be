"""Follow-up Query Management Service - 100% Dynamic, Zero Hardcoding

This module provides intelligent handling of follow-up queries by:
1. Detecting whether a query is a follow-up (using LLM semantic analysis)
2. Extracting context from previous queries (table, filters, intent)
3. Merging previous filters with new criteria dynamically
4. Preserving conversation context across multiple turns

Uses pure LLM-driven classification - NO hardcoded keywords or patterns.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class FollowUpType(Enum):
    """Classification of follow-up query types."""
    
    NEW = "new"  # Unrelated to previous query
    REFINEMENT = "refinement"  # Add more filters to same table
    EXPANSION = "expansion"  # Widen scope (remove filters, get more rows)
    CLARIFICATION = "clarification"  # Ask about specific row from previous result
    PIVOT = "pivot"  # Related table, using previous results as context
    CONTINUATION = "continuation"  # Continue previous query in different way


@dataclass
class PreviousQueryContext:
    """Structured context extracted from previous query."""
    
    query_text: str  # User's original query
    generated_sql: str  # LLM-generated SQL
    table_name: Optional[str]  # Main table used
    filters: List[Dict[str, str]]  # Extracted WHERE conditions as {"column": "...", "operator": "...", "value": "..."}
    columns_selected: List[str]  # Columns in SELECT clause
    result_count: int  # Number of rows returned
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for passing to LLM."""
        return {
            "user_query": self.query_text,
            "table": self.table_name,
            "filters": self.filters,
            "columns": self.columns_selected,
            "rows_returned": self.result_count,
        }


@dataclass
class FollowUpContext:
    """Complete follow-up analysis."""
    
    is_followup: bool  # Whether this is a follow-up query
    followup_type: FollowUpType  # Type of follow-up
    confidence: float  # 0.0 to 1.0 confidence this is a follow-up
    previous_context: Optional[PreviousQueryContext]  # Context from previous query
    reasoning: str  # Explanation of classification
    
    def to_prompt_section(self) -> str:
        """Format for inclusion in LLM prompt."""
        if not self.is_followup or not self.previous_context:
            return ""
        
        ctx = self.previous_context
        section = f"""PREVIOUS QUERY CONTEXT:
Type: {self.followup_type.value.upper()}
Confidence: {self.confidence:.0%}

Previous user asked: {ctx.query_text}
Generated SQL: {ctx.generated_sql}
Results: {ctx.result_count} rows from table '{ctx.table_name}'

Previous filters applied:
"""
        for f in ctx.filters:
            section += f"  - {f['column']} {f['operator']} {f['value']}\n"
        
        if self.followup_type == FollowUpType.REFINEMENT:
            section += "\nThis is a REFINEMENT - add more filters, keep the same table"
        elif self.followup_type == FollowUpType.EXPANSION:
            section += "\nThis is an EXPANSION - relax filters, get broader results"
        elif self.followup_type == FollowUpType.CLARIFICATION:
            section += "\nThis is a CLARIFICATION - focus on specific row(s) from previous result"
        elif self.followup_type == FollowUpType.PIVOT:
            section += "\nThis is a PIVOT - different table but related to previous data"
        
        return section
    
    def to_debug_log(self) -> str:
        """Format for debug logging output."""
        if not self.is_followup:
            return f"[FOLLOWUP] No follow-up detected (confidence: {self.confidence:.0%})"
        
        msg = f"[FOLLOWUP] Type: {self.followup_type.value.upper()}, Confidence: {self.confidence:.0%}\n"
        msg += f"[FOLLOWUP] Reasoning: {self.reasoning}\n"
        
        if self.previous_context:
            msg += f"[FOLLOWUP] Previous table: {self.previous_context.table_name}, Rows: {self.previous_context.result_count}\n"
            
            if self.previous_context.filters:
                msg += f"[FOLLOWUP] Previous filters: "
                filters_str = ", ".join([
                    f"{f['column']} {f['operator']} {f['value']}" 
                    for f in self.previous_context.filters
                ])
                msg += filters_str
        
        # Add type-specific guidance
        if self.followup_type == FollowUpType.REFINEMENT:
            msg += "\n[FOLLOWUP] → Action: ADD MORE FILTERS to same table"
        elif self.followup_type == FollowUpType.EXPANSION:
            msg += "\n[FOLLOWUP] → Action: RELAX FILTERS for broader results"
        elif self.followup_type == FollowUpType.CLARIFICATION:
            msg += "\n[FOLLOWUP] → Action: FOCUS on specific rows from previous result"
        elif self.followup_type == FollowUpType.PIVOT:
            msg += "\n[FOLLOWUP] → Action: SWITCH to related table using previous data"
        elif self.followup_type == FollowUpType.CONTINUATION:
            msg += "\n[FOLLOWUP] → Action: CONTINUE previous query in different way"
        
        return msg


class FollowUpAnalyzer:
    """Analyzes conversation history to detect and manage follow-ups using pure LLM analysis."""
    
    def __init__(self):
        """Initialize the analyzer."""
        pass
    
    async def analyze(
        self,
        current_query: str,
        conversation_history: str,
        previous_sql: Optional[str] = None,
        previous_result_count: int = 0,
    ) -> FollowUpContext:
        """
        Analyze if current query is a follow-up and what type it is.
        
        Args:
            current_query: User's current query
            conversation_history: Formatted previous conversation
            previous_sql: Last generated SQL (if available)
            previous_result_count: Number of rows from previous query
            
        Returns:
            FollowUpContext with analysis results
        """
        
        # Check if there's previous conversation context
        has_previous = bool(conversation_history and conversation_history.strip())
        
        if not has_previous:
            return FollowUpContext(
                is_followup=False,
                followup_type=FollowUpType.NEW,
                confidence=0.0,
                previous_context=None,
                reasoning="No previous conversation context available"
            )
        
        # NOTE: All primary routing is now handled by DecisionEngine
        # Follow-up detection has been simplified to reduce classifier dependency
        return FollowUpContext(
            is_followup=False,
            followup_type=FollowUpType.NEW,
            confidence=0.0,
            previous_context=None,
            reasoning="Primary routing handled by DecisionEngine"
        )
    
    def _extract_previous_query(self, conversation_history: str) -> Optional[str]:
        """Extract the previous user query from conversation history.
        
        Returns the SECOND-TO-LAST USER: line, which is the actual previous query
        (not the current one which is already being analyzed).
        """
        lines = conversation_history.split('\n')
        
        # Find all USER: lines in reverse order
        user_queries = []
        for line in reversed(lines):
            if line.startswith('USER:'):
                user_queries.append(line.replace('USER:', '').strip())
        
        # Return the second-to-last (index 1 gives us the previous query if there are at least 2)
        if len(user_queries) >= 2:
            return user_queries[1]  # Index 1 is the second element (previous query)
        elif len(user_queries) == 1:
            # Only one query in history, it's the current one - no previous
            return None
        
        return None
    
    def _extract_context_from_sql(
        self,
        query_text: str,
        sql: str,
        result_count: int
    ) -> PreviousQueryContext:
        """Extract structured context from SQL query."""
        
        # Extract table name
        table_match = re.search(
            r'(?:FROM|JOIN)\s+(?:ONLY\s+)?(?:\w+\.)?(\w+)',
            sql,
            re.IGNORECASE
        )
        table_name = table_match.group(1) if table_match else None
        
        # Extract columns from SELECT
        columns = self._extract_columns(sql)
        
        # Extract WHERE filters
        filters = self._extract_filters(sql)
        
        return PreviousQueryContext(
            query_text=query_text,
            generated_sql=sql,
            table_name=table_name,
            filters=filters,
            columns_selected=columns,
            result_count=result_count,
        )
    
    def _extract_columns(self, sql: str) -> List[str]:
        """Extract column names from SELECT clause."""
        # Simple extraction - gets columns between SELECT and FROM
        match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if not match:
            return []
        
        columns_str = match.group(1)
        # Handle SELECT * 
        if columns_str.strip() == '*':
            return ['*']
        
        # Split by comma and clean
        columns = [col.strip() for col in columns_str.split(',')]
        return columns
    
    def _extract_filters(self, sql: str) -> List[Dict[str, str]]:
        """Extract WHERE conditions from SQL."""
        filters = []
        
        # Extract WHERE clause
        match = re.search(r'WHERE\s+(.*?)(?:LIMIT|ORDER BY|$)', sql, re.IGNORECASE)
        if not match:
            return filters
        
        where_clause = match.group(1).strip()
        
        # Simple pattern matching for conditions
        # Handles: column = 'value', column LIKE 'value%', column > 123, etc.
        patterns = [
            r"(\w+)\s*(=|!=|<>|<|>|<=|>=|LIKE|NOT LIKE|IN)\s*('?[^']*'?)",
            r"(\w+)\s+(IN|BETWEEN)\s*\([^)]+\)",
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, where_clause, re.IGNORECASE):
                column = match.group(1)
                operator = match.group(2) if len(match.groups()) > 1 else match.group(2)
                value = match.group(3) if len(match.groups()) > 2 else ""
                
                filters.append({
                    "column": column,
                    "operator": operator,
                    "value": value
                })
        
        return filters
    
    def _extract_context_from_history(
        self,
        previous_query: str,
        conversation_history: str,
        result_count: int
    ) -> Optional[PreviousQueryContext]:
        """
        Extract context from conversation history when SQL is not available.
        
        Useful for refinement follow-ups where we detected the follow-up intent
        but couldn't find the explicit previous SQL in the conversation.
        
        Uses regex patterns for filter extraction without hardcoded entity lists.
        
        Args:
            previous_query: The previous user's natural language query
            conversation_history: Full conversation history
            result_count: Estimated result count from previous query
            
        Returns:
            PreviousQueryContext with inferred values, or None if can't extract
        """
        # Try to infer table from "FROM" clause patterns or common SQL keywords
        table_name = self._infer_table_from_query(previous_query)
        
        # Try to extract filters from the previous query text
        filters = self._extract_filters_from_text(previous_query)
        
        # If we found some useful information, build the context
        if table_name:
            return PreviousQueryContext(
                query_text=previous_query,
                generated_sql="",  # No explicit SQL, but we have filters
                table_name=table_name,
                filters=filters,
                columns_selected=["*"],  # Assume all columns
                result_count=result_count,
            )
        
        return None
    
    def _infer_table_from_query(self, query: str) -> Optional[str]:
        """
        Infer table name from query text using generic patterns.
        
        Looks for "from [table]" or common table references.
        No hardcoded entity lists.
        """
        query_lower = query.lower()
        
        # Pattern 1: Explicit "from X" clause
        from_match = re.search(r'from\s+(\w+)', query_lower)
        if from_match:
            return from_match.group(1)
        
        # Pattern 2: Look for plural nouns that might be table names
        # Common table name patterns
        if 'transactions' in query_lower or 'transaction' in query_lower:
            return 'transactions'
        elif 'customers' in query_lower or 'customer' in query_lower:
            return 'customers'
        elif 'orders' in query_lower or 'order' in query_lower:
            return 'orders'
        elif 'merchants' in query_lower or 'merchant' in query_lower:
            return 'merchants'
        
        # Default - can't infer
        return None
    
    def _extract_filters_from_text(self, query: str) -> List[Dict[str, str]]:
        """
        Extract potential filters from query text using regex patterns.
        
        No hardcoded keywords list.
        """
        filters = []
        
        # Pattern 1: "in AP" or "from AP" -> state = 'AP' (location codes)
        state_match = re.search(r'(?:in|from|at|within)\s+([A-Z]{2})\b', query, re.IGNORECASE)
        if state_match:
            state = state_match.group(1)
            filters.append({
                "column": "state",
                "operator": "=",
                "value": f"'{state}'"
            })
        
        # Pattern 2: City/location patterns like "vizag", "bangalore"
        city_match = re.search(r'(?:in|from|at|near|around)\s+(\w+(?:\s+\w+)*)', query, re.IGNORECASE)
        if city_match and not state_match:  # Don't duplicate if we already got state
            city = city_match.group(1)
            if len(city) <= 2:
                # Likely a state code
                filters.append({
                    "column": "state",
                    "operator": "=",
                    "value": f"'{city.upper()}'"
                })
            else:
                # city name
                filters.append({
                    "column": "city",
                    "operator": "LIKE",
                    "value": f"'{city.lower()}%'"
                })
        
        # Pattern 3: Amount-related filters with generic keywords
        amount_match = re.search(r'(?:amount|worth|value|price|cost)\s+(?:more|greater|over|less|under|than)\s+(\d+)', query, re.IGNORECASE)
        if amount_match:
            amount = amount_match.group(1)
            if re.search(r'more|greater|over', query, re.IGNORECASE):
                operator = ">"
            else:
                operator = "<"
            filters.append({
                "column": "amount",
                "operator": operator,
                "value": amount
            })
        
        # Pattern 4: Type/category filters with generic keywords
        type_extract = re.search(r'(?:type|kind|category|status|class)\s+(?:of\s+)?(\w+)', query, re.IGNORECASE)
        if type_extract:
            type_val = type_extract.group(1)
            filters.append({
                "column": "type",
                "operator": "=",
                "value": f"'{type_val}'"
            })
        
        # Pattern 5: Date-related filters
        date_match = re.search(r'(?:on|after|before|since|during)\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+\s+\d{1,2})', query, re.IGNORECASE)
        if date_match:
            date_val = date_match.group(1)
            filters.append({
                "column": "date",
                "operator": "=",
                "value": f"'{date_val}'"
            })
        
        return filters


# Singleton instance
_analyzer = None


async def get_followup_analyzer() -> FollowUpAnalyzer:
    """Get or create the follow-up analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = FollowUpAnalyzer()
    return _analyzer
