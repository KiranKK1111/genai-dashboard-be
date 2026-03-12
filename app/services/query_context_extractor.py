"""
Query Context Extractor - Unified SQL context extraction utilities.

This module provides EXTRACTION utilities only - NO decision logic.
All routing and follow-up decisions are made by DecisionArbiter.

Provides:
- SQL parsing: Extract tables, columns, filters from SQL
- Context building: Build structured context from previous queries
- History parsing: Extract previous query from conversation history

Used by:
- session_query_handler.py
- query_handler.py
- decision_arbiter.py
- rag_context_retriever.py
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class PreviousQueryContext:
    """Structured context extracted from previous query.
    
    This is CONTEXT data, not a decision.
    The DecisionArbiter uses this context to make routing decisions.
    """
    
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


class QueryContextExtractor:
    """
    Unified SQL context extraction utilities.
    
    This class provides ONLY extraction functionality.
    It does NOT make decisions about follow-ups or routing.
    Those decisions belong to DecisionArbiter.
    """
    
    def extract_context_from_sql(
        self,
        query_text: str,
        sql: str,
        result_count: int
    ) -> PreviousQueryContext:
        """
        Extract structured context from SQL query.
        
        Args:
            query_text: Original user query text
            sql: Generated SQL string
            result_count: Number of rows returned
            
        Returns:
            PreviousQueryContext with extracted table, columns, filters
        """
        # Extract table name
        table_match = re.search(
            r'(?:FROM|JOIN)\s+(?:ONLY\s+)?(?:\w+\.)?(\w+)',
            sql,
            re.IGNORECASE
        )
        table_name = table_match.group(1) if table_match else None
        
        # Extract columns from SELECT
        columns = self.extract_columns(sql)
        
        # Extract WHERE filters
        filters = self.extract_filters(sql)
        
        return PreviousQueryContext(
            query_text=query_text,
            generated_sql=sql,
            table_name=table_name,
            filters=filters,
            columns_selected=columns,
            result_count=result_count,
        )
    
    def extract_columns(self, sql: str) -> List[str]:
        """
        Extract column names from SELECT clause.
        
        Args:
            sql: SQL query string
            
        Returns:
            List of column names or ['*'] for SELECT *
        """
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
    
    def extract_filters(self, sql: str) -> List[Dict[str, str]]:
        """
        Extract WHERE conditions from SQL.
        
        Args:
            sql: SQL query string
            
        Returns:
            List of filter dicts: {"column": ..., "operator": ..., "value": ...}
        """
        filters = []
        
        # Extract WHERE clause
        match = re.search(r'WHERE\s+(.*?)(?:LIMIT|ORDER BY|GROUP BY|$)', sql, re.IGNORECASE)
        if not match:
            return filters
        
        where_clause = match.group(1).strip()
        
        # Pattern matching for conditions
        # Handles: column = 'value', column LIKE 'value%', column > 123, etc.
        patterns = [
            r"(\w+)\s*(=|!=|<>|<|>|<=|>=|LIKE|NOT LIKE|ILIKE)\s*('?[^']*'?)",
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
    
    def extract_previous_query(self, conversation_history: str) -> Optional[str]:
        """
        Extract the previous user query from conversation history.
        
        Returns the SECOND-TO-LAST USER query, which is the actual previous query
        (not the current one which is already being analyzed).
        
        Args:
            conversation_history: Formatted conversation history string
            
        Returns:
            Previous query string or None if not found
        """
        if not conversation_history:
            return None
            
        lines = conversation_history.split('\n')
        
        # Find all USER queries in various formats
        user_queries = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
                
            # Format 1: "USER:" or "User:" prefix
            if line.startswith(('USER:', 'User:')):
                query = line.split(':', 1)[1].strip()
                if query:
                    user_queries.append(query)
            
            # Format 2: Lines that look like questions (enhanced session format)
            elif ('?' in line and len(line) > 10 and 
                  not line.startswith(('SQL:', 'Results:', '[', 'INFO:', 'DEBUG:', 'WARNING:'))):
                user_queries.append(line)
            
            # Format 3: Lines that contain common query keywords
            elif any(keyword in line.lower() for keyword in ['how many', 'list', 'show', 'find', 'get', 'what', 'which']):
                if not line.startswith(('SQL:', 'Results:', '[', 'INFO:', 'DEBUG:', 'WARNING:')):
                    user_queries.append(line)
        
        # Return the second-to-last (index 1 gives us the previous query if there are at least 2)
        if len(user_queries) >= 2:
            return user_queries[1]  # Index 1 is the second element (previous query)
        elif len(user_queries) == 1:
            return user_queries[0]
        
        return None
    
    def extract_context_from_history(
        self,
        previous_query: str,
        conversation_history: str,
        result_count: int
    ) -> Optional[PreviousQueryContext]:
        """
        Extract context from conversation history when SQL is not available.
        
        Useful for refinement follow-ups where we detected the follow-up intent
        but couldn't find the explicit previous SQL in the conversation.
        
        Args:
            previous_query: The previous user's natural language query
            conversation_history: Full conversation history
            result_count: Estimated result count from previous query
            
        Returns:
            PreviousQueryContext with inferred values, or None if can't extract
        """
        # Try to infer table from query patterns
        table_name = self.infer_table_from_query(previous_query)
        
        # Try to extract filters from the previous query text
        filters = self.extract_filters_from_text(previous_query)
        
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
    
    def infer_table_from_query(self, query: str) -> Optional[str]:
        """
        Infer table name from query text using generic patterns.
        
        Looks for "from [table]" or common table references.
        NO hardcoded entity lists - returns None for dynamic resolution.
        
        Args:
            query: User query text
            
        Returns:
            Table name if found, None otherwise
        """
        query_lower = query.lower()
        
        # Pattern 1: Explicit "from X" clause
        from_match = re.search(r'from\s+(\w+)', query_lower)
        if from_match:
            return from_match.group(1)
        
        # No hardcoded table names - let calling code determine from context
        return None
    
    def extract_filters_from_text(self, query: str) -> List[Dict[str, str]]:
        """
        Extract potential filters from query text using regex patterns.

        Column names are set to None so the downstream LLM/SQL generator can
        determine which column the extracted value belongs to.

        Args:
            query: User query text

        Returns:
            List of filter dicts with column=None and raw_text for LLM resolution
        """
        filters = []

        # Pattern 1: Short location codes like "in AP" or "from TX" (2 uppercase letters)
        state_match = re.search(r'(?:in|from|at|within)\s+([A-Z]{2})\b', query, re.IGNORECASE)
        if state_match:
            filters.append({
                "column": None,
                "operator": "=",
                "value": f"'{state_match.group(1).upper()}'",
                "raw_text": state_match.group(0),
            })

        # Pattern 2: City/location patterns (longer than 2 chars, not already captured)
        city_match = re.search(r'(?:in|from|at|near|around)\s+(\w+(?:\s+\w+)*)', query, re.IGNORECASE)
        if city_match and not state_match:
            location = city_match.group(1)
            if len(location) <= 2:
                filters.append({
                    "column": None,
                    "operator": "=",
                    "value": f"'{location.upper()}'",
                    "raw_text": city_match.group(0),
                })
            else:
                filters.append({
                    "column": None,
                    "operator": "ILIKE",
                    "value": f"'{location.lower()}%'",
                    "raw_text": city_match.group(0),
                })

        # Pattern 3: Amount-related filters
        amount_match = re.search(
            r'(?:amount|worth|value|price|cost)\s+(?:more|greater|over|less|under|than)\s+(\d+)',
            query, re.IGNORECASE,
        )
        if amount_match:
            operator = ">" if re.search(r'more|greater|over', query, re.IGNORECASE) else "<"
            filters.append({
                "column": None,
                "operator": operator,
                "value": amount_match.group(1),
                "raw_text": amount_match.group(0),
            })

        # Pattern 4: Date-related filters
        date_match = re.search(
            r'(?:on|after|before|since|during)\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+\s+\d{1,2})',
            query, re.IGNORECASE,
        )
        if date_match:
            filters.append({
                "column": None,
                "operator": "=",
                "value": f"'{date_match.group(1)}'",
                "raw_text": date_match.group(0),
            })

        # Pattern 5: Year filters
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            filters.append({
                "column": None,
                "operator": "=",
                "value": year_match.group(1),
                "raw_text": year_match.group(0),
            })

        # Pattern 6: Quarter filters
        quarter_match = re.search(r'\b(Q[1-4])\b', query, re.IGNORECASE)
        if quarter_match:
            filters.append({
                "column": None,
                "operator": "=",
                "value": f"'{quarter_match.group(1).upper()}'",
                "raw_text": quarter_match.group(0),
            })

        return filters
    
    def extract_table_from_sql(self, sql: str) -> Optional[str]:
        """
        Extract the main table name from a SQL query.
        
        Args:
            sql: SQL query string
            
        Returns:
            Table name or None
        """
        # Match FROM clause, handling schema prefixes
        match = re.search(
            r'(?:FROM|JOIN)\s+(?:ONLY\s+)?(?:(\w+)\.)?(\w+)',
            sql,
            re.IGNORECASE
        )
        if match:
            # Return table name (group 2), could also return schema.table
            return match.group(2)
        return None


# Singleton instance
_extractor_instance: Optional[QueryContextExtractor] = None


def get_query_context_extractor() -> QueryContextExtractor:
    """Get or create the query context extractor singleton."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = QueryContextExtractor()
    return _extractor_instance
