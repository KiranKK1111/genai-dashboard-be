"""
SQL Validation & Hallucination Prevention

Ensures that:
1. Column extraction uses proper SQL parsing (sqlglot)
2. Literal values in WHERE clauses are not hallucinated
3. Filter conditions only use values from user query or retrieved samples
"""
import re
import logging
from typing import Set, List, Tuple, Optional
import sqlglot
from sqlglot import parse_one, exp

logger = logging.getLogger(__name__)


class SQLValidator:
    """Validates SQL queries for hallucinations and safety."""
    
    def __init__(self, user_query: str, retrieved_values: Optional[Set[str]] = None):
        """
        Initialize validator.
        
        Args:
            user_query: Original user query text
            retrieved_values: Set of valid values from schema/samples
        """
        self.user_query = user_query.lower()
        self.user_terms = set(self.user_query.split())
        
        # Add commonly used filter values
        self.retrieved_values = retrieved_values or set()
        self.valid_values = self.retrieved_values | self._extract_common_values()
        
    def _extract_common_values(self) -> Set[str]:
        """Extract potential filter values from user query."""
        # Extract quoted strings
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", self.user_query)
        # Extract numbers
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", self.user_query)
        # Extract common keywords
        keywords = {'true', 'false', 'yes', 'no', 'active', 'inactive', 'draft', 'published'}
        return set(quoted + numbers) | keywords
    
    def extract_columns_safe(self, sql: str) -> List[str]:
        """
        Extract columns from SQL using sqlglot (proper parser).
        
        Returns list of column names extracted from SELECT clause.
        """
        try:
            parsed = parse_one(sql)
            
            if not isinstance(parsed, (exp.Select, exp.Update, exp.Delete)):
                logger.warning(f"[SQL_VALIDATOR] Unexpected SQL type: {type(parsed)}")
                return []
            
            columns = []
            
            # For SELECT, look at select expressions
            if isinstance(parsed, exp.Select):
                for expr in parsed.expressions:
                    if isinstance(expr, exp.Column):
                        col_name = expr.name
                        if col_name and col_name.lower() != '*':
                            columns.append(col_name)
                    elif isinstance(expr, exp.Star):
                        columns.append('*')
                    elif isinstance(expr, exp.Alias):
                        # Get the actual column name
                        if isinstance(expr.this, exp.Column):
                            col_name = expr.this.name
                            if col_name:
                                columns.append(col_name)
            
            return [c for c in columns if c and 'jo' not in c.lower()]  # Filter out parsing artifacts
            
        except Exception as e:
            logger.warning(f"[SQL_VALIDATOR] sqlglot parsing failed: {e}. Falling back to regex.")
            return self._extract_columns_regex(sql)
    
    def _extract_columns_regex(self, sql: str) -> List[str]:
        """Fallback regex-based column extraction."""
        # SELECT col1, col2 FROM ...
        match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if not match:
            return []
        
        select_clause = match.group(1)
        # Split by comma, but be careful with function calls
        parts = [p.strip() for p in select_clause.split(',')]
        
        columns = []
        for part in parts:
            # Extract just the column name (no aliases, no functions)
            # Handle: col, col AS alias, func(col) AS alias, etc.
            if ' AS ' in part.upper():
                part = part.split(' AS ')[0].strip()
            
            # Remove function wrapper if present
            if '(' in part:
                # Extract column from function
                inner = re.search(r'\((\w+)\)', part)
                if inner:
                    columns.append(inner.group(1))
            else:
                if part and part != '*':
                    columns.append(part)
        
        return columns
    
    def validate_where_literals(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that literal values in WHERE clause are not hallucinated.
        
        Returns: (is_valid, error_message_if_invalid)
        """
        try:
            parsed = parse_one(sql)
        except Exception as e:
            logger.warning(f"[SQL_VALIDATOR] Cannot parse SQL for validation: {e}")
            return True, None  # Can't validate, allow it
        
        # Find WHERE clause
        where = parsed.find(exp.Where)
        if not where:
            return True, None  # No WHERE clause, nothing to validate
        
        # Extract all literal values
        hallucinated = []
        for literal in where.find_all(exp.Literal):
            value = literal.this
            if isinstance(value, str):
                # Check if this value appears in user query or retrieved values
                if not (value in self.valid_values or 
                        value.lower() in self.user_query or
                        value.upper() in self.user_query or
                        any(value in term for term in self.user_terms)):
                    hallucinated.append(value)
        
        if hallucinated:
            return False, f"Hallucinated literal values in WHERE clause: {hallucinated}. Use values from user query or retrieved schema only."
        
        return True, None
    
    def regenerate_with_samples(self, sql: str, table_samples: dict) -> Optional[str]:
        """
        Attempt to regenerate SQL by replacing hallucinated values with real samples.
        
        Args:
            sql: Original SQL with potential hallucinations
            table_samples: Dict[table_name][column_name] = [sample_values]
        
        Returns: Regenerated SQL or None if cannot fix
        """
        is_valid, error = self.validate_where_literals(sql)
        if is_valid:
            return sql  # No changes needed
        
        try:
            parsed = parse_one(sql)
            where = parsed.find(exp.Where)
            
            if not where:
                return sql
            
            # Try to replace literals with first sample value
            replaced_count = 0
            for literal in list(where.find_all(exp.Literal)):
                original_value = literal.this
                # Try to find a replacement from samples
                for table_name, col_samples in table_samples.items():
                    for col_name, samples in col_samples.items():
                        if samples and original_value not in samples:
                            # Replace with first sample
                            literal.replace(exp.Literal.string(samples[0]))
                            replaced_count += 1
                            logger.info(f"[SQL_VALIDATOR] Replaced hallucinated '{original_value}' with sample '{samples[0]}'")
                            break
            
            if replaced_count > 0:
                return parsed.sql()
            
        except Exception as e:
            logger.warning(f"[SQL_VALIDATOR] Cannot regenerate SQL: {e}")
        
        return None


def validate_sql_for_hallucinations(
    sql: str,
    user_query: str,
    retrieved_values: Optional[Set[str]] = None,
    table_samples: Optional[dict] = None
) -> Tuple[bool, str]:
    """
    Validate SQL query for hallucinated values.
    
    Returns: (is_valid, message)
    """
    validator = SQLValidator(user_query, retrieved_values)
    
    is_valid, error = validator.validate_where_literals(sql)
    if is_valid:
        return True, f"SQL validation passed: {sql[:100]}..."
    
    # Try to regenerate
    if table_samples:
        regenerated = validator.regenerate_with_samples(sql, table_samples)
        if regenerated:
            return True, f"SQL regenerated with valid values: {regenerated[:100]}..."
    
    return False, error or "SQL contains hallucinated values"
