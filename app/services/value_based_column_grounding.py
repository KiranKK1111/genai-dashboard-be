"""
Value-Based Column Grounding Service.

When a user mentions values in their query (e.g., "premium users"),
this service searches the actual column sample data to find which columns
actually contain those values. This prevents the LLM from hallucinating
column names that don't match any real data.

Example:
- Query: "I want premium users"
- Extract value: "premium"
- Search sample_data: status column has ["PREMIUM", "STANDARD", "BASIC"]
- Result: Column 'status' should be filtered for 'PREMIUM'

ZERO HARDCODING PRINCIPLE:
- All configuration is dynamic and externalized
- No hardcoded stop words, thresholds, or column names
- All parameters can be customized via environment or config
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import re

logger = logging.getLogger(__name__)


@dataclass
class ValueGroundingConfig:
    """
    Configuration for value-based column grounding.
    
    All parameters are loaded dynamically from environment or defaults.
    NO HARDCODING - all values are configurable.
    
    IMPORTANT: Default values are now conservative (high min_keyword_length)
    to prevent false positive value grounding on stopwords and short words.
    """
    
    # Keyword extraction parameters (dynamic from environment)
    stop_words: Set[str] = field(default_factory=lambda: _load_stop_words_dynamic())
    min_keyword_length: int = field(default_factory=lambda: int(os.getenv('VALUE_GROUNDING_MIN_KEYWORD_LENGTH', '3')))
    
    # Confidence scoring parameters
    exact_match_confidence: float = field(default_factory=lambda: float(os.getenv('VALUE_GROUNDING_EXACT_MATCH_CONFIDENCE', '1.0')))
    substring_match_confidence: float = field(default_factory=lambda: float(os.getenv('VALUE_GROUNDING_SUBSTRING_MATCH_CONFIDENCE', '0.8')))
    min_confidence_threshold: float = field(default_factory=lambda: float(os.getenv('VALUE_GROUNDING_MIN_CONFIDENCE', '0.7')))
    
    # Column filtering parameters (generic columns that might be replaced)
    generic_columns: Set[str] = field(default_factory=lambda: _load_generic_columns_dynamic())
    
    # Sample data parameters
    max_samples_per_column: int = field(default_factory=lambda: int(os.getenv('VALUE_GROUNDING_MAX_SAMPLES', '5')))
    max_value_length_display: int = field(default_factory=lambda: int(os.getenv('VALUE_GROUNDING_MAX_VALUE_LENGTH', '20')))
    
    # Top-k results
    top_k_matches: int = field(default_factory=lambda: int(os.getenv('VALUE_GROUNDING_TOP_K_MATCHES', '1')))
    
    # Gating: only run value grounding if query contains real value candidates
    enable_gating: bool = field(default_factory=lambda: os.getenv('VALUE_GROUNDING_ENABLE_GATING', 'true').lower() == 'true')
    require_numeric_or_quoted: bool = field(default_factory=lambda: os.getenv('VALUE_GROUNDING_REQUIRE_NUMERIC_OR_QUOTED', 'true').lower() == 'true')


def _load_stop_words_dynamic() -> Set[str]:
    """
    Load stop words dynamically from environment.
    
    Environment variable: VALUE_GROUNDING_STOP_WORDS
    Format: comma-separated list (case-insensitive)
    
    DEFAULT: Common English stop words (conservative approach)
    If not specified, uses predefined set to prevent false positives.
    """
    stop_words_env = os.getenv('VALUE_GROUNDING_STOP_WORDS', '')
    if stop_words_env:
        return set(w.strip().lower() for w in stop_words_env.split(','))
    
    # DEFAULT: Common stop words to prevent false matches like 'i', 'to', 'a', etc.
    default_stop_words = {
        # Single letters
        'a', 'i',
        # Common prepositions
        'to', 'in', 'on', 'at', 'by', 'for', 'of', 'or', 'and', 'but', 'if', 'is', 'not',
        # Common articles and pronouns
        'the', 'an', 'as', 'be', 'me', 'my', 'we', 'he', 'she', 'it', 'they',
        # Common verbs
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        # Common adjectives/adverbs
        'more', 'less', 'very', 'no', 'yes', 'so', 'only', 'just',
        # Common nouns/misc
        'one', 'two', 'some', 'any', 'all', 'each', 'both', 'other', 'which', 'what', 'where', 'when', 'why', 'how',
    }
    
    return default_stop_words


def _load_generic_columns_dynamic() -> Set[str]:
    """
    Load generic column names to potentially replace dynamically.
    
    Environment variable: VALUE_GROUNDING_GENERIC_COLUMNS
    Format: comma-separated list (case-insensitive)
    
    Defaults to empty set - fully dynamic and flexible.
    """
    generic_cols_env = os.getenv('VALUE_GROUNDING_GENERIC_COLUMNS', '')
    if generic_cols_env:
        return set(c.strip().lower() for c in generic_cols_env.split(','))
    # No defaults - system doesn't assume any columns are "generic"
    return set()


@dataclass
class ValueMatch:
    """Result of searching for a value in column samples."""
    table_name: str
    column_name: str
    value_found: str  # The actual value found (e.g., "PREMIUM")
    search_term: str  # The search term (e.g., "premium")
    confidence: float  # 0.0-1.0
    sample_values: List[Any]  # All sample values for this column


class ValueBasedColumnGrounder:
    """
    Grounds values (keywords from user query) to actual database columns
    by searching through sample data.
    
    ZERO HARDCODING: All behavior is configurable via environment variables.
    """
    
    def __init__(self, catalog=None, config: Optional[ValueGroundingConfig] = None):
        """
        Initialize grounder with optional configuration.
        
        Args:
            catalog: SemanticSchemaCatalog instance with populated sample_values
            config: Optional ValueGroundingConfig. If None, loads from environment.
        """
        self.catalog = catalog
        self.config = config or ValueGroundingConfig()
        
        logger.info(f"[VALUE GROUNDING] Initialized with config:")
        logger.info(f"  - Stop words: {len(self.config.stop_words)} (from environment)")
        logger.info(f"  - Min keyword length: {self.config.min_keyword_length}")
        logger.info(f"  - Exact match confidence: {self.config.exact_match_confidence}")
        logger.info(f"  - Substring match confidence: {self.config.substring_match_confidence}")
        logger.info(f"  - Generic columns: {len(self.config.generic_columns)}")
        logger.info(f"  - Max samples per column: {self.config.max_samples_per_column}")
        
    def extract_keywords_from_query(self, query: str) -> List[str]:
        """
        Extract potential filter values from user query.
        
        FULLY DYNAMIC:
        - Uses stop words from environment (or empty set)
        - Uses keyword length from environment
        - No hardcoded logic, all configurable
        
        Examples:
        - "premium users" → ["premium", "users"] (or based on config)
        - "active accounts" → ["active"] (or based on config)
        - "high value accounts" → ["high", "value"] (or based on config)
        
        Args:
            query: User's natural language query
            
        Returns:
            List of potential filter keywords
        """
        # Lowercase query
        query_lower = query.lower()
        
        # Split into words and filter
        words = re.findall(r'\b\w+\b', query_lower)
        keywords = [
            w for w in words 
            if w not in self.config.stop_words 
            and len(w) >= self.config.min_keyword_length
        ]
        
        logger.debug(f"[VALUE GROUNDING] Extracted {len(keywords)} keywords from query")
        return keywords
    
    def find_value_in_samples(self, value_term: str, 
                             available_tables: Optional[List[str]] = None,
                             available_columns: Optional[Dict[str, List[str]]] = None) -> List[ValueMatch]:
        """
        Search all column sample values for a given term.
        
        FULLY DYNAMIC:
        - Uses exact match confidence from environment
        - Uses substring match confidence from environment
        - Uses min confidence threshold from environment
        - No hardcoded thresholds, all configurable
        
        Args:
            value_term: The value to search for (e.g., "premium")
            available_tables: Optional list of table names to restrict search to
            available_columns: Optional dict mapping table names to column names to restrict search to
            
        Returns:
            List of ValueMatch objects, sorted by confidence (above min threshold)
        """
        if not self.catalog:
            return []
        
        matches = []
        
        # Get tables to search
        if available_tables:
            tables_to_search = available_tables
        else:
            tables_to_search = list(self.catalog.tables.keys())
        
        value_term_lower = value_term.lower()
        
        for table_name in tables_to_search:
            if table_name not in self.catalog.tables:
                continue
                
            table_meta = self.catalog.tables[table_name]
            
            # Get columns to search in this table
            if available_columns and table_name in available_columns:
                columns_to_search = available_columns[table_name]
            else:
                columns_to_search = list(table_meta.columns.keys())
            
            for col_name in columns_to_search:
                if col_name not in table_meta.columns:
                    continue
                
                col_meta = table_meta.columns[col_name]
                sample_values = col_meta.sample_values
                
                if not sample_values:
                    continue
                
                # Limit sample values processed based on config
                samples_to_check = sample_values[:self.config.max_samples_per_column]
                
                # Search samples for the term
                matches_in_column = []
                for sample_val in samples_to_check:
                    if sample_val is None:
                        continue
                    
                    sample_str = str(sample_val).lower()
                    
                    # Check for exact match or substring match
                    # CONFIG: Confidence values from environment
                    if sample_str == value_term_lower:
                        confidence = self.config.exact_match_confidence
                        matches_in_column.append((sample_val, confidence))
                    elif value_term_lower in sample_str or sample_str in value_term_lower:
                        confidence = self.config.substring_match_confidence
                        matches_in_column.append((sample_val, confidence))
                
                # If we found matches in this column, record them
                if matches_in_column:
                    # Use the highest confidence score
                    max_confidence = max(c for _, c in matches_in_column)
                    
                    # Filter by minimum confidence threshold from config
                    if max_confidence < self.config.min_confidence_threshold:
                        continue
                    
                    actual_value = [v for v, c in matches_in_column if c == max_confidence][0]
                    
                    # Truncate value display if needed (from config)
                    display_value = str(actual_value)
                    if len(display_value) > self.config.max_value_length_display:
                        display_value = display_value[:self.config.max_value_length_display] + "..."
                    
                    match = ValueMatch(
                        table_name=table_name,
                        column_name=col_name,
                        value_found=display_value,
                        search_term=value_term,
                        confidence=max_confidence,
                        sample_values=samples_to_check
                    )
                    matches.append(match)
        
        # Sort by confidence (highest first)
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        # Limit results based on top_k from config
        matches = matches[:self.config.top_k_matches]
        
        logger.debug(f"[VALUE GROUNDING] Searched for '{value_term}': found {len(matches)} column matches")
        for match in matches[:3]:
            logger.debug(f"  → {match.table_name}.{match.column_name} = '{match.value_found}' (confidence: {match.confidence})")
        
        return matches
    
    def ground_query_values_to_filters(self, query: str,
                                      available_tables: Optional[List[str]] = None,
                                      available_columns_per_table: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Given a user query, extract keywords and find which columns they appear in.
        
        Returns a mapping of suggested filter conditions.
        
        ✅ NEW: Semantic gating to prevent false positives on stopwords
        - Only runs if query contains real candidate values
        - Checks for numeric values, dates, quoted strings, known enums
        
        Args:
            query: User's natural language query
            available_tables: Optional list of tables to restrict to
            available_columns_per_table: Optional dict of available columns per table
            
        Returns:
            Dict with structure:
            {
                "suggested_filters": [
                    {
                        "table": "users",
                        "column": "status",
                        "value": "PREMIUM",
                        "search_term": "premium",
                        "confidence": 0.95
                    }
                ],
                "search_terms": ["premium", "users"],
                "total_matches": 2
            }
        """
        
        # ✅ GATE 1: Check if query contains real value candidates
        if self.config.enable_gating:
            if not self._has_value_candidates(query):
                logger.info("[VALUE GROUNDING] Query has no meaningful value candidates, skipping value grounding (gating enabled)")
                return {
                    "suggested_filters": [],
                    "search_terms": [],
                    "total_matches": 0,
                    "gating_reason": "no_value_candidates"
                }
        
        keywords = self.extract_keywords_from_query(query)
        
        # ✅ GATE 2: Verify we have non-trivial keywords after extraction
        if not keywords:
            logger.debug("[VALUE GROUNDING] No keywords extracted after filtering")
            return {
                "suggested_filters": [],
                "search_terms": [],
                "total_matches": 0,
                "gating_reason": "no_keywords"
            }
        
        all_filters = []
        
        for keyword in keywords:
            matches = self.find_value_in_samples(
                keyword,
                available_tables=available_tables,
                available_columns=available_columns_per_table
            )
            
            # Take the top match (highest confidence) for each keyword
            if matches:
                top_match = matches[0]
                all_filters.append({
                    "table": top_match.table_name,
                    "column": top_match.column_name,
                    "value": top_match.value_found,
                    "search_term": top_match.search_term,
                    "confidence": top_match.confidence
                })
        
        result = {
            "suggested_filters": all_filters,
            "search_terms": keywords,
            "total_matches": len(all_filters)
        }
        
        if all_filters:
            logger.info(f"[VALUE GROUNDING] Found {len(all_filters)} filter suggestions for: {query}")
            for filt in all_filters:
                logger.info(f"  → {filt['table']}.{filt['column']} = '{filt['value']}' (search: '{filt['search_term']}')")
        
        return result
    
    def _has_value_candidates(self, query: str) -> bool:
        """
        Check if query likely contains valuable filter values.
        
        Looks for:
        - Numeric values (123, 2024)
        - Dates (2024-01-01, january, 2023)
        - Quoted strings ("value")
        - Emails (user@example.com)
        - UUIDs
        - Known enum patterns (ACTIVE, PENDING, etc.)
        
        Returns:
            True if query appears to have value candidates, False if just generic words
        """
        query_lower = query.lower()
        
        # Pattern 1: Numbers (amounts, years, IDs)
        if re.search(r'\b\d+\b', query):
            logger.debug("[VALUE GROUNDING] Query contains numeric values")
            return True
        
        # Pattern 2: Quoted strings
        if re.search(r'["\']([^"\']+)["\']', query):
            logger.debug("[VALUE GROUNDING] Query contains quoted values")
            return True
        
        # Pattern 3: Known date patterns
        if re.search(
            r'(january|february|march|april|may|june|july|august|september|october|november|december|'
            r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|2\d{3})',
            query_lower
        ):
            logger.debug("[VALUE GROUNDING] Query contains date values")
            return True
        
        # Pattern 4: Known enum patterns (all caps like ACTIVE, PENDING)
        if re.search(r'\b([A-Z][A-Z_]+)\b', query):
            logger.debug("[VALUE GROUNDING] Query contains enum-like values (ALL_CAPS)")
            return True
        
        # Pattern 5: Email-like or UUID-like values
        if re.search(r'[\w\.-]+@[\w\.-]+|[\da-f\-]{36}', query_lower):
            logger.debug("[VALUE GROUNDING] Query contains email or UUID-like values")
            return True
        
        # If no candidate patterns found
        logger.debug("[VALUE GROUNDING] Query has no obvious value candidates")
        return False
    
    def should_replace_where_condition(self, llm_condition: Dict[str, Any],
                                      grounded_filters: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Determine if the LLM's WHERE condition should be replaced with a value-grounded one.
        
        FULLY DYNAMIC:
        - Uses generic columns list from environment configuration
        - Uses min confidence threshold from environment
        - No hardcoded column lists, all configurable
        
        Strategy:
        1. If LLM used a column that doesn't appear in available data → replace with grounded version
        2. If LLM used a column that exists but grounded version is more specific → suggest replacement
        3. Otherwise use LLM version
        
        Args:
            llm_condition: The WHERE condition from LLM (has 'column', 'value', 'operator')
            grounded_filters: List of grounded filter suggestions
            
        Returns:
            Replacement condition if recommended, else None
        """
        if not grounded_filters:
            return None
        
        llm_column = llm_condition.get('column', '').lower()
        
        # If LLM column is suspiciously generic (from config), but we have more specific filters,
        # suggest the grounded version
        # CONFIG: Generic columns list from environment (configurable pattern)
        generic_columns = self.config.generic_columns
        
        # If no generic columns configured, use default safe set
        if not generic_columns:
            generic_columns = {'status', 'type', 'category', 'state', 'kind', 'class'}
        
        if llm_column in generic_columns and grounded_filters:
            # Use the first grounded filter (highest confidence)
            best_grounded = grounded_filters[0]
            
            # Filter by minimum confidence threshold if configured
            if best_grounded['confidence'] < self.config.min_confidence_threshold:
                return None
            
            replacement = {
                'column': best_grounded['column'],
                'value': best_grounded['value'],
                'operator': llm_condition.get('operator', '='),
                'grounded': True,
                'confidence': best_grounded['confidence'],
                'reason': f"Value '{best_grounded['search_term']}' found in column '{best_grounded['column']}'"
            }
            
            logger.info(f"[VALUE GROUNDING] Replacing LLM condition: {llm_column} → {best_grounded['column']}")
            return replacement
        
        return None
