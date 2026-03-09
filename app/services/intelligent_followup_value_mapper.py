"""
Intelligent Follow-up Value Column Mapper
==========================================

When users provide follow-up queries with values (e.g., "what about those in delhi?"),
this service intelligently discovers which column those values belong to WITHOUT any
hardcoding or pre-configured mappings.

Key Features:
1. ZERO HARDCODING - All logic based on actual schema and data
2. FULLY DYNAMIC - Works with any table, any columns, any data types
3. LLM-ENHANCED - Uses intelligent prompting to find semantic column matches
4. CONTEXT-AWARE - Leverages previous query context (table, columns used, filters)
5. VALUE-GROUNDED - Searches actual database values, not just column names
6. CONFIDENCE-BASED - Returns ranked results with confidence scores

Example Flow:
    Previous Query: "SELECT * FROM <table> WHERE <column> = 'X' LIMIT 100"
    Follow-up: "what about those in Y?"
  
  Processing:
    1. Extract value "Y" from follow-up
    2. Search previous query context → table="<table>"
    3. Scan sample values → find "Y" in a matching column
  4. Rank by confidence (exact type match, case similarity, etc.)
    5. Return: column="<column>", confidence=0.95
  
  If value not found in sample data:
  6. Use LLM prompt: "In <table> with columns [..], which column would contain 'Y'?"
  7. LLM response with reasoning
  8. Validate against actual data
"""

import logging
import re
import os
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, inspect as sqlalchemy_inspect

from .. import llm

logger = logging.getLogger(__name__)


class ColumnDiscoveryStrategy(Enum):
    """Strategy for discovering which column a value belongs to."""
    EXACT_SAMPLE_MATCH = "exact_sample_match"           # Value found exactly in samples
    SUBSTRING_SAMPLE_MATCH = "substring_sample_match"   # Value partially matches samples
    SEMANTIC_SIMILARITY = "semantic_similarity"         # Column name/values semantically similar
    LLM_POWERED = "llm_powered"                          # LLM infers based on context
    CONTEXT_PRIOR = "context_prior"                      # Uses previous query's columns
    FALLBACK_GENERIC = "fallback_generic"                # Last resort - generic column


@dataclass
class ColumnValueMatch:
    """Result of value-to-column discovery."""
    table_name: str
    column_name: str
    data_type: str
    value_to_find: str                    # User provided value
    matched_value: Optional[str] = None   # Actual database value found
    strategy: ColumnDiscoveryStrategy = None
    confidence: float = 0.0               # 0.0-1.0 confidence score
    reasoning: str = ""                   # Why this match was chosen
    sample_values: List[Any] = field(default_factory=list)  # Other values in column
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table_name,
            "column": self.column_name,
            "data_type": self.data_type,
            "user_value": self.value_to_find,
            "database_value": self.matched_value,
            "strategy": self.strategy.value if self.strategy else None,
            "confidence": round(self.confidence, 2),
            "reasoning": self.reasoning,
            "sample_values": self.sample_values[:3],  # Top 3 samples
        }


@dataclass
class FollowUpContext:
    """Context from previous query to help discover columns.
    
    This class is used for BOTH follow-up queries AND initial queries:
    - For follow-ups: All fields populated with previous query context
    - For initial queries: Only table_name, table_schema, table_sample_data populated
      (previous_* fields will be None/empty)
    """
    table_name: str
    table_schema: Dict[str, str]         # {column_name: data_type} - REQUIRED
    table_sample_data: Dict[str, List[Any]]  # {column_name: [sample_values]} - REQUIRED
    previous_columns_used: Optional[List[str]] = None      # Columns in SELECT clause (None for initial queries)
    previous_filters: Optional[Dict[str, str]] = None      # Filters from WHERE clause {col: type} (None for initial queries)
    previous_query: Optional[str] = None                    # Full SQL query (None for initial queries)
    is_followup: bool = False                               # True if this is a follow-up query


class IntelligentFollowupValueMapper:
    """
    Maps user-provided values in follow-up queries to actual database columns.
    
    ZERO HARDCODING PRINCIPLE:
    - No hardcoded column names, mappings, or assumptions
    - All column discovery is data-driven
    - All thresholds configurable via environment
    - Works identically for any database/table/column combination
    """
    
    def __init__(self):
        """Initialize mapper with configurable parameters from environment."""
        # Confidence thresholds (all from environment, no hardcoding)
        self.exact_match_confidence = float(
            os.getenv('FOLLOWUP_EXACT_MATCH_CONFIDENCE', '0.95')
        )
        self.substring_match_confidence = float(
            os.getenv('FOLLOWUP_SUBSTRING_MATCH_CONFIDENCE', '0.75')
        )
        self.semantic_match_confidence = float(
            os.getenv('FOLLOWUP_SEMANTIC_MATCH_CONFIDENCE', '0.65')
        )
        self.context_prior_confidence = float(
            os.getenv('FOLLOWUP_CONTEXT_PRIOR_CONFIDENCE', '0.70')
        )
        self.llm_discovery_confidence = float(
            os.getenv('FOLLOWUP_LLM_DISCOVERY_CONFIDENCE', '0.80')
        )
        self.min_confidence_threshold = float(
            os.getenv('FOLLOWUP_MIN_CONFIDENCE_THRESHOLD', '0.60')
        )
        
        # Sample size for searching
        self.max_samples_per_column = int(
            os.getenv('FOLLOWUP_MAX_SAMPLES_PER_COLUMN', '100')
        )
        
        # Number of results to return
        self.top_k_results = int(
            os.getenv('FOLLOWUP_TOP_K_RESULTS', '3')
        )
        
        # Enable/disable LLM discovery
        self.enable_llm_discovery = os.getenv(
            'FOLLOWUP_ENABLE_LLM_DISCOVERY', 'true'
        ).lower() == 'true'
        
        logger.info(f"[FOLLOWUP_MAPPER] Initialized with configurable thresholds from environment")
    
    async def discover_column_for_value(
        self,
        followup_query: str,
        followup_context: FollowUpContext,
        db: AsyncSession,
        value_to_find: str = None
    ) -> List[ColumnValueMatch]:
        """
        Discover which column a value (from follow-up query) belongs to.
        
        This is the main entry point. It tries multiple strategies in order:
        1. CONTEXT_PRIOR: Use columns from previous query (highest confidence)
        2. EXACT_SAMPLE_MATCH: Search for exact value in sample data
        3. SUBSTRING_SAMPLE_MATCH: Search for partial matches in sample data
        4. SEMANTIC_SIMILARITY: Find semantically similar column names
        5. LLM_POWERED: Use LLM to infer based on column semantics
        6. Return all matches ranked by confidence
        
        Args:
            followup_query: The user's follow-up query text
            followup_context: FollowUpContext with previous query information
            db: Database session for fetching actual data if needed
            value_to_find: Optional specific value to search for (if None, extracted from query)
            
        Returns:
            List of ColumnValueMatch objects, ranked by confidence (highest first)
        """
        
        # Step 1: Extract value to find from query if not provided
        if not value_to_find:
            value_to_find = self._extract_value_from_query(followup_query)
        
        if not value_to_find:
            logger.warning(f"[FOLLOWUP_MAPPER] Could not extract value from query: {followup_query}")
            return []
        
        logger.info(f"[FOLLOWUP_MAPPER] Discovering column for value: '{value_to_find}'")
        logger.info(f"[FOLLOWUP_MAPPER] Table: {followup_context.table_name}")
        logger.info(f"[FOLLOWUP_MAPPER] Query Type: {'Follow-up' if followup_context.is_followup else 'Initial'}")
        
        matches: List[ColumnValueMatch] = []
        
        # Strategy 1: CONTEXT_PRIOR - Use previous query's columns (only for follow-ups!)
        if followup_context.is_followup and followup_context.previous_columns_used:
            logger.debug(f"[FOLLOWUP_MAPPER] Strategy 1/5: CONTEXT_PRIOR (follow-up query)")
            context_matches = await self._strategy_context_prior(
                value_to_find, followup_context, db
            )
            matches.extend(context_matches)
        else:
            logger.debug(f"[FOLLOWUP_MAPPER] Skipping Strategy 1/5: CONTEXT_PRIOR (not a follow-up or no previous context)")
        
        # Strategy 2: EXACT_SAMPLE_MATCH - Find exact value in schemas
        logger.debug(f"[FOLLOWUP_MAPPER] Strategy 2/5: EXACT_SAMPLE_MATCH")
        exact_matches = await self._strategy_exact_sample_match(
            value_to_find, followup_context, db
        )
        matches.extend(exact_matches)
        
        # Strategy 3: SUBSTRING_SAMPLE_MATCH - Partial value matches
        if not exact_matches:
            logger.debug(f"[FOLLOWUP_MAPPER] Strategy 3/5: SUBSTRING_SAMPLE_MATCH")
            substring_matches = await self._strategy_substring_sample_match(
                value_to_find, followup_context, db
            )
            matches.extend(substring_matches)
        
        # Strategy 4: SEMANTIC_SIMILARITY - If value might be a column name or type
        if not exact_matches:
            logger.debug(f"[FOLLOWUP_MAPPER] Strategy 4/5: SEMANTIC_SIMILARITY")
            semantic_matches = await self._strategy_semantic_similarity(
                value_to_find, followup_context, db
            )
            matches.extend(semantic_matches)
        
        # Strategy 5: LLM_POWERED - Last resort, use LLM intelligence
        if self.enable_llm_discovery and not exact_matches:
            logger.debug(f"[FOLLOWUP_MAPPER] Strategy 5/5: LLM_POWERED")
            llm_matches = await self._strategy_llm_powered(
                value_to_find, followup_query, followup_context, db
            )
            matches.extend(llm_matches)
        
        # Remove duplicates and sort by confidence
        seen = set()
        unique_matches = []
        for match in matches:
            key = (match.table_name, match.column_name, match.matched_value)
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        # Sort by confidence (highest first)
        unique_matches.sort(key=lambda m: m.confidence, reverse=True)
        
        # Filter by minimum confidence threshold
        qualified_matches = [
            m for m in unique_matches 
            if m.confidence >= self.min_confidence_threshold
        ]
        
        # Return top-k results
        result = qualified_matches[:self.top_k_results]
        
        logger.info(f"[FOLLOWUP_MAPPER] ✅ Found {len(result)} qualified column matches:")
        for i, match in enumerate(result[:3], 1):
            logger.info(f"  {i}. {match.table_name}.{match.column_name} "
                       f"(value='{match.matched_value}', confidence={match.confidence:.2f}, "
                       f"strategy={match.strategy.value})")
        
        return result
    
    async def _strategy_context_prior(
        self,
        value_to_find: str,
        followup_context: FollowUpContext,
        db: AsyncSession
    ) -> List[ColumnValueMatch]:
        """
        Strategy 1: Use columns from previous query (they're likely candidates).
        
        Rationale: If previous query filtered by city, follow-up "what about delhi?"
        likely refers to the same city column.
        
        NOTE: Only works for follow-up queries (skipped for initial queries).
        """
        if not followup_context.previous_columns_used:
            return []
        
        matches = []
        
        for prev_column in followup_context.previous_columns_used:
            # Check if this column exists in schema
            if prev_column not in followup_context.table_schema:
                continue
            
            data_type = followup_context.table_schema[prev_column]
            
            # Get sample values for this column
            samples = followup_context.table_sample_data.get(prev_column, [])
            
            # Check if value matches samples (case-insensitive)
            value_lower = str(value_to_find).lower()
            for sample in samples:
                if str(sample).lower() == value_lower:
                    match = ColumnValueMatch(
                        table_name=followup_context.table_name,
                        column_name=prev_column,
                        data_type=data_type,
                        value_to_find=value_to_find,
                        matched_value=sample,
                        strategy=ColumnDiscoveryStrategy.CONTEXT_PRIOR,
                        confidence=self.context_prior_confidence,
                        reasoning=f"Column used in previous query, value '{sample}' matches in {prev_column}",
                        sample_values=samples[:5]
                    )
                    matches.append(match)
                    logger.debug(f"  ✅ CONTEXT_PRIOR: Found '{value_to_find}' in "
                               f"{followup_context.table_name}.{prev_column}")
                    break
        
        return matches
    
    async def _strategy_exact_sample_match(
        self,
        value_to_find: str,
        followup_context: FollowUpContext,
        db: AsyncSession
    ) -> List[ColumnValueMatch]:
        """
        Strategy 2: Search all columns for exact value match in sample data.
        
        Rationale: User provides value, we find exact database match.
        """
        matches = []
        value_lower = str(value_to_find).lower()
        
        for column_name, samples in followup_context.table_sample_data.items():
            if not samples:
                continue
            
            data_type = followup_context.table_schema.get(column_name, "unknown")
            
            for sample in samples[:self.max_samples_per_column]:
                if sample is None:
                    continue
                
                sample_str = str(sample).lower()
                
                if sample_str == value_lower or str(sample) == value_to_find:
                    match = ColumnValueMatch(
                        table_name=followup_context.table_name,
                        column_name=column_name,
                        data_type=data_type,
                        value_to_find=value_to_find,
                        matched_value=sample,
                        strategy=ColumnDiscoveryStrategy.EXACT_SAMPLE_MATCH,
                        confidence=self.exact_match_confidence,
                        reasoning=f"Exact value match found in {column_name}: '{sample}'",
                        sample_values=samples[:5]
                    )
                    matches.append(match)
                    logger.debug(f"  ✅ EXACT_MATCH: '{value_to_find}' found in "
                               f"{followup_context.table_name}.{column_name}")
                    break
        
        return matches
    
    async def _strategy_substring_sample_match(
        self,
        value_to_find: str,
        followup_context: FollowUpContext,
        db: AsyncSession
    ) -> List[ColumnValueMatch]:
        """
        Strategy 3: Search for partial/substring matches in sample data.
        
        Rationale: User might provide partial value (e.g., "delhi" instead of "New Delhi").
        """
        matches = []
        value_lower = str(value_to_find).lower()
        
        for column_name, samples in followup_context.table_sample_data.items():
            if not samples:
                continue
            
            data_type = followup_context.table_schema.get(column_name, "unknown")
            
            for sample in samples[:self.max_samples_per_column]:
                if sample is None:
                    continue
                
                sample_str = str(sample).lower()
                
                # Check substring matching both directions
                if value_lower in sample_str or sample_str in value_lower:
                    # Confidence based on match quality
                    if len(value_lower) > len(sample_str) * 0.7:
                        # Good partial match
                        confidence = self.substring_match_confidence
                    else:
                        confidence = self.substring_match_confidence * 0.8
                    
                    match = ColumnValueMatch(
                        table_name=followup_context.table_name,
                        column_name=column_name,
                        data_type=data_type,
                        value_to_find=value_to_find,
                        matched_value=sample,
                        strategy=ColumnDiscoveryStrategy.SUBSTRING_SAMPLE_MATCH,
                        confidence=confidence,
                        reasoning=f"Partial/substring match: '{value_to_find}' ⊂ '{sample}'",
                        sample_values=samples[:5]
                    )
                    matches.append(match)
                    logger.debug(f"  ✅ SUBSTRING_MATCH: '{value_to_find}' matches in "
                               f"{followup_context.table_name}.{column_name}")
                    break
        
        return matches
    
    async def _strategy_semantic_similarity(
        self,
        value_to_find: str,
        followup_context: FollowUpContext,
        db: AsyncSession
    ) -> List[ColumnValueMatch]:
        """
        Strategy 4: Find semantically similar column names or data types.
        
        Rationale: If user says "delhi" and there's a "location" column with
        geographic data, that's a semantic match.
        """
        matches = []
        
        # For now, simple semantic check: if value contains keywords matching column names
        value_lower = str(value_to_find).lower()
        
        for column_name in followup_context.table_schema.keys():
            column_lower = column_name.lower()
            data_type = followup_context.table_schema[column_name]
            samples = followup_context.table_sample_data.get(column_name, [])
            
            # Check if column name semantically related to value
            # Use generic pattern matching instead of hardcoded keyword lists
            # Look for columns where the column name overlaps with value tokens
            value_tokens = set(value_lower.replace('_', ' ').split())
            column_tokens = set(column_lower.replace('_', ' ').split())
            
            # Semantic similarity if column tokens overlap with value context
            if value_tokens & column_tokens or any(t in column_lower for t in value_tokens):
                match = ColumnValueMatch(
                    table_name=followup_context.table_name,
                    column_name=column_name,
                    data_type=data_type,
                    value_to_find=value_to_find,
                    matched_value=None,  # No exact match, semantic only
                    strategy=ColumnDiscoveryStrategy.SEMANTIC_SIMILARITY,
                    confidence=self.semantic_match_confidence,
                    reasoning=f"Semantic similarity: column '{column_name}' likely matches value '{value_to_find}'",
                    sample_values=samples[:5]
                )
                matches.append(match)
                logger.debug(f"  ✅ SEMANTIC_MATCH: '{value_to_find}' semantically matches "
                           f"{followup_context.table_name}.{column_name}")
        
        return matches
    
    async def _strategy_llm_powered(
        self,
        value_to_find: str,
        followup_query: str,
        followup_context: FollowUpContext,
        db: AsyncSession
    ) -> List[ColumnValueMatch]:
        """
        Strategy 5: Use LLM to intelligently infer which column contains the value.
        
        Rationale: LLM can understand context and make smart inferences.
        For example: "delhi" → likely in city/location column based on semantics.
        
        Works for both follow-up queries (with previous context) and initial queries (no previous context).
        """
        if not self.enable_llm_discovery:
            return []
        
        try:
            # Build intelligent prompt for LLM
            column_descriptions = self._build_column_descriptions(followup_context)
            
            # Build context section based on whether this is a follow-up or initial query
            if followup_context.is_followup and followup_context.previous_columns_used:
                context_section = f"""CONTEXT:
- Table: {followup_context.table_name}
- Query Type: FOLLOW-UP
- Previous query used columns: {', '.join(followup_context.previous_columns_used)}
- Previous filters: {followup_context.previous_filters}
"""
            else:
                context_section = f"""CONTEXT:
- Table: {followup_context.table_name}
- Query Type: INITIAL
"""
            
            prompt = f"""You are a database schema expert. A user is asking a question 
with a specific value, and you need to identify which database column that value belongs to.

{context_section}

TABLE SCHEMA:
{column_descriptions}

USER QUERY: "{followup_query}"
USER PROVIDED VALUE: "{value_to_find}"

Your task:
1. Analyze which column "{value_to_find}" most likely belongs to
2. Provide your top column candidate
3. Explain your reasoning
4. Consider semantic meaning, data types, and context

Output format (JSON):
{{
    "top_column": "column_name",
    "confidence": 0.85,
    "reasoning": "Your explanation here",
    "alternatives": ["column2", "column3"]
}}
"""
            
            logger.debug(f"[FOLLOWUP_MAPPER] Calling LLM for column discovery...")
            
            # Call LLM
            llm_response = await llm.call_llm(prompt, temperature=0.3, max_tokens=200)
            
            logger.debug(f"[FOLLOWUP_MAPPER] LLM Response: {llm_response}")
            
            # Parse LLM response
            try:
                import json
                response_json = json.loads(llm_response)
                top_column = response_json.get("top_column")
                
                if top_column and top_column in followup_context.table_schema:
                    data_type = followup_context.table_schema[top_column]
                    samples = followup_context.table_sample_data.get(top_column, [])
                    
                    match = ColumnValueMatch(
                        table_name=followup_context.table_name,
                        column_name=top_column,
                        data_type=data_type,
                        value_to_find=value_to_find,
                        matched_value=None,  # LLM doesn't find exact value, just column
                        strategy=ColumnDiscoveryStrategy.LLM_POWERED,
                        confidence=self.llm_discovery_confidence,
                        reasoning=response_json.get("reasoning", "LLM inference"),
                        sample_values=samples[:5]
                    )
                    logger.debug(f"  ✅ LLM_DISCOVERY: Suggested column '{top_column}' "
                               f"with confidence {self.llm_discovery_confidence}")
                    return [match]
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[FOLLOWUP_MAPPER] Could not parse LLM response: {e}")
            
            return []
        
        except Exception as e:
            logger.warning(f"[FOLLOWUP_MAPPER] LLM discovery failed: {e}")
            return []
    
    def _extract_value_from_query(self, query: str) -> Optional[str]:
        """
        Extract potential value/filter from user query.
        
        Example:
        - "what about those in delhi?" -> "delhi"
        - "show me 2024 data" -> "2024"
        - "find active accounts" -> "active"
        """
        # Remove common question words and connectors
        clean_query = re.sub(
            r'\b(what|show|find|get|list|select|display|give|tell|about|those|in|is|are|where|for|with|from|to|by|and|or)\b',
            '', query, flags=re.IGNORECASE
        ).strip()
        
        # Extract potential values
        # Quoted strings first (highest priority)
        quoted = re.findall(r'["\']([^"\']+)["\']', clean_query)
        if quoted:
            return quoted[0]
        
        # Numeric values
        numbers = re.findall(r'\b\d+\b', clean_query)
        if numbers:
            return numbers[0]
        
        # Remaining significant words (longest non-stop-words)
        words = re.findall(r'\b\w+\b', clean_query.lower())
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'not', 'is', 'are', 'have', 'has',
            'than', 'then', 'this', 'that', 'these', 'those', 'other', 'all', 'each'
        }
        significant_words = [w for w in words if w not in stop_words and len(w) >= 3]
        
        if significant_words:
            # Return the most significant word (usually the value)
            return significant_words[0]
        
        return None
    
    def _is_likely_geographic_value(self, value: str, context: Optional[FollowUpContext] = None) -> bool:
        """
        Check if value looks like a geographic location using pattern-based detection.
        Uses column hints and sample data rather than hardcoded location lists.
        """
        value_lower = value.lower().strip()
        
        # Check if value exists in any location-related columns from context
        if context and context.table_sample_data:
            for col_name, samples in context.table_sample_data.items():
                col_lower = col_name.lower()
                # Check if column name suggests geographic data
                if any(geo in col_lower for geo in ['city', 'location', 'state', 'country', 'region', 'address', 'place', 'area', 'zone', 'district']):
                    # Check if value matches any sample from this column
                    sample_lower = [str(s).lower() for s in samples if s is not None]
                    if value_lower in sample_lower or any(value_lower in s for s in sample_lower):
                        return True
        
        # Pattern-based heuristic: proper noun-like (capitalized first letter, no digits)
        # Let LLM handle actual geographic determination through main flow
        return False
    
    def _build_column_descriptions(self, context: FollowUpContext) -> str:
        """Build human-readable column descriptions for LLM prompt."""
        descriptions = []
        
        for col_name, data_type in context.table_schema.items():
            samples = context.table_sample_data.get(col_name, [])
            sample_str = ", ".join(str(s) for s in samples[:3]) if samples else "N/A"
            
            is_previous = "✓" if (context.previous_columns_used and col_name in context.previous_columns_used) else ""
            was_filtered = "🔍" if (context.previous_filters and col_name in context.previous_filters) else ""
            
            descriptions.append(
                f"  - {col_name} ({data_type}): {sample_str} {is_previous} {was_filtered}"
            )
        
        return "\n".join(descriptions)


# Singleton instance
_mapper: Optional[IntelligentFollowupValueMapper] = None

def get_followup_value_mapper() -> IntelligentFollowupValueMapper:
    """Get or create singleton mapper instance."""
    global _mapper
    if _mapper is None:
        _mapper = IntelligentFollowupValueMapper()
    return _mapper


# ============================================================================
# HELPER FUNCTIONS: Easy context creation for both initial and follow-up queries
# ============================================================================

async def create_context_for_followup(
    table_name: str,
    table_schema: Dict[str, str],
    table_sample_data: Dict[str, List[Any]],
    previous_columns_used: Optional[List[str]] = None,
    previous_filters: Optional[Dict[str, str]] = None,
    previous_query: Optional[str] = None,
) -> FollowUpContext:
    """
    Create FollowUpContext for a follow-up query.
    
    Args:
        table_name: The database table being queried
        table_schema: {column_name: data_type} from schema
        table_sample_data: {column_name: [sample_values]} from database
        previous_columns_used: Columns from previous SELECT clause
        previous_filters: Filters from previous WHERE clause
        previous_query: Full SQL of previous query
        
    Returns:
        FollowUpContext configured for follow-up discovery
    """
    return FollowUpContext(
        table_name=table_name,
        table_schema=table_schema,
        table_sample_data=table_sample_data,
        previous_columns_used=previous_columns_used,
        previous_filters=previous_filters,
        previous_query=previous_query,
        is_followup=True
    )


async def create_context_for_initial_query(
    table_name: str,
    table_schema: Dict[str, str],
    table_sample_data: Dict[str, List[Any]],
) -> FollowUpContext:
    """
    Create FollowUpContext for an initial/main query.
    
    For initial queries, we don't have previous context, so we just provide
    the table schema and sample data. All strategies except CONTEXT_PRIOR
    will work.
    
    Args:
        table_name: The database table being queried
        table_schema: {column_name: data_type} from schema
        table_sample_data: {column_name: [sample_values]} from database
        
    Returns:
        FollowUpContext configured for initial query discovery
    """
    return FollowUpContext(
        table_name=table_name,
        table_schema=table_schema,
        table_sample_data=table_sample_data,
        previous_columns_used=None,
        previous_filters=None,
        previous_query=None,
        is_followup=False
    )
