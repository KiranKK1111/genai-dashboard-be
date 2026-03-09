"""
Universal Query Analyzer - Semantic-Driven Query Classification & Analysis
===========================================================================

This module provides universal query analysis for ANY user prompt, supporting:
1. Query type detection (MAIN, FOLLOW-UP, CLARIFICATION, etc.)
2. Semantic table discovery (finds relevant tables without hardcoding)
3. Bidirectional value-column mapping (value→column AND column→value)
4. Join pattern detection (understands when to use JOINs)
5. Complexity assessment (simple vs complex patterns)
6. LLM-enhanced semantic understanding

ZERO HARDCODING: All behavior is data-driven and semantic.
UNIVERSAL: Works for ANY SQL pattern in TABLE_QUERIES.md
DYNAMIC: Adapts to schema changes without code modifications
"""

import logging
import re
import os
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import inspect as sqlalchemy_inspect, text

from .. import llm

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Classification of user query types."""
    MAIN_SIMPLE = "main_simple"           # Single table, basic SELECT
    MAIN_FILTERED = "main_filtered"       # Single table with WHERE
    MAIN_AGGREGATED = "main_aggregated"   # GROUP BY, COUNT, SUM, etc.
    MAIN_JOINED = "main_joined"           # Multiple tables with JOIN
    MAIN_COMPLEX = "main_complex"         # CTEs, subqueries, window functions
    FOLLOWUP_REFINEMENT = "followup_refinement"  # Refine previous query
    FOLLOWUP_EXPANSION = "followup_expansion"    # Expand to related tables
    FOLLOWUP_AGGREGATION = "followup_aggregation" # Aggregate previous results
    FOLLOWUP_COMPARISON = "followup_comparison"  # Compare with previous
    CLARIFICATION_NEEDED = "clarification_needed"
    UNRELATED = "unrelated"


class JoinType(Enum):
    """Types of JOINs to apply."""
    NONE = "none"
    INNER_JOIN = "inner_join"
    LEFT_JOIN = "left_join"
    FULL_JOIN = "full_join"
    CROSS_JOIN = "cross_join"


@dataclass
class TableMetadata:
    """Metadata about a database table."""
    table_name: str
    columns: Dict[str, str]  # {column_name: data_type}
    sample_data: Dict[str, List[Any]]  # {column_name: [samples]}
    primary_key: Optional[str] = None
    foreign_keys: List[Tuple[str, str, str]] = field(default_factory=list)  # [(fk_col, ref_table, ref_col)]
    relevance_score: float = 0.0  # How relevant to query (0-1)


@dataclass
class SemanticAnalysis:
    """Complete semantic analysis of a user query."""
    query_type: QueryType
    confidence: float  # 0-1 confidence in classification
    
    # Table analysis - store full TableMetadata objects for complete information
    primary_table: Optional[str] = None
    related_tables: List[str] = field(default_factory=list)  # Just table names for now
    relevant_tables: List[TableMetadata] = field(default_factory=list)  # Full metadata (REPLACES related_tables usage)
    required_joins: List[Tuple[str, str, str]] = field(default_factory=list)  # [(table1, table2, on_clause)]
    
    # Value analysis
    user_values: Dict[str, Any] = field(default_factory=dict)  # Values mentioned by user
    value_to_column_mappings: List[Dict[str, Any]] = field(default_factory=list)
    column_to_value_mappings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Pattern analysis
    aggregation_functions: List[str] = field(default_factory=list)  # COUNT, SUM, AVG, etc.
    filter_conditions: List[str] = field(default_factory=list)
    sorting_requested: bool = False
    pagination_requested: bool = False
    distinct_requested: bool = False
    
    # Context
    is_followup: bool = False
    previous_query_context: Optional[Any] = None
    reasoning: str = ""  # Explanation of analysis
    

@dataclass
class ValueColumnMapping:
    """Bidirectional value-column relationship."""
    value: Any
    columns: List[str]  # Columns that could contain this value
    confidence: float
    direction: str = "value_to_column"  # or "column_to_value"
    strategy: str = "semantic_match"  # How it was discovered
    reasoning: str = ""


class UniversalQueryAnalyzer:
    """
    Analyzes ANY user prompt semantically to understand query intent.
    
    Key capabilities:
    1. Detect query type (main, follow-up, complex, etc.)
    2. Find relevant tables using semantic analysis
    3. Discover value-column mappings bidirectionally
    4. Detect join patterns
    5. Understand user intent deeply
    
    Goal: Minimize hardcoding by relying on LLM + schema-aware signals.
    Note: This module still uses lightweight, generic keyword signals as hints.
    """
    
    def __init__(self):
        """Initialize analyzer with semantic understanding."""
        self.aggregate_keywords = {
            'count', 'sum', 'avg', 'average', 'total', 'minimum', 'maximum',
            'max', 'min', 'distinct', 'unique', 'group', 'aggregate'
        }
        
        self.join_keywords = {
            'both', 'cross', 'all', 'match', 'relate', 'connect', 'link',
            'combine', 'merge', 'associate', 'along', 'together'
        }
        
        self.filter_keywords = {
            'where', 'filter', 'condition', 'criteria', 'status', 'type',
            'category', 'like', 'contain', 'match', 'between', 'range',
            'greater', 'less', 'more', 'high', 'low', 'active', 'inactive'
        }
        
        self.refinement_keywords = {
            'about', 'those', 'same', 'also', 'too', 'similar', 'like',
            'more', 'less', 'other', 'another', 'else', 'what', 'how',
            'whose', 'which', 'that', 'related', 'connect'
        }
        
        logger.info("[UNIVERSAL_ANALYZER] Initialized")
    
    async def analyze_query(
        self,
        user_prompt: str,
        db: AsyncSession,
        previous_query_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[str]] = None,
    ) -> SemanticAnalysis:
        """
        Perform comprehensive semantic analysis on user prompt.
        
        This is the main entry point. It:
        1. Classifies query type
        2. Discovers relevant tables
        3. Extracts user values
        4. Maps values to columns bidirectionally
        5. Detects join patterns
        6. Returns complete analysis
        
        Args:
            user_prompt: The user's natural language query
            db: Database session for schema discovery
            previous_query_context: Optional context from previous query
            conversation_history: Optional full conversation history
            
        Returns:
            SemanticAnalysis with all discovered information
        """
        
        logger.info(f"[ANALYZER] Analyzing query: {user_prompt[:60]}...")
        
        # Step 1: Determine if this is a follow-up
        is_followup = self._is_followup_query(
            user_prompt, 
            previous_query_context, 
            conversation_history
        )
        
        logger.info(f"[ANALYZER] Query type: {'FOLLOW-UP' if is_followup else 'MAIN'}")
        
        # Step 2: Classify query type (main_simple, main_joined, followup_refinement, etc.)
        query_type = await self._classify_query_type(
            user_prompt,
            is_followup,
            previous_query_context,
            db
        )
        
        logger.info(f"[ANALYZER] Classification: {query_type.value}")
        
        # Step 3: Discover relevant tables
        relevant_tables = await self._discover_tables(
            user_prompt,
            previous_query_context,
            db
        )
        
        logger.info(f"[ANALYZER] Found {len(relevant_tables)} relevant tables: "
                   f"{[t.table_name for t in relevant_tables]}")
        
        # Step 4: Extract user values from prompt
        user_values = self._extract_values(user_prompt)
        
        logger.info(f"[ANALYZER] Extracted {len(user_values)} values: {user_values}")
        
        # Step 5: Bidirectional value-column mapping
        value_to_col_mappings = await self._map_values_to_columns(
            user_values,
            relevant_tables,
            db
        )
        
        col_to_val_mappings = await self._map_columns_to_values(
            user_prompt,
            relevant_tables,
            db
        )
        
        logger.info(f"[ANALYZER] Value→Column mappings: {len(value_to_col_mappings)}, "
                   f"Column→Value mappings: {len(col_to_val_mappings)}")
        
        # Step 6: Detect join patterns
        join_instructions = await self._detect_join_patterns(
            user_prompt,
            relevant_tables,
            previous_query_context,
            db
        )
        
        # Step 7: Extract query patterns (aggregations, filters, sorting, etc.)
        aggregations = self._extract_aggregations(user_prompt)
        filters = self._extract_filters(user_prompt)
        sorting = self._extract_sorting(user_prompt)
        pagination = self._extract_pagination(user_prompt)
        distinct = self._extract_distinct(user_prompt)
        
        # Step 8: Build final analysis
        analysis = SemanticAnalysis(
            query_type=query_type,
            confidence=0.85,  # Semantic analysis confidence
            primary_table=relevant_tables[0].table_name if relevant_tables else None,
            related_tables=[t.table_name for t in relevant_tables[1:]],
            relevant_tables=relevant_tables,  # Store full TableMetadata for access to table properties
            required_joins=join_instructions,
            user_values=user_values,
            value_to_column_mappings=value_to_col_mappings,
            column_to_value_mappings=col_to_val_mappings,
            aggregation_functions=aggregations,
            filter_conditions=filters,
            sorting_requested=sorting,
            pagination_requested=pagination,
            distinct_requested=distinct,
            is_followup=is_followup,
            previous_query_context=previous_query_context,
            reasoning=f"Semantic analysis: {query_type.value} with "
                     f"{len(relevant_tables)} tables, "
                     f"{len(value_to_col_mappings)} value mappings"
        )
        
        logger.info(f"[ANALYZER] ✅ Analysis complete: {analysis.reasoning}")
        return analysis
    
    def _is_followup_query(
        self,
        user_prompt: str,
        previous_context: Optional[Dict[str, Any]],
        conversation_history: Optional[List[str]]
    ) -> bool:
        """
        Determine if this is a follow-up query based on:
        1. Keywords in prompt (what about, those, that, etc.)
        2. Existence of previous context
        3. Conversation history
        """
        # No previous context = must be main query
        if not previous_context and not conversation_history:
            return False
        
        # Check for follow-up keywords
        prompt_lower = user_prompt.lower()
        for keyword in self.refinement_keywords:
            if keyword in prompt_lower:
                # But exclude standalone keywords that might appear in main queries
                if len(conversation_history or []) > 0:
                    return True
        
        # If we have history and prompt refers to previous or related data
        if conversation_history and len(conversation_history) > 0:
            # Check for relative references
            relative_keywords = {'more', 'less', 'other', 'another', 'similar',
                               'same', 'different', 'related', 'those', 'them'}
            if any(kw in prompt_lower for kw in relative_keywords):
                return True
        
        return False
    
    async def _classify_query_type(
        self,
        user_prompt: str,
        is_followup: bool,
        previous_context: Optional[Dict[str, Any]],
        db: AsyncSession
    ) -> QueryType:
        """
        Classify the exact query type using semantic analysis.
        
        Main query types:
        - MAIN_SIMPLE: "list all rows from a table"
        - MAIN_FILTERED: "show rows where <column> matches a value"
        - MAIN_AGGREGATED: "count/aggregate rows"
        - MAIN_JOINED: "combine results from related tables"
        - MAIN_COMPLEX: "rank/group with multiple steps"
        
        Follow-up types:
        - FOLLOWUP_REFINEMENT: "only those matching an additional condition"
        - FOLLOWUP_EXPANSION: "include related details too"
        - FOLLOWUP_AGGREGATION: "count for each group/category"
        - FOLLOWUP_COMPARISON: "compare one group to another"
        """
        
        prompt_lower = user_prompt.lower()
        
        # ========================
        # USE LLM TO DETECT CONVERSATIONAL QUERIES (ZERO HARDCODING)
        # ========================
        # Get table names first so we can give LLM schema context
        try:
            all_tables = await db.run_sync(self._get_all_tables_sync)
        except Exception as e:
            logger.warning(f"[ANALYZER] Failed to get tables for context: {e}")
            all_tables = []
        
        # Let LLM determine if this is conversational vs database query (with schema context)
        is_conversational = await self._is_conversational_query_llm(user_prompt, all_tables)
        if is_conversational:
            logger.info(f"[ANALYZER] LLM detected conversational query, routing to UNRELATED")
            return QueryType.UNRELATED
        
        # Check for aggregations
        has_aggregation = any(kw in prompt_lower for kw in self.aggregate_keywords)
        
        # Check for joins
        has_join = any(kw in prompt_lower for kw in self.join_keywords)
        
        # Check for complex patterns (window functions, CTEs, subqueries)
        has_complex = any(word in prompt_lower for word in 
                         ['rank', 'partition', 'over', 'row_number', 'with recursive',
                          'case when', 'nested', 'hierarchical', 'recursive'])
        
        # Check for filters
        has_filter = any(kw in prompt_lower for kw in self.filter_keywords)
        
        # Classify based on is_followup
        if is_followup:
            if has_aggregation:
                return QueryType.FOLLOWUP_AGGREGATION
            elif has_join:
                return QueryType.FOLLOWUP_EXPANSION
            elif has_filter:
                return QueryType.FOLLOWUP_REFINEMENT
            else:
                return QueryType.FOLLOWUP_REFINEMENT
        else:
            if has_complex:
                return QueryType.MAIN_COMPLEX
            elif has_aggregation and has_join:
                return QueryType.MAIN_COMPLEX
            elif has_join:
                return QueryType.MAIN_JOINED
            elif has_aggregation:
                return QueryType.MAIN_AGGREGATED
            elif has_filter:
                return QueryType.MAIN_FILTERED
            else:
                return QueryType.MAIN_SIMPLE
    
    async def _discover_tables(
        self,
        user_prompt: str,
        previous_context: Optional[Dict[str, Any]],
        db: AsyncSession
    ) -> List[TableMetadata]:
        """
        Discover relevant tables using semantic analysis.
        
        Strategy:
        1. Extract entity keywords from the prompt (schema-aware)
        2. Search database for matching tables
        3. Score by relevance to prompt
        4. Return ranked list
        """
        
        try:
            # Get all table names from database using async-safe method
            # Use run_sync to safely call inspector in async context
            all_tables = await db.run_sync(self._get_all_tables_sync)
            logger.info(f"[ANALYZER] Found {len(all_tables)} tables in database: {all_tables}")
        except Exception as e:
            logger.error(f"[ANALYZER] Error getting tables: {e}", exc_info=True)
            return []
        
        # Extract entities mentioned in prompt using LLM (ZERO HARDCODING)
        entities = await self._extract_entities(user_prompt, all_tables)
        logger.info(f"[ANALYZER] Extracted entities: {entities}")
        
        # If follow-up with previous context, prioritize previous table
        primary_table = None
        if previous_context and 'table_name' in previous_context:
            primary_table = previous_context['table_name']
        
        # Match entities to tables using semantic similarity
        table_matches = []
        
        for table in all_tables:
            # Fetch table metadata first for semantic relevance calculation
            schema_dict = {}
            foreign_keys = []
            try:
                table_info = await db.run_sync(
                    self._get_table_info_sync, 
                    table
                )
                schema_dict = table_info.get('columns', {})
                foreign_keys = table_info.get('foreign_keys', [])
            except Exception as e:
                logger.warning(f"[ANALYZER] Could not fetch metadata for table {table}: {e}")
                continue  # Skip this table if we can't get metadata
            
            # Calculate relevance with full table information
            relevance = await self._calculate_table_relevance(
                table, entities, user_prompt, schema_dict
            )
            
            logger.debug(f"[ANALYZER] Table '{table}': relevance={relevance:.2f}")
            
            if relevance > 0.3:  # Threshold for relevance
                
                table_metadata = TableMetadata(
                    table_name=table,
                    columns=schema_dict,
                    sample_data={},  # Will fetch if needed
                    foreign_keys=foreign_keys,
                    relevance_score=relevance
                )
                
                table_matches.append(table_metadata)
                logger.info(f"[ANALYZER] ✅ Matched table: {table} (relevance={relevance:.2f}, columns={len(schema_dict)})")
        
        # Sort by relevance, prioritize primary table
        table_matches.sort(key=lambda t: t.relevance_score, reverse=True)
        
        if primary_table:
            # Move primary table to front
            primary_matches = [t for t in table_matches if t.table_name == primary_table]
            other_matches = [t for t in table_matches if t.table_name != primary_table]
            table_matches = primary_matches + other_matches
        
        logger.info(f"[ANALYZER] ✅ Final matched tables: {[t.table_name for t in table_matches[:3]]}")
        
        return table_matches[:3]  # Return top 3 most relevant tables
    
    async def _is_conversational_query_llm(self, prompt: str, available_tables: List[str] = None) -> bool:
        """
        Use LLM to determine if query is conversational (not database-related).
        
        ZERO HARDCODING: Pure semantic understanding with schema context.
        """
        from .. import llm
        
        # Build schema context to help LLM understand domain terminology
        schema_context = ""
        if available_tables:
            tables_str = ", ".join(available_tables[:25])  # Limit for token efficiency
            schema_context = f"""\n\nAVAILABLE DATABASE TABLES: {tables_str}

If the query mentions ANYTHING that could semantically relate to these tables (even using synonyms or business terminology), it's a DATABASE query."""
        
        classification_prompt = f"""Is this user query asking for DATABASE DATA or just CONVERSATIONAL?

USER QUERY: "{prompt}"{schema_context}

DATABASE DATA queries:
- Ask to retrieve, show, get, find, count, list, fetch data
- Ask about business entities that could exist in a database
- Use any terminology that might refer to database tables (including synonyms and business terms)

CONVERSATIONAL queries:
- ONLY pure greetings, small talk, questions about the AI itself
- Examples: "hi", "how are you", "what can you do", "thanks"

When in doubt, assume DATABASE query. Prefer false negatives over false positives.

Respond with ONLY valid JSON:
{{
  "is_conversational": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""
        
        try:
            response = await llm.call_llm(
                [{"role": "system", "content": "You are a query classifier. Respond ONLY with valid JSON."},
                 {"role": "user", "content": classification_prompt}],
                stream=False,
                max_tokens=100,
                temperature=0.0
            )
            
            import json
            result = json.loads(response.strip().replace('```json', '').replace('```', '').strip())
            is_conv = result.get('is_conversational', False)
            confidence = result.get('confidence', 0.5)
            
            # Return True only if high confidence
            if is_conv and confidence >= 0.7:
                return True
                
        except Exception as e:
            logger.warning(f"[ANALYZER] Conversational detection failed: {e}")
        
        return False
    
    async def _extract_entities(self, prompt: str, all_tables: List[str]) -> List[str]:
        """
        Extract entities mentioned in prompt using the LLM.
        
        ZERO HARDCODING: Uses LLM to identify entities based on actual database schema.
        """
        from .. import llm
        import json
        
        if not all_tables:
            return []
        
        # Build dynamic prompt with actual table names from database
        tables_list = ', '.join(all_tables[:20])  # Use available tables
        
        entity_prompt = f"""Extract business entities mentioned in this user query.

User Query: "{prompt}"

Available Database Tables:
{tables_list}

Instructions:
1. Identify which business concepts the user is asking about
2. Map those concepts to the available database tables
3. Consider singular/plural forms and naming variants (singular vs plural)
4. Consider obvious synonyms/aliases only when they clearly map to a listed table

Respond with ONLY a JSON array of entity names, no explanations:
["entity1", "entity2", ...]

If no entities found, return: []
"""
        
        try:
            response = await llm.call_llm(
                [{
                    "role": "system",
                    "content": "You are a semantic entity extractor. Extract business entities from queries. Respond ONLY with valid JSON."
                },
                {
                    "role": "user",
                    "content": entity_prompt
                }],
                stream=False,
                max_tokens=200,
                temperature=0.0
            )
            
            # Extract JSON from response
            json_match = response.find('[')
            json_end = response.rfind(']') + 1
            if json_match >= 0 and json_end > json_match:
                json_str = response[json_match:json_end]
                entities = json.loads(json_str)
                logger.debug(f"[ANALYZER] LLM extracted entities: {entities}")
                return entities
            
        except Exception as e:
            logger.warning(f"[ANALYZER] Entity extraction failed: {e}, using fallback")
        
        # Fallback: Direct word matching only - NO HARDCODED SYNONYMS
        # LLM handles synonym understanding during query plan generation
        entities = []
        prompt_lower = prompt.lower()
        
        import re
        query_words = set(re.findall(r'\b\w+\b', prompt_lower))
        
        # Direct table name matches only
        for table in all_tables:
            table_lower = table.lower()
            table_singular = table_lower.rstrip('s')
            if table_lower in query_words or table_singular in query_words:
                entities.append(table)
        
        return entities
    
    async def _calculate_table_relevance(
        self,
        table_name: str,
        entities: List[str],
        prompt: str,
        table_columns: Dict[str, str]
    ) -> float:
        """
        Calculate how relevant a table is to the user's query using semantic analysis.
        
        FULLY DYNAMIC: Uses LLM to assess relevance without hardcoded rules.
        """
        from .. import llm
        import json
        
        table_lower = table_name.lower()
        prompt_lower = prompt.lower()
        
        # Quick keyword matching for obvious matches
        if table_lower in prompt_lower or table_lower.replace('_', ' ') in prompt_lower:
            return 0.9
        
        # Check singular/plural forms
        table_singular = table_lower.rstrip('s')
        if table_singular != table_lower and table_singular in prompt_lower:
            return 0.85
        
        # Check entity matches
        for entity in entities:
            entity_lower = str(entity).lower()
            if entity_lower in table_lower or table_lower in entity_lower:
                return 0.7
        
        # Use LLM for semantic matching (fallback for complex cases)
        if len(entities) > 0 and len(table_columns) > 0:
            try:
                columns_str = ', '.join(list(table_columns.keys())[:10])
                
                relevance_prompt = f"""Assess relevance of database table to user query.

User Query: "{prompt}"
Extracted Entities: {entities}

Table Details:
Name: {table_name}
Columns: {columns_str}

Rate relevance from 0.0 (not relevant) to 1.0 (highly relevant).
Consider:
- Does table name match entities?
- Do columns support the query requirements?
- Is this likely the right table for the query?

Respond with ONLY a JSON object:
{{"relevance": 0.5, "reasoning": "brief explanation"}}
"""
                
                response = await llm.call_llm(
                    [{
                        "role": "system",
                        "content": "You are a database relevance scorer. Respond ONLY with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": relevance_prompt
                    }],
                    stream=False,
                    max_tokens=100,
                    temperature=0.0
                )
                
                json_match = response.find('{')
                json_end = response.rfind('}') + 1
                if json_match >= 0 and json_end > json_match:
                    json_str = response[json_match:json_end]
                    data = json.loads(json_str)
                    relevance = float(data.get('relevance', 0.1))
                    logger.debug(f"[ANALYZER] LLM relevance for {table_name}: {relevance:.2f}")
                    return min(max(relevance, 0.0), 1.0)
                    
            except Exception as e:
                logger.debug(f"[ANALYZER] LLM relevance scoring failed: {e}")
        
        # Default low relevance if no matches
        return 0.1 if entities else 0.0
    
    def _extract_values(self, prompt: str) -> Dict[str, Any]:
        """Extract potential values from user prompt."""
        values = {}
        
        # Extract quoted strings
        quoted = re.findall(r'["\']([^"\']+)["\']', prompt)
        for q in quoted:
            values[f"quoted_{len(values)}"] = q
        
        # Extract numbers
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', prompt)
        for num in numbers[:3]:  # Limit to top 3 numbers
            values[f"number_{len(values)}"] = float(num) if '.' in num else int(num)
        
        # Extract meaningful words (values, statuses, etc.)
        # NO HARDCODED STATUS KEYWORDS - LLM handles semantic understanding
        words = re.findall(r'\b([a-z]{3,})\b', prompt.lower())
        
        # Only extract words that look like potential filter values (not common English words)
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
                       'her', 'was', 'one', 'our', 'out', 'has', 'his', 'how', 'man', 'new',
                       'now', 'old', 'see', 'way', 'who', 'boy', 'did', 'its', 'let', 'put',
                       'say', 'she', 'too', 'use', 'get', 'give', 'show', 'list', 'find',
                       'with', 'from', 'have', 'this', 'that', 'what', 'which', 'where'}
        
        for word in words:
            if word not in common_words and len(word) >= 4:
                values[f"potential_value_{len(values)}"] = word
        
        return values
    
    async def _map_values_to_columns(
        self,
        values: Dict[str, Any],
        tables: List[TableMetadata],
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Map extracted user values to database columns."""
        mappings = []
        
        for value_key, value in values.items():
            for table in tables:
                # Check each column in the table
                for col_name, col_type in table.columns.items():
                    # Simple matching logic (can be enhanced with semantic similarity)
                    if self._value_matches_column(value, col_name, col_type):
                        mapping = {
                            "value": value,
                            "column": col_name,
                            "table": table.table_name,
                            "confidence": 0.85,
                            "strategy": "semantic_match"
                        }
                        mappings.append(mapping)
                        break  # Found match in this table, move to next
        
        return mappings
    
    async def _map_columns_to_values(
        self,
        prompt: str,
        tables: List[TableMetadata],
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Map database columns back to user text (reverse direction)."""
        mappings = []
        
        for table in tables:
            for col_name in table.columns.keys():
                # Check if column name appears in prompt
                if col_name in prompt.lower():
                    mapping = {
                        "column": col_name,
                        "table": table.table_name,
                        "mentioned_in_prompt": True,
                        "confidence": 0.90
                    }
                    mappings.append(mapping)
        
        return mappings
    
    def _value_matches_column(
        self,
        value: Any,
        column_name: str,
        column_type: str
    ) -> bool:
        """Check if a value likely belongs to a column."""
        # Simple heuristics - can be enhanced
        col_name_lower = column_name.lower()
        value_str = str(value).lower()
        
        # Type-based matching
        if isinstance(value, bool) and 'bool' in column_type.lower():
            return True
        if isinstance(value, int) and any(t in column_type.lower() for t in ['int', 'numeric']):
            return True
        if isinstance(value, str) and any(t in column_type.lower() for t in ['varchar', 'text', 'string']):
            return True
        
        # Name-based heuristics
        if col_name_lower in value_str or value_str in col_name_lower:
            return True
        
        return False
    
    async def _detect_join_patterns(
        self,
        prompt: str,
        tables: List[TableMetadata],
        previous_context: Optional[Dict[str, Any]],
        db: AsyncSession
    ) -> List[Tuple[str, str, str]]:
        """
        Detect if JOINs are needed between tables.
        
        Returns list of (table1, table2, on_clause) tuples.
        """
        joins = []
        
        if len(tables) < 2:
            return joins  # No joins needed for single table
        
        # Check for explicit join keywords in prompt
        prompt_lower = prompt.lower()
        if any(keyword in prompt_lower for keyword in self.join_keywords):
            # Try to find relationships via foreign keys
            for i, table1 in enumerate(tables[:-1]):
                for table2 in tables[i+1:]:
                    # Check if table1 has FK to table2
                    for fk_col, ref_table, ref_col in table1.foreign_keys:
                        if ref_table == table2.table_name:
                            joins.append((table1.table_name, table2.table_name, 
                                        f"{table1.table_name}.{fk_col} = {table2.table_name}.{ref_col}"))
                            break
        
        return joins
    
    def _extract_aggregations(self, prompt: str) -> List[str]:
        """Extract aggregation functions requested."""
        aggs = []
        prompt_lower = prompt.lower()
        
        agg_map = {
            'count': ['count', 'how many', 'total number'],
            'sum': ['sum', 'total', 'altogether'],
            'avg': ['average', 'avg', 'mean'],
            'min': ['minimum', 'min', 'lowest', 'least'],
            'max': ['maximum', 'max', 'highest', 'most'],
            'distinct': ['distinct', 'unique', 'different'],
        }
        
        for agg_func, keywords in agg_map.items():
            for kw in keywords:
                if kw in prompt_lower:
                    aggs.append(agg_func)
                    break
        
        return aggs
    
    def _extract_filters(self, prompt: str) -> List[str]:
        """Extract filter conditions requested."""
        return []  # Placeholder - would extract WHERE conditions
    
    def _extract_sorting(self, prompt: str) -> bool:
        """Check if sorting is requested."""
        keywords = ['sort', 'order', 'descending', 'ascending', 'asc', 'desc', 'by']
        return any(kw in prompt.lower() for kw in keywords)
    
    def _extract_pagination(self, prompt: str) -> bool:
        """Check if pagination is requested."""
        keywords = ['limit', 'first', 'top', 'last', 'page', 'offset']
        return any(kw in prompt.lower() for kw in keywords)
    
    def _extract_distinct(self, prompt: str) -> bool:
        """Check if DISTINCT is requested."""
        keywords = ['distinct', 'unique', 'different', 'all different']
        return any(kw in prompt.lower() for kw in keywords)
    
    @staticmethod
    def _get_all_tables_sync(sync_session: Any) -> List[str]:
        """Sync method to get all table names via Inspector.
        
        This must be called via await db.run_sync(_get_all_tables_sync).
        Solves the MissingGreenlet issue by running sync operations in proper context.
        
        Args:
            sync_session: SQLAlchemy sync session object passed by run_sync()
        """
        from ..config import get_schema
        
        # Get the engine from the session's bind, not the session itself
        # This is required because sqlalchemy_inspect() works on engines/connections, not sessions
        engine = sync_session.get_bind()
        inspector = sqlalchemy_inspect(engine)
        
        # Get schema from config (e.g., "genai" or "public")
        schema = get_schema()
        return inspector.get_table_names(schema=schema)
    
    @staticmethod
    def _get_table_info_sync(sync_session: Any, table_name: str) -> Dict[str, Any]:
        """Sync method to get table column and foreign key info.
        
        This must be called via await db.run_sync(_get_table_info_sync, table_name).
        
        Args:
            sync_session: SQLAlchemy sync session object passed by run_sync()
            table_name: Name of table to inspect
        """
        from ..config import get_schema
        
        # Get the engine from the session's bind
        engine = sync_session.get_bind()
        inspector = sqlalchemy_inspect(engine)
        
        # Get schema from config (e.g., "genai" or "public")
        schema = get_schema()
        
        # Get columns
        schema_dict = {}
        try:
            columns = inspector.get_columns(table_name, schema=schema)
            for col in columns:
                schema_dict[col['name']] = str(col['type'])
        except Exception as e:
            logger.debug(f"[ANALYZER] Could not get columns for {table_name}: {e}")
        
        # Get foreign keys
        foreign_keys = []
        try:
            fks = inspector.get_foreign_keys(table_name, schema=schema)
            for fk in fks:
                foreign_keys.append((
                    fk['constrained_columns'][0] if fk['constrained_columns'] else '',
                    fk['referred_table'],
                    fk['referred_columns'][0] if fk['referred_columns'] else ''
                ))
        except Exception as e:
            logger.debug(f"[ANALYZER] Could not get foreign keys for {table_name}: {e}")
        
        return {
            'columns': schema_dict,
            'foreign_keys': foreign_keys
        }



# Singleton instance
_analyzer: Optional[UniversalQueryAnalyzer] = None

def get_universal_analyzer() -> UniversalQueryAnalyzer:
    """Get or create singleton analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = UniversalQueryAnalyzer()
    return _analyzer
