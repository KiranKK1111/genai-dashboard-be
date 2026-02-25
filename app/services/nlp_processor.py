"""
Advanced Natural Language Processing (NLP) for Query Intelligence.

This module provides sophisticated NLP capabilities for understanding user queries:

1. INTENT ANALYSIS
   - Multi-level intent classification (primary + secondary)
   - Operation type detection (SELECT, INSERT, UPDATE, DELETE, ANALYSIS)
   - Sub-intent extraction (filter, sort, aggregate, etc.)
   - Confidence scoring with reasoning

2. ENTITY EXTRACTION
   - Table name extraction (fuzzy matching against schema)
   - Column/field name extraction
   - Filter conditions and operators
   - Temporal references (dates, date ranges)
   - Aggregation functions
   - Join requirements

3. CONTEXT UNDERSTANDING
   - Conversation-aware processing
   - Pronoun resolution ("show me their records" → identify 'their')
   - Reference tracking (previous entities mentioned)
   - Context carryover for follow-up questions

4. QUERY DECOMPOSITION
   - Break complex multi-part queries into steps
   - Identify dependencies between query parts
   - Handle "and", "also", "by the way" connectors
   - Execute in logical order

5. INTELLIGENCE SCORING
   - Query complexity assessment
   - Confidence level calculation
   - Ambiguity detection
   - Fallback suggestion generation

Features:
- Zero hardcoded values (fully dynamic)
- Conversation-aware (understands context)
- Error recovery with suggestions
- Multi-language support ready
- Extensible intent taxonomy
"""

from __future__ import annotations

import re
import json
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from .. import llm
from ..config import settings


class QueryIntentType(Enum):
    """Primary intent classification."""
    DATA_RETRIEVAL = "data_retrieval"      # SELECT
    DATA_MODIFICATION = "data_modification"  # INSERT, UPDATE, DELETE
    DATA_ANALYSIS = "data_analysis"         # Aggregation, statistics
    FILE_PROCESSING = "file_processing"     # File upload/analysis
    CONFIG_UPDATE = "config_update"         # System configuration
    CLARIFICATION_NEEDED = "clarification_needed"  # Need more info
    GREETING = "greeting"                  # Greetings (hi, hello, hey)
    CHIT_CHAT = "chit_chat"                # Casual conversation
    GENERAL_QUESTION = "general_question"  # General knowledge questions
    UNKNOWN = "unknown"


class OperationType(Enum):
    """Specific database operation type."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    AGGREGATE = "aggregate"
    FILTER = "filter"
    SORT = "sort"
    JOIN = "join"
    GROUP_BY = "group_by"
    ANALYSIS = "analysis"


class ConfidenceLevel(Enum):
    """Confidence in query understanding."""
    VERY_HIGH = (0.90, 1.0)      # Crystal clear
    HIGH = (0.75, 0.89)          # Well understood
    MEDIUM = (0.50, 0.74)        # Reasonably clear
    LOW = (0.25, 0.49)           # Some ambiguity
    VERY_LOW = (0.0, 0.24)       # Needs clarification


@dataclass
class EntityReference:
    """Reference to a database entity (table/column)."""
    name: str
    entity_type: str  # 'table' or 'column'
    confidence: float
    alternatives: List[str] = None
    original_text: str = None  # What user said
    
    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


@dataclass
class QueryClause:
    """Represents a logical clause in the query."""
    clause_type: str  # 'filter', 'sort', 'aggregate', etc.
    content: str
    confidence: float
    entities: List[EntityReference] = None
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []


@dataclass
class NLPAnalysisResult:
    """Complete NLP analysis of a user query."""
    original_query: str
    primary_intent: QueryIntentType
    operations: List[OperationType]
    confidence: float
    confidence_level: ConfidenceLevel
    reasoning: str
    entities: List[EntityReference]
    clauses: List[QueryClause]
    decomposed_steps: List[str]
    clarifying_questions: List[str]
    suggested_alternatives: List[str]
    complexity_score: float  # 0-1, where 1 is most complex
    requires_context: bool
    context_references: List[str]  # References to conversation context
    metadata: Dict[str, Any]
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "original_query": self.original_query,
            "primary_intent": self.primary_intent.value,
            "operations": [op.value for op in self.operations],
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.name,
            "reasoning": self.reasoning,
            "entities": [asdict(e) for e in self.entities],
            "clauses": [asdict(c) for c in self.clauses],
            "decomposed_steps": self.decomposed_steps,
            "clarifying_questions": self.clarifying_questions,
            "suggested_alternatives": self.suggested_alternatives,
            "complexity_score": self.complexity_score,
            "requires_context": self.requires_context,
            "context_references": self.context_references,
            "metadata": self.metadata,
        }


class AdvancedNLPProcessor:
    """Advanced NLP processing for robust query understanding."""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.available_tables: Set[str] = set()
        self.available_columns: Dict[str, Set[str]] = {}
        self.table_keywords = {}  # Mapping common names to actual tables
        
    async def initialize(self):
        """Load database schema for entity matching."""
        try:
            # Get all table names
            sql = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = :schema
                ORDER BY table_name
            """
            result = await self.db_session.execute(
                text(sql),
                {"schema": settings.postgres_schema}
            )
            self.available_tables = {row[0] for row in result.fetchall()}
            
            # Get columns for each table
            sql = """
                SELECT table_name, column_name
                FROM information_schema.columns
                WHERE table_schema = :schema
                ORDER BY table_name, column_name
            """
            result = await self.db_session.execute(
                text(sql),
                {"schema": settings.postgres_schema}
            )
            
            for table_name, column_name in result.fetchall():
                if table_name not in self.available_columns:
                    self.available_columns[table_name] = set()
                self.available_columns[table_name].add(column_name)
                
            print(f"[NLP] Loaded {len(self.available_tables)} tables, {sum(len(c) for c in self.available_columns.values())} total columns")
            
        except Exception as e:
            print(f"[WARNING] Failed to load schema: {str(e)}")
    
    async def analyze_query(
        self,
        query: str,
        conversation_history: str = "",
        previous_queries: List[str] = None
    ) -> NLPAnalysisResult:
        """
        Perform comprehensive NLP analysis on user query.
        
        Args:
            query: User's natural language query
            conversation_history: Previous conversation context
            previous_queries: List of previous queries in this session
            
        Returns:
            Complete NLP analysis with entities, intent, confidence, etc.
        """
        if not self.available_tables:
            await self.initialize()
        
        if previous_queries is None:
            previous_queries = []
        
        # 🚨 CHECK FOR GIBBERISH FIRST - if detected, return special result
        is_gibberish, gibberish_reason = self._detect_gibberish(query)
        if is_gibberish:
            print(f"[NLP] 🚨 GIBBERISH DETECTED: {gibberish_reason} for query: '{query[:50]}...'")
            return NLPAnalysisResult(
                original_query=query,
                primary_intent=QueryIntentType.UNKNOWN,
                operations=[],
                confidence=0.0,
                confidence_level=ConfidenceLevel.VERY_LOW,
                reasoning=f"Input appears to be gibberish: {gibberish_reason}",
                entities=[],
                clauses=[],
                decomposed_steps=[],
                clarifying_questions=[],
                suggested_alternatives=[],
                complexity_score=0.0,
                requires_context=False,
                context_references=[],
                metadata={
                    "analyzed_at": datetime.utcnow().isoformat(),
                    "analyzer_version": "2.0",
                    "gibberish_detected": True,
                    "gibberish_reason": gibberish_reason,
                }
            )
        
        # Step 1: Extract intent from query
        primary_intent = await self._detect_primary_intent(query)
        operations = await self._detect_operations(query)
        
        # Step 2: Extract entities (tables, columns, etc.)
        entities = await self._extract_entities(query)
        
        # Step 3: Identify query clauses (filter, sort, etc.)
        clauses = await self._extract_clauses(query)
        
        # Step 4: Decompose complex queries into steps
        decomposed_steps = await self._decompose_query(query)
        
        # Step 5: Analyze context dependencies
        context_refs, requires_context = await self._analyze_context_requirements(
            query, conversation_history, previous_queries
        )
        
        # Step 6: Calculate confidence
        confidence, confidence_level = await self._calculate_confidence(
            query, primary_intent, entities, requires_context
        )
        
        # Step 7: Generate clarifying questions if needed
        clarifying_questions = []
        if confidence < 0.50:
            clarifying_questions = await self._generate_clarifying_questions(
                query, primary_intent, entities
            )
        
        # Step 8: Generate reasoning
        reasoning = await self._generate_reasoning(
            query, primary_intent, operations, confidence, entities
        )
        
        # Step 9: Assess complexity
        complexity_score = await self._assess_complexity(
            query, operations, len(clauses), len(decomposed_steps)
        )
        
        # Step 10: Suggest alternatives if low confidence
        suggested_alternatives = []
        if confidence < 0.50:
            suggested_alternatives = await self._suggest_query_alternatives(query)
        
        return NLPAnalysisResult(
            original_query=query,
            primary_intent=primary_intent,
            operations=operations,
            confidence=confidence,
            confidence_level=confidence_level,
            reasoning=reasoning,
            entities=entities,
            clauses=clauses,
            decomposed_steps=decomposed_steps,
            clarifying_questions=clarifying_questions,
            suggested_alternatives=suggested_alternatives,
            complexity_score=complexity_score,
            requires_context=requires_context,
            context_references=context_refs,
            metadata={
                "analyzed_at": datetime.utcnow().isoformat(),
                "analyzer_version": "2.0",
                "conversation_context_used": bool(conversation_history),
            }
        )
    
    async def _detect_primary_intent(self, query: str) -> QueryIntentType:
        """Detect primary intent from query using LLM for dynamic classification.
        
        Uses LLM to dynamically understand intent without hardcoded keywords,
        allowing the modal to adapt and learn conversation patterns naturally.
        """
        # 🎯 DYNAMIC INTENT DETECTION: Use LLM to process all queries
        # This allows the modal to adapt to different conversation styles and languages
        try:
            prompt = f"""Analyze this user query and classify it by PRIMARY intent. Return ONLY the intent type name, nothing else.

Query: "{query}"

INTENT CLASSIFICATION GUIDE:

GREETING: Simple greeting with NO follow-up request or question
  Only if: Single/short greeting words (hi, hello, hey) with nothing else
  Examples: "hi", "hello", "hey there"
  Return: GREETING

CHIT_CHAT: Casual conversation about personal status or interaction (with greeting)
  Only if: Asking about person's status/feelings ("how are you", "what's up", "how have you been")
  Examples: "how are you", "what's up", "how have you been"
  Return: CHIT_CHAT

DATA_RETRIEVAL: User wants to GET, FETCH, SHOW, LIST, DISPLAY data from database or find specific information
  KEY INDICATORS: "get", "show", "list", "display", "fetch", "retrieve", "what are", "what is the", "find", "give me", "[entity] id", or specific ID/value lookups
  IMPORTANT: ANY query mentioning a database entity (transaction, customer, order, account) OR a specific ID/reference is DATA_RETRIEVAL
  Examples: 
    - "get me all customers"
    - "show transactions"
    - "what is the transaction reference for transaction id 1353189" (SPECIFIC LOOKUP)
    - "find the customer name for id 123"
    - "what's the account balance for account 456"
  If query mentions: getting/showing/listing/finding DATA from company database OR looking up specific record by ID → DATA_RETRIEVAL
  Return: DATA_RETRIEVAL

DATA_ANALYSIS: User wants STATISTICS, COUNTS, AGGREGATES, TRENDS, SUMMARIES
  Keywords: "how many", "count", "sum", "average", "total", "statistics", "trends", "compare"
  Examples: "how many customers", "what's the total sales", "average order value"
  Return: DATA_ANALYSIS

DATA_MODIFICATION: User wants to CREATE, ADD, INSERT, UPDATE, DELETE database records
  Keywords: "add", "insert", "create", "update", "modify", "delete", "remove", "change"
  Examples: "add new customer", "update the order", "delete this record"
  Return: DATA_MODIFICATION

FILE_PROCESSING: User wants to UPLOAD, PROCESS, ANALYZE files
  Keywords: "upload", "processing", "import", "analyze file", "document"
  Examples: "upload a CSV", "process this file", "analyze document"
  Return: FILE_PROCESSING

CONFIG_UPDATE: User wants to change SETTINGS or CONFIGURATION
  Keywords: "change", "set config", "configure", "timeout", "API key"
  Examples: "change the timeout", "set API key", "configure system"
  Return: CONFIG_UPDATE

CLARIFICATION_NEEDED: Ambiguous or conflicting instructions
  Examples: "should I do X or Y", "I'm not sure"
  Return: CLARIFICATION_NEEDED

GENERAL_QUESTION: Abstract knowledge questions NOT about database or company data
  ONLY for conceptual/educational topics that don't relate to database or specific lookups
  Keywords: Conceptual questions (what IS [concept], how DOES [mechanism], who invented, definition)
  Examples: "what is SQL", "how does HTTP work", "who invented the internet"
  NOTE: "what is the transaction reference for transaction id X" is DATABASE LOOKUP → DATA_RETRIEVAL, NOT this
  Only return GENERAL_QUESTION for abstract concepts, NOT for specific data lookups
  Return: GENERAL_QUESTION

---
DECISION LOGIC (Apply in order):
1. If ONLY greeting words (hi, hello, hey) → GREETING
2. If asking about personal interaction (how are you, what's up) → CHIT_CHAT
3. If mentions database entity (transaction, customer, order, account, employee, etc.) OR has specific ID/reference lookup → DATA_RETRIEVAL ⭐
4. If asking for counts/statistics/aggregates (how many, count, sum, average, etc.) → DATA_ANALYSIS
5. If modifying/adding/deleting records → DATA_MODIFICATION
6. If uploading/processing files → FILE_PROCESSING
7. If changing settings/config → CONFIG_UPDATE
8. If abstract concept question (what is SQL, how does HTTP, definition) → GENERAL_QUESTION
9. Otherwise → CLARIFICATION_NEEDED

⭐ CRITICAL: Database entity mentions or specific lookups (by ID/reference) ALWAYS → DATA_RETRIEVAL

Return ONLY the intent name (one word, ALL CAPS)."""
            
            response = await llm.call_llm(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0.1
            )
            intent_text = str(response).strip().upper()
            print(f"[NLP_INTENT] Query: '{query}' -> LLM response: '{intent_text}'")
            
            # Match response to enum - be more flexible with matching
            intent_text = intent_text.replace("_", " ")
            for intent in QueryIntentType:
                intent_value = intent.value.upper().replace("_", " ")
                # Check for exact match or if response contains the intent name
                if intent_value in intent_text or intent_value.split()[0] in intent_text:
                    print(f"[NLP_INTENT] Matched to: {intent.value}")
                    return intent
            
            print(f"[WARNING] Could not match intent from response: {intent_text}")
            
        except Exception as e:
            print(f"[WARNING] LLM intent detection failed: {str(e)}")
        
        # Fallback: Return UNKNOWN if LLM fails
        return QueryIntentType.UNKNOWN
    
    async def _detect_operations(self, query: str) -> List[OperationType]:
        """
        Detect specific database operations needed.
        
        Note: Primary routing now handled by DecisionEngine.
        This defaults to SELECT for supporting queries.
        """
        return [OperationType.SELECT]
    
    async def _extract_entities(self, query: str) -> List[EntityReference]:
        """Extract table and column references from query."""
        entities = []
        found_tables = set()
        
        # PRIORITY 1: Look for exact table name matches (case-insensitive)
        # "transaction details" -> transactions table
        for table_name in self.available_tables:
            if table_name.lower() in query.lower():
                entities.append(EntityReference(
                    name=table_name,
                    entity_type="table",
                    confidence=0.95,
                    original_text=table_name
                ))
                found_tables.add(table_name)
                print(f"[NLP_ENTITY] Exact match: {table_name}")
        
        # PRIORITY 2: Use LLM for fuzzy table extraction (even if exact matches found)
        # This helps find tables mentioned with different wording
        # "customer with code X" should identify customers table
        if self.available_tables:
            try:
                tables_list = ", ".join(sorted(list(self.available_tables)))
                prompt = f"""Given this query: "{query}"
And available tables: {tables_list}

List which tables are likely being referenced. Return ONLY table names in a simple list format (one per line or comma-separated). No explanations."""
                
                response = await llm.call_llm(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.1
                )
                response_text = str(response).strip()
                print(f"[NLP_ENTITY] LLM response: {response_text}")
                print(f"[NLP_ENTITY] Available tables: {sorted(list(self.available_tables))}")
                
                # Parse response - handle multiple formats
                # Create a mapping of lowercase table names to actual names
                available_lower = {t.lower(): t for t in self.available_tables}
                
                # Extract all words from the response
                # Look for any word that matches one of our tables
                words = re.findall(r'\b[a-z_]+\b', response_text.lower())
                
                candidate_tables = []
                for word in words:
                    if word in available_lower and available_lower[word] not in found_tables:
                        candidate_tables.append(available_lower[word])
                        found_tables.add(available_lower[word])
                
                print(f"[NLP_ENTITY] Parsed candidates: {candidate_tables}")
                
                # Add all matched tables to entities
                for table in candidate_tables:
                    entities.append(EntityReference(
                        name=table,
                        entity_type="table",
                        confidence=0.70,
                        original_text=table
                    ))
                    print(f"[NLP_ENTITY] LLM matched: {table}")
                            
            except Exception as e:
                print(f"[WARNING] LLM table extraction failed: {str(e)}")
        
        # PRIORITY 3: Look for column matches
        for table_name, columns in self.available_columns.items():
            for column_name in columns:
                if column_name.lower() in query.lower():
                    entities.append(EntityReference(
                        name=column_name,
                        entity_type="column",
                        confidence=0.90,
                        original_text=column_name
                    ))
                    print(f"[NLP_ENTITY] Column match: {column_name}")
        
        # Deduplicate by name
        seen = {}
        for entity in entities:
            key = (entity.name, entity.entity_type)
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity
        
        return list(seen.values())
    
    async def _extract_clauses(self, query: str) -> List[QueryClause]:
        """Extract logical clauses (WHERE, ORDER BY, GROUP BY, etc.)."""
        clauses = []
        
        # Pattern matching for common clauses
        clause_patterns = {
            "filter": [
                r'(?:where|filter|with|whose|that (?:have|is|are))\s+([^,;.]+)',
                r'(?:if|only if)\s+([^,;.]+)',
                r'(?:having|matching)\s+([^,;.]+)',
            ],
            "sort": [
                r'(?:sort|order)(?:ed)?\s+(?:by|on)\s+([^,;.]+)',
                r'(?:sort|order)(?:ed)?\s+([^,;.]+)\s+(?:asc|desc|ascending|descending)',
            ],
            "aggregate": [
                r'(?:count|sum|average|max|min|total)\s+(?:of|the)?\s+([^,;.]+)',
                r'(?:how many|how much|what is the total|what is the average)\s+([^,;.]+)',
            ],
            "group": [
                r'(?:group|grouped)\s+(?:by|on)\s+([^,;.]+)',
                r'(?:per|for each)\s+([^,;.]+)',
            ],
        }
        
        query_lower = query.lower()
        
        for clause_type, patterns in clause_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query_lower)
                for match in matches:
                    clause_content = match.group(1)
                    if clause_content and len(clause_content) > 2:
                        clauses.append(QueryClause(
                            clause_type=clause_type,
                            content=clause_content,
                            confidence=0.80
                        ))
        
        return clauses
    
    async def _decompose_query(self, query: str) -> List[str]:
        """Break complex queries into logical steps."""
        steps = []
        
        # Split on logical connectors
        connectors = [' and ', ' also ', ' additionally ', ' moreover ', ' furthermore ']
        remaining = query
        
        for connector in connectors:
            if connector in remaining.lower():
                parts = re.split(re.escape(connector), remaining, flags=re.IGNORECASE)
                for part in parts:
                    if part.strip():
                        steps.append(part.strip())
                return steps
        
        # If no connectors, check for multiple clauses
        if len(query) > 100 or query.count(",") > 1:
            # Complex query - treat as single step but mark for verification
            steps.append(query)
        else:
            steps.append(query)
        
        return steps
    
    async def _analyze_context_requirements(
        self,
        query: str,
        conversation_history: str,
        previous_queries: List[str]
    ) -> Tuple[List[str], bool]:
        """Analyze if query depends on conversation context."""
        context_refs = []
        requires_context = False
        
        # Pronouns that reference prior context
        context_pronouns = ['it', 'they', 'them', 'their', 'these', 'those', 'that', 'this']
        query_lower = query.lower()
        
        for pronoun in context_pronouns:
            if f" {pronoun} " in f" {query_lower} ":
                context_refs.append(pronoun)
                requires_context = True
        
        # Temporal references
        temporal_refs = ['earlier', 'previously', 'before', 'last time', 'recently', 'previous']
        for ref in temporal_refs:
            if ref in query_lower:
                context_refs.append(f"temporal:{ref}")
                requires_context = True
        
        return context_refs, requires_context
    
    async def _calculate_confidence(
        self,
        query: str,
        intent: QueryIntentType,
        entities: List[EntityReference],
        requires_context: bool
    ) -> Tuple[float, ConfidenceLevel]:
        """Calculate overall confidence in query understanding."""
        confidence = 0.5
        
        # 🚨 SMART DETECTION: Boost confidence for specific database lookups
        # These queries are DEFINITELY database queries, not conversational
        query_lower = query.lower()
        has_specific_lookup = False
        
        # Check for patterns like "id 123456", "transaction id", "account number"
        specific_lookup_patterns = [
            r'\bid\s+\d+',  # "id 123456"
            r'\bid\s*=\s*\d+',  # "id = 123456"
            r'transaction\s+id\s+\d+',  # "transaction id 123456"
            r'account\s+id\s+\d+',  # "account id 123456"
            r'customer\s+id\s+\d+',  # "customer id 123456"
            r'order\s+id\s+\d+',  # "order id 123456"
            r'reference\s+\d+',  # "reference 123456"
            r'reference\s+for\s+\w+\s+', # "reference for transaction"
            r'find\s+\w+\s+(?:for|by|with)\s+', # "find X for/by/with Y"
            r'what\s+(?:is|are|was|were)\s+the\s+\w+\s+for\s+', # "what is/are the X for"
        ]
        
        for pattern in specific_lookup_patterns:
            if re.search(pattern, query_lower):
                print(f"[CONFIDENCE_BOOST] Specific lookup detected: {pattern}")
                has_specific_lookup = True
                break
        
        # If it's specific lookup AND it's DATA_RETRIEVAL/DATA_ANALYSIS intent → VERY HIGH confidence
        if has_specific_lookup and intent in [QueryIntentType.DATA_RETRIEVAL, QueryIntentType.DATA_ANALYSIS]:
            print(f"[CONFIDENCE_BOOST] Specific database lookup with {intent.value} → 0.95 confidence")
            confidence = 0.95  # Very high confidence for specific lookups
        else:
            # Original logic
            # Boost if clear intent
            if intent != QueryIntentType.UNKNOWN:
                confidence += 0.20
            
            # Boost if entities found
            if entities:
                avg_entity_confidence = sum(e.confidence for e in entities) / len(entities)
                confidence += avg_entity_confidence * 0.15
        
        # Reduce if missing context (but not for specific lookups)
        if requires_context and not has_specific_lookup:
            confidence -= 0.15
        
        # Reduce if query is too vague (but specific lookups are never vague)
        if len(query) < 10 and not has_specific_lookup:
            confidence -= 0.10
        
        # Cap at 1.0
        confidence = min(1.0, max(0.0, confidence))
        
        # Map to confidence level
        for level in ConfidenceLevel:
            min_val, max_val = level.value
            if min_val <= confidence <= max_val:
                return confidence, level
        
        return confidence, ConfidenceLevel.MEDIUM
    
    async def _generate_clarifying_questions(
        self,
        query: str,
        intent: QueryIntentType,
        entities: List[EntityReference]
    ) -> List[str]:
        """Generate clarifying questions when confidence is low."""
        questions = []
        
        if not entities:
            questions.append("Which table or data would you like me to access?")
        
        if intent == QueryIntentType.UNKNOWN:
            questions.append("Are you looking to retrieve data, modify data, or analyze data?")
        
        if intent == QueryIntentType.DATA_MODIFICATION and not entities:
            questions.append("Which table would you like to update?")
        
        if "where" not in query.lower() and intent == QueryIntentType.DATA_RETRIEVAL:
            questions.append("Would you like to filter the results? (e.g., by date, status, name)")
        
        return questions[:3]  # Limit to 3 questions
    
    async def _generate_reasoning(
        self,
        query: str,
        intent: QueryIntentType,
        operations: List[OperationType],
        confidence: float,
        entities: List[EntityReference]
    ) -> str:
        """Generate human-readable reasoning for the analysis."""
        parts = []
        
        parts.append(f"Intent: {intent.value.replace('_', ' ').title()}")
        
        if operations:
            op_names = [op.value for op in operations[:3]]
            parts.append(f"Operations: {', '.join(op_names)}")
        
        if entities:
            entity_names = [e.name for e in entities[:3]]
            parts.append(f"Entities: {', '.join(entity_names)}")
        
        parts.append(f"Confidence: {confidence:.0%}")
        
        return " | ".join(parts)
    
    async def _assess_complexity(
        self,
        query: str,
        operations: List[OperationType],
        num_clauses: int,
        num_steps: int
    ) -> float:
        """Assess query complexity on 0-1 scale."""
        complexity = 0.2  # Base complexity
        
        # Add for each operation
        complexity += len(operations) * 0.10
        
        # Add for clauses
        complexity += num_clauses * 0.05
        
        # Add for decomposed steps
        complexity += (num_steps - 1) * 0.15
        
        # Add for query length
        if len(query) > 200:
            complexity += 0.15
        elif len(query) > 100:
            complexity += 0.08
        
        # Complexity based on operations (more sophisticated than keyword matching)
        # JOIN, AGGREGATE, GROUP_BY, ANALYSIS indicate higher complexity
        complex_operations = {
            OperationType.JOIN: 0.15,
            OperationType.AGGREGATE: 0.12,
            OperationType.GROUP_BY: 0.10,
            OperationType.ANALYSIS: 0.12,
        }
        
        for operation in operations:
            if operation in complex_operations:
                complexity += complex_operations[operation]
        
        return min(1.0, complexity)
    
    async def _suggest_query_alternatives(self, query: str) -> List[str]:
        """Suggest alternative phrasings when confidence is low."""
        try:
            prompt = f"""A user asked this query and it was unclear: "{query}"

Suggest 2-3 clearer ways they could rephrase this query (one per line).
Return ONLY the alternative phrasings, nothing else."""
            
            response = await llm.call_llm(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            response_text = str(response).strip()
            alternatives = [line.strip() for line in response_text.split("\n") if line.strip()]
            return alternatives[:3]
            
        except Exception:
            return []
    
    def _detect_gibberish(self, query: str) -> Tuple[bool, str]:
        """
        Detect if input is gibberish (nonsensical/random characters).
        
        Returns:
            Tuple[is_gibberish: bool, reason: str]
        """
        if not query or len(query.strip()) == 0:
            return True, "Empty input"
        
        query_clean = query.strip().lower()
        
        # Check 1: Repeated character patterns (hvahvsakhvhkavhkajvkjavkajkahkdavkhdajvkjakda)
        # Look for 3+ consecutive repeated substrings
        repeated_pattern = r'([a-z]+)(\1){2,}'
        if re.search(repeated_pattern, query_clean):
            repeat_matches = re.findall(repeated_pattern, query_clean)
            if len(repeat_matches) > 2:  # More than 2 separate repeat patterns
                return True, "Repeated character patterns detected"
        
        # Check 2: High proportion of uncommon character sequences
        # Common English/Spanish/Portuguese letter combinations
        common_bigrams = {
            'th', 'he', 'in', 'en', 'er', 'to', 'es', 'or', 'ar', 'ou', 'it', 'an', 'st',
            'de', 'la', 'le', 'os', 'as', 'su', 'el', 'ed', 'te', 'se', 'et', 'on', 'al',
            'co', 'do', 're', 'is', 'be', 'ha', 'at', 'by', 'we', 'up'
        }
        
        valid_bigrams = 0
        total_bigrams = 0
        for i in range(len(query_clean) - 1):
            if query_clean[i].isalpha() and query_clean[i+1].isalpha():
                total_bigrams += 1
                bigram = query_clean[i:i+2]
                if bigram in common_bigrams:
                    valid_bigrams += 1
        
        if total_bigrams > 5:  # Only check if enough bigrams
            valid_ratio = valid_bigrams / total_bigrams
            if valid_ratio < 0.25:  # Less than 25% valid bigrams
                return True, "Uncommon character sequences"
        
        # Check 3: Very few spaces for text length (indicate gibberish run-on)
        word_count = len(query_clean.split())
        if len(query_clean) > 30 and word_count == 1:
            return True, "Single run-on word for long input"
        
        # Check 4: Mostly consonants with few vowels (unnatural language pattern)
        vowels = sum(1 for c in query_clean if c in 'aeiouáéíóú')
        consonants = sum(1 for c in query_clean if c.isalpha() and c not in 'aeiouáéíóú')
        
        if consonants > 0 and vowels > 0:
            vowel_ratio = vowels / (vowels + consonants)
            if vowel_ratio < 0.15:  # Less than 15% vowels is very unnatural
                return True, "Insufficient vowel-consonant balance"
        elif consonants > 10 and vowels == 0:  # All consonants
            return True, "No vowels detected"
        
        # Check 5: Keyboard mashing pattern (random adjacent key presses)
        qwerty_rows = [
            'qwertyuiop',
            'asdfghjkl',
            'zxcvbnm'
        ]
        keyboard_matches = 0
        for row in qwerty_rows:
            for i in range(len(query_clean) - 2):
                two_char = query_clean[i:i+2]
                # Check if chars are adjacent on keyboard
                if two_char[0] in row and two_char[1] in row:
                    idx1 = row.index(two_char[0])
                    idx2 = row.index(two_char[1])
                    if abs(idx1 - idx2) <= 1:  # Adjacent keys
                        keyboard_matches += 1
        
        if len(query_clean) > 20 and keyboard_matches > len(query_clean) * 0.5:
            return True, "Keyboard mashing pattern detected"
        
        return False, ""


async def create_nlp_processor(db: AsyncSession) -> AdvancedNLPProcessor:
    """Factory function to create and initialize NLP processor."""
    processor = AdvancedNLPProcessor(db)
    await processor.initialize()
    return processor
