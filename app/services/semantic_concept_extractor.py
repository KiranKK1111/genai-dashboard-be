"""
Semantic Concept Extractor - Stage 1 of Query Understanding Pipeline

Converts natural language queries into structured semantic concepts BEFORE SQL generation.
This addresses the core issue: the system was jumping from prompt → SQL instead of
prompt → semantic concepts → grounded plan → SQL.

Based on ChatGPT-style query understanding:
1. Extract semantic intent (count, filter, group, etc.)
2. Identify entity concepts (customers, accounts, etc.)  
3. Extract filter concepts (gender, temporal, etc.)
4. Normalize temporal expressions (January → month 1, birthday → dob)
5. Output structured intent for downstream grounding

Example transformation:
"How many male and female clients have a birthday in January?"

Input: Natural language
Output: {
  "intent": "count",
  "entity": "clients",
  "filters": [
    {"concept": "gender", "operator": "in", "values": ["male", "female"]},
    {"concept": "birth_month", "operator": "equals", "value": 1}
  ]
}

This fixes the missing temporal concept extraction that caused the system to miss
"birthday in January" → EXTRACT(MONTH FROM dob) = 1
"""

import logging
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Supported query intents"""
    COUNT = "count"
    LIST = "list" 
    AGGREGATE = "aggregate"
    FILTER = "filter"
    GROUP_BY = "group_by"
    COMPARE = "compare"
    TREND = "trend"


class OperatorType(Enum):
    """Supported filter operators"""
    EQUALS = "equals"
    IN = "in"
    NOT_IN = "not_in"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    BETWEEN = "between"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    MONTH_EQUALS = "month_equals"  # For temporal: EXTRACT(MONTH FROM col) = val
    YEAR_EQUALS = "year_equals"    # For temporal: EXTRACT(YEAR FROM col) = val
    DATE_RANGE = "date_range"      # For temporal: DATE BETWEEN start AND end


@dataclass
class FilterConcept:
    """A semantic filter extracted from user query"""
    concept: str                    # Semantic concept (e.g., 'gender', 'birth_month')
    operator: OperatorType         # How to filter
    values: Optional[List[Any]] = None     # For IN, NOT_IN operators
    value: Optional[Any] = None            # For single value operators
    raw_text: Optional[str] = None         # Original text for debugging
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "concept": self.concept,
            "operator": self.operator.value,
            "values": self.values,
            "value": self.value,
            "raw_text": self.raw_text
        }


@dataclass  
class SemanticIntent:
    """Structured semantic intent extracted from query"""
    intent: IntentType
    entity: Optional[str] = None           # Main entity (customers, accounts, etc.)
    filters: List[FilterConcept] = field(default_factory=list)
    aggregation: Optional[str] = None      # COUNT, SUM, AVG, etc.
    grouping: Optional[List[str]] = None   # GROUP BY concepts
    sorting: Optional[Dict[str, str]] = None  # ORDER BY concepts
    limit: Optional[int] = None
    confidence: float = 1.0
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "intent": self.intent.value,
            "entity": self.entity,
            "filters": [f.to_dict() for f in self.filters],
            "aggregation": self.aggregation,
            "grouping": self.grouping,
            "sorting": self.sorting,
            "limit": self.limit,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


class SemanticConceptExtractor:
    """
    Extracts structured semantic concepts from natural language queries.
    
    This is the CRITICAL missing piece that caused temporal concept extraction
    to fail. Instead of jumping straight to SQL generation, this creates a
    semantic understanding layer first.
    """
    
    def __init__(self):
        """Initialize with semantic temporal and entity mappings"""
        
        # Temporal concept mappings (semantic normalization - not configuration dependency)
        self.month_mapping = {
            "january": 1, "jan": 1,
            "february": 2, "feb": 2, 
            "march": 3, "mar": 3,
            "april": 4, "apr": 4,
            "may": 5,
            "june": 6, "jun": 6,
            "july": 7, "jul": 7,
            "august": 8, "aug": 8,
            "september": 9, "sep": 9, "sept": 9,
            "october": 10, "oct": 10,
            "november": 11, "nov": 11,
            "december": 12, "dec": 12
        }
        
        # Temporal field concepts (semantic mapping)
        self.temporal_concepts = {
            "birthday": "birth_month",
            "birth date": "birth_month", 
            "born": "birth_month",
            "dob": "birth_month",
            "birth": "birth_month",
            "birthdate": "birth_month"
        }
        
        # Gender concept mappings
        self.gender_concepts = {
            "male": ["M", "male", "Male"],
            "female": ["F", "female", "Female"], 
            "men": ["M", "male", "Male"],
            "women": ["F", "female", "Female"],
            "man": ["M", "male", "Male"],
            "woman": ["F", "female", "Female"]
        }
        
        # Intent detection patterns
        self.intent_patterns = {
            IntentType.COUNT: [
                r"how many", r"count", r"number of", r"total"
            ],
            IntentType.LIST: [
                r"list", r"show", r"get", r"find", r"retrieve"
            ],
            IntentType.AGGREGATE: [
                r"average", r"sum", r"total", r"maximum", r"minimum"
            ]
        }
        
        # Entity patterns (semantic business entities)
        self.entity_patterns = {
            "customers": ["client", "clients", "customer", "customers"],
            "accounts": ["account", "accounts"], 
            "transactions": ["transaction", "transactions", "payment", "payments"],
            "employees": ["employee", "employees", "staff", "worker", "workers"],
            "users": ["user", "users"]
        }
        
        logger.info(f"[CONCEPT_EXTRACTOR] Loaded {len(self.month_mapping)} month mappings")
        logger.info(f"[CONCEPT_EXTRACTOR] Loaded {len(self.entity_patterns)} entity patterns")
    
    def extract_semantic_intent(self, query: str) -> SemanticIntent:
        """
        Extract structured semantic intent from natural language query.
        
        This is the key function that converts:
        "How many male and female clients have a birthday in January?"
        
        Into:
        {
          "intent": "count",
          "entity": "customers", 
          "filters": [
            {"concept": "gender", "operator": "in", "values": ["M", "F"]},
            {"concept": "birth_month", "operator": "month_equals", "value": 1}
          ]
        }
        """
        query_lower = query.lower()
        
        # Extract intent
        intent = self._extract_intent(query_lower)
        
        # Extract entity
        entity = self._extract_entity(query_lower)
        
        # Extract all filter concepts
        filters = []
        
        # Extract gender filters
        gender_filter = self._extract_gender_filter(query_lower)
        if gender_filter:
            filters.append(gender_filter)
            
        # Extract temporal filters (CRITICAL for the birthday/January case)
        temporal_filter = self._extract_temporal_filter(query_lower)
        if temporal_filter:
            filters.append(temporal_filter)
            
        # Extract other value-based filters
        value_filters = self._extract_value_filters(query_lower) 
        filters.extend(value_filters)
        
        reasoning = f"Extracted {len(filters)} filter concepts from query"
        if temporal_filter:
            reasoning += f" (including temporal filter: {temporal_filter.concept})"
            
        return SemanticIntent(
            intent=intent,
            entity=entity,
            filters=filters,
            aggregation="COUNT" if intent == IntentType.COUNT else None,
            confidence=0.9,
            reasoning=reasoning
        )
    
    def _extract_intent(self, query: str) -> IntentType:
        """Extract the main intent from query"""
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return intent_type
        return IntentType.LIST  # Default
    
    def _extract_entity(self, query: str) -> Optional[str]:
        """Extract the main entity being queried"""
        for entity, aliases in self.entity_patterns.items():
            for alias in aliases:
                if alias in query:
                    return entity
        return None
    
    def _extract_gender_filter(self, query: str) -> Optional[FilterConcept]:
        """
        Extract gender filter concepts.
        
        Examples:
        - "male and female" → gender IN ['M', 'F'] 
        - "men" → gender = 'M'
        - "women and men" → gender IN ['M', 'F']
        """
        found_genders = []
        raw_parts = []
        
        for gender_term, values in self.gender_concepts.items():
            if gender_term in query:
                found_genders.extend(values[:1])  # Take first canonical value
                raw_parts.append(gender_term)
        
        if not found_genders:
            return None
            
        # Remove duplicates while preserving order
        unique_genders = []
        for g in found_genders:
            if g not in unique_genders:
                unique_genders.append(g)
        
        if len(unique_genders) == 1:
            return FilterConcept(
                concept="gender",
                operator=OperatorType.EQUALS,
                value=unique_genders[0], 
                raw_text=" and ".join(raw_parts)
            )
        else:
            return FilterConcept(
                concept="gender",
                operator=OperatorType.IN,
                values=unique_genders,
                raw_text=" and ".join(raw_parts)
            )
    
    def _extract_temporal_filter(self, query: str) -> Optional[FilterConcept]:
        """
        Extract temporal filter concepts - THE KEY MISSING PIECE!
        
        Examples:
        - "birthday in January" → birth_month = 1
        - "born in March" → birth_month = 3  
        - "birthdate in December" → birth_month = 12
        
        This is what was missing that caused the system to generate:
        SELECT COUNT(*) FROM customers WHERE gender IN ('M','F')
        Instead of:
        SELECT COUNT(*) FROM customers WHERE gender IN ('M','F') AND EXTRACT(MONTH FROM dob) = 1
        """
        # Check for temporal field mentions
        temporal_field = None
        for concept, mapped_field in self.temporal_concepts.items():
            if concept in query:
                temporal_field = mapped_field
                break
                
        if not temporal_field:
            return None
            
        # Look for month references
        for month_name, month_num in self.month_mapping.items():
            if month_name in query:
                return FilterConcept(
                    concept=temporal_field,
                    operator=OperatorType.MONTH_EQUALS,
                    value=month_num,
                    raw_text=f"{temporal_field} in {month_name}"
                )
        
        return None
    
    def _extract_value_filters(self, query: str) -> List[FilterConcept]:
        """Extract other value-based filters (can be expanded)"""
        filters = []
        
        # This can be expanded to handle other concepts like:
        # - Status filters ("active users")
        # - Category filters ("premium accounts") 
        # - etc.
        
        return filters


def get_concept_extractor() -> SemanticConceptExtractor:
    """Get singleton instance of concept extractor"""
    global _concept_extractor
    if '_concept_extractor' not in globals():
        _concept_extractor = SemanticConceptExtractor()
    return _concept_extractor