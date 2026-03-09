"""
Intent Classifier - Classifies query intent using pure semantic analysis.

Replaces hardcoded keyword checking with LLM-based semantic inference.
Classifies queries into: COUNT, AGGREGATE, FILTER, GROUP_BY, GET_BY_ID, LIST, JOIN

No hardcoded keyword lists. Pure semantic understanding.
"""

from __future__ import annotations

import logging
import json
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .. import llm

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """High-level query intent (matches entity_parser.QueryIntent)."""
    GET_BY_ID = "get_by_id"
    LIST = "list"
    COUNT = "count"
    AGGREGATE = "aggregate"  # sum, avg, min, max
    FILTER = "filter"
    GROUP_BY = "group_by"
    JOIN = "join"


@dataclass
class IntentClassification:
    """Result of intent classification."""
    intent: QueryIntent
    confidence: float  # 0.0-1.0
    reasoning: str
    key_signals: list


class IntentClassifier:
    """Pure semantic intent classification (replaces hardcoded keywords)."""
    
    def __init__(self, cache_size: int = 100):
        self.cache_size = cache_size
        self._intent_cache = {}  # Simple LRU cache
    
    async def classify_intent(
        self,
        query: str,
        schema_context: Optional[str] = None
    ) -> IntentClassification:
        """
        Classify query intent WITHOUT keywords.
        
        Uses LLM semantic analysis to determine:
        - What the user wants to do (COUNT, AGGREGATE, FILTER, etc.)
        - Confidence level
        - Reasoning and signals used
        
        Examples (using generic entities):
        - "How many [entities]?" → COUNT (high confidence)
        - "What's the [entity] count?" → COUNT (high confidence, same meaning, different wording)
        - "Total [metric]" → AGGREGATE (high confidence)
        - "[Status] [entities]" → FILTER (high confidence)
        - "[Entities] by [category]" → GROUP_BY (high confidence)
        - "Show all" → LIST (high confidence)
        - "[Entity] #123" → GET_BY_ID (high confidence)
        
        Works by: Pure LLM semantic analysis, not keyword matching
        """
        
        # Check cache
        cache_key = query.lower()
        if cache_key in self._intent_cache:
            logger.debug(f"Cache hit for intent: {query}")
            return self._intent_cache[cache_key]
        
        logger.debug(f"Classifying intent for: {query}")
        
        schema_context_str = schema_context or "No schema context provided."
        
        prompt = f"""
You are a SQL query intent classifier. Analyze this query and determine the user's intent.

Query: {query}

Schema Context:
{schema_context_str}

Classify the intent based on SEMANTIC meaning, NOT keywords.

Possible intents:
1. COUNT - User asks for QUANTITY/NUMBER of records (not specific records)
   Pattern: "How many?", "Total number?", "What is the count?"
   
2. AGGREGATE - User asks for COMPUTED VALUE (sum, avg, min, max, etc.)
   Pattern: "Total amount", "Average value", "Highest balance", "Sum of records"
   
3. FILTER - User wants SPECIFIC records with criteria
   Pattern: "[status] [entities]", "[entities] > [value]", "[filter] [entities]"
   
4. GROUP_BY - User wants DATA GROUPED by category
   Pattern: "By [category]", "Grouped by [field]", "Per [entity]", "Count by [type]"
   
5. GET_BY_ID - User wants SINGLE record by identifier
   Pattern: "[Entity] #123", "[Entity] ID 456", "[Entity] [code]"
   
6. LIST - User wants ALL or MANY records (no specific filtering, just show them)
   Pattern: "Show all", "List [entities]", "Get [entities]"
   
7. JOIN - User combines data from MULTIPLE tables
   Pattern: "[EntityA] and their [EntityB]", "[EntityA] with [EntityB]"

Respond with ONLY valid JSON (no markdown, no explanation):
{{
  "intent": "COUNT|AGGREGATE|FILTER|GROUP_BY|GET_BY_ID|LIST|JOIN",
  "confidence": 0.0-1.0,
  "reasoning": "Why you selected this intent",
  "key_signals": ["signal1", "signal2"]
}}
"""
        
        try:
            response = await llm.call_llm([
                {
                    "role": "system",
                    "content": "You are an intent classifier. Respond with ONLY valid JSON."
                },
                {"role": "user", "content": prompt}
            ], max_tokens=300, temperature=0.3)
            
            response_text = str(response)
            result = json.loads(response_text)
            
            # Map string to enum
            intent_str = result.get("intent", "LIST").upper()
            intent = QueryIntent[intent_str]
            
            classification = IntentClassification(
                intent=intent,
                confidence=float(result.get("confidence", 0.5)),
                reasoning=result.get("reasoning", ""),
                key_signals=result.get("key_signals", [])
            )
            
            # Cache result (simple size limit)
            if len(self._intent_cache) >= self.cache_size:
                # Remove oldest entry (naive approach)
                self._intent_cache.pop(next(iter(self._intent_cache)))
            
            self._intent_cache[cache_key] = classification
            
            intent_val = intent.value if hasattr(intent, 'value') else str(intent)
            logger.debug(
                f"Classified intent for '{query}': {intent_val} "
                f"(confidence: {classification.confidence})"
            )
            
            return classification
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return IntentClassification(
                intent=QueryIntent.LIST,
                confidence=0.3,
                reasoning="Failed to parse intent classification, defaulting to LIST",
                key_signals=["parse_error"]
            )
        
        except Exception as e:
            logger.error(f"Error classifying intent: {e}", exc_info=True)
            return IntentClassification(
                intent=QueryIntent.LIST,
                confidence=0.2,
                reasoning=f"Error during classification: {str(e)}",
                key_signals=["error"]
            )
    
    def clear_cache(self) -> None:
        """Clear intent cache."""
        self._intent_cache.clear()
        logger.debug("Cleared intent cache")
