"""
Entity Parser - Extracts entities and filter values from user queries.

Analyzes user queries to identify:
- The entity/table type they're asking about
- Filter values (ID, date range, amount thresholds, etc.)
- Query intent (GET_BY_ID, LIST, AGGREGATE, FILTER, etc.)

Works with Schema Discovery & Schema Normalizer to map to DB columns.
Optional semantic intent classification (replaces hardcoded keywords if enabled).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .. import llm
# NEW PHASE 2: Optional semantic intent classification
from .intent_classifier import IntentClassifier, IntentClassification
from ..config import settings

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """High-level query intent."""
    GET_BY_ID = "get_by_id"
    LIST = "list"
    COUNT = "count"
    AGGREGATE = "aggregate"  # sum, avg, min, max
    FILTER = "filter"
    GROUP_BY = "group_by"
    JOIN = "join"


@dataclass
class EntityReference:
    """A reference to a domain entity in a query."""
    entity_name: str
    filters: Dict[str, Any]  # column_name -> value
    aggregation: Optional[str] = None  # COUNT, SUM, AVG, etc.


@dataclass
class ParsedQuery:
    """Result of parsing a user query."""
    intent: QueryIntent
    entities: List[EntityReference]
    filter_conditions: Dict[str, Tuple[str, Any]]  # column -> (operator, value)
    group_by_column: Optional[str] = None
    order_by_column: Optional[str] = None
    limit: Optional[int] = None
    reasoning: str = ""


class EntityParser:
    """
    Parses natural language queries to extract entities and intents.
    
    Uses LLM for semantic understanding + simple regex for structured extraction.
    Optionally uses semantic intent classifier (if enabled in config) instead of hardcoded keywords.
    """
    
    def __init__(self):
        # Common patterns for structured data extraction
        self.id_pattern = re.compile(
            r'(?:id|for|with)\s+(?:id\s+)?(?:is\s+)?(["\']?)(\d+)\1',
            re.IGNORECASE
        )
        self.date_pattern = re.compile(
            r'(?:on|since|from|between)\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            re.IGNORECASE
        )
        self.amount_pattern = re.compile(
            r'(?:amount|total|value|above|below|greater|less|more|less)\s+(?:of|than)?\s*([₹$€£])?(\d+(?:[.,]\d+)*)',
            re.IGNORECASE
        )
        
        # NEW PHASE 2: Optional semantic intent classifier
        self.intent_classifier: Optional[IntentClassifier] = None
        if settings.enable_semantic_intent_classifier:
            try:
                self.intent_classifier = IntentClassifier()
                logger.debug("[ENTITY PARSER] Semantic intent classifier enabled")
            except Exception as e:
                logger.warning(f"[ENTITY PARSER] Could not initialize semantic intent classifier: {e}")
    
    async def parse_query(self, user_query: str) -> ParsedQuery:
        """
        Parse a user query into structured components.
        
        Args:
            user_query: Natural language query
        
        Returns:
            ParsedQuery with extracted entities and intents
        """
        
        # Use LLM for semantic parsing
        return await self._parse_with_llm(user_query)
    
    async def _parse_with_llm(self, user_query: str) -> ParsedQuery:
        """Use LLM to parse the query."""
        
        system_prompt = """You are a query parser that analyzes natural language database queries.
Extract structured information and respond with ONLY valid JSON.

Respond with this structure:
{
  "intent": "get_by_id|list|count|aggregate|filter|group_by|join",
  "primary_entity": "<the main entity/table user is asking about>",
  "filters": {
    "field_name": "extracted_value" (e.g., {"id": "12345"})
  },
  "aggregation": "COUNT|SUM|AVG|MIN|MAX|null",
  "group_by": null or "column_name",
  "order_by": null or "column_name",
  "limit": null or number,
  "reasoning": "brief explanation"
}

Examples:
{"intent":"get_by_id","primary_entity":"record","filters":{"id":"12345"},"reasoning":"User asking for specific record"}
{"intent":"list","primary_entity":"entity","filters":{},"reasoning":"User wants a list"}
{"intent":"aggregate","primary_entity":"data","aggregation":"SUM","filters":{},"reasoning":"User wants total"}
"""
        
        user_prompt = f"Parse this query: {user_query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            response = await llm.call_llm(messages, stream=False, max_tokens=300)
            
            # Parse JSON response
            import json
            json_str = response.strip()
            if "```" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                json_str = json_str[start:end]
            
            parsed = json.loads(json_str)
            
            intent_str = parsed.get("intent", "list").upper().replace("-", "_")
            try:
                intent = QueryIntent[intent_str]
            except KeyError:
                intent = QueryIntent.LIST
            
            return ParsedQuery(
                intent=intent,
                entities=[EntityReference(
                    entity_name=parsed.get("primary_entity", "unknown"),
                    filters=parsed.get("filters", {}),
                    aggregation=parsed.get("aggregation")
                )],
                filter_conditions={
                    k: ("=", v) for k, v in parsed.get("filters", {}).items()
                },
                group_by_column=parsed.get("group_by"),
                order_by_column=parsed.get("order_by"),
                limit=parsed.get("limit"),
                reasoning=parsed.get("reasoning", "")
            )
            
        except Exception as e:
            logger.warning(f"LLM parsing failed: {e}, falling back to regex")
            return await self._parse_with_regex(user_query)
    
    async def _parse_with_regex(self, user_query: str) -> ParsedQuery:
        """Fallback: use regex patterns for simple extraction."""
        
        filters: Dict[str, Tuple[str, Any]] = {}
        
        # Extract IDs
        id_match = self.id_pattern.search(user_query)
        if id_match:
            filters["id"] = ("=", int(id_match.group(2)))
        
        # Extract dates
        date_match = self.date_pattern.search(user_query)
        if date_match:
            filters["date"] = (">=", date_match.group(1))
        
        # Extract amounts
        amount_match = self.amount_pattern.search(user_query)
        if amount_match:
            amount_str = amount_match.group(2).replace(",", "").replace(".", "")
            filters["amount"] = ("=", float(amount_str))
        
        # Determine intent
        query_lower = user_query.lower()
        
        # NEW PHASE 2: Try semantic intent classifier first (if enabled)
        intent = None
        classification: Optional[IntentClassification] = None
        
        if self.intent_classifier:
            try:
                classification = await self.intent_classifier.classify_intent(user_query, None)
                if classification and classification.confidence >= settings.intent_classification_confidence_threshold:
                    intent = classification.intent
                    intent_val = intent.value if hasattr(intent, 'value') else str(intent)
                    logger.debug(
                        f"[ENTITY PARSER] Semantic classification: {intent_val} "
                        f"(confidence: {classification.confidence})"
                    )
            except Exception as e:
                logger.debug(f"[ENTITY PARSER] Semantic classification failed: {e}")
        
        # Fallback to hardcoded keywords if classification didn't work or disabled
        if intent is None:
            if any(kw in query_lower for kw in ["count", "how many"]):
                intent = QueryIntent.COUNT
            elif any(kw in query_lower for kw in ["sum", "total", "average", "avg", "min", "max"]):
                intent = QueryIntent.AGGREGATE
            elif any(kw in query_lower for kw in ["group", "by", "grouped"]):
                intent = QueryIntent.GROUP_BY
            elif filters and len(filters) == 1 and any(kw in str(list(filters.keys())[0]).lower() for kw in ["id", "code", "ref"]):
                intent = QueryIntent.GET_BY_ID
            elif filters:
                intent = QueryIntent.FILTER
            else:
                intent = QueryIntent.LIST
        
        reasoning = f"Semantic classification (confidence: {classification.confidence} )" if classification else "Extracted using hardcoded patterns"
        
        return ParsedQuery(
            intent=intent,
            entities=[EntityReference(
                entity_name="unknown",
                filters={k: v[1] for k, v in filters.items()}
            )],
            filter_conditions=filters,
            reasoning=reasoning,
        )
    
    def extract_filter_value(self, query: str, field_name: str) -> Optional[Any]:
        """Extract a specific filter value from query."""
        # Try ID extraction
        if "id" in field_name.lower():
            match = self.id_pattern.search(query)
            if match:
                return int(match.group(2))
        
        # Try amount extraction
        if any(x in field_name.lower() for x in ["amount", "value", "total"]):
            match = self.amount_pattern.search(query)
            if match:
                return float(match.group(2).replace(",", ""))
        
        return None


async def create_entity_parser() -> EntityParser:
    """Factory function to create an entity parser."""
    return EntityParser()
