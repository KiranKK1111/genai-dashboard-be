"""
Advanced Semantic Analysis Engine - ChatGPT-Level Intelligence

Multi-layered semantic understanding with contextual reasoning,
dynamic intent classification, and intelligent query enhancement.
"""

import logging
import re
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .. import llm

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Advanced intent classification"""
    CONTINUATION = "continuation"  # "show me more", "next 10"
    REFINEMENT = "refinement"     # "only the active ones", "exclude January"
    TRANSFORMATION = "transformation"  # "group by age", "show as chart" 
    EXPANSION = "expansion"       # "include their addresses", "with details"
    COMPARISON = "comparison"     # "compare with last year", "vs competitors"
    DRILL_DOWN = "drill_down"     # "who are those clients?", "list their names"
    AGGREGATION = "aggregation"   # "sum those amounts", "average age"
    TEMPORAL_SHIFT = "temporal_shift"  # "for 2023", "last month instead"
    FILTER_MODIFY = "filter_modify"    # "only premium customers", "exclude inactive"
    NEW_QUESTION = "new_question"      # Completely new request


class ConversationState(Enum):
    """Conversation context states"""
    FRESH_START = "fresh_start"
    EXPLORING_DATA = "exploring_data"  
    REFINING_RESULTS = "refining_results"
    COMPARING_METRICS = "comparing_metrics"
    ANALYZING_TRENDS = "analyzing_trends"
    DRILLING_DOWN = "drilling_down"


@dataclass
class SemanticContext:
    """Rich semantic context with multi-dimensional understanding"""
    
    # Core detection
    is_followup: bool = False
    confidence: float = 0.0
    intent_type: IntentType = IntentType.NEW_QUESTION
    
    # Conversation understanding
    conversation_state: ConversationState = ConversationState.FRESH_START
    context_continuity: float = 0.0  # How much context carries over
    topic_drift: float = 0.0         # How much topic has changed
    
    # Entity understanding
    referenced_entities: List[str] = field(default_factory=list)
    entity_resolution: Dict[str, str] = field(default_factory=dict) 
    implicit_references: List[str] = field(default_factory=list)
    
    # Query enhancement context
    missing_context: List[str] = field(default_factory=list)
    inherited_filters: Dict[str, Any] = field(default_factory=dict)
    enhanced_query: str = ""
    
    # Reasoning chain
    reasoning_steps: List[str] = field(default_factory=list)
    evidence_scores: Dict[str, float] = field(default_factory=dict)
    
    # Previous context
    previous_query: str = ""
    previous_results_schema: Optional[Dict] = None
    previous_intent: Optional[IntentType] = None


class AdvancedSemanticAnalyzer:
    """
    ChatGPT-Level Semantic Intelligence
    
    Multi-layered understanding with contextual reasoning,
    conversation state tracking, and intelligent enhancement.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize advanced semantic intelligence"""
        
        # Initialize embedding model
        if SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                self.use_embeddings = True
                logger.info("✅ Advanced embeddings loaded for semantic analysis")
            except Exception as e:
                logger.warning(f"Embedding model failed, using LLM fallback: {e}")
                self.embedding_model = None
                self.use_embeddings = False
        else:
            logger.warning("sentence-transformers not available, using LLM-only analysis")
            self.embedding_model = None
            self.use_embeddings = False
        
        # Advanced thresholds (tuned for high accuracy)
        self.semantic_thresholds = {
            "intent_confidence": 0.6,      # Lowered for better detection
            "context_continuity": 0.5,
            "entity_resolution": 0.6,
            "topic_stability": 0.4,
            "reference_strength": 0.55     # Lowered for better detection
        }
        
        # Conversation state tracking
        self.conversation_memory = {}
        
        logger.info("[ADVANCED_SEMANTIC] Initialized ChatGPT-level semantic intelligence")
    
    async def analyze_semantic_context(
        self, 
        query: str, 
        conversation_history: List[Dict] = None,
        session_context: Dict = None
    ) -> SemanticContext:
        """
        Primary method: Deep semantic analysis with ChatGPT-level intelligence
        """
        
        context = SemanticContext()
        context.reasoning_steps.append("🧠 Starting advanced semantic analysis")
        
        if not conversation_history:
            context.intent_type = IntentType.NEW_QUESTION
            context.conversation_state = ConversationState.FRESH_START
            return context
        
        # Stage 1: Multi-modal Intent Classification
        intent_analysis = await self._classify_intent_advanced(query, conversation_history)
        context.intent_type = intent_analysis["intent"]
        context.confidence = intent_analysis["confidence"]
        context.reasoning_steps.extend(intent_analysis["reasoning"])
        
        # Stage 2: Conversation State Understanding
        state_analysis = self._analyze_conversation_state(query, conversation_history)
        context.conversation_state = state_analysis["state"]
        context.context_continuity = state_analysis["continuity"]
        context.topic_drift = state_analysis["drift"]
        
        # Stage 3: Advanced Entity Resolution
        entity_analysis = await self._resolve_entities_advanced(query, conversation_history)
        context.referenced_entities = entity_analysis["entities"]
        context.entity_resolution = entity_analysis["resolutions"]
        context.implicit_references = entity_analysis["implicit"]
        
        # Stage 4: Dynamic Context Enhancement
        enhancement = await self._enhance_query_context(query, conversation_history, context)
        context.enhanced_query = enhancement["enhanced_query"]
        context.inherited_filters = enhancement["filters"]
        context.missing_context = enhancement["missing"]
        
        # Stage 5: Confidence Synthesis
        overall_confidence = self._synthesize_confidence(context)
        context.confidence = overall_confidence
        context.is_followup = overall_confidence > self.semantic_thresholds["intent_confidence"]
        
        context.reasoning_steps.append(f"🎯 Final confidence: {overall_confidence:.2f} ({'Follow-up' if context.is_followup else 'New query'})")
        
        return context
    
    async def _classify_intent_advanced(self, query: str, history: List[Dict]) -> Dict:
        """Advanced intent classification using LLM + embeddings"""
        
        reasoning = []
        
        # Get recent conversation context
        recent_queries = [h.get("user_query", "") for h in history[-5:]]
        recent_context = " | ".join(recent_queries)
        
        # LLM-powered intent classification
        prompt = f"""
Analyze this query in conversation context to determine the user's intent.

CONVERSATION HISTORY:
{recent_context}

CURRENT QUERY: "{query}"

Classify the intent as one of:
- CONTINUATION: User wants more of the same data
- REFINEMENT: User wants to filter/narrow existing results  
- TRANSFORMATION: User wants different format/grouping of same data
- EXPANSION: User wants additional fields/details for same data
- DRILL_DOWN: User wants to see individual records from aggregated results
- COMPARISON: User wants to compare with different time periods/segments
- AGGREGATION: User wants to summarize/calculate from existing results
- TEMPORAL_SHIFT: User wants same query for different time period
- FILTER_MODIFY: User wants to change filters on existing query
- NEW_QUESTION: Completely new, unrelated question

Return JSON:
{{
    "intent": "INTENT_TYPE",
    "confidence": 0.0-1.0,
    "reasoning": "Explanation of why this intent was chosen",
    "context_indicators": ["specific words/phrases that indicate this intent"]
}}
"""
        
        try:
            response = await llm.call_llm(
                [{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=300,
                temperature=0.1
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                intent_data = json.loads(json_match.group())
                
                # Map string to enum
                intent_str = intent_data.get("intent", "NEW_QUESTION")
                try:
                    intent_enum = IntentType(intent_str.lower())
                except ValueError:
                    intent_enum = IntentType.NEW_QUESTION
                
                reasoning.append(f"🤖 LLM Intent: {intent_data.get('reasoning', 'No reasoning')}")
                
                return {
                    "intent": intent_enum,
                    "confidence": intent_data.get("confidence", 0.5),
                    "reasoning": reasoning,
                    "indicators": intent_data.get("context_indicators", [])
                }
                
        except Exception as e:
            reasoning.append(f"⚠️ LLM intent classification failed: {e}")
            logger.warning(f"Intent classification failed: {e}")
        
        # Fallback: Embedding similarity analysis
        if self.use_embeddings:
            similarity_score = self._compute_semantic_similarity(query, recent_queries)
            if similarity_score > 0.7:
                reasoning.append(f"📊 High embedding similarity: {similarity_score:.2f}")
                return {
                    "intent": IntentType.CONTINUATION,
                    "confidence": similarity_score,
                    "reasoning": reasoning,
                    "indicators": []
                }
        
        # Ultimate fallback: Rule-based detection
        return self._fallback_intent_detection(query, reasoning)
    
    def _compute_semantic_similarity(self, query: str, recent_queries: List[str]) -> float:
        """Compute embedding similarity with recent queries"""
        if not self.use_embeddings or not recent_queries:
            return 0.0
        
        try:
            query_embedding = self.embedding_model.encode(query)
            max_similarity = 0.0
            
            for prev_query in recent_queries[-3:]:
                if prev_query.strip():
                    prev_embedding = self.embedding_model.encode(prev_query)
                    similarity = np.dot(query_embedding, prev_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(prev_embedding)
                    )
                    max_similarity = max(max_similarity, similarity)
            
            return float(max_similarity)
            
        except Exception as e:
            logger.warning(f"Embedding similarity computation failed: {e}")
            return 0.0
    
    def _fallback_intent_detection(self, query: str, reasoning: List[str]) -> Dict:
        """Rule-based fallback for intent detection"""
        
        query_lower = query.lower()
        
        # Strong follow-up indicators - boost confidence
        strong_indicators = ["those", "them", "these", "that data", "previous", "above", "earlier"]
        if any(indicator in query_lower for indicator in strong_indicators):
            reasoning.append("🔍 Strong referential language detected")
            return {
                "intent": IntentType.DRILL_DOWN,
                "confidence": 0.85,  # High confidence for strong indicators
                "reasoning": reasoning,
                "indicators": []
            }
        
        # Medium follow-up indicators
        medium_indicators = ["show me more", "next", "additional", "also show", "and", "list", "names", "details"]
        if any(indicator in query_lower for indicator in medium_indicators):
            reasoning.append("➡️ Continuation language detected")
            return {
                "intent": IntentType.CONTINUATION,
                "confidence": 0.75,  # Medium-high confidence
                "reasoning": reasoning,
                "indicators": []
            }
        
        # Weak follow-up indicators
        weak_indicators = ["what", "can you", "show", "get"]
        if any(indicator in query_lower for indicator in weak_indicators):
            reasoning.append("❔ Weak follow-up indicators detected")
            return {
                "intent": IntentType.CONTINUATION,
                "confidence": 0.6,   # Medium confidence
                "reasoning": reasoning,
                "indicators": []
            }
        
        # Default to new question
        reasoning.append("❓ No clear follow-up indicators, treating as new")
        return {
            "intent": IntentType.NEW_QUESTION,
            "confidence": 0.9,
            "reasoning": reasoning,
            "indicators": []
        }
    
    def _analyze_conversation_state(self, query: str, history: List[Dict]) -> Dict:
        """Analyze conversation flow and state transitions"""
        
        if not history:
            return {
                "state": ConversationState.FRESH_START,
                "continuity": 0.0,
                "drift": 0.0
            }
        
        # Analyze query patterns in recent history
        recent_queries = [h.get("user_query", "") for h in history[-5:]]
        
        # Check for data exploration patterns
        if any("show" in q.lower() or "list" in q.lower() for q in recent_queries):
            if len([q for q in recent_queries if "where" in q.lower() or "filter" in q.lower()]) > 1:
                return {
                    "state": ConversationState.REFINING_RESULTS,
                    "continuity": 0.8,
                    "drift": 0.2
                }
            else:
                return {
                    "state": ConversationState.EXPLORING_DATA,
                    "continuity": 0.7,
                    "drift": 0.3
                }
        
        # Check for analysis patterns
        if any(word in " ".join(recent_queries).lower() for word in ["count", "sum", "average", "total"]):
            return {
                "state": ConversationState.ANALYZING_TRENDS,
                "continuity": 0.6,
                "drift": 0.4
            }
        
        return {
            "state": ConversationState.FRESH_START,
            "continuity": 0.0,
            "drift": 1.0
        }
    
    async def _resolve_entities_advanced(self, query: str, history: List[Dict]) -> Dict:
        """Advanced entity resolution with contextual understanding"""
        
        # Extract entities from current query
        current_entities = self._extract_entities_nlp(query)
        
        # Get entities from recent history
        historical_entities = []
        for h in history[-3:]:
            prev_query = h.get("user_query", "")
            if prev_query:
                historical_entities.extend(self._extract_entities_nlp(prev_query))
        
        # Resolve pronouns and references
        resolutions = {}
        implicit_refs = []
        
        # Detect referential language
        referential_patterns = [
            r'\bthose\s+(\w+)',
            r'\bthese\s+(\w+)', 
            r'\bthat\s+(\w+)',
            r'\bthe\s+(previous|earlier|above)\s+(\w+)',
            r'\bthem\b',
            r'\bit\b'
        ]
        
        for pattern in referential_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                implicit_refs.append(match.group())
                
                # Try to resolve to historical entities
                if historical_entities:
                    # Simple resolution: map to most recent similar entity
                    resolutions[match.group()] = historical_entities[-1]
        
        return {
            "entities": current_entities,
            "resolutions": resolutions,
            "implicit": implicit_refs
        }
    
    def _extract_entities_nlp(self, text: str) -> List[str]:
        """Extract business entities using NLP patterns"""
        
        entities = []
        
        # Business entity patterns
        entity_patterns = {
            "customers": r'\b(customer|client|user|person|people|individual)s?\b',
            "products": r'\b(product|item|good|service|offering)s?\b',
            "transactions": r'\b(transaction|payment|purchase|sale|order)s?\b',
            "accounts": r'\b(account|profile|record)s?\b',
            "metrics": r'\b(revenue|sales|profit|cost|amount|value|price)s?\b'
        }
        
        for entity_type, pattern in entity_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                entities.append(entity_type)
        
        return entities
    
    async def _enhance_query_context(self, query: str, history: List[Dict], context: SemanticContext) -> Dict:
        """Enhance query with missing context using ChatGPT-style reasoning"""
        
        enhanced_query = query
        inherited_filters = {}
        missing_context = []
        
        if context.intent_type in [IntentType.CONTINUATION, IntentType.DRILL_DOWN, IntentType.REFINEMENT]:
            
            # Get the most recent query with results
            for h in reversed(history[-5:]):
                if h.get("sql_query") and h.get("results"):
                    previous_query = h.get("user_query", "")
                    previous_sql = h.get("sql_query", "")
                    
                    # Extract filters from previous SQL
                    filters = self._extract_filters_from_sql(previous_sql)
                    inherited_filters.update(filters)
                    
                    # Enhance current query with context
                    if context.intent_type == IntentType.DRILL_DOWN:
                        if "list" in query.lower() or "show" in query.lower():
                            enhanced_query = f"{query} (from previous result set: {previous_query})"
                    
                    break
        
        return {
            "enhanced_query": enhanced_query,
            "filters": inherited_filters,
            "missing": missing_context
        }
    
    def _extract_filters_from_sql(self, sql: str) -> Dict[str, Any]:
        """Extract WHERE clause filters from SQL query"""
        
        filters = {}
        
        # Simple WHERE clause extraction
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            filters["where_clause"] = where_clause.strip()
        
        return filters
    
    def _synthesize_confidence(self, context: SemanticContext) -> float:
        """Synthesize overall confidence from multiple signals"""
        
        confidence_factors = []
        
        # Intent confidence
        base_confidence = context.confidence
        confidence_factors.append(("base_intent", base_confidence, 0.3))
        
        # Context continuity
        continuity_score = context.context_continuity
        confidence_factors.append(("context_continuity", continuity_score, 0.25))
        
        # Entity resolution strength
        entity_score = len(context.implicit_references) * 0.2
        entity_score = min(entity_score, 1.0)
        confidence_factors.append(("entity_references", entity_score, 0.25))
        
        # Topic stability (inverse of drift)
        stability_score = 1.0 - context.topic_drift
        confidence_factors.append(("topic_stability", stability_score, 0.2))
        
        # Weighted combination
        weighted_sum = sum(score * weight for _, score, weight in confidence_factors)
        
        # Evidence scoring
        context.evidence_scores = {name: score for name, score, _ in confidence_factors}
        
        return min(weighted_sum, 1.0)


# Factory function
def get_advanced_semantic_analyzer() -> AdvancedSemanticAnalyzer:
    """Get singleton instance of advanced semantic analyzer"""
    global _advanced_analyzer
    if '_advanced_analyzer' not in globals():
        _advanced_analyzer = AdvancedSemanticAnalyzer()
    return _advanced_analyzer