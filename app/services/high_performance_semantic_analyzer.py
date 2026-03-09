"""
Simplified High-Performance Semantic Analyzer

Focuses on rule-based detection that works reliably without LLM dependencies.
Optimized for high accuracy follow-up detection.
"""

import logging
import re  
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    SentenceTransformer = None
    np = None

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Simplified intent classification"""
    DRILL_DOWN = "drill_down"         # "list those clients", "show me the names"
    CONTINUATION = "continuation"      # "show me more", "next 10"
    REFINEMENT = "refinement"         # "only the active ones", "exclude January"
    NEW_QUESTION = "new_question"     # Completely new request


@dataclass
class HighPerformanceSemanticContext:
    """Simplified semantic context focused on accuracy"""
    
    is_followup: bool = False
    confidence: float = 0.0
    intent_type: IntentType = IntentType.NEW_QUESTION
    
    # Detection reasoning
    reasoning_steps: List[str] = field(default_factory=list)
    detected_indicators: List[str] = field(default_factory=list)
    
    # Context elements
    referenced_entities: List[str] = field(default_factory=list)
    embedding_similarity: float = 0.0


class HighPerformanceSemanticAnalyzer:
    """
    High-Performance Semantic Analyzer
    
    Focuses on reliable rule-based detection with optional embeddings boost.
    Designed for 90%+ accuracy without external dependencies.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize with optional embeddings"""
        
        # Try to initialize embeddings (optional boost)
        if SentenceTransformer and np:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                self.use_embeddings = True
                logger.info("✅ Embeddings loaded for semantic boost")
            except Exception as e:
                logger.info(f"Embeddings not available, using rule-based only: {e}")
                self.embedding_model = None
                self.use_embeddings = False
        else:
            self.embedding_model = None
            self.use_embeddings = False
        
        # High-accuracy detection patterns
        self.strong_followup_patterns = [
            # Explicit references
            r'\b(those|them|these)\s+(client|customer|user|people|person|result)s?\b',
            r'\b(that|the)\s+(data|result|query|search)\b',
            r'\b(previous|earlier|above|prior)\s+',
            
            # List expansion requests  
            r'\b(list|show|display|get).*\b(those|them|these|that)\b',
            r'\b(list|show|display)\s+(all|the)?\s*(client|customer|user|people|name|detail)s?\b',
            
            # Detail drilling
            r'\b(name|detail|info|information)s?\s*(of|for|about)?\s*(those|them|these)?\b',
            r'\b(can you|could you).*\b(show|get|list|tell)\b',
            r'\b(what about|how about)\s+',
        ]
        
        self.medium_followup_patterns = [
            # Continuation language
            r'\b(more|additional|also|next|further)\b',
            r'\b(and|plus|additionally|furthermore)\b',
            r'\b(expand|elaborate|break down|drill down)\b',
            
            # Simple requests
            r'^\s*(show|list|get|display)\s+',
            r'^\s*(what|which|who|how)\s+',
        ]
        
        self.weak_followup_patterns = [
            # Vague references
            r'\bit\b|\bthat\b|\bthey\b',
            r'\b(more|other|different)\b',
        ]
        
        logger.info("🎯 High-Performance Semantic Analyzer initialized")
    
    async def analyze_semantic_context(
        self, 
        query: str, 
        conversation_history: List[Dict] = None,
        session_context: Dict = None
    ) -> HighPerformanceSemanticContext:
        """
        High-accuracy semantic analysis optimized for follow-up detection
        """
        
        context = HighPerformanceSemanticContext()
        context.reasoning_steps.append("🎯 Starting high-performance semantic analysis")
        
        if not conversation_history:
            context.intent_type = IntentType.NEW_QUESTION
            context.reasoning_steps.append("❓ No conversation history - new question")
            return context
        
        # Stage 1: Pattern-based detection (primary method)
        pattern_score, pattern_indicators, intent = self._analyze_patterns(query)
        context.detected_indicators = pattern_indicators
        context.intent_type = intent
        
        # Stage 2: Entity continuity analysis  
        entity_score, entities = self._analyze_entity_continuity(query, conversation_history)
        context.referenced_entities = entities
        
        # Stage 3: Embedding similarity (if available)
        embedding_score = 0.0
        if self.use_embeddings:
            embedding_score = self._compute_semantic_similarity(query, conversation_history)
            context.embedding_similarity = embedding_score
        
        # Stage 4: Confidence synthesis
        final_confidence = self._synthesize_confidence(pattern_score, entity_score, embedding_score, context)
        context.confidence = final_confidence
        context.is_followup = final_confidence > 0.6  # Tuned threshold
        
        # Add final reasoning
        status = "Follow-up detected" if context.is_followup else "New query"
        context.reasoning_steps.append(f"🎯 {status} with {final_confidence:.2f} confidence")
        
        return context
    
    def _analyze_patterns(self, query: str) -> Tuple[float, List[str], IntentType]:
        """Analyze query patterns for follow-up indicators"""
        
        query_lower = query.lower().strip()
        indicators = []
        score = 0.0
        intent = IntentType.NEW_QUESTION
        
        # Check strong patterns (high confidence)
        for pattern in self.strong_followup_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                score += 0.4  # Each strong pattern adds significant confidence
                indicators.extend([f"strong:{match}" if isinstance(match, str) else f"strong:{pattern[:20]}..." for match in matches])
                
                # Determine specific intent
                if any(word in query_lower for word in ["list", "show", "display", "name", "detail"]):
                    intent = IntentType.DRILL_DOWN
                elif any(word in query_lower for word in ["more", "additional", "next"]):
                    intent = IntentType.CONTINUATION
                elif any(word in query_lower for word in ["only", "filter", "exclude", "include"]):
                    intent = IntentType.REFINEMENT
        
        # Check medium patterns (medium confidence)  
        for pattern in self.medium_followup_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                score += 0.25  # Medium confidence boost
                indicators.extend([f"medium:{match}" if isinstance(match, str) else f"medium:{pattern[:20]}..." for match in matches])
                
                if intent == IntentType.NEW_QUESTION:  # Only set if not already set
                    intent = IntentType.CONTINUATION
        
        # Check weak patterns (low confidence)
        for pattern in self.weak_followup_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                score += 0.1  # Small confidence boost
                indicators.extend([f"weak:{match}" if isinstance(match, str) else f"weak:{pattern[:20]}..." for match in matches])
        
        # Cap the score
        score = min(score, 1.0)
        
        return score, indicators, intent
    
    def _analyze_entity_continuity(self, query: str, history: List[Dict]) -> Tuple[float, List[str]]:
        """Analyze entity continuity across conversation"""
        
        # Extract entities from current query
        current_entities = self._extract_business_entities(query)
        
        # Extract entities from recent history
        historical_entities = set()
        for h in history[-3:]:  # Last 3 turns
            prev_query = h.get("user_query", "")
            if prev_query:
                historical_entities.update(self._extract_business_entities(prev_query))
        
        # Calculate overlap
        if not historical_entities:
            return 0.0, current_entities
        
        entity_overlap = len(set(current_entities) & historical_entities) / max(len(historical_entities), 1)
        
        # Boost score if query is short but refers to same entities
        if len(query.split()) <= 5 and entity_overlap > 0:
            entity_overlap = min(entity_overlap + 0.3, 1.0)
        
        return entity_overlap, current_entities
    
    def _extract_business_entities(self, text: str) -> List[str]:
        """Extract business entities from text"""
        
        entities = []
        text_lower = text.lower()
        
        entity_patterns = {
            "customers": r'\b(customer|client|user|person|people|individual)s?\b',
            "products": r'\b(product|item|service|offering)s?\b',
            "transactions": r'\b(transaction|payment|purchase|sale|order)s?\b', 
            "accounts": r'\b(account|profile|record)s?\b',
            "employees": r'\b(employee|staff|worker|member)s?\b'
        }
        
        for entity_type, pattern in entity_patterns.items():
            if re.search(pattern, text_lower):
                entities.append(entity_type)
        
        return entities
    
    def _compute_semantic_similarity(self, query: str, history: List[Dict]) -> float:
        """Compute embedding similarity if available"""
        
        if not self.use_embeddings or not history:
            return 0.0
        
        try:
            query_embedding = self.embedding_model.encode(query)
            max_similarity = 0.0
            
            for h in history[-3:]:  # Last 3 queries
                prev_query = h.get("user_query", "")
                if prev_query and len(prev_query.strip()) > 3:
                    prev_embedding = self.embedding_model.encode(prev_query)
                    similarity = np.dot(query_embedding, prev_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(prev_embedding)
                    )
                    max_similarity = max(max_similarity, float(similarity))
            
            return max_similarity
            
        except Exception as e:
            logger.warning(f"Embedding similarity failed: {e}")
            return 0.0
    
    def _synthesize_confidence(self, pattern_score: float, entity_score: float, embedding_score: float, context: HighPerformanceSemanticContext) -> float:
        """Synthesize final confidence from multiple signals"""
        
        # Pattern-based detection is primary (most reliable)
        base_confidence = pattern_score * 0.6
        
        # Entity continuity provides boost
        entity_boost = entity_score * 0.25
        
        # Embedding similarity provides additional boost (if available)
        embedding_boost = embedding_score * 0.15 if self.use_embeddings else 0.0
        
        # Strong pattern bonus
        strong_indicators = [i for i in context.detected_indicators if i.startswith("strong:")]
        if strong_indicators:
            base_confidence = min(base_confidence + 0.2, 1.0)
        
        # Short query with entities bonus (likely referential)
        if len(context.referenced_entities) > 0 and len(context.reasoning_steps) > 0:
            query_length = len(context.reasoning_steps[0].split())  # Approximate
            if query_length <= 5:
                base_confidence = min(base_confidence + 0.15, 1.0)
        
        total_confidence = base_confidence + entity_boost + embedding_boost
        
        # Add reasoning
        context.reasoning_steps.append(f"📊 Pattern: {pattern_score:.2f}, Entity: {entity_score:.2f}, Embedding: {embedding_score:.2f}")
        context.reasoning_steps.append(f"🎯 Indicators: {len(context.detected_indicators)} ({len(strong_indicators)} strong)")
        
        return min(total_confidence, 1.0)


# Factory function
def get_high_performance_semantic_analyzer() -> HighPerformanceSemanticAnalyzer:
    """Get singleton instance of high-performance semantic analyzer"""
    global _hp_analyzer
    if '_hp_analyzer' not in globals():
        _hp_analyzer = HighPerformanceSemanticAnalyzer()
    return _hp_analyzer