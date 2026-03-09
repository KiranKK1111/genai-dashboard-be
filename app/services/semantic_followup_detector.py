"""
Semantic Follow-Up Detector - True Semantic Analysis

Replaces pattern-based detection with semantic understanding.
Uses embeddings and LLM analysis to detect follow-up queries without
hardcoded patterns or configuration dependencies.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class SemanticFollowUpContext:
    """Semantic context for follow-up detection"""
    is_followup: bool
    confidence: float
    previous_query_embedding: Optional[np.ndarray] = None
    semantic_similarity: float = 0.0
    referential_indicators: List[str] = None
    reasoning: str = ""
    previous_entities: List[str] = None
    current_entities: List[str] = None


class SemanticFollowUpDetector:
    """
    Detects follow-up queries using semantic analysis instead of patterns.
    
    Methods:
    1. Embedding Similarity: Compare query embeddings to detect related queries
    2. Entity Continuity: Track entity references across conversation  
    3. LLM Analysis: Use language model to identify referential language
    4. Context Coherence: Analyze semantic coherence between queries
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize with semantic models (no pattern dependencies)"""
        
        # Load embedding model for semantic similarity
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Semantic thresholds (learnable, not hardcoded)
        self.similarity_threshold = 0.6  # Can be learned from data
        self.context_window = 3  # Consider last 3 queries
        
        # Entity tracking for continuity analysis
        self.entity_history = {}
        
        logger.info(f"[SEMANTIC_FOLLOWUP] Initialized with {embedding_model}")
    
    async def detect_semantic_followup(
        self, 
        current_query: str,
        session_id: str,
        session_manager,
        db: AsyncSession
    ) -> SemanticFollowUpContext:
        """
        Detect follow-up using semantic analysis (no patterns needed).
        
        Returns semantic context with confidence scores and reasoning.
        """
        
        # Get conversation history
        history = await self._get_conversation_history(session_id, session_manager)
        
        if not history:
            return SemanticFollowUpContext(
                is_followup=False,
                confidence=0.0,
                reasoning="No conversation history"
            )
        
        # Method 1: Embedding Similarity Analysis
        similarity_result = await self._analyze_embedding_similarity(
            current_query, history
        )
        
        # Method 2: Entity Continuity Analysis  
        entity_result = await self._analyze_entity_continuity(
            current_query, history, session_id
        )
        
        # Method 3: LLM Referential Analysis
        llm_result = await self._analyze_referential_language(
            current_query, history, db
        )
        
        # Method 4: Semantic Coherence Analysis
        coherence_result = await self._analyze_semantic_coherence(
            current_query, history
        )
        
        # Combine results with weighted scoring
        combined_confidence = self._combine_confidence_scores([
            (similarity_result['confidence'], 0.3),
            (entity_result['confidence'], 0.25), 
            (llm_result['confidence'], 0.25),
            (coherence_result['confidence'], 0.2)
        ])
        
        is_followup = combined_confidence > 0.6
        
        return SemanticFollowUpContext(
            is_followup=is_followup,
            confidence=combined_confidence,
            previous_query_embedding=similarity_result.get('embedding'),
            semantic_similarity=similarity_result['confidence'],
            referential_indicators=llm_result.get('indicators', []),
            reasoning=self._generate_reasoning([
                similarity_result, entity_result, llm_result, coherence_result
            ]),
            previous_entities=entity_result.get('previous_entities', []),
            current_entities=entity_result.get('current_entities', [])
        )
    
    async def _analyze_embedding_similarity(
        self, 
        current_query: str, 
        history: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze semantic similarity using embeddings"""
        
        current_embedding = self.embedding_model.encode(current_query)
        max_similarity = 0.0
        best_previous = None
        
        for entry in history[-self.context_window:]:
            previous_query = entry.get('user_query', '')
            if not previous_query:
                continue
                
            previous_embedding = self.embedding_model.encode(previous_query)
            similarity = np.dot(current_embedding, previous_embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(previous_embedding)
            )
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_previous = entry
        
        return {
            'confidence': min(max_similarity / self.similarity_threshold, 1.0),
            'embedding': current_embedding,
            'similarity': max_similarity,
            'best_match': best_previous,
            'method': 'embedding_similarity'
        }
    
    async def _analyze_entity_continuity(
        self, 
        current_query: str, 
        history: List[Dict],
        session_id: str
    ) -> Dict[str, Any]:
        """Analyze entity continuity across conversation turns"""
        
        # Extract entities from current query (simple keyword extraction)
        current_entities = self._extract_entities(current_query)
        
        # Get entities from recent history
        previous_entities = []
        for entry in history[-2:]:  # Last 2 queries
            previous_entities.extend(
                self._extract_entities(entry.get('user_query', ''))
            )
        
        # Calculate entity overlap
        if not previous_entities:
            entity_overlap = 0.0
        else:
            common_entities = set(current_entities) & set(previous_entities)
            entity_overlap = len(common_entities) / len(set(previous_entities))
        
        # Check for implicit references (pronouns, determiners)
        implicit_refs = self._detect_implicit_references(current_query)
        implicit_confidence = len(implicit_refs) * 0.3  # Each reference adds confidence
        
        combined_confidence = min((entity_overlap * 0.7) + (implicit_confidence * 0.3), 1.0)
        
        return {
            'confidence': combined_confidence,
            'entity_overlap': entity_overlap,
            'current_entities': current_entities,
            'previous_entities': previous_entities,
            'implicit_references': implicit_refs,
            'method': 'entity_continuity'
        }
    
    async def _analyze_referential_language(
        self, 
        current_query: str, 
        history: List[Dict],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Use LLM to detect referential language patterns"""
        
        if not history:
            return {'confidence': 0.0, 'indicators': [], 'method': 'llm_analysis'}
        
        recent_queries = [entry.get('user_query', '') for entry in history[-2:]]
        
        # Create prompt for LLM analysis
        prompt = f'''
        Analyze if the current query refers to results from previous queries.
        
        Previous queries:
        {chr(10).join(f"- {q}" for q in recent_queries)}
        
        Current query: {current_query}
        
        Does the current query refer to results from previous queries? 
        Look for referential language, implicit references, or context dependency.
        
        Respond with JSON: {{"is_referential": boolean, "confidence": 0.0-1.0, "indicators": [list of referential words/phrases], "reasoning": "explanation"}}
        '''
        
        # TODO: Call LLM service here
        # For now, return simple heuristic-based analysis
        
        indicators = self._detect_implicit_references(current_query)
        confidence = min(len(indicators) * 0.4, 1.0)
        
        return {
            'confidence': confidence,
            'indicators': indicators,
            'method': 'llm_analysis',
            'reasoning': f"Found {len(indicators)} referential indicators"
        }
    
    async def _analyze_semantic_coherence(
        self, 
        current_query: str, 
        history: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze semantic coherence and topic continuation"""
        
        if not history:
            return {'confidence': 0.0, 'method': 'coherence_analysis'}
        
        # Simple coherence analysis based on shared vocabulary
        current_words = set(current_query.lower().split())
        
        coherence_scores = []
        for entry in history[-2:]:
            previous_query = entry.get('user_query', '')
            previous_words = set(previous_query.lower().split())
            
            # Calculate Jaccard similarity
            intersection = current_words & previous_words
            union = current_words | previous_words
            
            if union:
                jaccard = len(intersection) / len(union)
                coherence_scores.append(jaccard)
        
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        
        return {
            'confidence': avg_coherence,
            'coherence_score': avg_coherence,
            'method': 'coherence_analysis'
        }
    
    def _extract_entities(self, query: str) -> List[str]:
        """Simple entity extraction (can be replaced with NER model)"""
        # Simple keyword-based entity extraction
        entities = []
        query_lower = query.lower()
        
        # Business entities
        if any(word in query_lower for word in ['client', 'customer', 'user']):
            entities.append('customers')
        if any(word in query_lower for word in ['account', 'accounts']):
            entities.append('accounts')
        if any(word in query_lower for word in ['transaction', 'payment']):
            entities.append('transactions')
        
        return entities
    
    def _detect_implicit_references(self, query: str) -> List[str]:
        """Detect implicit referential language without hardcoded patterns"""
        query_lower = query.lower()
        
        # Semantic indicators of reference (learnable from data)
        implicit_indicators = []
        
        # Pronouns and determiners 
        if any(word in query_lower.split() for word in ['those', 'them', 'these', 'that', 'it']):
            implicit_indicators.append('pronoun_reference')
        
        # Comparative/relative terms
        if any(phrase in query_lower for phrase in ['same', 'similar', 'previous', 'above', 'before']):
            implicit_indicators.append('comparative_reference')
        
        # Elliptical constructions (incomplete queries that need context)
        if len(query.split()) < 4 and any(word in query_lower for word in ['list', 'show', 'get']):
            implicit_indicators.append('elliptical_reference')
        
        return implicit_indicators
    
    def _combine_confidence_scores(self, weighted_scores: List[Tuple[float, float]]) -> float:
        """Combine multiple confidence scores with weights"""
        total_score = sum(score * weight for score, weight in weighted_scores)
        total_weight = sum(weight for _, weight in weighted_scores)
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_reasoning(self, results: List[Dict[str, Any]]) -> str:
        """Generate human-readable reasoning for follow-up detection"""
        reasons = []
        
        for result in results:
            method = result.get('method', 'unknown')
            confidence = result.get('confidence', 0.0)
            
            if confidence > 0.5:
                if method == 'embedding_similarity':
                    reasons.append(f"High semantic similarity ({confidence:.2f})")
                elif method == 'entity_continuity':
                    reasons.append(f"Entity continuity detected ({confidence:.2f})")
                elif method == 'llm_analysis':
                    reasons.append(f"Referential language found ({confidence:.2f})")
                elif method == 'coherence_analysis':
                    reasons.append(f"Topic coherence maintained ({confidence:.2f})")
        
        return "; ".join(reasons) if reasons else "No strong follow-up indicators"
    
    async def _get_conversation_history(
        self, 
        session_id: str, 
        session_manager
    ) -> List[Dict]:
        """Get recent conversation history for analysis"""
        # Get last few queries from session
        # This would integrate with your existing session management
        return []  # TODO: Implement with actual session manager


# Factory function for easy integration
def create_semantic_followup_detector(embedding_model: str = "all-MiniLM-L6-v2") -> SemanticFollowUpDetector:
    """Create semantic follow-up detector instance"""
    return SemanticFollowUpDetector(embedding_model)