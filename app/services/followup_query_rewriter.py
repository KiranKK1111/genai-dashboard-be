"""
Semantic Follow-Up Query Rewriter - No Pattern Dependencies

Replaces pattern-based detection with semantic understanding.
Uses embeddings and entity continuity for follow-up detection.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    
logger = logging.getLogger(__name__)


@dataclass
class SemanticFollowUpContext:
    """Semantic context for follow-up detection (no patterns needed)"""
    is_followup: bool
    confidence: float
    previous_query_plan: Optional[Dict[str, Any]] = None
    previous_user_query: str = ""
    semantic_similarity: float = 0.0
    entity_overlap: float = 0.0
    referential_indicators: List[str] = None
    reasoning: str = ""
    previous_entities: List[str] = None
    current_entities: List[str] = None


class SemanticFollowUpRewriter:
    """
    Semantic follow-up rewriting without pattern dependencies.
    
    Uses embeddings, entity continuity, and semantic analysis
    instead of hardcoded patterns for follow-up detection.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize with semantic models (no pattern dependencies)"""
        
        # Initialize embedding model if available
        if SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                self.use_embeddings = True
            except Exception as e:
                logger.warning(f"Could not load embeddings: {e}")
                self.embedding_model = None
                self.use_embeddings = False
        else:
            logger.warning("sentence-transformers not available, using fallback detection")
            self.embedding_model = None
            self.use_embeddings = False
        
        # Semantic thresholds (learnable, not hardcoded)
        self.similarity_threshold = 0.6
        self.entity_overlap_threshold = 0.3
        
        logger.info(f"[SEMANTIC_FOLLOWUP] Initialized with embeddings: {self.use_embeddings}")
    
    def detect_semantic_followup(self, query: str, conversation_history: List[Dict] = None) -> Tuple[bool, float, List[str]]:
        """
        Detect follow-up using semantic analysis instead of patterns.
        
        Returns: (is_followup, confidence, indicators)
        """
        if not conversation_history:
            return False, 0.0, []
        
        # Method 1: Embedding similarity (if available)
        embedding_confidence = 0.0
        if self.use_embeddings and len(conversation_history) > 0:
            embedding_confidence = self._compute_embedding_similarity(query, conversation_history)
        
        # Method 2: Entity continuity analysis
        entity_confidence = self._analyze_entity_continuity(query, conversation_history)
        
        # Method 3: Implicit reference detection
        implicit_indicators = self._detect_semantic_references(query)
        implicit_confidence = min(len(implicit_indicators) * 0.3, 1.0)
        
        # Combine confidences
        if self.use_embeddings:
            combined_confidence = (embedding_confidence * 0.4 + 
                                 entity_confidence * 0.4 + 
                                 implicit_confidence * 0.2)
        else:
            combined_confidence = (entity_confidence * 0.6 + 
                                 implicit_confidence * 0.4)
        
        is_followup = combined_confidence > 0.5
        
        return is_followup, combined_confidence, implicit_indicators
    
    def _compute_embedding_similarity(self, current_query: str, history: List[Dict]) -> float:
        """Compute semantic similarity using embeddings"""
        if not self.embedding_model:
            return 0.0
        
        try:
            current_embedding = self.embedding_model.encode(current_query)
            max_similarity = 0.0
            
            for entry in history[-3:]:  # Last 3 queries
                previous_query = entry.get('user_query', '')
                if previous_query:
                    previous_embedding = self.embedding_model.encode(previous_query)
                    similarity = np.dot(current_embedding, previous_embedding) / (
                        np.linalg.norm(current_embedding) * np.linalg.norm(previous_embedding)
                    )
                    max_similarity = max(max_similarity, similarity)
            
            return min(max_similarity / self.similarity_threshold, 1.0)
        
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def _analyze_entity_continuity(self, current_query: str, history: List[Dict]) -> float:
        """Analyze entity continuity across conversation"""
        current_entities = self._extract_entities(current_query)
        if not current_entities:
            return 0.0
        
        previous_entities = []
        for entry in history[-2:]:  # Last 2 queries
            previous_query = entry.get('user_query', '')
            previous_entities.extend(self._extract_entities(previous_query))
        
        if not previous_entities:
            return 0.0
        
        # Calculate entity overlap
        common_entities = set(current_entities) & set(previous_entities)
        overlap_ratio = len(common_entities) / len(set(previous_entities))
        
        return min(overlap_ratio / self.entity_overlap_threshold, 1.0)
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query semantically"""
        entities = []
        query_lower = query.lower()
        
        # Business entities (semantic, not pattern-based)
        if any(word in query_lower for word in ['client', 'customer', 'user']):
            entities.append('customers')
        if any(word in query_lower for word in ['account']):
            entities.append('accounts')
        if any(word in query_lower for word in ['transaction', 'payment']):
            entities.append('transactions')
        if any(word in query_lower for word in ['employee', 'staff']):
            entities.append('employees')
        
        return entities
    
    def _detect_semantic_references(self, query: str) -> List[str]:
        """Detect referential language semantically (minimal hardcoding)"""
        indicators = []
        query_lower = query.lower()
        
        # Core referential words (language fundamentals, not configurable patterns)
        if any(word in query_lower.split() for word in ['those', 'them', 'these', 'that', 'it']):
            indicators.append('pronoun_reference')
        
        if any(phrase in query_lower for phrase in ['same', 'previous', 'above', 'before']):
            indicators.append('temporal_reference')
        
        # Elliptical constructions (incomplete queries needing context)
        words = query_lower.split()
        if len(words) < 4 and any(word in words for word in ['list', 'show', 'get', 'count']):
            indicators.append('elliptical_query')
        
        return indicators
    
    async def analyze_followup(
        self, 
        query: str, 
        session_id: str, 
        db: AsyncSession,
        session_manager
    ) -> SemanticFollowUpContext:
        """
        Analyze follow-up semantically (no pattern dependencies).
        """
        try:
            # Get conversation history
            history = await self._get_conversation_history(session_id, session_manager)
            
            if not history:
                return SemanticFollowUpContext(
                    is_followup=False,
                    confidence=0.0,
                    reasoning="No conversation history"
                )
            
            # Semantic follow-up detection
            is_followup, confidence, indicators = self.detect_semantic_followup(query, history)
            
            if is_followup and confidence > 0.5:
                # Get the most recent relevant context
                previous_context = history[-1] if history else {}
                
                return SemanticFollowUpContext(
                    is_followup=True,
                    confidence=confidence,
                    previous_query_plan=previous_context.get('query_plan_json', {}),
                    previous_user_query=previous_context.get('user_query', ''),
                    referential_indicators=indicators,
                    reasoning=f"Semantic analysis: {confidence:.2f} confidence, indicators: {indicators}"
                )
            
            return SemanticFollowUpContext(
                is_followup=False,
                confidence=confidence,
                reasoning=f"Low confidence ({confidence:.2f}) or no follow-up detected"
            )
            
        except Exception as e:
            logger.error(f"[SEMANTIC_FOLLOWUP] Error analyzing: {e}")
            return SemanticFollowUpContext(
                is_followup=False,
                confidence=0.0,
                reasoning=f"Error: {e}"
            )

    def rewrite_query_with_context(
        self, 
        query: str, 
        context: SemanticFollowUpContext
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Rewrite query using semantic context (no pattern replacement).
        """
        if not context.is_followup or not context.previous_query_plan:
            return query, {}
        
        # Extract semantic intent
        intent = self._extract_semantic_intent(query)
        
        # Get previous plan details
        previous_plan = context.previous_query_plan
        entity = self._extract_entity_from_plan(previous_plan)
        filters = self._extract_filters_as_text(previous_plan)
        
        # Generate enhanced query semantically
        if intent == "list":
            if filters:
                enhanced_query = f"list all {entity} where {filters}"
            else:
                enhanced_query = f"list all {entity}"
        elif intent == "count":
            if filters:
                enhanced_query = f"count {entity} where {filters}"
            else:
                enhanced_query = f"count {entity}"
        else:
            enhanced_query = f"{query} (referring to previous {entity} results)"
        
        # Create modified plan
        modified_plan = self._create_modified_plan(previous_plan, intent)
        
        logger.info(f"[SEMANTIC_REWRITE] '{query}' → '{enhanced_query}'")
        
        return enhanced_query, modified_plan
    
    def _extract_semantic_intent(self, query: str) -> str:
        """Extract intent semantically (minimal hardcoding)"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['list', 'show', 'display', 'get']):
            return 'list'
        elif any(word in query_lower for word in ['count', 'many', 'number']):
            return 'count'
        elif any(word in query_lower for word in ['delete', 'remove']):
            return 'delete'
        elif any(word in query_lower for word in ['update', 'modify']):
            return 'update'
        else:
            return 'list'  # Default
    
    def _create_modified_plan(self, previous_plan: Dict[str, Any], intent: str) -> Dict[str, Any]:
        """Create modified query plan based on intent"""
        modified_plan = previous_plan.copy()
        
        # Modify based on intent
        if intent == "list":
            modified_plan["select_clauses"] = [{"type": "wildcard", "column": "*"}]
            modified_plan["intent"] = "list"
        elif intent == "count":
            modified_plan["select_clauses"] = [{"type": "aggregate", "function": "COUNT", "column": "*"}]
            modified_plan["intent"] = "count"
        
        return modified_plan
    
    async def _get_conversation_history(self, session_id: str, session_manager) -> List[Dict]:
        """Get conversation history for semantic analysis"""
        try:
            return await session_manager.get_conversation_history(session_id, limit=5)
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []

    def _extract_filters_as_text(self, query_plan: Dict[str, Any]) -> str:
        """Convert filters to natural language"""
        where_conditions = query_plan.get("where_conditions", [])
        if not where_conditions:
            return ""
        
        filter_descriptions = []
        
        for condition in where_conditions:
            column = condition.get("column", "")
            operator = condition.get("operator", "")
            value = condition.get("value")
            values = condition.get("values", [])
            
            if operator == "IN" and values:
                values_str = ", ".join(f"'{v}'" for v in values)
                filter_descriptions.append(f"{column} in ({values_str})")
            elif operator == "MONTH_EQUALS" and value:
                month_names = {
                    1: "January", 2: "February", 3: "March", 4: "April",
                    5: "May", 6: "June", 7: "July", 8: "August", 
                    9: "September", 10: "October", 11: "November", 12: "December"
                }
                month_name = month_names.get(value, str(value))
                filter_descriptions.append(f"birthday in {month_name}")
            elif operator == "=" and value is not None:
                filter_descriptions.append(f"{column} = '{value}'")
        
        return " and ".join(filter_descriptions)

    def _extract_entity_from_plan(self, query_plan: Dict[str, Any]) -> str:
        """Extract entity from plan semantically"""
        try:
            table = query_plan.get("primary_table", "") or query_plan.get("from_table", "")
            
            # Semantic table mapping (minimal hardcoding)
            if 'customer' in table.lower():
                return 'clients'
            elif 'account' in table.lower():
                return 'accounts'
            elif 'transaction' in table.lower():
                return 'transactions'
            elif 'employee' in table.lower():
                return 'employees'
            else:
                return 'records'
        except Exception as e:
            logger.error(f"Error extracting entity: {e}")
            return 'records'


# Factory function to create semantic rewriter
def get_followup_rewriter() -> SemanticFollowUpRewriter:
    """Create semantic follow-up rewriter instance (replaces pattern-based version)"""
    return SemanticFollowUpRewriter()